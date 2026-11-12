# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from S-FLM's `sandbox_gsm8k.py` (https://github.com/jdeschena/s-flm),
# Copyright (c) the S-FLM authors, licensed under the Apache License, Version 2.0,
# which is itself adapted from https://github.com/JaeyeonKim01/PUMA. Modified.

"""Score GSM8K by executing generated Python code.

TinyGSM-style completions define ``def simple_math_problem() -> int: ... return result``;
a problem is scored by calling that function and comparing its return value with
the gold answer. Extraction and scoring follow the S-FLM reference implementation
(copied, not imported) so that scores are comparable with the ones it reports.

Minimal sandboxing, since generated code is exec'd:

- `__builtins__` is replaced by an allow-list; only `import math` is permitted
- a SIGALRM wall-clock timeout (effective only in the main thread)
- stdout / stderr are discarded (only the return value is scored; stdout is read
  only when the return value is not a number)
"""

from __future__ import annotations

import contextlib
import io
import math
import re
import signal
import warnings

__all__ = ["extract_code", "run_code", "to_number", "gold_number",
           "numbers_equal", "score_code", "MAX_ANSWER_DIGITS"]

#: Per-problem time limit for generated code, in seconds.
DEFAULT_TIMEOUT_S = 2.0

#: Function name every TinyGSM program defines (as in S-FLM).
ENTRY_POINT = "simple_math_problem"


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):  # noqa: ARG001
    raise _Timeout()


@contextlib.contextmanager
def _time_limit(timeout_s: float):
    """Wall-clock limit via SIGALRM. Only effective in the main thread."""
    has_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    old = None
    if has_alarm:
        old = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)


_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    "range": range, "enumerate": enumerate, "int": int, "float": float,
    "str": str, "bool": bool, "round": round, "print": print,
    "sorted": sorted, "zip": zip, "list": list, "dict": dict, "set": set,
    "tuple": tuple, "divmod": divmod, "pow": pow, "any": any, "all": all,
}


def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
    if name == "math":
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import blocked: {name}")


def extract_code(text: str) -> str:
    """Extract the part of a completion that parses as Python (S-FLM's heuristics).

    1. If there is a ```` ```python ... ``` ```` block, take its contents
    2. Drop any prose before the first `def `
    3. If trailing garbage causes a SyntaxError, keep the lines before the failing one
    """
    fence = re.search(r"```(?:python)?\s*(.*?)```", text,
                      flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    i = text.find("def ")
    if i != -1:
        text = text[i:]
    text = text.strip()
    if not text:
        return text
    try:
        compile(text, "<sample>", "exec")
        return text
    except SyntaxError as err:
        if err.lineno is None or err.lineno <= 1:
            return text
        lineno = err.lineno
    except (ValueError, MemoryError, RecursionError):
        return text
    lines = text.splitlines()
    candidate = "\n".join(lines[: lineno - 1]).strip()
    if not candidate:
        return text
    try:
        compile(candidate, "<sample>", "exec")
        return candidate
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return text


def run_code(code: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Exec `code` in a restricted namespace and call `ENTRY_POINT`.

    Returns `{"compiled", "has_entry", "ran", "value", "stdout", "error"}`.
    All exceptions are swallowed (generated code often fails); the flags record
    the stage reached, so "not valid code" and "ran but wrong answer" can be
    reported separately.
    """
    out = {"compiled": False, "has_entry": False, "ran": False,
           "value": None, "stdout": "", "error": ""}
    if not code.strip():
        out["error"] = "empty"
        return out
    try:
        compile(code, "<sample>", "exec")
        out["compiled"] = True
    except Exception as e:  # noqa: BLE001 - generated code may raise anything
        out["error"] = f"compile: {type(e).__name__}"
        return out

    ns = {"__builtins__": {**_SAFE_BUILTINS, "__import__": _limited_import},
          "math": math}
    stdout = io.StringIO()
    try:
        with _time_limit(timeout_s):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    exec(code, ns, ns)  # noqa: S102 - intentional sandboxed exec
                fn = ns.get(ENTRY_POINT)
                if fn is None:
                    out["error"] = "no entry point"
                    return out
                out["has_entry"] = True
                with contextlib.redirect_stdout(stdout), \
                     contextlib.redirect_stderr(io.StringIO()):
                    out["value"] = fn()
                out["ran"] = True
    except _Timeout:
        out["error"] = "timeout"
    except BaseException as e:  # noqa: BLE001 - also catches SystemExit etc.
        out["error"] = f"run: {type(e).__name__}"
    out["stdout"] = stdout.getvalue()[:4096]
    return out


_NUM_RE = re.compile(r"[-+]?\$?\d[\d,]*\.?\d*")


#: Maximum number of digits accepted for an integer answer. GSM8K answers have
#: at most about 7 digits, so anything longer is wrong anyway. The cap also guards
#: against Python 3.11+ refusing int -> str beyond 4,300 digits: if generated code
#: returns e.g. `10**99999`, `str(...)` raises `ValueError: Exceeds the limit
#: (4300 digits)`, which would kill one rank and leave the others hanging in NCCL.
MAX_ANSWER_DIGITS = 100


def _too_big(n: int) -> bool:
    """Check the digit count without calling `str()`, which may itself raise."""
    return n.bit_length() > int(MAX_ANSWER_DIGITS * 3.33) + 8


def to_number(x):
    """Normalize a return value to int / float, or None if not possible.

    Integers longer than `MAX_ANSWER_DIGITS` map to None.
    """
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, int):
        return None if _too_big(x) else int(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        if abs(x - round(x)) < 1e-6:
            r = int(round(x))
            return None if _too_big(r) else r
        return float(x)
    if isinstance(x, str):
        m = _NUM_RE.search(x)
        if not m:
            return None
        s = m.group(0).replace(",", "").replace("$", "")
        if len(s) > MAX_ANSWER_DIGITS + 2:
            return None
        try:
            if s.count(".") == 1:
                f = float(s)
                return int(round(f)) if abs(f - round(f)) < 1e-6 else f
            return int(s)
        except ValueError:
            return None
    return None  # tuples / lists etc. are not scored


def gold_number(answer: str):
    """Extract the `#### <number>` from a GSM8K `answer`."""
    m = re.search(r"####\s*(.+)", answer or "")
    return to_number(m.group(1) if m else answer)


def last_number(text: str):
    """Last number in `text` (lenient fallback when the code does not run)."""
    ms = _NUM_RE.findall(text or "")
    return to_number(ms[-1]) if ms else None


def numbers_equal(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    if isinstance(pred, float) or isinstance(gold, float):
        return abs(float(pred) - float(gold)) <= 1e-3
    return int(pred) == int(gold)


def score_code(completion: str, gold_answer: str,
               timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Score one problem. `strict` is the metric comparable with S-FLM.

    - `strict`: the executed return value (or the number in stdout if the value
      is not a number) equals gold
    - `flexible`: `strict`, or the last number in the completion equals gold
      (a lenient control)
    """
    code = extract_code(completion)
    res = run_code(code, timeout_s=timeout_s)
    value = to_number(res["value"])
    if value is None and res["ran"]:
        value = last_number(res["stdout"])
    gold = gold_number(gold_answer)
    strict = numbers_equal(value, gold)
    return {
        "code": code,
        "compiled": bool(res["compiled"]),
        "has_entry": bool(res["has_entry"]),
        "ran": bool(res["ran"]),
        "error": res["error"],
        "pred_value": value,
        "gold_value": gold,
        "strict": strict,
        "flexible": strict or numbers_equal(last_number(completion), gold),
    }
