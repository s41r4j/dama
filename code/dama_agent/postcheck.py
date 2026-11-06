"""Deterministic check of the reader's CALC line (prompt P2 in render.py; docs/research/DAMA_v3_research_and_architecture.md
§5.10). No model, no I/O.

P2 makes the reader end its working with one CALC line: COUNT (over its ITEM: lines), SUM(...), DATEDIFF(d1, d2, unit),
plain arithmetic, LATEST(#id) or NONE. Python recomputes it, and if the first number in the ANSWER line disagrees, that
number is replaced. LATEST and NONE never change the answer (a line id cannot be mapped back to a value), nor does an
abstention. Day counts within one day are left alone: "how many days" is often counted inclusively, and LongMemEval's
judge accepts off-by-one day counts. The policy is an ablation arm on dev before it is frozen.
"""
from __future__ import annotations
import ast
import operator
import re
from dataclasses import dataclass
from datetime import date

NUM_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_NUM = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?!\w|[.,]\d)|\b(" + "|".join(NUM_WORDS) + r")\b", re.I)
_MONTH_AFTER = re.compile(r"\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)
_ISO = r"(\d{4}-\d{2}-\d{2})"
_ABSTAIN = re.compile(r"not mentioned|not available|no information|does not (?:say|mention)", re.I)


@dataclass
class PostcheckResult:
    answer: str
    changed: bool
    reason: str
    kind: str = "NONE"
    value: float | None = None


def _line_body(line: str, tag: str) -> str | None:
    # also "4. ANSWER: ..." (P2's numbered steps) and "2. RESOLVE: CALC: ..." (a step label before the tag)
    m = re.match(r"^[\s*#>\-|]*(?:\d+[.)]\s*)?(?:\**[A-Z]+\**\s*:\s*)?\**" + tag + r"\s*\**\s*:\s*\**\s*(.*)$", line, re.I)
    return m.group(1).strip().strip("*").strip() if m else None


def last_line(output: str, tag: str) -> str | None:
    found = [b for line in (output or "").splitlines() if (b := _line_body(line, tag)) is not None]
    return found[-1] if found else None


def items(output: str) -> list[str]:
    """ITEM: lines, de-duplicated after dropping line refs, parentheses, punctuation and leading articles."""
    seen, out = set(), []
    for line in (output or "").splitlines():
        b = _line_body(line, "ITEM")
        if b is None:
            continue
        key = re.sub(r"\([^)]*\)|#\d+", " ", b.lower())
        key = re.sub(r"^(the|a|an)\s+", "", re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", key)).strip())
        if key and key not in seen:
            seen.add(key); out.append(b)
    return out


# ----------------------------------------------------------------------------------------------- CALC
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    """+ - * / and parentheses over number literals; anything else raises ValueError (no names, calls or powers)."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError(f"not allowed: {type(n).__name__}")
    try:
        return float(ev(ast.parse(expr.replace("×", "*").replace("÷", "/"), mode="eval")))
    except (SyntaxError, ZeroDivisionError, RecursionError) as e:
        raise ValueError(str(e)) from e


def date_diff(d1: str, d2: str, unit: str) -> float:
    """Unsigned difference. days exact; weeks = days / 7; months and years in whole calendar units when the day of the
    month (and month) line up, else months = days / 30.4375 and years = completed years (as for an age)."""
    a, b = sorted((date.fromisoformat(d1), date.fromisoformat(d2)))
    days = (b - a).days
    unit = unit.lower().rstrip("s")
    if unit == "day":
        return float(days)
    if unit == "week":
        return days / 7
    if unit == "month":
        m = (b.year - a.year) * 12 + b.month - a.month
        return float(m) if b.day == a.day else days / 30.4375
    if unit == "year":
        return float(b.year - a.year - ((b.month, b.day) < (a.month, a.day)))
    raise ValueError(f"unknown unit {unit!r}")


def parse_calc(line: str | None):
    """(kind, args): NONE | COUNT (int or None) | SUM (list) | DATEDIFF ((d1, d2, unit)) | LATEST (line id) | ARITH (expr)."""
    s = (line or "").strip().rstrip(".").strip()
    if not s or s.upper().startswith("NONE"):
        return "NONE", None
    if m := re.fullmatch(r"COUNT\s*(?:\(\s*(\d+)\s*\))?(?:\s*=\s*(\d+))?", s, re.I):
        n = m.group(1) or m.group(2)
        return "COUNT", int(n) if n else None
    if m := re.fullmatch(r"SUM\s*\((.*)\)(?:\s*=.*)?", s, re.I):
        nums = [re.search(r"-?\d+(?:\.\d+)?", x) for x in m.group(1).split(",")]
        return ("SUM", [float(n.group()) for n in nums]) if nums and all(nums) else ("NONE", None)
    if m := re.fullmatch(r"DATEDIFF\s*\(\s*" + _ISO + r"\s*,\s*" + _ISO + r"\s*,\s*(\w+)\s*\)(?:\s*=.*)?", s, re.I):
        return "DATEDIFF", (m.group(1), m.group(2), m.group(3))
    if m := re.fullmatch(r"LATEST\s*\(\s*#?\s*(\d+)\s*\)", s, re.I):
        return "LATEST", int(m.group(1))
    expr = s.split("=")[0].strip()
    if re.fullmatch(r"[\d\s.+\-*/()×÷]+", expr) and re.search(r"\d", expr):
        return "ARITH", expr
    return "NONE", None


# ----------------------------------------------------------------------------------------------- answer numbers
def _value(m: re.Match) -> float:
    return float(NUM_WORDS[m.group(2).lower()]) if m.group(2) else float(m.group(1).replace(",", ""))


def first_number(answer: str) -> re.Match | None:
    """The first number in the answer that is not a year (1900-2100) or the day of a date ("7 May 2023")."""
    for m in _NUM.finditer(answer):
        v = _value(m)
        if m.group(1) and v == int(v) and 1900 <= v <= 2100:
            continue
        if _MONTH_AFTER.match(answer, m.end()):
            continue
        return m
    return None


def fmt(v: float) -> str:
    return str(int(round(v))) if abs(v - round(v)) < 0.05 else f"{v:.1f}"


def postcheck(output: str) -> PostcheckResult:
    answer = last_line(output, "ANSWER")
    if answer is None:
        return PostcheckResult("", False, "no_answer")
    kind, args = parse_calc(last_line(output, "CALC"))
    if kind in ("NONE", "LATEST"):
        return PostcheckResult(answer, False, kind.lower(), kind)
    try:
        if kind == "COUNT":
            found = items(output)
            if not found and args is None:
                return PostcheckResult(answer, False, "count_without_items", kind)
            v = float(len(found) if found else args)
        elif kind == "SUM":
            v = float(sum(args))
        elif kind == "DATEDIFF":
            v = date_diff(*args)
        else:
            v = safe_eval(args)
    except ValueError as e:
        return PostcheckResult(answer, False, f"calc_error: {e}", kind)
    if _ABSTAIN.search(answer):
        return PostcheckResult(answer, False, "abstained", kind, v)
    m = first_number(answer)
    if m is None:
        return PostcheckResult(answer, False, "no_number", kind, v)
    n = _value(m)
    unit = args[2].lower().rstrip("s") if kind == "DATEDIFF" else ""
    tol = 1.0 if unit == "day" else 0.5 if unit else 0.05
    if abs(n - v) <= tol or m.group() == fmt(v):
        return PostcheckResult(answer, False, "consistent", kind, v)
    return PostcheckResult(answer[:m.start()] + fmt(v) + answer[m.end():], True, f"{kind.lower()}_mismatch", kind, v)
