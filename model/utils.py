"""
utils.py  –  shared helpers for parsing, vocabulary, and encoding.

Date is generated as yyyy-mm-dd (year first) so the model predicts
the most-constrained part (decade/leap year → year digits) first,
then month, then day.
"""

import re
from datetime import date as dt_date
from typing import Optional


# ── constants ─────────────────────────────────────────────────────────────────

DAYS   = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

CONDITION_TOKENS = (
    DAYS
    + MONTHS
    + ["True", "False"]
    + [str(d) for d in range(180, 221)]
)

DATE_TOKENS = [str(i) for i in range(10)] + ["-"]

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
SPECIAL    = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]

ALL_TOKENS = SPECIAL + CONDITION_TOKENS + DATE_TOKENS


def build_vocab() -> tuple[dict[str, int], dict[int, str]]:
    char2idx = {tok: i for i, tok in enumerate(ALL_TOKENS)}
    idx2char  = {i: tok for tok, i in char2idx.items()}
    return char2idx, idx2char


PAD_IDX = ALL_TOKENS.index(PAD_TOKEN)
SOS_IDX = ALL_TOKENS.index(SOS_TOKEN)
EOS_IDX = ALL_TOKENS.index(EOS_TOKEN)

LINE_RE = re.compile(
    r'\[(\w+)\]\s+\[(\w+)\]\s+\[(\w+)\]\s+\[(\w+)\](?:\s+(\S+))?'
)


def parse_line(line: str) -> tuple[list[str], Optional[str]]:
    m = LINE_RE.match(line.strip())
    if not m:
        raise ValueError(f"Cannot parse line: {line!r}")
    day, month, leap, decade, date = m.groups()
    return [day, month, leap, decade], date


def date_to_tokens(date_str: str) -> list[str]:
    """dd-mm-yyyy → yyyy-mm-dd token list (year first for better learning)."""
    parts = date_str.split("-")
    if len(parts) != 3:
        return list(date_str)
    dd, mm, yyyy = parts[0], parts[1], parts[2]
    return list(yyyy) + ["-"] + list(mm) + ["-"] + list(dd)


def tokens_to_date(tokens: list[str]) -> str:
    """yyyy-mm-dd token list → dd-mm-yyyy string."""
    clean  = [t for t in tokens if t not in SPECIAL]
    joined = "".join(clean)
    parts  = joined.split("-")
    if len(parts) == 3:
        yyyy, mm, dd = parts[0], parts[1], parts[2]
        return f"{dd}-{mm}-{yyyy}"
    return joined


# ── validation ────────────────────────────────────────────────────────────────

_MONTH_MAP = {m: i + 1 for i, m in enumerate(MONTHS)}
_DAY_MAP   = {d: i for i, d in enumerate(DAYS)}


def is_valid_date(date_str: str) -> bool:
    try:
        parts = date_str.split("-")
        if len(parts) != 3:
            return False
        dd, mm, yyyy = int(parts[0]), int(parts[1]), int(parts[2])
        if not (1 <= dd <= 31 and 1 <= mm <= 12 and 1800 <= yyyy <= 2200):
            return False
        dt_date(yyyy, mm, dd)
        return True
    except (ValueError, IndexError, OverflowError):
        return False


def check_conditions(cond_tokens: list[str], date_str: str) -> dict[str, bool]:
    if not is_valid_date(date_str):
        return {"day": False, "month": False, "leap": False, "decade": False, "valid": False}
    day_tok, month_tok, leap_tok, decade_tok = cond_tokens
    parts = date_str.split("-")
    dd, mm, yyyy = int(parts[0]), int(parts[1]), int(parts[2])
    d = dt_date(yyyy, mm, dd)
    is_leap = (yyyy % 4 == 0 and (yyyy % 100 != 0 or yyyy % 400 == 0))
    return {
        "day":    d.weekday() == _DAY_MAP[day_tok],
        "month":  mm == _MONTH_MAP[month_tok],
        "leap":   is_leap == (leap_tok == "True"),
        "decade": yyyy // 10 == int(decade_tok),
        "valid":  True,
    }


def all_conditions_pass(cond_tokens: list[str], date_str: str) -> bool:
    return all(check_conditions(cond_tokens, date_str).values())