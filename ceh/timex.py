import re
from typing import Callable, List, Optional, Tuple

Date = Tuple[Optional[int], Optional[int], Optional[int]]

_EN_MONTHS = {
    name: i + 1
    for i, name in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
    )
}
_EN_MONTHS.update({name[:3]: num for name, num in list(_EN_MONTHS.items())})
_EN_MONTHS["sept"] = 9

_ES_MONTHS = {
    name: i + 1
    for i, name in enumerate(
        [
            "enero",
            "febrero",
            "marzo",
            "abril",
            "mayo",
            "junio",
            "julio",
            "agosto",
            "septiembre",
            "octubre",
            "noviembre",
            "diciembre",
        ]
    )
}
_ES_MONTHS["setiembre"] = 9

_EN = "|".join(sorted(_EN_MONTHS, key=len, reverse=True))
_ES = "|".join(sorted(_ES_MONTHS, key=len, reverse=True))


def _en(name: str) -> int:
    return _EN_MONTHS[name.lower().rstrip(".")]


def _es(name: str) -> int:
    return _ES_MONTHS[name.lower()]


def _num(s: Optional[str]) -> Optional[int]:
    return int(s) if s else None


_PATTERNS: List[Tuple[re.Pattern, Callable[[re.Match], Date]]] = [
    (
        re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})"),
        lambda m: (int(m[1]), int(m[2]), int(m[3])),
    ),
    (
        re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]"),
        lambda m: (int(m[1]), int(m[2]), int(m[3])),
    ),
    (
        re.compile(r"\b(%s)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b" % _EN, re.I),
        lambda m: (int(m[3]), _en(m[1]), int(m[2])),
    ),
    (
        re.compile(r"\b(\d{1,2})\s+(%s)\.?,?\s+(\d{4})\b" % _EN, re.I),
        lambda m: (int(m[3]), _en(m[2]), int(m[1])),
    ),
    (
        re.compile(r"\b(\d{1,2})\s+de\s+(%s)(?:\s+del?\s+(\d{4}))?" % _ES, re.I),
        lambda m: (_num(m[3]), _es(m[2]), int(m[1])),
    ),
    (
        re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月"),
        lambda m: (int(m[1]), int(m[2]), None),
    ),
    (
        re.compile(r"\b(%s)\.?\s+(\d{4})\b" % _EN, re.I),
        lambda m: (int(m[2]), _en(m[1]), None),
    ),
    (
        re.compile(r"\b(%s)\s+del?\s+(\d{4})\b" % _ES, re.I),
        lambda m: (int(m[2]), _es(m[1]), None),
    ),
    (
        re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]"),
        lambda m: (None, int(m[1]), int(m[2])),
    ),
    (
        re.compile(r"\b(%s)\.?\s+(\d{1,2})\b" % _EN, re.I),
        lambda m: (None, _en(m[1]), int(m[2])),
    ),
    (re.compile(r"(\d{4})\s*年"), lambda m: (int(m[1]), None, None)),
    (re.compile(r"\b(1[89]\d{2}|20\d{2})\b"), lambda m: (int(m[1]), None, None)),
]


def normalize(text: str) -> Optional[Date]:
    if not text:
        return None
    for pattern, convert in _PATTERNS:
        m = pattern.search(text)
        if m:
            date = convert(m)
            if _valid(date):
                return date
    return None


def not_later(a: Date, b: Date) -> Optional[bool]:
    compared = False
    for x, y in zip(a, b):
        if x is None or y is None:
            break
        compared = True
        if x != y:
            return x < y
    return True if compared else None


def _valid(date: Date) -> bool:
    _, month, day = date
    if month is not None and not 1 <= month <= 12:
        return False
    if day is not None and not 1 <= day <= 31:
        return False
    return any(x is not None for x in date)
