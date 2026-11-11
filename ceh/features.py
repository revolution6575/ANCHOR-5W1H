import re
from typing import Optional

from . import timex
from .parsing import ParsedDoc
from .schema import ENTITY_SLOTS, Event, Span, overlaps

_CAUSAL = re.compile(
    r"因为|由于|所以|因此|因而|从而|导致|致使|引发|为了|以便|鉴于|"
    r"\bbecause\b|\bsince\b|\btherefore\b|\bthus\b|\bhence\b|\bdue to\b|"
    r"\bas a result\b|\bin order to\b|\bso that\b|\bled to\b|\bcaused\b|"
    r"\bporque\b|\bdebido a\b|\bpor lo tanto\b|\bya que\b|\bpara que\b|\bpor eso\b",
    re.I,
)

_MANNER_CUES = {
    "measure": r"措施|举措|办法|对策|\bmeasures?\b|\bsteps?\b|\bmedidas?\b",
    "method": r"通过|采用|采取|以.{0,4}方式|\bby\b|\bthrough\b|\bmethods?\b|"
    r"\bmediante\b|\ba través de\b|\bmétodos?\b",
    "instrument": r"使用|利用|借助|\busing\b|\bwith\b|\bvia\b|\busando\b|\bcon\b",
    "procedure": r"流程|程序|步骤|先.{0,20}再|\bprocedures?\b|\bprocess\b|"
    r"\bprocedimientos?\b|\bproceso\b",
    "mechanism": r"机制|制度|体制|\bmechanisms?\b|\bmecanismos?\b",
}
_MANNER = {k: re.compile(v, re.I) for k, v in _MANNER_CUES.items()}

_ARGUMENT_SLOTS = ("who", "when", "where")


def _saturate(count: int, full: int = 2) -> float:
    return min(1.0, count / float(full))


def position(sent_idx: int, n_sents: int) -> float:
    if n_sents <= 1:
        return 1.0
    return max(0.0, 1.0 - sent_idx / float(min(n_sents, 6)))


def trigger_mainness(parsed: ParsedDoc, span: Span) -> float:
    main = parsed.is_main_predicate(span)
    if main is None:
        return 0.5
    return 1.0 if main else 0.0


def coverage(event: Event) -> float:
    filled = sum(1 for role in ENTITY_SLOTS if event.slot_text(role))
    return filled / float(len(ENTITY_SLOTS))


def centrality(in_degree: int) -> float:
    return _saturate(in_degree)


def direction(
    target: Event, cand_event: Optional[Event], relation: str, mode: str = "explicit"
) -> float:
    if (
        mode == "explicit"
        and relation in ("direct_reason", "trigger_event")
        and cand_event is not None
    ):
        a = timex.normalize(cand_event.slot_text("when"))
        b = timex.normalize(target.slot_text("when"))
        if a and b and all(v is not None for v in a + b):
            from datetime import date

            try:
                return 1.0 if date(*a) <= date(*b) else 0.0
            except ValueError:
                pass
    return 0.5


def causal_cue(text: str, a: Span, b: Span) -> float:
    lo, hi = min(a[0], b[0]), max(a[1], b[1])
    return _saturate(len(_CAUSAL.findall(text[lo:hi])))


def distance(i: int, j: int, n_sents: int) -> float:
    if n_sents <= 1:
        return 0.0
    return min(1.0, abs(i - j) / float(n_sents))


def subordination(
    parsed: ParsedDoc,
    cand_span: Span,
    cand_head: Optional[Span],
    cand_sent: int,
    event: Event,
) -> float:
    evidence = {event.sent_idx}
    arguments = []
    for role in ENTITY_SLOTS:
        el = event.elements.get(role)
        if el is None or not el.is_span():
            continue
        evidence.add(el.sent_idx)
        if role in _ARGUMENT_SLOTS:
            arguments.append(el.span)
    if cand_sent not in evidence:
        return 0.0
    shares_argument = any(overlaps(cand_span, arg) for arg in arguments)
    governed = bool(parsed.governed_by(cand_head or cand_span, event.span))
    return 1.0 if shares_argument or governed else 0.0


def manner_affinity(text: str) -> float:
    return max(_saturate(len(p.findall(text))) for p in _MANNER.values())
