from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

VERSION = "anchor-c00-ordered-4w-v1"
ELEMENTS = ["who", "what", "when", "where", "why", "how"]
ENTITY_SLOTS = ELEMENTS[:4]
EDGE_ROLES = ["why", "how"]
SPAN, NO_EVIDENCE, ERROR = "SPAN", "NO_EVIDENCE", "ERROR"
MISSING, DEFER = "MISSING", "DEFER"
WHY, HOW = "why", "how"
FORM_SAME_SENT, FORM_CROSS_SENT = "same_sent", "cross_sent"
FORM_ANOTHER_EVENT, FORM_NP = "another_event", "np"
Span = Tuple[int, int]


def overlaps(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def grounded(text: str, doc: str, start: int, end: int) -> bool:
    return bool(text) and 0 <= start < end <= len(doc) and doc[start:end] == text


@dataclass
class Sentence:
    idx: int
    text: str
    start: int
    end: int


@dataclass
class Token:
    idx: int
    text: str
    start: int
    end: int
    pos: str
    dep: str
    head: int
    sent_idx: int


@dataclass
class Clause:
    cid: int
    text: str
    sent_idx: int
    start: int
    end: int
    head_start: int = -1
    head_end: int = -1
    pred_lemma: str = ""
    kind: str = "clause"
    marker: str = ""
    unit_id: str = ""
    source_event_id: int = -1

    @property
    def span(self) -> Span:
        return self.start, self.end

    @property
    def head_span(self) -> Optional[Span]:
        return None if self.head_start < 0 else (self.head_start, self.head_end)


@dataclass
class Evidence:
    text: str
    start: int
    end: int
    sent_idx: int
    candidate_id: int
    unit_id: str
    source_event_id: int
    score: float
    gain: float
    form: str
    rel_type: str = ""
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class Element:
    state: str = NO_EVIDENCE
    text: str = ""
    sent_idx: int = -1
    start: int = -1
    end: int = -1
    score: float = 0.0
    form: str = ""
    rel_type: str = ""
    values: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    reason: str = "not_extracted"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def span(self) -> Span:
        return self.start, self.end

    def is_span(self) -> bool:
        return self.state == SPAN and bool(self.text or self.values)


@dataclass
class Event:
    eid: int
    trigger: str
    sent_idx: int
    start: int
    end: int
    ev_score: float = 0.0
    is_central: bool = False
    features: Dict[str, float] = field(default_factory=dict)
    elements: Dict[str, Element] = field(default_factory=dict)

    @property
    def span(self) -> Span:
        return self.start, self.end

    def ensure_elements(self) -> None:
        for role in ELEMENTS:
            self.elements.setdefault(role, Element())

    def slot_text(self, role: str) -> str:
        el = self.elements.get(role)
        return el.text if el is not None and el.is_span() else ""


@dataclass
class DocResult:
    doc_id: str
    lang: str
    system: str = "ours"
    schema_version: str = VERSION
    status: str = "ok"
    events: List[Event] = field(default_factory=list)
    n_llm_calls: int = 0
    elapsed_s: float = 0.0
    error: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    def central(self) -> Optional[Event]:
        return next(
            (ev for ev in self.events if ev.is_central),
            self.events[0] if self.events else None,
        )
