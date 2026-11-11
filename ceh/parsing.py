import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .schema import Clause, Sentence, Span, Token, overlaps

logger = logging.getLogger(__name__)

SPACY_MODELS = {"zh": "zh_core_web_sm", "en": "en_core_web_sm", "es": "es_core_news_sm"}

_PREDICATE_POS = {"VERB", "AUX"}
_DEPENDENT_PREDICATE = ("aux", "cop")
_COORD_DEPS = {"conj", "parataxis"}

MARKERS = frozenset(
    {
        "由于",
        "因为",
        "因",
        "鉴于",
        "为了",
        "because",
        "due",
        "owing",
        "thanks",
        "porque",
        "debido",
        "gracias",
        "por",
        "通过",
        "经过",
        "凭借",
        "借助",
        "依靠",
        "以",
        "by",
        "through",
        "via",
        "with",
        "mediante",
        "través",
    }
)
_AGENT_DEPS = {"agent"}
_ATTACHED_MARKER_DEPS = ("case", "mark", "fixed")

_CORE_DEPS = {
    "dobj",
    "obj",
    "iobj",
    "ccomp",
    "xcomp",
    "attr",
    "dative",
    "prt",
    "acomp",
    "oprd",
    "aux:asp",
    "range",
    "nmod:range",
}

_SENT_END = re.compile(r"(?<=[。！？!?；;])|(?<=[.!?])(?=\s+[A-Z“\"])")
_CLAUSE_SEP = re.compile(
    r"[，,；;、]|\band\b|\bbut\b|\bbecause\b|\bso\b|"
    r"因为|由于|所以|从而|并且|然后|以便|为了"
)
_EDGE_CHARS = " \t\r\n，,。.；;：:！!？?、\"'“”‘’()（）"
_MIN_CLAUSE_LEN = 3

_models: Dict[str, object] = {}


def load_spacy(lang: str):
    if lang not in _models:
        name = SPACY_MODELS.get(lang, SPACY_MODELS["en"])
        try:
            import spacy

            _models[lang] = spacy.load(name, disable=["ner", "lemmatizer"])
        except Exception as exc:
            logger.warning("spaCy 模型 %s 加载失败，退化为标点切分：%s", name, exc)
            _models[lang] = None
    return _models[lang]


@dataclass
class ParsedDoc:

    text: str
    lang: str
    sentences: List[Sentence]
    tokens: List[Token] = field(default_factory=list)
    clauses: List[Clause] = field(default_factory=list)
    diagnostics: Dict[str, int] = field(default_factory=dict)
    _by_start: Dict[int, Token] = field(default_factory=dict, repr=False)

    @property
    def has_syntax(self) -> bool:
        return bool(self.tokens)

    @property
    def has_complete_syntax(self) -> bool:
        return bool(self.sentences) and {t.sent_idx for t in self.tokens} == {
            s.idx for s in self.sentences
        }

    def token_at(self, offset: int) -> Optional[Token]:
        if not self._by_start and self.tokens:
            self._by_start.update((t.start, t) for t in self.tokens)
        return self._by_start.get(offset)

    def sentence_of(self, offset: int) -> int:
        for s in self.sentences:
            if s.start <= offset < s.end:
                return s.idx
        return -1

    def tokens_in(self, span: Span) -> List[Token]:
        return [t for t in self.tokens if overlaps((t.start, t.end), span)]

    def head_token(self, span: Span) -> Optional[Token]:
        inside = self.tokens_in(span)
        ids = {t.idx for t in inside}
        for t in inside:
            if t.head == t.idx or t.head not in ids:
                return t
        return None

    def is_main_predicate(self, span: Span) -> Optional[bool]:
        if not self.has_syntax:
            return None
        for t in self.tokens_in(span):
            if t.dep == "ROOT":
                return True
            if t.dep in _COORD_DEPS and self.tokens[t.head].dep == "ROOT":
                return True
        return False

    def governed_by(self, span: Span, governor: Span) -> Optional[bool]:
        if not self.has_syntax:
            return None
        tok = self.head_token(span)
        if tok is None:
            return False
        visited = set()
        while tok.head != tok.idx and tok.idx not in visited:
            visited.add(tok.idx)
            tok = self.tokens[tok.head]
            if overlaps((tok.start, tok.end), governor):
                return True
        return False


def split_sentences(text: str) -> List[Sentence]:
    sentences, pos = [], 0
    for piece in _SENT_END.split(text):
        body = piece.strip()
        if body:
            start = pos + len(piece) - len(piece.lstrip())
            sentences.append(
                Sentence(
                    idx=len(sentences), text=body, start=start, end=start + len(body)
                )
            )
        pos += len(piece)
    return sentences


def parse_document(text: str, lang: str) -> ParsedDoc:
    doc = ParsedDoc(text=text, lang=lang, sentences=split_sentences(text))
    nlp = load_spacy(lang)
    if nlp is not None:
        _attach_syntax(doc, nlp)
    if doc.has_syntax:
        doc.clauses = _syntactic_clauses(doc)
    if not doc.clauses:
        doc.clauses = _punctuation_clauses(doc)
    return doc


def clause_around(doc: ParsedDoc, span: Span, cid: int) -> Clause:
    if doc.has_syntax:
        tok = doc.head_token(span)
        if tok is not None:
            start, end = _with_preposition(doc, tok, *_subtree_span(doc, tok.idx))
            start, end = _trim(doc.text, start, end)
            return Clause(
                cid=cid,
                text=doc.text[start:end],
                sent_idx=tok.sent_idx,
                start=start,
                end=end,
                head_start=tok.start,
                head_end=tok.end,
                pred_lemma=tok.text,
                marker=_leading_marker(doc, start),
            )

    sid = doc.sentence_of(span[0])
    sent = doc.sentences[sid] if sid >= 0 else None
    start, end = span
    if sent is not None:
        for a, b in _punctuation_pieces(sent):
            if a <= span[0] < b:
                start, end = a, max(b, span[1])
                break
    start, end = _trim(doc.text, start, end)
    return Clause(
        cid=cid,
        text=doc.text[start:end],
        sent_idx=sid if sid >= 0 else 0,
        start=start,
        end=end,
        head_start=span[0],
        head_end=span[1],
        pred_lemma=doc.text[span[0] : span[1]],
    )




def _attach_syntax(doc: ParsedDoc, nlp) -> None:
    for s in doc.sentences:
        try:
            parsed = nlp(s.text)
        except Exception as exc:
            logger.debug("句子 %d 依存分析失败：%s", s.idx, exc)
            continue
        base = len(doc.tokens)
        for tok in parsed:
            start = s.start + tok.idx
            doc.tokens.append(
                Token(
                    idx=base + tok.i,
                    text=tok.text,
                    start=start,
                    end=start + len(tok.text),
                    pos=tok.pos_,
                    dep=tok.dep_,
                    head=base + tok.head.i,
                    sent_idx=s.idx,
                )
            )


def _children(doc: ParsedDoc) -> Dict[int, List[int]]:
    kids: Dict[int, List[int]] = defaultdict(list)
    for t in doc.tokens:
        if t.head != t.idx:
            kids[t.head].append(t.idx)
    return kids


def _subtree_span(
    doc: ParsedDoc, root: int, kids: Optional[Dict[int, List[int]]] = None
) -> Tuple[int, int]:
    kids = kids if kids is not None else _children(doc)
    start, end = doc.tokens[root].start, doc.tokens[root].end
    stack = [root]
    while stack:
        i = stack.pop()
        start, end = min(start, doc.tokens[i].start), max(end, doc.tokens[i].end)
        stack.extend(kids.get(i, ()))
    return start, end


def _syntactic_clauses(doc: ParsedDoc) -> List[Clause]:
    kids = _children(doc)
    seen, clauses = set(), []

    def add(head: Token, start: int, end: int, kind: str) -> None:
        start, end = _trim(doc.text, start, end)
        if end - start < _MIN_CLAUSE_LEN or (start, end) in seen:
            return
        seen.add((start, end))
        clauses.append(
            Clause(
                cid=len(clauses),
                text=doc.text[start:end],
                sent_idx=head.sent_idx,
                start=start,
                end=end,
                head_start=head.start,
                head_end=head.end,
                pred_lemma=head.text,
                kind=kind,
                marker=_leading_marker(doc, start),
            )
        )

    predicates = [
        t
        for t in doc.tokens
        if t.pos in _PREDICATE_POS and not t.dep.startswith(_DEPENDENT_PREDICATE)
    ]
    for t in predicates:
        add(t, *_with_preposition(doc, t, *_subtree_span(doc, t.idx, kids)), "clause")
    for t in doc.tokens:
        phrase = _marker_phrase(doc, t, kids)
        if phrase is not None:
            add(*phrase, "marker")
    for t in predicates:
        core = _predicate_core(doc, t, kids)
        if core is not None:
            add(t, *core, "core")
    return clauses


def _with_preposition(
    doc: ParsedDoc, tok: Token, start: int, end: int
) -> Tuple[int, int]:
    head = doc.tokens[tok.head]
    if (
        head.idx != tok.idx
        and head.pos in ("ADP", "SCONJ")
        and head.dep not in _AGENT_DEPS
        and head.end <= start
        and not doc.text[head.end : start].strip()
    ):
        return head.start, end
    return start, end


def _marker_phrase(
    doc: ParsedDoc, tok: Token, kids: Dict[int, List[int]]
) -> Optional[Tuple[Token, int, int]]:
    if tok.text.lower() not in MARKERS or tok.dep in _AGENT_DEPS:
        return None
    if tok.dep.startswith(_ATTACHED_MARKER_DEPS):
        head = doc.tokens[tok.head]
        if head.idx == tok.idx or head.dep == "ROOT":
            return None
        start, end = _subtree_span(doc, head.idx, kids)
        return head, max(start, tok.start), end
    if kids.get(tok.idx):
        start, end = _subtree_span(doc, tok.idx, kids)
        return tok, start, end
    return None


def _predicate_core(
    doc: ParsedDoc, tok: Token, kids: Dict[int, List[int]]
) -> Optional[Tuple[int, int]]:
    children = [doc.tokens[i] for i in kids.get(tok.idx, ())]
    right = [c for c in children if c.start > tok.start and c.dep in _CORE_DEPS]
    if not right:
        return None
    core = (tok.start, max(_subtree_span(doc, c.idx, kids)[1] for c in right))
    if core == _subtree_span(doc, tok.idx, kids):
        return None
    return core


def _leading_marker(doc: ParsedDoc, start: int) -> str:
    tok = doc.token_at(start)
    return tok.text if tok is not None and tok.text.lower() in MARKERS else ""


def _punctuation_pieces(sent: Sentence) -> List[Tuple[int, int]]:
    pieces, last = [], 0
    for m in _CLAUSE_SEP.finditer(sent.text):
        if m.start() > last:
            pieces.append((sent.start + last, sent.start + m.start()))
        last = m.end()
    if last < len(sent.text):
        pieces.append((sent.start + last, sent.end))
    return pieces or [(sent.start, sent.end)]


def _punctuation_clauses(doc: ParsedDoc) -> List[Clause]:
    clauses = []
    for s in doc.sentences:
        for a, b in _punctuation_pieces(s):
            start, end = _trim(doc.text, a, b)
            if end - start < _MIN_CLAUSE_LEN:
                continue
            clauses.append(
                Clause(
                    cid=len(clauses),
                    text=doc.text[start:end],
                    sent_idx=s.idx,
                    start=start,
                    end=end,
                )
            )
    return clauses


def _trim(text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and text[start] in _EDGE_CHARS:
        start += 1
    while end > start and text[end - 1] in _EDGE_CHARS:
        end -= 1
    return start, end
