from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

RELEASE = Path(
    os.environ.get(
        "ANCHOR_RELEASE_DIR", Path(__file__).resolve().parents[1] / "datasets"
    )
)
ELEMENTS = ("who", "what", "when", "where", "why", "how")


def catalog():
    with (Path(RELEASE) / "manifest.json").open(encoding="utf-8") as stream:
        return {item["name"]: item for item in json.load(stream)["datasets"]}


def records(name):
    item = catalog()[name]
    path = Path(RELEASE) / item["path"]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        try:
            import ijson
        except ImportError:
            yield from json.load(stream)["records"]
        else:
            yield from ijson.items(stream, "records.item", use_float=True)


def doc_lang(doc):
    return doc.lang


@dataclass
class Span:
    text: str
    start: int
    end: int

    def valid(self, doc_text: str) -> bool:
        return doc_text[self.start : self.end] == self.text


@dataclass
class Ev:

    id: str
    cluster_id: str
    level: str
    text: str
    trigger: Span
    roles: Dict[str, List[Span]] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


@dataclass
class Doc:
    id: str
    corpus: str
    tier: str
    lang: str
    text: str
    sentences: List[Span] = field(default_factory=list)
    events: List[Ev] = field(default_factory=list)
    gate: str = ""


def load_corpus(name: str, limit: Optional[int] = None) -> List[Doc]:
    item = catalog()[name]
    tier, lang = item["tier"], item["language"]
    out = []
    for rec in records(name):
        text = rec.get("Text") or ""
        nodes = {n.get("Node_Id"): n for n in rec.get("Nodes") or []}
        doc = Doc(
            id=rec.get("Id"),
            corpus=name,
            tier=tier,
            lang=lang,
            text=text,
            gate=(rec.get("Review") or {}).get("Gate", ""),
        )
        for s in rec.get("Sentences") or []:
            doc.sentences.append(
                Span(
                    s.get("Tag_Text") or "",
                    s.get("Tag_Start", -1),
                    s.get("Tag_End", -1),
                )
            )
        for cl in rec.get("Event_Clusters") or []:
            for e in cl.get("Events") or []:
                t = e.get("Trigger") or {}
                ev = Ev(
                    id=e.get("Event_Id"),
                    cluster_id=cl.get("Cluster_Id"),
                    level=e.get("Event_Level") or "supporting",
                    text=e.get("Event_Text") or "",
                    trigger=Span(
                        t.get("Tag_Text") or "",
                        t.get("Tag_Start", -1),
                        t.get("Tag_End", -1),
                    ),
                    missing=list(e.get("Missing") or []),
                    evidence=list(e.get("Evidence") or []),
                )
                slots = (e.get("Event_Hyperedge") or {}).get("Nodes") or {}
                for r in ELEMENTS:
                    ev.roles[r] = []
                    for nid in slots.get(r) or []:
                        n = nodes.get(nid)
                        if n:
                            ev.roles[r].append(
                                Span(
                                    n.get("Tag_Text") or "",
                                    n.get("Tag_Start", -1),
                                    n.get("Tag_End", -1),
                                )
                            )
                doc.events.append(ev)
        out.append(doc)
        if limit and len(out) >= limit:
            break
    return out
