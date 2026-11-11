import hashlib, json
from dataclasses import asdict
from pathlib import Path
from ..parsing import ParsedDoc
from ..schema import Sentence, Token, Clause, Event, Element, Evidence


def filename(doc):
    return (
        hashlib.sha256((doc["key"] + "\0" + doc["text_sha256"]).encode()).hexdigest()
        + ".json"
    )


def parsed_from(data):
    return ParsedDoc(
        text=data["text"],
        lang=data["lang"],
        sentences=[Sentence(**x) for x in data["sentences"]],
        tokens=[Token(**x) for x in data["tokens"]],
        clauses=[Clause(**x) for x in data["clauses"]],
        diagnostics=data.get("diagnostics", {}),
    )


def parsed_dict(parsed):
    return {k: v for k, v in asdict(parsed).items() if k != "_by_start"}


def events_from(rows):
    events = []
    for data in rows:
        d = dict(data)
        d["elements"] = {
            r: Element(
                **{
                    **v,
                    "evidence": [Evidence(**item) for item in v.get("evidence", [])],
                }
            )
            for r, v in d["elements"].items()
        }
        events.append(Event(**d))
    return events


def load_parsed(root, doc):
    data = json.loads(
        (Path(root) / "parsed" / filename(doc)).read_text(encoding="utf-8")
    )
    assert hashlib.sha256(data["text"].encode()).hexdigest() == doc["text_sha256"]
    return parsed_from(data)


def load_s1(root, doc):
    data = json.loads((Path(root) / "s1" / filename(doc)).read_text(encoding="utf-8"))
    assert data["text_sha256"] == doc["text_sha256"]
    return data, events_from(data.get("events", []))
