from typing import Any, Dict, List, Tuple

from . import features, prompts, languages
from .entities import fill_entities
from .llm import LLM
from .parsing import ParsedDoc, clause_around
from .schema import Clause, Event, overlaps


def build_cluster(parsed: ParsedDoc, llm: LLM, cfg: dict) -> List[Event]:
    if cfg.get("s1_factorial", {}).get("enabled", False):
        from .s1_factorial import run_s1

        return run_s1(parsed, llm, cfg)
    cc = cfg["cluster"]
    max_events = cc["max_events"] if cc.get("enabled", True) else 1

    if hasattr(llm, "context"):
        llm.context["stage"] = "S1_trigger"
    proposals, relations = _propose(
        parsed, llm, max_events + 2, cfg["input"]["max_listed_sentences"]
    )
    events = _locate(proposals, parsed)
    parsed.diagnostics["proposed_events"] = len(proposals)
    parsed.diagnostics["grounded_proposals"] = len(events)
    parsed.diagnostics["unlocated_triggers"] = sum(
        _find(parsed, p["text"].strip(), _as_int(p.get("sent"), 0)) < 0
        for p in proposals
    )
    if proposals and not events:
        raise ValueError("S1 proposed triggers could not be grounded")
    if not events:
        return []

    if hasattr(llm, "context"):
        llm.context["stage"] = "S1_entity"
    fill_entities(events, parsed, llm, cfg)
    in_degree = _in_degree(relations, events)
    weights = cc["ev_weights"]
    for ev in events:
        ev.features = {
            "pos": features.position(ev.sent_idx, len(parsed.sentences)),
            "trig": features.trigger_mainness(parsed, ev.span),
            "cov": features.coverage(ev),
            "cent": features.centrality(in_degree.get(ev.eid, 0)),
        }
        ev.ev_score = round(sum(weights[k] * v for k, v in ev.features.items()), 4)

    events.sort(key=lambda e: -e.ev_score)
    admitted = [e for e in events if e.ev_score >= cc["admit_threshold"]][:max_events]
    if not admitted:
        admitted = events[:1]
    for i, ev in enumerate(admitted):
        ev.eid = i
        ev.is_central = i == 0
    return admitted


def mark_candidate_pool(
    parsed: ParsedDoc, events: List[Event], cfg: dict
) -> List[Clause]:
    from dataclasses import replace

    clauses, seen = [], set()
    for candidate in parsed.clauses:
        if candidate.span not in seen:
            clauses.append(replace(candidate))
            seen.add(candidate.span)
    for ev in events:
        if not any(_headed_by(c, ev) for c in clauses):
            next_cid = max((c.cid for c in clauses), default=-1) + 1
            owner = clause_around(parsed, ev.span, cid=next_cid)
            if owner.span not in seen:
                clauses.append(owner)
                seen.add(owner.span)
    for c in clauses:
        owners = [ev for ev in events if _headed_by(c, ev)]
        c.source_event_id = owners[0].eid if len(owners) == 1 else -1
        if not cfg["candidates"]["group_variants"]:
            c.unit_id = "span:%d:%d" % c.span
        elif c.source_event_id >= 0:
            c.unit_id = "event:%d" % c.source_event_id
        elif c.head_span is not None and c.kind in ("clause", "core"):
            c.unit_id = "predicate:%d:%d" % c.head_span
        else:
            c.unit_id = "span:%d:%d" % c.span
    return clauses




def _propose(
    parsed: ParsedDoc, llm: LLM, k: int, limit: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    reply = llm.json_chat(
        languages.template("trigger", parsed.lang).format(
            k=k,
            sents=prompts.numbered(
                parsed.sentences, limit=limit or len(parsed.sentences)
            ),
        ),
        languages.system("trigger", parsed.lang),
        default=None,
    )
    if (
        not isinstance(reply, dict)
        or not isinstance(reply.get("triggers"), list)
        or not isinstance(reply.get("relations"), list)
    ):
        raise ValueError("S1 trigger response failed or has invalid schema")
    triggers = reply["triggers"][:k]
    if any(
        not isinstance(t, dict)
        or not isinstance(t.get("text"), str)
        or not t["text"].strip()
        for t in triggers
    ):
        raise ValueError("S1 trigger list contains malformed proposals")
    relations = [r for r in reply["relations"] if isinstance(r, dict)]
    return triggers, relations


def _locate(proposals: List[Dict[str, Any]], parsed: ParsedDoc) -> List[Event]:
    events, taken = [], []
    for i, prop in enumerate(proposals):
        trigger = str(prop.get("text") or "").strip()
        if not trigger:
            continue
        start = _find(parsed, trigger, _as_int(prop.get("sent"), 0))
        if start < 0:
            continue
        span = (start, start + len(trigger))
        if any(overlaps(span, t) for t in taken):
            continue
        taken.append(span)
        events.append(
            Event(
                eid=i,
                trigger=trigger,
                sent_idx=parsed.sentence_of(start),
                start=span[0],
                end=span[1],
            )
        )
    return events


def _find(parsed: ParsedDoc, trigger: str, sent_idx: int) -> int:
    if 0 <= sent_idx < len(parsed.sentences):
        sent = parsed.sentences[sent_idx]
        pos = parsed.text.find(trigger, sent.start, sent.end)
        if pos >= 0:
            return pos
    return parsed.text.find(trigger)


def _in_degree(relations: List[Dict[str, Any]], events: List[Event]) -> Dict[int, int]:
    alive = {ev.eid for ev in events}
    degree: Dict[int, int] = {}
    seen = set()
    for rel in relations:
        src, dst = _as_int(rel.get("source"), -1), _as_int(rel.get("target"), -1)
        if (
            rel.get("type") in prompts.RELATION_TYPES
            and src != dst
            and src in alive
            and dst in alive
            and (src, dst) not in seen
        ):
            degree[dst] = degree.get(dst, 0) + 1
            seen.add((src, dst))
    return degree


def _headed_by(clause: Clause, event: Event) -> bool:
    head = clause.head_span
    if head is not None:
        return overlaps(head, event.span)
    return clause.start <= event.start < clause.end


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
