import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from . import features, prompts, languages
from .parsing import ParsedDoc
from .schema import Clause, Event
from bench.roles import role_defs


@dataclass
class PairScore:
    event_id: int
    clause: Clause
    why: Optional[float]
    how: Optional[float]
    p_llm: Optional[float]
    q_llm: Optional[float]
    rel_type: Optional[str]
    self_ref: bool
    feats: Dict[str, Optional[float]] = field(default_factory=dict)
    valid_why: Optional[bool] = True
    valid_how: Optional[bool] = True
    errors: Dict[str, str] = field(default_factory=dict)
    query_status: Dict[str, str] = field(
        default_factory=lambda: {"why": "evaluated", "how": "evaluated"}
    )
    upper_bounds: Dict[str, float] = field(default_factory=dict)


class ScoreTable:
    def __init__(self):
        self._pairs: Dict[Tuple[int, int], PairScore] = {}
        self.audit = {}

    def __len__(self):
        return len(self._pairs)

    def add(self, pair: PairScore):
        key = pair.event_id, pair.clause.cid
        if key in self._pairs:
            raise ValueError("Duplicate (event, candidate) identifier")
        self._pairs[key] = pair

    def of_event(self, event_id: int) -> List[PairScore]:
        return [p for (eid, _), p in self._pairs.items() if eid == event_id]

    def pairs(self):
        return list(self._pairs.values())


def _priority(c):
    return 0 if c.marker else (2 if c.kind == "core" else 1)


def candidate_lists(events: List[Event], clauses: List[Clause], cfg: dict):
    cc = cfg["candidates"]
    table = {}
    for ev in events:
        units = defaultdict(list)
        for c in clauses:
            if c.source_event_id == ev.eid:
                continue
            if (
                c.source_event_id < 0
                and c.kind == "core"
                and c.sent_idx != ev.sent_idx
                and not cc["cross_sentence_core"]
            ):
                continue
            units[c.unit_id].append(c)
        key = lambda c: (abs(c.sent_idx - ev.sent_idx), _priority(c), c.start, c.cid)
        event_units = [u for u, cs in units.items() if cs[0].source_event_id >= 0]
        ordinary = [u for u in units if u not in event_units]
        event_units.sort(key=lambda u: (min(key(c) for c in units[u]), u))
        ordinary.sort(key=lambda u: (min(key(c) for c in units[u]), u))
        selected = event_units + ordinary[: cc["max_non_event_units"]]
        chosen = []
        for u in selected:
            variants = sorted(units[u], key=key)
            chosen.extend(variants[: cc["max_variants_per_unit"]])
        table[ev.eid] = chosen
    return table


def event_anchor(ev: Event, parsed: ParsedDoc, full: bool) -> str:
    anchor = {"event_id": ev.eid, "trigger": ev.trigger, "sentence": ev.sent_idx}
    if full:
        anchor.update(
            {role: ev.slot_text(role) for role in ("who", "what", "when", "where")}
        )
        anchor["evidence_sentence"] = parsed.sentences[ev.sent_idx].text
    return json.dumps(anchor, ensure_ascii=False)


def role_prompt(role: str, ev: Event, c: Clause, events, parsed, cfg):
    full = cfg["scoring"]["full_event_context"]
    others = "\n".join(event_anchor(e, parsed, True) for e in events if e.eid != ev.eid)
    return languages.template(role, parsed.lang).format(
        definition=role_defs(parsed.lang)[role],
        cand=c.text,
        si=c.sent_idx,
        target=event_anchor(ev, parsed, full),
        others=(others or languages.EMPTY_CONTEXT[parsed.lang][0])
        if full
        else languages.EMPTY_CONTEXT[parsed.lang][1],
        ctx=prompts.context(
            parsed.sentences, c.sent_idx, ev.sent_idx, pad=cfg["input"]["context_pad"]
        ),
    )


def parse_support(reply, key: str):
    if not isinstance(reply, dict):
        return 0.0, False, "missing_or_non_object_response"
    value = reply.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0, False, "support_must_be_number"
    if not math.isfinite(value) or not 0 <= value <= 1:
        return 0.0, False, "support_out_of_range"
    if key == "p" and reply.get("relation") not in prompts.WHY_RELATIONS:
        return float(value), False, "invalid_relation_type"
    return float(value), True, ""


def score_candidates(events, clauses, parsed, llm, cfg) -> ScoreTable:
    if cfg["scoring"].get("evaluation", "exhaustive") == "certified":
        from .certified import score_certified

        return score_certified(events, clauses, parsed, llm, cfg)
    table = score_exhaustive(events, clauses, parsed, llm, cfg)
    table.audit = {
        "mode": "exhaustive",
        "queried_roles": 2 * len(table),
        "certified_skipped_roles": 0,
        "rounds": 1 if len(table) else 0,
    }
    return table


def score_exhaustive(events, clauses, parsed, llm, cfg) -> ScoreTable:
    lists = candidate_lists(events, clauses, cfg)
    jobs = [(ev, c) for ev in events for c in lists[ev.eid]]
    table = ScoreTable()
    if not jobs:
        return table
    requests = [role_prompt("why", ev, c, events, parsed, cfg) for ev, c in jobs]
    requests += [role_prompt("how", ev, c, events, parsed, cfg) for ev, c in jobs]
    labels = [
        {"stage": "S2", "event_id": ev.eid, "candidate_id": c.cid, "role": role}
        for role in ("why", "how")
        for ev, c in jobs
    ]
    contextual = getattr(llm, "json_chat_many_context", None)
    replies = (
        contextual(
            requests, languages.system("role", parsed.lang), labels, default=None
        )
        if contextual
        else llm.json_chat_many(
            requests, languages.system("role", parsed.lang), default=None
        )
    )
    if len(replies) != len(requests):
        raise ValueError("S2 response count mismatch")
    sc = cfg["scoring"]
    w, v = sc["why_weights"], sc["how_weights"]
    by_id = {e.eid: e for e in events}
    for i, (ev, c) in enumerate(jobs):
        wr, hr = replies[i], replies[i + len(jobs)]
        p, wp, we = parse_support(wr, "p")
        q, hp, he = parse_support(hr, "q")
        relation = wr["relation"] if wp else "none"
        cue_target = c.span if sc["cue_scope"] == "candidate" else ev.span
        feats = {
            "dir": features.direction(
                ev, by_id.get(c.source_event_id), relation, sc["temporal"]
            ),
            "cue": features.causal_cue(parsed.text, c.span, cue_target),
            "dist": features.distance(c.sent_idx, ev.sent_idx, len(parsed.sentences)),
            "sub": features.subordination(parsed, c.span, c.head_span, c.sent_idx, ev),
            "sty": features.manner_affinity(c.text),
        }
        if sc["typed"]:
            sw = (
                w["llm"] * p
                + w["dir"] * feats["dir"]
                + w["cue"] * feats["cue"]
                - w["dist"] * feats["dist"]
            )
            sh = (
                v["llm"] * q
                + v["sub"] * feats["sub"]
                + v["sty"] * feats["sty"]
                - v["cue"] * feats["cue"]
            )
        else:
            shared = v["sub"] * feats["sub"] - w["dist"] * feats["dist"]
            sw, sh = w["llm"] * p + shared, w["llm"] * q + shared
        table.add(
            PairScore(
                ev.eid,
                c,
                round(sw, 4),
                round(sh, 4),
                p,
                q,
                relation,
                c.source_event_id == ev.eid,
                feats,
                wp,
                hp,
                {r: err for r, err in (("why", we), ("how", he)) if err},
                query_status={
                    "why": "evaluated" if wp else "invalid",
                    "how": "evaluated" if hp else "invalid",
                },
            )
        )
    return table
