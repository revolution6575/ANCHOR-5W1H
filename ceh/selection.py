from collections import defaultdict
from dataclasses import dataclass
from typing import List
from .schema import (
    ERROR,
    NO_EVIDENCE,
    SPAN,
    EDGE_ROLES,
    Element,
    Evidence,
    FORM_ANOTHER_EVENT,
    FORM_SAME_SENT,
    FORM_CROSS_SENT,
    grounded,
)
from .scoring import PairScore


@dataclass
class Choice:
    pair: PairScore
    role: str
    score: float
    gain: float


def span_valid(c, parsed):
    sid = parsed.sentence_of(c.start)
    return (
        grounded(c.text, parsed.text, c.start, c.end)
        and sid >= 0
        and sid == c.sent_idx
        and c.end <= parsed.sentences[sid].end
    )


def _choice_key(ch):
    c = ch.pair.clause
    return (-ch.score, c.end - c.start, c.start, c.cid, ch.role)


def _path_key(path):
    return (
        len(path),
        tuple(
            (
                c.pair.clause.unit_id,
                c.role,
                c.pair.clause.end - c.pair.clause.start,
                c.pair.clause.start,
                c.pair.clause.cid,
            )
            for c in path
        ),
    )


def _better(new, old):
    if old is None:
        return True
    if abs(new[0] - old[0]) > 1e-12:
        return new[0] > old[0]
    return _path_key(new[1]) < _path_key(old[1])


def solve(groups, capacities, exclusive=True) -> List[Choice]:
    kw, kh = capacities["why"], capacities["how"]
    states = {(0, 0): (0.0, [])}
    for unit in sorted(groups):
        best = {}
        for ch in groups[unit]:
            if ch.gain <= 0:
                continue
            if ch.role not in best or _choice_key(ch) < _choice_key(best[ch.role]):
                best[ch.role] = ch
        options = [[], *([best[r]] for r in EDGE_ROLES if r in best)]
        if not exclusive and len(best) == 2:
            options.append([best["why"], best["how"]])
        updated = {}
        for (nw, nh), (value, path) in states.items():
            for option in options:
                counts = (
                    nw + sum(c.role == "why" for c in option),
                    nh + sum(c.role == "how" for c in option),
                )
                if counts[0] > kw or counts[1] > kh:
                    continue
                proposal = value + sum(c.gain for c in option), path + option
                if _better(proposal, updated.get(counts)):
                    updated[counts] = proposal
        states = updated
    result = None
    for value in states.values():
        if _better(value, result):
            result = value
    return result[1] if result is not None else []


def select_evidence(events, table, parsed, cfg):
    sc = cfg["selection"]
    report = {}
    for ev in events:
        ev.ensure_elements()
        pairs = table.of_event(ev.eid)
        groups = defaultdict(list)
        diag = {
            r: {
                "candidates": len(pairs),
                "invalid_responses": 0,
                "invalid_span": 0,
                "self_reference": 0,
                "no_relation": 0,
                "below_threshold": 0,
                "eligible": 0,
                "certified_unqueried": 0,
                "queried": 0,
            }
            for r in EDGE_ROLES
        }
        for pair in pairs:
            for role in EDGE_ROLES:
                d = diag[role]
                if pair.query_status.get(role) in (
                    "certified_skipped",
                    "certified_inadmissible",
                ):
                    d["certified_unqueried"] += 1
                    continue
                if pair.query_status.get(role) == "pending":
                    raise ValueError("Uncertified pending role reached final selection")
                d["queried"] += 1
                if not getattr(pair, "valid_" + role):
                    d["invalid_responses"] += 1
                    continue
                if not span_valid(pair.clause, parsed):
                    d["invalid_span"] += 1
                    continue
                if pair.self_ref or pair.clause.source_event_id == ev.eid:
                    d["self_reference"] += 1
                    continue
                if role == "why" and pair.rel_type == "none":
                    d["no_relation"] += 1
                    continue
                score = getattr(pair, role)
                gain = score - sc["thresholds"][role]
                if gain <= 0:
                    d["below_threshold"] += 1
                    continue
                d["eligible"] += 1
                groups[pair.clause.unit_id].append(Choice(pair, role, score, gain))
        chosen = solve(groups, sc["max_fillers"], sc["exclusive_roles"])
        for role in EDGE_ROLES:
            picks = sorted((ch for ch in chosen if ch.role == role), key=_choice_key)
            d = diag[role]
            d["selected"] = len(picks)
            d["incomplete"] = d["invalid_responses"] > 0
            if not picks:
                failed = bool(pairs) and d["invalid_responses"] == len(pairs)
                reason = (
                    "all_role_responses_invalid"
                    if failed
                    else "no_candidates"
                    if not pairs
                    else "budget_or_role_competition"
                    if d["eligible"]
                    else "certified_no_accepted_evidence"
                    if d["certified_unqueried"]
                    else "no_accepted_evidence"
                )
                ev.elements[role] = Element(
                    state=ERROR if failed else NO_EVIDENCE, reason=reason, diagnostics=d
                )
                continue
            evidences = [_evidence(ev, ch) for ch in picks]
            first = evidences[0]
            ev.elements[role] = Element(
                state=SPAN,
                text=first.text,
                start=first.start,
                end=first.end,
                sent_idx=first.sent_idx,
                score=first.score,
                form=first.form,
                rel_type=first.rel_type,
                values=[x.text for x in evidences],
                evidence=evidences,
                reason="selected",
                diagnostics=d,
            )
        report[str(ev.eid)] = {
            "objective": round(sum(ch.gain for ch in chosen), 8),
            "roles": diag,
        }
    return report


def _evidence(event, choice):
    c = choice.pair.clause
    form = (
        FORM_ANOTHER_EVENT
        if c.source_event_id >= 0
        else FORM_SAME_SENT
        if c.sent_idx == event.sent_idx
        else FORM_CROSS_SENT
    )
    return Evidence(
        c.text,
        c.start,
        c.end,
        c.sent_idx,
        c.cid,
        c.unit_id,
        c.source_event_id,
        choice.score,
        round(choice.gain, 8),
        form,
        choice.pair.rel_type if choice.role == "why" else "",
        dict(choice.pair.feats),
    )
