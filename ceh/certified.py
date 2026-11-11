import math
from collections import defaultdict
from . import features, languages, prompts
from .scoring import PairScore, ScoreTable, candidate_lists, parse_support, role_prompt
from .selection import Choice, solve, span_valid

ROLES = ("why", "how")


def upper_bound(role, ev, c, events_by_id, feats, cfg):
    sc = cfg["scoring"]
    w, v = sc["why_weights"], sc["how_weights"]
    if not sc["typed"]:
        value = w["llm"] + v["sub"] * feats["sub"] - w["dist"] * feats["dist"]
    elif role == "why":
        direction = max(
            features.direction(
                ev, events_by_id.get(c.source_event_id), r, sc["temporal"]
            )
            for r in prompts.WHY_RELATIONS
            if r != "none"
        )
        value = (
            w["llm"]
            + w["dir"] * direction
            + w["cue"] * feats["cue"]
            - w["dist"] * feats["dist"]
        )
    else:
        value = (
            v["llm"]
            + v["sub"] * feats["sub"]
            + v["sty"] * feats["sty"]
            - v["cue"] * feats["cue"]
        )
    return max(round(value, 4), math.ceil(value * 10000) / 10000)


def _known(pair, role):
    return pair.query_status[role] != "pending"


def optimistic_choice(pairs, cfg, optimistic=True):
    groups = defaultdict(list)
    for p in pairs:
        for role in ROLES:
            state = p.query_status[role]
            if state == "pending":
                if not optimistic:
                    continue
                score = p.upper_bounds[role]
            elif state != "evaluated" or not getattr(p, "valid_" + role):
                continue
            elif role == "why" and p.rel_type == "none":
                continue
            else:
                score = getattr(p, role)
            groups[p.clause.unit_id].append(
                Choice(p, role, score, score - cfg["selection"]["thresholds"][role])
            )
    return solve(
        groups, cfg["selection"]["max_fillers"], cfg["selection"]["exclusive_roles"]
    )


def path_signature(choices):
    return [(c.pair.event_id, c.pair.clause.cid, c.role) for c in choices]


def score_certified(events, clauses, parsed, llm, cfg):
    table = ScoreTable()
    by_id = {e.eid: e for e in events}
    lists = candidate_lists(events, clauses, cfg)
    for ev in events:
        for c in lists[ev.eid]:
            cue_target = (
                c.span if cfg["scoring"]["cue_scope"] == "candidate" else ev.span
            )
            f = {
                "dir": None,
                "cue": features.causal_cue(parsed.text, c.span, cue_target),
                "dist": features.distance(
                    c.sent_idx, ev.sent_idx, len(parsed.sentences)
                ),
                "sub": features.subordination(
                    parsed, c.span, c.head_span, c.sent_idx, ev
                ),
                "sty": features.manner_affinity(c.text),
            }
            p = PairScore(
                ev.eid,
                c,
                None,
                None,
                None,
                None,
                None,
                c.source_event_id == ev.eid,
                f,
                None,
                None,
                query_status={r: "pending" for r in ROLES},
            )
            p.upper_bounds = {r: upper_bound(r, ev, c, by_id, f, cfg) for r in ROLES}
            if p.self_ref or not span_valid(c, parsed):
                p.query_status = {r: "certified_inadmissible" for r in ROLES}
            table.add(p)
    rounds = queried = 0
    certificates = {}
    window = cfg["scoring"].get("query_window", 8)
    while True:
        frontier = []
        remaining = []
        for ev in events:
            if str(ev.eid) in certificates:
                continue
            pairs = table.of_event(ev.eid)
            best = optimistic_choice(pairs, cfg)
            unknown = [
                (ch.pair, ch.role) for ch in best if not _known(ch.pair, ch.role)
            ]
            if not unknown:
                lower = optimistic_choice(pairs, cfg, optimistic=False)
                if path_signature(lower) != path_signature(best):
                    raise AssertionError(
                        "Optimistic certificate disagrees with feasible tie-breaking"
                    )
                certificates[str(ev.eid)] = {
                    "round": rounds,
                    "objective": round(sum(c.gain for c in best), 8),
                    "selected": path_signature(best),
                    "all_selected_roles_observed": True,
                }
                for p in pairs:
                    for r in ROLES:
                        if p.query_status[r] == "pending":
                            p.query_status[r] = "certified_skipped"
                continue
            frontier.extend(unknown)
            frontier_keys = {(p.event_id, p.clause.cid, r) for p, r in unknown}
            remaining.extend(
                (p, r)
                for p in pairs
                for r in ROLES
                if p.query_status[r] == "pending"
                and (p.event_id, p.clause.cid, r) not in frontier_keys
            )
        if not frontier:
            break

        def priority(item):
            p, r = item
            c = p.clause
            return (
                -(p.upper_bounds[r] - cfg["selection"]["thresholds"][r]),
                c.end - c.start,
                c.start,
                c.cid,
                r,
                p.event_id,
            )

        jobs = (sorted(frontier, key=priority) + sorted(remaining, key=priority))[
            :window
        ]
        requests = [
            role_prompt(r, by_id[p.event_id], p.clause, events, parsed, cfg)
            for p, r in jobs
        ]
        labels = [
            {
                "stage": "S2",
                "event_id": p.event_id,
                "candidate_id": p.clause.cid,
                "role": r,
            }
            for p, r in jobs
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
        if len(replies) != len(jobs):
            raise ValueError("S2 response count mismatch")
        rounds += 1
        queried += len(jobs)
        sc = cfg["scoring"]
        w, v = sc["why_weights"], sc["how_weights"]
        for (p, role), reply in zip(jobs, replies):
            support, valid, error = parse_support(reply, "p" if role == "why" else "q")
            p.query_status[role] = "evaluated" if valid else "invalid"
            setattr(p, "valid_" + role, valid)
            setattr(p, "p_llm" if role == "why" else "q_llm", support)
            if error:
                p.errors[role] = error
            f = p.feats
            if role == "why":
                p.rel_type = reply["relation"] if valid else "none"
                f["dir"] = features.direction(
                    by_id[p.event_id],
                    by_id.get(p.clause.source_event_id),
                    p.rel_type,
                    sc["temporal"],
                )
                value = (
                    w["llm"] * support
                    + w["dir"] * f["dir"]
                    + w["cue"] * f["cue"]
                    - w["dist"] * f["dist"]
                )
            else:
                value = (
                    v["llm"] * support
                    + v["sub"] * f["sub"]
                    + v["sty"] * f["sty"]
                    - v["cue"] * f["cue"]
                )
            if not sc["typed"]:
                value = w["llm"] * support + v["sub"] * f["sub"] - w["dist"] * f["dist"]
            value = round(value, 4)
            setattr(p, role, value)
            if valid and value > p.upper_bounds[role] + 1e-12:
                raise AssertionError("Observed score exceeds conservative bound")
    skipped = sum(
        p.query_status[r].startswith("certified_") for p in table.pairs() for r in ROLES
    )
    table.audit = {
        "mode": "certified",
        "queried_roles": queried,
        "certified_skipped_roles": skipped,
        "rounds": rounds,
        "query_window": window,
        "certificates": certificates,
    }
    return table
