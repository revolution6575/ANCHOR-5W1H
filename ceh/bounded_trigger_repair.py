import copy, json
from .cluster import _find, _locate, _as_int
from .trigger_grounding_repair import locate_literal, INSTRUCTION, OUTPUT
from . import languages

VERSION = "c00-remaining-partial-trigger-v1"


def correction_request(parsed, ps, invalid, previous, llm):
    capacity = max(s["safe_tokens"] for s in llm.pool.services)
    sids = sorted({s.idx for s in parsed.sentences})
    claimed = [_as_int(ps[i].get("sent"), 0) for i in invalid]
    ordered = sorted(sids, key=lambda sid: (min(abs(sid - c) for c in claimed), sid))
    tokens = [t for t in parsed.tokens if parsed.text[t.start : t.end].strip()]
    payload = {
        "document": [{"sent": s.idx, "text": s.text} for s in parsed.sentences],
        "invalid_proposals": [{"index": i, **ps[i]} for i in invalid],
        "token_columns": ["token_id", "original_text", "sentence_id"],
    }
    if previous:
        payload["previous_invalid_response"] = {
            "reply": previous["reply"],
            "rejected": previous["rejected"],
        }
    system = languages.system("trigger", parsed.lang)

    def build(n):
        chosen = set(ordered[:n])
        ts = [t for t in tokens if t.sent_idx in chosen]
        p = {
            **payload,
            "document_tokens": [
                [t.idx, parsed.text[t.start : t.end], t.sent_idx] for t in ts
            ],
        }
        prompt = (
            INSTRUCTION[parsed.lang]
            + "\n"
            + json.dumps(p, ensure_ascii=False)
            + "\n"
            + OUTPUT[parsed.lang]
        )
        budget = (
            llm.pool.tokens(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
            )
            + llm.max_tokens
        )
        return prompt, ts, budget

    prompt, ts, budget = build(len(ordered))
    if budget > capacity:
        low, high = 0, len(ordered)
        while low < high:
            mid = (low + high + 1) // 2
            if build(mid)[2] <= capacity:
                low = mid
            else:
                high = mid - 1
        prompt, ts, budget = build(low)
        if not ts or budget > capacity:
            raise ValueError(
                "Full text plus minimum trigger token index exceeds service budget"
            )
    return (
        prompt,
        system,
        {t.idx: t for t in ts},
        {
            "request_token_budget": budget,
            "service_capacity": capacity,
            "full_text_sentences": len(parsed.sentences),
            "indexed_sentences": sorted({t.sent_idx for t in ts}),
            "indexed_tokens": len(ts),
            "full_text_truncated": False,
        },
    )


def repair_proposals(parsed, original, llm, max_attempts=2):
    ps = copy.deepcopy(original["triggers"])
    relations = copy.deepcopy(original["relations"])
    for p in ps:
        p["text"] = p["text"].strip()
    initial = copy.deepcopy(ps)
    valid = {
        i
        for i, p in enumerate(ps)
        if _find(parsed, p["text"], _as_int(p.get("sent"), 0)) >= 0
    }
    audit = {
        "version": VERSION,
        "original_proposals": copy.deepcopy(ps),
        "attempts": [],
        "deterministic_repairs": [],
        "discarded_indices": [],
        "initial_valid_indices": sorted(valid),
    }
    for i, p in enumerate(ps):
        m = locate_literal(parsed, p["text"], _as_int(p.get("sent"), 0))
        if m and m[2] != "exact":
            a, b, kind = m
            audit["deterministic_repairs"].append(
                {
                    "index": i,
                    "before": p["text"],
                    "after": parsed.text[a:b],
                    "start": a,
                    "end": b,
                    "kind": kind,
                }
            )
            p["text"] = parsed.text[a:b]
            p["sent"] = parsed.sentence_of(a)
    discarded = set()
    for attempt in range(max_attempts):
        invalid = [
            i
            for i, p in enumerate(ps)
            if i not in discarded
            and _find(parsed, p["text"], _as_int(p.get("sent"), 0)) < 0
        ]
        if not invalid:
            break
        request, system, tokens, budget = correction_request(
            parsed,
            ps,
            invalid,
            audit["attempts"][-1] if audit["attempts"] else None,
            llm,
        )
        llm.context.update(
            stage="S1_trigger_copy_repair",
            repair_protocol=VERSION,
            repair_attempt=attempt + 1,
        )
        reply = llm.json_chat(request, system, default=None)
        record = {
            "attempt": attempt + 1,
            "reply": reply,
            "invalid_indices": invalid,
            "accepted": [],
            "rejected": [],
            "budget": budget,
        }
        audit["attempts"].append(record)
        changes = reply.get("repairs") if isinstance(reply, dict) else None
        if not isinstance(changes, list):
            record["rejected"].append({"reason": "invalid_schema"})
            continue
        counts = {
            i: sum(
                isinstance(c, dict)
                and not isinstance(c.get("index"), bool)
                and c.get("index") == i
                for c in changes
            )
            for i in invalid
        }
        for c in changes:
            if not isinstance(c, dict):
                record["rejected"].append({"reason": "invalid_item"})
                continue
            i = c.get("index")
            first = c.get("start_token")
            last = c.get("end_token")
            if (
                isinstance(i, bool)
                or not isinstance(i, int)
                or i not in invalid
                or counts[i] != 1
                or not {"start_token", "end_token"} <= set(c)
            ):
                record["rejected"].append(
                    {"reason": "invalid_or_duplicate_index", "item": c}
                )
                continue
            if first is None and last is None:
                discarded.add(i)
                ps[i]["text"] = ""
                audit["discarded_indices"].append(i)
                record["accepted"].append(
                    {"index": i, "action": "discard_unsubstantiated"}
                )
                continue
            if any(
                isinstance(t, bool) or not isinstance(t, int) or t not in tokens
                for t in (first, last)
            ):
                record["rejected"].append({"index": i, "reason": "unlisted_token_id"})
                continue
            t0, t1 = tokens[first], tokens[last]
            if t0.start > t1.start or t0.sent_idx != t1.sent_idx:
                record["rejected"].append(
                    {"index": i, "reason": "reversed_or_cross_sentence"}
                )
                continue
            a, b = t0.start, t1.end
            before = ps[i]["text"]
            ps[i]["text"] = parsed.text[a:b]
            ps[i]["sent"] = parsed.sentence_of(a)
            record["accepted"].append(
                {
                    "index": i,
                    "before": before,
                    "after": ps[i]["text"],
                    "start": a,
                    "end": b,
                }
            )
    assert all(ps[i] == initial[i] for i in valid), (
        "Originally grounded proposal changed"
    )
    events = _locate(ps, parsed)
    audit["unresolved_indices"] = [
        i
        for i, p in enumerate(ps)
        if i not in discarded
        and _find(parsed, p["text"], _as_int(p.get("sent"), 0)) < 0
    ]
    audit["final_proposals"] = ps
    parsed.diagnostics.update(
        proposed_events=len(ps),
        grounded_proposals=len(events),
        unlocated_triggers=len(audit["unresolved_indices"]),
        trigger_repair=audit,
    )
    if not events:
        raise ValueError("No grounded event after bounded recovery")
    return events, relations, audit
