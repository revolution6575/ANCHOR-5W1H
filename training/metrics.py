from collections import Counter, defaultdict
import evaluator_frozen as legacy

ROLES = ("who", "what", "when", "where", "why", "how")


def f1(c):
    return (
        2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"])
        if 2 * c["tp"] + c["fp"] + c["fn"]
        else 0.0
    )


def union_chars(spans):
    return set(i for a, b in spans for i in range(a, b))


def count_doc(doc, events):
    gold = {r: set() for r in ROLES}
    pred = {r: set() for r in ROLES}
    values = {r.upper(): [] for r in ROLES}
    for n in doc["nodes"]:
        gold[n["Node_Type"]].add((n["Tag_Start"], n["Tag_End"]))
    for ev in events:
        for r, el in ev.elements.items():
            if el.is_span():
                spans = [(v.start, v.end, v.text) for v in el.evidence] or [
                    (el.start, el.end, el.text)
                ]
                for a, b, t in spans:
                    if not (0 <= a < b <= len(doc["text"]) and doc["text"][a:b] == t):
                        raise ValueError("Invalid predicted span")
                    pred[r].add((a, b))
                    values[r.upper()].append(t)
    tags = [
        legacy.Tag(
            n["Tag_Text"], n["Node_Type"].upper(), n["Tag_Start"], n["Tag_End"], i
        )
        for i, n in enumerate(doc["nodes"])
    ]
    guesses, _, _ = legacy.prediction_tags(values, doc["text"])
    out = {
        "position": legacy.position_counts(tags, guesses),
        "soft": legacy.soft_counts(tags, guesses),
        "exact": {},
        "character": {},
    }
    for r in ROLES:
        a, b = gold[r], pred[r]
        out["exact"][r.upper()] = {"tp": len(a & b), "fp": len(b - a), "fn": len(a - b)}
        a, b = union_chars(a), union_chars(b)
        out["character"][r.upper()] = {
            "tp": len(a & b),
            "fp": len(b - a),
            "fn": len(a - b),
        }
    return out


def summarize(rows):
    counts = defaultdict(Counter)
    gold_counts = Counter()
    how_empty = how_false = 0
    for row in rows:
        for role, c in row["counts"]["exact"].items():
            for scope in (
                "ALL",
                row["dataset"],
                "lang:" + row["lang"],
                "source:" + row["source"],
            ):
                gold_counts[scope + "/" + role] += c["tp"] + c["fn"]
        h = row["counts"]["exact"]["HOW"]
        if h["tp"] + h["fn"] == 0:
            how_empty += 1
            how_false += h["fp"] > 0
        for mode, roles in row["counts"].items():
            for role, c in roles.items():
                for scope in (
                    "ALL",
                    row["dataset"],
                    "lang:" + row["lang"],
                    "source:" + row["source"],
                ):
                    counts[scope, mode, role].update(c)
                    counts[scope, mode, "ALL"].update(c)
    scores = {
        scope + "/" + mode + "/" + role: {
            "counts": dict(c),
            "F1": f1(c) if mode in ("exact", "character") else legacy.metrics(c)["F1"],
        }
        for (scope, mode, role), c in counts.items()
    }
    objective = (
        sum(
            scores.get("ALL/" + m + "/" + r, {"F1": 0})["F1"]
            for m in ("exact", "character")
            for r in ("WHY", "HOW")
        )
        / 4
    )
    from selection import objective as multilingual_objective

    return {
        "objective": multilingual_objective(scores, gold_counts),
        "legacy_objective": objective,
        "documents": len(rows),
        "scores": scores,
        "gold_counts": dict(gold_counts),
        "how_no_gold_documents": how_empty,
        "how_no_gold_fp_documents": how_false,
        "how_no_gold_fp_rate": how_false / how_empty if how_empty else 0.0,
    }


def reference_diagnostics(doc, events, rows, parsed):
    from ceh.discriminative.data import align, role_label

    linked = {ev.eid: g for ev, g in align(events, doc["events"], parsed)}
    gold = set()
    pred = set()
    bins = {
        r: [{"n": 0, "sum_confidence": 0.0, "sum_label": 0.0} for _ in range(10)]
        for r in ("why", "how")
    }
    lengths = {r: [] for r in ("why", "how")}
    boundary_bins = {
        r: [{"n": 0, "sum_support": 0.0, "exact_pair_labels": 0} for _ in range(10)]
        for r in ("why", "how")
    }
    for g in doc["events"]:
        for r in ("why", "how"):
            if g["known"][r]:
                gold.update((g["eid"], r, *span) for span in g["slots"][r])
    for ev in events:
        for r in ("why", "how"):
            if ev.eid in linked and not linked[ev.eid]["known"][r]:
                continue
            for v in ev.elements[r].evidence:
                gid = (
                    linked[ev.eid]["eid"]
                    if ev.eid in linked
                    else "unmatched:" + str(ev.eid)
                )
                pred.add((gid, r, v.start, v.end))
                lengths[r].append(v.end - v.start)
    for p in rows:
        if p.event_id not in linked:
            continue
        label = role_label((p.start, p.end), linked[p.event_id], p.role)
        if label is None:
            continue
        b = bins[p.role][min(9, int(p.semantic_score * 10))]
        b["n"] += 1
        b["sum_confidence"] += p.semantic_score
        b["sum_label"] += label
        b = boundary_bins[p.role][min(9, int(p.boundary_score * 10))]
        b["n"] += 1
        b["sum_support"] += p.boundary_score
        b["exact_pair_labels"] += int(
            (p.start, p.end) in [tuple(s) for s in linked[p.event_id]["slots"][p.role]]
        )
    return {
        "reference_joint_exact": {
            "tp": len(gold & pred),
            "fp": len(pred - gold),
            "fn": len(gold - pred),
        },
        "event_association_reliability": doc["event_association_reliability"],
        "human_verified_joint_metric": None,
        "calibration_bins": bins,
        "boundary_bins": boundary_bins,
        "predicted_lengths": lengths,
    }
