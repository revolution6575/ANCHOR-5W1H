from functools import lru_cache
from .policy import iou
from ..schema import Event, Element, Evidence, SPAN


def valid_span(text, node, prefix="Tag_"):
    a, b = node.get(prefix + "Start"), node.get(prefix + "End")
    value = node.get(prefix + "Text")
    return (
        isinstance(a, int)
        and not isinstance(a, bool)
        and isinstance(b, int)
        and not isinstance(b, bool)
        and isinstance(value, str)
        and 0 <= a < b <= len(text)
        and text[a:b] == value
    )


def normalize(record, dataset, lang):
    text = record["Text"]
    issues = []
    nodes = {}
    ambiguous = set()
    for n in record.get("Nodes", []):
        nid = n.get("Node_Id")
        if nid in nodes:
            ambiguous.add(nid)
            issues.append("duplicate_node:" + str(nid))
        if not valid_span(text, n) or n.get("Node_Type") not in (
            "who",
            "what",
            "when",
            "where",
            "why",
            "how",
        ):
            ambiguous.add(nid)
            issues.append("invalid_node:" + str(nid))
        nodes[nid] = n
    events = []
    keys = set()
    for cl in record.get("Event_Clusters", []):
        for e in cl.get("Events", []):
            key = (cl.get("Cluster_Id"), e.get("Event_Id"))
            if key in keys:
                issues.append("duplicate_event:" + str(key))
                continue
            keys.add(key)
            tr = e.get("Trigger", {})
            if not valid_span(text, tr):
                issues.append("invalid_trigger:" + str(key))
                continue
            he = e.get("Event_Hyperedge", {})
            refs = he.get("Nodes", {})
            slots = {}
            known = {}
            valid_link = he.get("Event") == e.get("Event_Id")
            for r in ("who", "what", "when", "where", "why", "how"):
                ids = refs.get(r)
                ok = valid_link and isinstance(ids, list)
                spans = []
                for nid in ids if isinstance(ids, list) else []:
                    if (
                        nid not in nodes
                        or nid in ambiguous
                        or nodes[nid].get("Node_Type") != r
                    ):
                        ok = False
                        continue
                    n = nodes[nid]
                    spans.append([n["Tag_Start"], n["Tag_End"]])
                if spans and r in e.get("Missing", []):
                    ok = False
                if not ok:
                    issues.append("unknown_association:" + str(key) + ":" + r)
                slots[r] = sorted(set(map(tuple, spans)))
                known[r] = ok
            events.append(
                {
                    "eid": len(events),
                    "original_event_id": list(key),
                    "trigger": tr["Tag_Text"],
                    "span": [tr["Tag_Start"], tr["Tag_End"]],
                    "slots": slots,
                    "known": known,
                }
            )
    review = record.get("Review", {})
    return {
        "doc_id": str(record["Id"]),
        "key": dataset + "/" + str(record["Id"]),
        "dataset": dataset,
        "lang": lang,
        "text": text,
        "events": events,
        "nodes": [n for k, n in nodes.items() if k not in ambiguous],
        "review": review,
        "event_association_reliability": "derived"
        if review.get("Trigger_Source") == "first_what"
        else "unverified",
        "annotation_issues": issues,
    }


def gold_event(data, parsed):
    a, b = data["span"]
    ev = Event(data["eid"], data["trigger"], parsed.sentence_of(a), a, b)
    ev.ensure_elements()
    for r in ("who", "what", "when", "where"):
        spans = data["slots"][r]
        if spans and data["known"][r]:
            spans = sorted(
                spans, key=lambda s: (abs(s[0] - ev.start), s[1] - s[0], s[0])
            )
            a, b = spans[0]
            evidence = [
                Evidence(
                    parsed.text[s:t],
                    s,
                    t,
                    parsed.sentence_of(s),
                    -1,
                    "",
                    -1,
                    0.0,
                    0.0,
                    "",
                )
                for s, t in spans
            ]
            ev.elements[r] = Element(
                state=SPAN,
                text=parsed.text[a:b],
                start=a,
                end=b,
                sent_idx=parsed.sentence_of(a),
                values=[v.text for v in evidence],
                evidence=evidence,
            )
    return ev


def align(predicted, gold, parsed):
    scores = []
    for ev in predicted:
        row = []
        for g in gold:
            ti = iou(ev.span, g["span"])
            what = ev.elements.get("what")
            wi = 0.0
            if (
                what
                and what.is_span()
                and ev.sent_idx == parsed.sentence_of(g["span"][0])
            ):
                wi = max((iou(what.span, s) for s in g["slots"]["what"]), default=0.0)
            row.append(2 * ti + wi if ti >= 0.5 or wi >= 0.5 else 0.0)
        positive = sorted((x for x in row if x > 0), reverse=True)
        if len(positive) > 1 and abs(positive[0] - positive[1]) < 1e-9:
            row = [0.0] * len(row)
        scores.append(row)

    @lru_cache(None)
    def dp(i, used):
        if i == len(predicted):
            return 0.0, ()
        best = dp(i + 1, used)
        for j, value in enumerate(scores[i]):
            if value <= 0 or j in used:
                continue
            val, path = dp(i + 1, tuple(sorted((*used, j))))
            proposal = (val + value, ((i, j),) + path)
            if proposal[0] > best[0] + 1e-12:
                best = proposal
        return best

    return [(predicted[i], gold[j]) for i, j in dp(0, ())[1]]


def role_label(span, gold, role):
    if not gold["known"][role]:
        return None
    overlaps = [iou(span, g) for g in gold["slots"][role]]
    best = max(overlaps, default=0.0)
    return 1.0 if best >= 0.5 else 0.0 if best == 0 else None
