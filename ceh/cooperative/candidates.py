import math

ROLES = ("who", "what", "when", "where")


def boundary_spans(probabilities, offsets, limit=6, top_positions=32, max_tokens=256):
    if limit <= 0:
        return []
    valid = [i for i, (a, b) in enumerate(offsets) if a < b]
    k = min(top_positions, len(valid))
    starts = sorted(valid, key=lambda i: (-probabilities[0][i], i))[:k]
    ends = sorted(valid, key=lambda i: (-probabilities[1][i], i))[:k]
    items = []
    for i in starts:
        for j in ends:
            if i <= j and j - i + 1 <= max_tokens:
                a, b = offsets[i][0], offsets[j][1]
                if a < b:
                    items.append(
                        (math.sqrt(probabilities[0][i] * probabilities[1][j]), a, b)
                    )
    seen = set()
    result = []
    for score, a, b in sorted(items, key=lambda x: (-x[0], x[2] - x[1], x[1], x[2])):
        if (a, b) in seen:
            continue
        seen.add((a, b))
        result.append((a, b, score))
        if len(result) >= limit:
            break
    return result


def syntax_spans(parsed, event, role):
    spans = set()
    if role == "what":
        spans.update(c.span for c in parsed.clauses)
    else:
        allowed = (
            {"NOUN", "PROPN", "ADJ", "DET"}
            if role in ("who", "where")
            else {"NUM", "PROPN", "NOUN", "ADJ", "ADP", "PUNCT"}
        )
        run = []

        def flush():
            if run and (role != "when" or any(t.pos == "NUM" for t in run)):
                spans.add((run[0].start, run[-1].end))
            run.clear()

        for t in parsed.tokens:
            if run and (run[-1].sent_idx != t.sent_idx or len(run) >= 8):
                flush()
            if t.pos in allowed:
                run.append(t)
            else:
                flush()
        flush()
    return sorted(
        (
            x
            for x in spans
            if 0 <= x[0] < x[1] <= len(parsed.text) and parsed.text[x[0] : x[1]].strip()
        ),
        key=lambda x: (
            abs(parsed.sentence_of(x[0]) - event.sent_idx),
            abs(x[0] - event.start),
            x[1] - x[0],
            x,
        ),
    )


def propose(parsed, event, offsets, probabilities, options):
    result = {}
    for r, role in enumerate(ROLES):
        spans = {}

        def add(a, b, source, support=None):
            if not 0 <= a < b <= len(parsed.text) or not parsed.text[a:b].strip():
                return
            if (a, b) in spans:
                spans[a, b]["sources"].append(source)
                if support is not None:
                    spans[a, b]["boundary_support"] = support
            else:
                spans[a, b] = {
                    "text": parsed.text[a:b],
                    "start": a,
                    "end": b,
                    "sent_id": parsed.sentence_of(a),
                    "sources": [source],
                    "boundary_support": support,
                }

        old = event.elements.get(role)
        if old and old.is_span() and parsed.text[old.start : old.end] == old.text:
            add(old.start, old.end, "llm_draft")
        for a, b, score in boundary_spans(
            probabilities[r],
            offsets,
            options.get("learned_candidates", 6),
            options.get("top_positions", 32),
            options.get("max_span_tokens", 256),
        ):
            add(a, b, "boundary", score)
        for a, b in syntax_spans(parsed, event, role)[
            : options.get("syntax_candidates", 2)
        ]:
            add(a, b, "syntax")
        result[role] = [
            dict(item, id=f"e{event.eid}.{role}.c{i}")
            for i, (_, item) in enumerate(sorted(spans.items()))
        ]
    return result
