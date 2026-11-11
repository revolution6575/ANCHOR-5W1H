OFFSET_POLICY = "content-whitespace-v1"


def content_offsets(text, raw):
    result = []
    for start, end in raw:
        if not (0 <= start <= end <= len(text)):
            raise ValueError("Tokenizer offset outside original text")
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        result.append((start, end))
    return result


def boundary_targets(offsets, spans):
    from .policy import token_span

    valid = [a < b for a, b in offsets]
    target = [[0.0] * len(offsets) for _ in range(2)]
    mask = [valid.copy(), valid.copy()]
    unknown = []
    known = []
    for span in spans:
        ids = token_span(offsets, span, exact=True)
        if ids is None:
            overlap = token_span(offsets, span)
            unknown.append(
                {
                    "span": list(span),
                    "reason": "inside_token_boundary"
                    if overlap
                    else "no_content_token",
                    "overlap": overlap,
                }
            )
        else:
            known.append(ids)
    for item in unknown:
        if item["overlap"]:
            a, b = item["overlap"]
            for k in (0, 1):
                for i in range(a, b + 1):
                    mask[k][i] = False
    for a, b in known:
        target[0][a] = target[1][b] = 1.0
        mask[0][a] = mask[1][b] = True
    return target, mask, known, unknown


def validate_mapping_options(options):
    if options.get("offset_policy") != OFFSET_POLICY:
        raise ValueError(
            "Offset policy mismatch; legacy checkpoints require the 10/25 code"
        )


def compatible_options(trained, decision):
    validate_mapping_options(trained)
    validate_mapping_options(decision)
    keys = (
        "use_4w",
        "hard_negatives",
        "boundary_head",
        "boundary_candidates",
        "window",
        "overlap",
        "boundary_top_positions",
        "boundary_top_spans",
        "max_span_tokens",
        "projection",
        "rank",
    )
    for key in keys:
        if trained.get(key) != decision.get(key):
            raise ValueError("Checkpoint/decision configuration mismatch: " + key)
