import torch
from ..schema import Clause, Element, Evidence, SPAN, NO_EVIDENCE
from .. import features, timex
from ..scoring import candidate_lists
from .policy import RoleScore, choose
from .offsets import compatible_options, OFFSET_POLICY


def syntax_candidates(event, events, clauses, cfg):
    from dataclasses import replace

    relabel = [
        replace(c, source_event_id=-1) if c.source_event_id == event.eid else c
        for c in clauses
    ]
    result = candidate_lists([event], relabel, cfg)[event.eid]
    return [c for c in result if c.span != event.span]


def expand(boundaries, offsets, parsed, clauses, event, options):
    seen = {c.span for c in clauses}
    out = list(clauses)
    next_id = max((c.cid for c in clauses), default=-1) + 1
    if not options.get("boundary_candidates", True):
        return out
    probs = boundaries.detach().float().sigmoid().cpu()
    valid = [i for i, (a, b) in enumerate(offsets) if a < b]
    k = min(options.get("boundary_top_positions", 32), len(valid))
    maxlen = options.get("max_span_tokens", 256)
    for r in range(2):
        starts = sorted(valid, key=lambda i: (-float(probs[r, 0, i]), i))[:k]
        ends = sorted(valid, key=lambda i: (-float(probs[r, 1, i]), i))[:k]
        candidates = []
        for a in starts:
            for b in ends:
                if b < a or b - a + 1 > maxlen:
                    continue
                lo, hi = offsets[a][0], offsets[b][1]
                if lo >= hi or (lo, hi) == event.span or not parsed.text[lo:hi].strip():
                    continue
                candidates.append(
                    (
                        -((float(probs[r, 0, a]) * float(probs[r, 1, b])) ** 0.5),
                        hi - lo,
                        lo,
                        hi,
                    )
                )
        accepted = set()
        for _, _, lo, hi in sorted(candidates):
            if (lo, hi) in accepted:
                continue
            accepted.add((lo, hi))
            if (lo, hi) not in seen:
                parent = min(
                    (c for c in clauses if c.start <= lo and c.end >= hi),
                    key=lambda c: c.end - c.start,
                    default=None,
                )
                unit = parent.unit_id if parent else f"boundary:{lo}:{hi}"
                owner = (
                    parent.source_event_id
                    if parent and parent.source_event_id != event.eid
                    else -1
                )
                seen.add((lo, hi))
                out.append(
                    Clause(
                        next_id,
                        parsed.text[lo:hi],
                        parsed.sentence_of(lo),
                        lo,
                        hi,
                        kind="boundary",
                        unit_id=unit,
                        source_event_id=owner,
                    )
                )
                next_id += 1
            if len(accepted) == options.get("boundary_top_spans", 8):
                break
    return out


def feature_rows(event, clauses, events, parsed, use_4w=True):
    by = {e.eid: e for e in events}
    out = []
    for c in clauses:
        cue = features.causal_cue(parsed.text, c.span, c.span)
        dist = features.distance(event.sent_idx, c.sent_idx, len(parsed.sentences))
        other = by.get(c.source_event_id)
        order = known = 0.0
        if other and use_4w:
            a = timex.normalize(other.slot_text("when"))
            b = timex.normalize(event.slot_text("when"))
            if a and b and all(x is not None for x in a + b):
                from datetime import date

                try:
                    order = 1.0 if date(*a) <= date(*b) else -1.0
                    known = 1.0
                except ValueError:
                    pass
        sub = features.subordination(parsed, c.span, c.head_span, c.sent_idx, event)
        out.append(
            [
                [cue, dist, order, known],
                [sub, features.manner_affinity(c.text), dist, cue],
            ]
        )
    return out


def event_forward(model, h, offsets, ev, events, clauses, parsed, cfg, gold_spans=None):
    u = model.event_vector(h, offsets, ev)
    boundary = model.boundaries(h, u)
    cs = expand(
        boundary,
        offsets,
        parsed,
        syntax_candidates(ev, events, clauses, cfg),
        ev,
        model.options,
    )
    if gold_spans is not None:
        seen = {c.span for c in cs}
        cid = max((c.cid for c in cs), default=-1) + 1
        for a, b in gold_spans:
            if (a, b) not in seen and (a, b) != ev.span:
                cs.append(
                    Clause(
                        cid,
                        parsed.text[a:b],
                        parsed.sentence_of(a),
                        a,
                        b,
                        kind="gold_training_only",
                        unit_id=f"gold:{a}:{b}",
                    )
                )
                cid += 1
                seen.add((a, b))
    cs, v, ranges = model.candidates(h, offsets, cs)
    fs = feature_rows(ev, cs, events, parsed, model.options.get("use_4w", True))
    f = h.new_tensor(fs).reshape(len(cs), 2, 4)
    logits = model.role_logits(u, v, f)
    if ranges:
        a = torch.tensor([x[0] for x in ranges], device=h.device)
        b = torch.tensor([x[1] for x in ranges], device=h.device)
        bp = boundary.float().sigmoid()
        support = (bp[:, 0, a] * bp[:, 1, b]).sqrt().transpose(0, 1)
    else:
        support = h.new_empty((0, 2))
    return cs, logits, support, boundary, fs


@torch.inference_mode()
def predict(model, parsed, events, clauses, cfg):
    compatible_options(model.options, cfg["discriminator"])
    before = model.encoding_calls
    with torch.autocast(
        device_type=model.device.type,
        dtype=torch.bfloat16,
        enabled=model.device.type == "cuda",
    ):
        h, offsets, n = model.encode(parsed.text)
        rows = []
        candidate_records = []
        for ev in events:
            cs, logits, boundary, _, fs = event_forward(
                model, h, offsets, ev, events, clauses, parsed, cfg
            )
            for i, c in enumerate(cs):
                candidate_records.append(
                    {
                        "event_id": ev.eid,
                        "candidate_id": c.cid,
                        "source": c.kind,
                        "start": c.start,
                        "end": c.end,
                    }
                )
                for j, r in enumerate(("why", "how")):
                    rows.append(
                        RoleScore(
                            ev.eid,
                            c.cid,
                            c.unit_id,
                            r,
                            c.start,
                            c.end,
                            c.text,
                            c.kind,
                            float(logits[i, j]),
                            float(boundary[i, j]),
                            fs[i][j],
                            c.source_event_id,
                        )
                    )
    tokenized = model.tokenize(parsed.text)
    return rows, {
        "encoder_windows": n,
        "encoder_forward_windows": model.encoding_calls - before,
        "role_pairs": len(rows),
        "candidates": candidate_records,
        "s2_llm_calls": 0,
        "offset_policy": OFFSET_POLICY,
        "whitespace_adjusted_tokens": sum(
            x != y for x, y in zip(tokenized["raw_offsets"], offsets)
        ),
        "invalid_boundary_positions": sum(not x for x in tokenized["valid_boundaries"]),
    }


def assemble(events, rows, parsed, cfg):
    text = parsed.text
    report = {}
    for ev in events:
        ev.ensure_elements()
        pairs = [p for p in rows if p.event_id == ev.eid]
        selected = choose(pairs, ev, text, cfg)
        for r in ("why", "how"):
            picks = [p for p in selected if p.role == r]
            if not picks:
                ev.elements[r] = Element(
                    state=NO_EVIDENCE, reason="no_accepted_evidence"
                )
                continue
            p = picks[0]
            rel = "unspecified" if r == "why" else ""
            sid = parsed.sentence_of(p.start)
            form = (
                "another_event"
                if p.source_event_id >= 0
                else "same_sent"
                if sid == ev.sent_idx
                else "cross_sent"
            )
            evidence = Evidence(
                p.text,
                p.start,
                p.end,
                sid,
                p.candidate_id,
                p.unit_id,
                p.source_event_id,
                p.score,
                p.gain,
                form,
                rel,
                {
                    "semantic_score": p.semantic_score,
                    "boundary_score": p.boundary_score,
                    "threshold": p.threshold,
                },
            )
            ev.elements[r] = Element(
                state=SPAN,
                text=p.text,
                start=p.start,
                end=p.end,
                sent_idx=sid,
                score=p.score,
                form=form,
                rel_type=rel,
                values=[p.text],
                evidence=[evidence],
                reason="selected",
            )
        report[str(ev.eid)] = {
            "objective": sum(p.gain for p in selected),
            "selected": [(p.candidate_id, p.role) for p in selected],
        }
    return report


class Backend:
    def __init__(self, checkpoint, device):
        from .model import SharedDiscriminator

        self.model, self.metadata = SharedDiscriminator.load(checkpoint, device)

    def predict(self, parsed, events, clauses, cfg):
        rows, audit = predict(self.model, parsed, events, clauses, cfg)
        audit.update(
            model_metadata=self.metadata, decision_parameters=cfg["discriminator"]
        )
        return rows, audit


_BACKENDS = {}


def get_backend(cfg):
    c = cfg["discriminator"]
    key = (c.get("checkpoint"), c.get("device", "cuda"), c.get("offset_policy"))
    if not key[0]:
        raise ValueError(
            "No trained discriminator checkpoint configured; no LLM fallback"
        )
    if key not in _BACKENDS:
        _BACKENDS[key] = Backend(*key[:2])
    compatible_options(_BACKENDS[key].model.options, c)
    return _BACKENDS[key]
