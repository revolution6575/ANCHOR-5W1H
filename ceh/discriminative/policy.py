import math
from dataclasses import dataclass
from itertools import product

ROLES = ("why", "how")


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0])) if inter else 0.0


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def score(
    logit, boundary, temperature, weight, fusion="gated", features=None, role=None
):
    if not all(math.isfinite(float(v)) for v in (logit, boundary, temperature, weight)):
        raise ValueError("Nonfinite role output/configuration")
    if temperature <= 0 or not 0 <= boundary <= 1 or not 0 <= weight <= 1:
        raise ValueError("Invalid score configuration")
    p = sigmoid(logit / temperature)
    value = (
        p * ((1 - weight) + weight * boundary)
        if fusion == "gated"
        else (1 - weight) * p + weight * boundary
    )
    if fusion == "legacy_additive":
        if role == "why":
            cue, dist, order, known = features
            value = 0.5 * p + 0.2 * 0.5 + 0.2 * cue - 0.1 * dist
        else:
            sub, sty, dist, cue = features
            value = 0.5 * p + 0.25 * sub + 0.15 * sty - 0.1 * cue
    if fusion == "gated" and value > p + 1e-12:
        raise AssertionError("Semantic gate violated")
    return p, value


@dataclass
class RoleScore:
    event_id: int
    candidate_id: int
    unit_id: str
    role: str
    start: int
    end: int
    text: str
    source: str
    logit: float
    boundary_score: float
    features: list
    source_event_id: int = -1
    semantic_score: float = 0.0
    score: float = 0.0
    threshold: float = 0.0
    gain: float = 0.0
    eligible: bool = False
    selected: bool = False
    reason: str = "unscored"


def choose(rows, event, text, cfg):
    for p in rows:
        p.selected = False
        p.eligible = False
        p.semantic_score, p.score = score(
            p.logit,
            p.boundary_score,
            cfg["temperatures"][p.role],
            cfg["boundary_lambdas"][p.role],
            cfg.get("fusion", "gated"),
            p.features,
            p.role,
        )
        p.threshold = cfg["thresholds"][p.role]
        p.gain = p.score - p.threshold
        if not (0 <= p.start < p.end <= len(text) and text[p.start : p.end] == p.text):
            p.reason = "invalid_span"
        elif (p.start, p.end) == event.span:
            p.reason = "trigger_self_reference"
        elif p.gain <= 0:
            p.reason = "below_threshold"
        else:
            p.eligible = True
            p.reason = "eligible"
    options = {
        r: [None] + [x for x in rows if x.role == r and x.eligible] for r in ROLES
    }
    if cfg.get("conflict", "exact") == "exact":
        for r in ROLES:
            seen = set()
            top = []
            for p in sorted(
                options[r][1:],
                key=lambda p: (
                    -p.gain,
                    p.end - p.start,
                    p.start,
                    p.end,
                    p.candidate_id,
                ),
            ):
                if (p.start, p.end) not in seen:
                    top.append(p)
                    seen.add((p.start, p.end))
                if len(top) == 2:
                    break
            options[r] = [None] + top
    best = None
    answer = []
    for w, h in product(options["why"], options["how"]):
        if w and h:
            rule = cfg.get("conflict", "exact")
            if rule == "exact" and (w.start, w.end) == (h.start, h.end):
                continue
            if rule == "unit" and w.unit_id == h.unit_id:
                continue
            if rule == "span" and w.start < h.end and h.start < w.end:
                continue
        path = [p for p in (w, h) if p]
        key = (
            -sum(p.gain for p in path),
            len(path),
            sum(p.end - p.start for p in path),
            tuple((p.start, p.end, p.role, p.candidate_id) for p in path),
        )
        if best is None or key < best:
            best = key
            answer = path
    for p in rows:
        if any(p is x for x in answer):
            p.selected = True
            p.reason = "selected"
        elif p.eligible:
            p.reason = "capacity_or_conflict"
    return answer


def window_ranges(n, capacity, overlap):
    if capacity <= overlap or overlap < 0:
        raise ValueError("Invalid window overlap")
    start = 0
    while start < n:
        end = min(n, start + capacity)
        yield start, end
        if end == n:
            break
        start = end - overlap


def token_span(offsets, span, exact=False):
    ids = [
        i for i, (a, b) in enumerate(offsets) if a < b and a < span[1] and span[0] < b
    ]
    if not ids:
        return None
    if exact and (offsets[ids[0]][0] != span[0] or offsets[ids[-1]][1] != span[1]):
        return None
    return ids[0], ids[-1]
