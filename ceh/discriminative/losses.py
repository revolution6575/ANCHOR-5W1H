import random
import torch
from torch.nn import functional as F
from .data import role_label
from .offsets import boundary_targets


def balanced_bce(logits, targets, mask=None):
    if mask is not None:
        logits, targets = logits[mask], targets[mask]
    if not logits.numel():
        return logits.sum() * 0
    loss = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none"
    )
    terms = []
    for positive in (True, False):
        selected = (targets > 0.5) == positive
        if selected.any():
            terms.append(loss[selected].mean())
    return torch.stack(terms).mean()


def event_loss(
    clauses, logits, boundary_support, boundary_logits, offsets, gold, options, seed
):
    zero = boundary_logits.sum() * 0
    roles = []
    bounds = []
    ranking = []
    audit = {
        "masked_partial": 0,
        "unalignable_boundaries": 0,
        "overlong_gold": 0,
        "rank_pairs": 0,
    }
    lam = logits.new_tensor(
        [options.get("training_rank_lambdas", {}).get(r, 0.5) for r in ("why", "how")]
    )
    gated = logits.float().sigmoid() * ((1 - lam) + lam * boundary_support.float())
    for r, role in enumerate(("why", "how")):
        if not gold["known"][role]:
            continue
        targets = [role_label(c.span, gold, role) for c in clauses]
        if targets:
            mask = torch.tensor([t is not None for t in targets], device=logits.device)
            y = logits.new_tensor([t if t is not None else 0 for t in targets])
            roles.append(balanced_bce(logits[:, r], y, mask))
            audit["masked_partial"] += sum(t is None for t in targets)
        target, mask, known, unknown = boundary_targets(offsets, gold["slots"][role])
        boundary_target = boundary_logits.new_tensor(target)
        boundary_mask = torch.tensor(
            mask, device=boundary_logits.device, dtype=torch.bool
        )
        audit["unalignable_boundaries"] += len(unknown)
        audit["alignable_boundaries"] = audit.get("alignable_boundaries", 0) + len(
            known
        )
        audit["overlong_gold"] += sum(
            b - a + 1 > options.get("max_span_tokens", 256) for a, b in known
        )
        if options.get("boundary_head", True):
            bounds.extend(
                balanced_bce(
                    boundary_logits[r, k], boundary_target[k], boundary_mask[k]
                )
                for k in (0, 1)
            )
        if options.get("hard_negatives", True) and len(clauses):
            rng = random.Random(seed + r)
            positives = [
                i
                for i, c in enumerate(clauses)
                if c.span in [tuple(s) for s in gold["slots"][role]]
            ]
            for positive in positives:
                span = clauses[positive].span
                gold_spans = {tuple(s) for s in gold["slots"][role]}
                negatives = [
                    i
                    for i, t in enumerate(targets)
                    if t == 0
                    or (
                        clauses[i].span not in gold_spans
                        and clauses[i].start <= span[0]
                        and clauses[i].end >= span[1]
                    )
                ]
                rng.shuffle(negatives)
                negatives.sort(key=lambda i: -float(gated[i, r].detach()))
                for negative in negatives[:4]:
                    ranking.append(
                        F.relu(0.1 + gated[negative, r] - gated[positive, r])
                    )
                    audit["rank_pairs"] += 1
    mean = lambda xs: torch.stack(xs).mean() if xs else zero
    lr, lb, lk = mean(roles), mean(bounds), mean(ranking)
    return lr + 0.5 * lb + 0.2 * lk, {
        "role": float(lr.detach()),
        "boundary": float(lb.detach()),
        "rank": float(lk.detach()),
        **audit,
    }
