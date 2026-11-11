import argparse, json, random, sys, time, math
from pathlib import Path
from common import R, read, save, sha, documents, configuration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ceh.discriminative.model import SharedDiscriminator
from ceh.discriminative.packets import load_parsed, filename, events_from
from ceh.discriminative.data import gold_event, align
from ceh.cooperative.heads import FourWHead, boundary_loss, ROLES
from ceh.cooperative.candidates import propose


def encoding(base, meta, doc, cache):
    path = cache / filename(doc).replace(".json", ".pt")
    if path.exists():
        value = torch.load(path, map_location="cpu", weights_only=True)
        assert (
            value["base_sha"] == meta["weights_sha256"]
            and value["text_sha"] == doc["text_sha256"]
            and value["offset_policy"] == base.options["offset_policy"]
        )
        return value["h"].to(base.device), value["offsets"], value["windows"]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        h, offsets, n = base.encode(doc["text"])
    h = h.clone().detach()
    cache.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "h": h.cpu(),
            "offsets": offsets,
            "windows": n,
            "base_sha": meta["weights_sha256"],
            "text_sha": doc["text_sha256"],
            "offset_policy": base.options["offset_policy"],
        },
        path,
    )
    return h, offsets, n


@torch.inference_mode()
def evaluate(base, head, meta, docs, cache, cfg):
    head.eval()
    counts = {r: {"gold": 0, "exact": 0, "union_exact": 0} for r in ROLES}
    rows = []
    for doc in docs:
        h, offsets, _ = encoding(base, meta, doc, cache)
        parsed = load_parsed(R, doc)
        packet = read(R / "s1" / filename(doc))
        events = events_from(packet["events"])
        candidate_sets = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for e in events:
                candidate_sets[e.eid] = propose(
                    parsed,
                    e,
                    offsets,
                    head(h, offsets, e, parsed, base).float().sigmoid().cpu().tolist(),
                    cfg["cooperative"],
                )
        for role in ROLES:
            gold = {
                (n["Tag_Start"], n["Tag_End"])
                for n in doc["nodes"]
                if n["Node_Type"] == role
            }
            learned = {
                (c["start"], c["end"])
                for cs in candidate_sets.values()
                for c in cs[role]
                if "boundary" in c["sources"]
            }
            union = {
                (c["start"], c["end"])
                for cs in candidate_sets.values()
                for c in cs[role]
            }
            counts[role]["gold"] += len(gold)
            counts[role]["exact"] += len(gold & learned)
            counts[role]["union_exact"] += len(gold & union)
        rows.append(
            {
                "key": doc["key"],
                "text_sha256": doc["text_sha256"],
                "fold": doc["fold"],
                "candidates": candidate_sets,
            }
        )
    objective = sum(c["exact"] / c["gold"] for c in counts.values() if c["gold"]) / sum(
        bool(c["gold"]) for c in counts.values()
    )
    return objective, counts, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--fold", type=int, default=-1)
    p.add_argument("--epochs", type=int, default=11)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    out = (
        R
        / ("smoke_heads" if a.smoke else "heads")
        / ("full" if a.fold < 0 else f"fold{a.fold}")
    )
    out.mkdir(parents=True, exist_ok=True)
    cache = R / "feature_cache" / ("full" if a.fold < 0 else f"fold{a.fold}")

    if (out / "completed.json").exists():
        return
    ds = documents()
    train = [d for d in ds if a.fold < 0 or d["fold"] != a.fold]
    val = [d for d in ds if d["fold"] == a.fold]
    if a.smoke:
        train = train[:3]
        val = val[:2]
    assert not {d["key"] for d in train} & {d["key"] for d in val}
    random.seed(20261028)
    torch.manual_seed(20261028)
    torch.cuda.manual_seed_all(20261028)
    cfg = configuration()
    base, meta = SharedDiscriminator.load(a.base_checkpoint, "cuda")
    base.eval()
    assert meta["fold"] == a.fold and meta["seed"] == 20261025, (
        "Matched-fold encoder required"
    )
    for param in base.parameters():
        param.requires_grad_(False)
    head = FourWHead(base.hidden).float().cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.01)
    state = {
        "state": "caching",
        "fold": a.fold,
        "seed": 20261028,
        "training_documents": len(train),
        "validation_documents": len(val),
        "base_weights_sha256": meta["weights_sha256"],
        "base_frozen": True,
        "train_sha256": sha(R / "data/train.jsonl"),
        "history": [],
        "started_unix": time.time(),
    }
    save(out / "progress.json", state)
    for i, d in enumerate(train + val):
        encoding(base, meta, d, cache)
        if i % 25 == 0:
            state.update(cache_documents=i + 1)
            save(out / "progress.json", state)
    best = -1.0
    stale = 0
    steps = 0
    expected = sum(len(base.tokenize(d["text"])["windows"]) for d in train)
    total_steps = math.ceil(expected / 16) * a.epochs

    def lr(step):
        return min(1, (step + 1) / max(1, int(total_steps * 0.1))) * max(
            0, (total_steps - step) / max(1, total_steps - int(total_steps * 0.1))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr)
    for epoch in range(1, a.epochs + 1):
        head.train()
        order = list(train)
        random.Random(20261028 + epoch).shuffle(order)
        losses = []
        accum = 0
        optimizer.zero_grad(set_to_none=True)
        audit = {"aligned": 0, "unalignable": 0, "unknown_roles": 0}
        beg = time.time()

        def step():
            nonlocal accum, steps
            if not accum:
                return
            for param in head.parameters():
                if param.grad is not None:
                    param.grad.div_(accum)
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            accum = 0

        for i, d in enumerate(order):
            h, offsets, n = encoding(base, meta, d, cache)
            parsed = load_parsed(R, d)
            events = events_from(read(R / "s1" / filename(d))["events"])
            gold_events = [gold_event(g, parsed) for g in d["events"]]
            pairs = list(zip(gold_events, d["events"])) + align(
                events, d["events"], parsed
            )
            values = []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for e, g in pairs:
                    loss, info = boundary_loss(
                        head(h, offsets, e, parsed, base), offsets, g
                    )
                    if any(g["known"][r] for r in ROLES):
                        values.append(loss)
                    for k, v in info.items():
                        audit[k] += v
                if not values:
                    continue
                loss = torch.stack(values).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(d["key"])
            (loss * n).backward()
            accum += n
            losses.append(float(loss.detach()))
            if accum >= 16:
                step()
            if i % 40 == 0:
                state.update(
                    state="training",
                    epoch=epoch,
                    document_index=i,
                    current_loss=losses[-1],
                )
                save(out / "progress.json", state)
        step()
        if val:
            objective, coverage, rows = evaluate(base, head, meta, val, cache, cfg)
        else:
            objective, coverage, rows = 0.0, {}, []
        row = {
            "epoch": epoch,
            "loss": sum(losses) / len(losses),
            "objective": objective,
            "coverage": coverage,
            "audit": audit,
            "elapsed_s": time.time() - beg,
        }
        state["history"].append(row)
        if objective > best + 1e-9 if val else epoch == a.epochs:
            target = out / f"checkpoint_{epoch:02d}_{time.time_ns()}"
            head.save(
                target,
                {
                    "base_weights_sha256": meta["weights_sha256"],
                    "fold": a.fold,
                    "seed": 20261028,
                    "epoch": epoch,
                    "train_sha256": state["train_sha256"],
                    "training_keys_sha256": __import__("hashlib")
                    .sha256("\n".join(sorted(d["key"] for d in train)).encode())
                    .hexdigest(),
                    "supervision": "original valid aligned 4W labels; no LLM score targets",
                },
            )
            predictions = out / f"epoch_{epoch:02d}.candidates.jsonl"
            predictions.write_text(
                "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                encoding="utf-8",
            )
            state["best"] = {
                "checkpoint": str(target),
                "epoch": epoch,
                "objective": objective,
                "predictions": str(predictions),
            }
            best = objective
            stale = 0
        else:
            stale += 1
        save(out / "progress.json", state)
        print(json.dumps(row), flush=True)
        if val and stale >= 2:
            break
    assert not any(p.requires_grad for p in base.parameters())
    state.update(state="completed", finished_unix=time.time())
    save(out / "progress.json", state)
    save(out / "completed.json", state)


if __name__ == "__main__":
    main()
