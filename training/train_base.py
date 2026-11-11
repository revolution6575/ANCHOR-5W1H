import argparse, copy, hashlib, json, math, os, random, sys, time
from pathlib import Path
from dataclasses import asdict

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
import torch
from ceh.cluster import mark_candidate_pool
from ceh.discriminative.model import SharedDiscriminator
from ceh.discriminative.packets import load_parsed, load_s1
from ceh.discriminative.data import gold_event, align
from ceh.discriminative.engine import event_forward, predict, assemble
from ceh.discriminative.losses import event_loss
from metrics import count_doc, summarize


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save(p, d):
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    t.replace(p)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def profile(cfg, variant):
    if variant not in ("R", "A", "B", "C"):
        raise ValueError("Unknown preregistered architecture")
    cfg = copy.deepcopy(cfg)
    o = cfg["discriminator"]
    o.update(
        boundary_head=True,
        source_weight="equal",
        use_4w=variant in ("R", "B"),
        hard_negatives=variant in ("R", "A"),
        architecture=variant,
        training_rank_lambdas={"why": 0.5, "how": 0.5},
    )
    return cfg


def views(doc, parsed, predicted):
    gold_events = [gold_event(g, parsed) for g in doc["events"]]
    return [
        (
            gold_events,
            [(e, g) for e, g in zip(gold_events, doc["events"])],
            "annotated",
        ),
        (predicted, align(predicted, doc["events"], parsed), "predicted_aligned"),
    ]


def evaluate(model, docs, root, cfg, path):
    model.eval()
    results = []
    raw = []
    start = time.perf_counter()
    coverage = {"gold": 0, "exact": 0, "iou50": 0}
    from ceh.discriminative.policy import iou

    for doc in docs:
        parsed = load_parsed(root, doc)
        packet, events = load_s1(root, doc)
        rows = []
        audit = {}
        if packet["status"] != "error":
            clauses = mark_candidate_pool(parsed, events, cfg)
            rows, audit = predict(model, parsed, events, clauses, cfg)
            assemble(events, rows, parsed, cfg["discriminator"])
        results.append(
            {
                "dataset": doc["dataset"],
                "doc_id": doc["doc_id"],
                "lang": doc["lang"],
                "source": doc["review"].get("Tier", "unknown"),
                "counts": count_doc(doc, events),
            }
        )
        candidates = {(p.start, p.end) for p in rows}
        for node in doc["nodes"]:
            if node["Node_Type"] in ("why", "how"):
                span = (node["Tag_Start"], node["Tag_End"])
                coverage["gold"] += 1
                coverage["exact"] += span in candidates
                coverage["iou50"] += any(iou(span, c) >= 0.5 for c in candidates)
        raw.append(
            {
                "key": doc["key"],
                "dataset": doc["dataset"],
                "doc_id": doc["doc_id"],
                "lang": doc["lang"],
                "text_sha256": doc["text_sha256"],
                "fold": doc["fold"],
                "events": [asdict(e) for e in events],
                "scores": [asdict(p) for p in rows],
                "s1_status": packet["status"],
                "audit": audit,
                "offset_policy": cfg["discriminator"]["offset_policy"],
            }
        )
    summary = summarize(results)
    summary.update(
        elapsed_s=time.perf_counter() - start,
        candidate_coverage=coverage,
        peak_gpu_bytes=torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else 0,
    )
    save(path.with_suffix(".metrics.json"), summary)
    path.write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in raw), encoding="utf-8"
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=os.environ.get("ANCHOR_TRAIN_DIR", str(R / "runs" / "training")),
    )
    p.add_argument("--model", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--fold", type=int, default=-1)
    p.add_argument("--seed", type=int, default=20261025)
    p.add_argument("--variant", default="R")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    root = Path(a.root)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed)
    if (out / "completed.json").exists():
        return
    manifest = read(root / "data/manifest.json")
    assert (
        hashlib.sha256((root / "data/train.jsonl").read_bytes()).hexdigest()
        == manifest["file_hashes"]["train.jsonl"]
    )
    all_docs = [
        json.loads(s)
        for s in (root / "data/train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(all_docs) == 722 and all(d["split"] == "train" for d in all_docs)
    train_docs = [d for d in all_docs if a.fold < 0 or d["fold"] != a.fold]
    val_docs = [d for d in all_docs if d["fold"] == a.fold]
    if a.smoke:
        train_docs = train_docs[:4]
        val_docs = val_docs[:2]
    assert not {d["key"] for d in train_docs} & {d["key"] for d in val_docs}
    cfg = profile(
        read(
            root / "experiment_config.json"
            if (root / "experiment_config.json").exists()
            else R / "config/anchor.json"
        ),
        a.variant,
    )
    model = SharedDiscriminator(a.model, cfg["discriminator"]).float().cuda()
    assert all(p.dtype == torch.float32 for p in model.parameters()), (
        "AdamW requires FP32 master parameters"
    )
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    groups = [
        {"params": list(model.encoder.parameters()), "lr": 1e-5},
        {
            "params": [
                v for k, v in model.named_parameters() if not k.startswith("encoder.")
            ],
            "lr": 1e-4,
        },
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    expected_windows = sum(
        len(model.tokenize(d["text"])["windows"]) for d in train_docs
    )
    total_steps = max(1, math.ceil(expected_windows / 16) * a.epochs)

    def schedule(step):
        return min(1, (step + 1) / max(1, int(total_steps * 0.1))) * max(
            0, (total_steps - step) / max(1, total_steps - int(total_steps * 0.1))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    state = {
        "state": "training",
        "model": a.model_name,
        "variant": a.variant,
        "fold": a.fold,
        "seed": a.seed,
        "training_documents": len(train_docs),
        "validation_documents": len(val_docs),
        "train_manifest_sha256": hashlib.sha256(
            (root / "data/manifest.json").read_bytes()
        ).hexdigest(),
        "best": None,
        "started_unix": time.time(),
        "history": [],
    }
    state["precision_policy"] = "fp32_master_weights_and_adamw_moments_bf16_autocast_r1"
    state["offset_policy"] = cfg["discriminator"]["offset_policy"]
    state["early_stopping_objective"] = "legacy_exact_character_unchanged_from_1025"
    save(out / "config.json", cfg)
    save(out / "progress.json", state)
    best = -1.0
    stale = 0
    accum = 0
    step = 0
    optimizer.zero_grad(set_to_none=True)
    first_epoch = 1
    if (out / "resume.pt").exists():
        resume = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        state = resume["state"]
        best = resume["best"]
        stale = resume["stale"]
        step = resume["step"]
        first_epoch = resume["epoch"] + 1
        torch.set_rng_state(resume["rng"])
        torch.cuda.set_rng_state_all(resume["cuda_rng"])
        del resume

    def update():
        nonlocal accum, step
        if not accum:
            return
        for v in model.parameters():
            if v.grad is not None:
                v.grad.div_(accum)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        accum = 0
        state["optimizer_steps"] = step

    for epoch in range(first_epoch, a.epochs + 1):
        model.train()
        docs = list(train_docs)
        random.Random(a.seed + epoch).shuffle(docs)
        losses = []
        audit_totals = {}
        epoch_start = time.time()
        used = 0
        for index, doc in enumerate(docs):
            if (
                a.variant == "human_only"
                and doc["review"].get("Provenance") != "human_gold"
            ):
                continue
            parsed = load_parsed(root, doc)
            _, predicted = load_s1(root, doc)
            source_weight = (
                0.5
                if a.variant == "l3_half" and doc["review"].get("Tier") == "L3"
                else 1.0
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h, offsets, nwindows = model.encode(doc["text"])
                doc_losses = []
                for events, pairs, view in views(doc, parsed, predicted):
                    clauses = mark_candidate_pool(parsed, events, cfg)
                    for event, gold in pairs:
                        spans = [
                            tuple(s)
                            for role in ("why", "how")
                            for s in gold["slots"][role]
                            if gold["known"][role]
                        ]
                        spans.extend(
                            (node["Tag_Start"], node["Tag_End"])
                            for node in doc["nodes"]
                        )
                        cs, logits, bs, bl, features = event_forward(
                            model,
                            h,
                            offsets,
                            event,
                            events,
                            clauses,
                            parsed,
                            cfg,
                            spans,
                        )
                        loss, audit = event_loss(
                            cs,
                            logits,
                            bs,
                            bl,
                            offsets,
                            gold,
                            cfg["discriminator"],
                            a.seed + epoch * 1000 + index,
                        )
                        doc_losses.append(loss)
                        for k, v in audit.items():
                            audit_totals[k] = audit_totals.get(k, 0) + v
                if doc_losses:
                    loss = torch.stack(doc_losses).mean() * source_weight
                else:
                    continue
            if not loss.requires_grad:
                audit_totals["no_supervision_skipped"] = (
                    audit_totals.get("no_supervision_skipped", 0) + 1
                )
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss: " + doc["key"])
            (loss * nwindows).backward()
            accum += nwindows
            losses.append(float(loss.detach()))
            used += 1
            if accum >= 16:
                update()
            if index % 20 == 0:
                state.update(
                    epoch=epoch,
                    document_index=index,
                    optimizer_steps=step,
                    last_key=doc["key"],
                    current_loss=losses[-1],
                )
                save(out / "progress.json", state)
        update()
        if not losses:
            raise ValueError("No usable training supervision")
        summary = (
            evaluate(
                model, val_docs, root, cfg, out / f"epoch_{epoch:02d}.predictions.jsonl"
            )
            if val_docs
            else None
        )
        objective = (
            summary["legacy_objective"] if summary else -sum(losses) / len(losses)
        )
        history = {
            "epoch": epoch,
            "loss": sum(losses) / len(losses),
            "objective": objective,
            "train_used": used,
            "elapsed_s": time.time() - epoch_start,
            "loss_audit": audit_totals,
        }
        state["history"].append(history)
        improving = objective > best + 1e-8 if val_docs else epoch == a.epochs
        if improving:
            checkpoint = out / f"checkpoint_{epoch:02d}_{time.time_ns()}"
            metadata = {
                "seed": a.seed,
                "variant": a.variant,
                "fold": a.fold,
                "epoch": epoch,
                "manifest_sha256": state["train_manifest_sha256"],
                "model_name": a.model_name,
                "split": "training_pool_only",
                "precision_policy": state["precision_policy"],
                "s1_protocol": cfg.get("s1_factorial"),
                "s1_snapshot_manifest_sha256": hashlib.sha256(
                    (root / "s1_manifest.json").read_bytes()
                ).hexdigest()
                if (root / "s1_manifest.json").exists()
                else None,
            }
            model.save(checkpoint, metadata)
            state["best"] = {
                "checkpoint": str(checkpoint),
                "epoch": epoch,
                "objective": objective,
                "predictions": str(out / f"epoch_{epoch:02d}.predictions.jsonl")
                if val_docs
                else None,
            }
            best = objective
            stale = 0
        else:
            stale += 1
        save(out / "progress.json", state)
        print(json.dumps(history), flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "state": state,
                "best": best,
                "stale": stale,
                "step": step,
                "epoch": epoch,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            out / "resume.tmp",
        )
        (out / "resume.tmp").replace(out / "resume.pt")
        if val_docs and stale >= 2:
            break
    state.update(state="completed", finished_unix=time.time())
    save(out / "progress.json", state)
    save(out / "completed.json", state)


if __name__ == "__main__":
    main()
