import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from common import PROJECT, R, configuration, read, save, sha

sys.path.insert(0, str(PROJECT))
from bench.release import catalog, records
from ceh.cluster import build_cluster
from ceh.discriminative.data import normalize
from ceh.discriminative.packets import filename, parsed_dict
from ceh.llm import LLM
from ceh.parsing import parse_document
from ceh.runner import apply_overrides, source_hash
from ceh.config import validate_config


def training_documents():
    pool = read(PROJECT / "training" / "pool.json")
    by_dataset = {}
    for row in pool:
        by_dataset.setdefault(row["dataset"], {})[row["doc_id"]] = row
    selected = {}
    corpora = catalog()
    for dataset, wanted in by_dataset.items():
        for record in records(dataset):
            ref = wanted.get(record["Id"])
            if ref is None:
                continue
            doc = normalize(record, dataset, corpora[dataset]["language"])
            digest = hashlib.sha256(doc["text"].encode()).hexdigest()
            if digest != ref["text_sha256"] or doc["key"] in selected:
                raise ValueError(f"Pool mismatch for {doc['key']}")
            doc.update(text_sha256=digest, split="train", fold=ref["fold"])
            selected[doc["key"]] = doc
    return [selected[row["dataset"] + "/" + row["doc_id"]] for row in pool]


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild the training pool, parse its documents and collect S1 drafts."
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Rebuild the annotated pool and skip model requests",
    )
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    docs = training_documents()
    folder = R / "data"
    folder.mkdir(parents=True, exist_ok=True)
    dataset = folder / "train.jsonl"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=folder, delete=False, newline="\n"
    ) as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    Path(f.name).replace(dataset)
    save(
        folder / "manifest.json",
        {"records": len(docs), "file_hashes": {dataset.name: sha(dataset)}},
    )
    if args.data_only:
        print(f"{len(docs)} training documents in {dataset}")
        return

    cfg = configuration()
    apply_overrides(cfg, args.set)
    validate_config(cfg)
    signature = hashlib.sha256(
        json.dumps([source_hash(), cfg], sort_keys=True).encode()
    ).hexdigest()
    if os.getenv("ANCHOR_API_KEY"):
        cfg["llm"]["api_key"] = os.environ["ANCHOR_API_KEY"]
    llm = LLM(cfg)
    for doc in docs:
        packet_name = filename(doc)
        s1 = R / "s1" / packet_name
        syntax = R / "parsed" / packet_name
        if s1.exists() and syntax.exists() and read(s1).get("signature") == signature:
            continue
        parsed = parse_document(doc["text"], doc["lang"])
        if cfg["input"]["require_syntax"] and not parsed.has_complete_syntax:
            raise RuntimeError(f"Missing spaCy parser for {doc['lang']}")
        events = build_cluster(parsed, llm, cfg)
        save(syntax, parsed_dict(parsed))
        save(
            s1,
            {
                "signature": signature,
                "text_sha256": doc["text_sha256"],
                "status": "ok",
                "events": [asdict(e) for e in events],
            },
        )


if __name__ == "__main__":
    main()
