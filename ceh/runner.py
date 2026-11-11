import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from .config import load_config, validate_config
from .runtime import run_stream
from collections import Counter

CODE_DIR = Path(__file__).resolve().parents[1]


def apply_overrides(cfg, overrides):
    for item in overrides or []:
        if "=" not in item:
            raise ValueError("--set requires key=value")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise ValueError("Unknown configuration key: " + key)
            node = node[part]
        if parts[-1] not in node:
            raise ValueError("Unknown configuration key: " + key)
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        node[parts[-1]] = value


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def source_hash():
    sha = hashlib.sha256()
    for package in ("ceh", "bench"):
        for path in sorted((CODE_DIR / package).rglob("*.py")):
            sha.update(path.relative_to(CODE_DIR).as_posix().encode("utf-8"))
            sha.update(path.read_bytes())
    return sha.hexdigest()


def read_jsonl(path, limit=None):
    docs, seen = [], set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            did = row.get("doc_id", row.get("id"))
            text, lang = row.get("text"), row.get("lang", "zh")
            if not isinstance(did, str) or not did or did in seen:
                raise ValueError("Invalid/duplicate doc_id on line %d" % line_no)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Missing text on line %d" % line_no)
            if lang not in ("zh", "en", "es"):
                raise ValueError("Unsupported language on line %d" % line_no)
            seen.add(did)
            docs.append(SimpleNamespace(id=did, text=text, lang=lang))
            if limit and len(docs) >= limit:
                break
    if not docs:
        raise ValueError("Input contains no documents")
    return docs


def _existing_rows(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                did = row["doc_id"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    "Invalid output line %d; preserve it and repair before resume."
                    % line_no
                ) from exc
            if did in rows:
                raise ValueError("Duplicate doc_id in output: " + did)
            rows[did] = row
    return rows


def run_documents(
    name,
    docs,
    cfg,
    dry=False,
    tag="anchor",
    output_root=None,
    resume=False,
    overwrite=False,
):
    validate_config(cfg)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag) or tag in (".", ".."):
        raise ValueError("tag must be a simple directory name")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
        raise ValueError("name must be a simple file stem")
    if resume and overwrite:
        raise ValueError("Choose resume or overwrite, not both")
    run_tag = tag + "_dryrun" if dry else tag
    outdir = Path(output_root or CODE_DIR / "outputs" / "raw") / run_tag
    outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / (name + ".jsonl")
    manifest_path = outdir / (name + ".manifest.json")
    from bench.release import doc_lang

    inputs = [(d.id, d.text, doc_lang(d)) for d in docs]
    if len({d[0] for d in inputs}) != len(inputs):
        raise ValueError("Duplicate input document IDs")
    signature = {
        "schema": cfg["method_version"],
        "config_hash": _hash(cfg),
        "source_sha256": source_hash(),
        "input_hash": _hash(inputs),
        "dry_run": dry,
        "n_documents": len(inputs),
    }
    existing = {}
    if outp.exists() and not overwrite:
        if not resume:
            raise FileExistsError(
                "Output exists. Use a new --tag, --resume or --overwrite."
            )
        if not manifest_path.exists():
            raise ValueError("Missing run manifest; cannot safely resume")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["signature"] != signature:
            raise ValueError(
                "Resume refused: code, config, input or dry-run mode changed"
            )
        existing = _existing_rows(outp)
        if not set(existing).issubset({d[0] for d in inputs}):
            raise ValueError("Output has document IDs absent from this input")
    public_cfg = json.loads(json.dumps(cfg))
    if public_cfg["llm"].get("api_key") not in (None, "", "EMPTY"):
        public_cfg["llm"]["api_key"] = "<redacted>"
    if not existing:
        manifest_path.write_text(
            json.dumps(
                {"signature": signature, "config": public_cfg},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    totals = Counter()
    counts = {"ok": 0, "partial": 0, "error": 0}
    for row in existing.values():
        counts[row.get("status", "error")] += 1
    mode = "a" if existing else "w"
    pending = [item for item in inputs if item[0] not in existing]
    with outp.open(mode, encoding="utf-8", newline="\n") as stream:
        for i, (result, counters, records) in enumerate(
            run_stream(pending, cfg, dry), len(existing) + 1
        ):
            row = result.to_json()
            totals.update(counters)
            if records:
                with (outdir / (name + ".requests.jsonl")).open(
                    "a", encoding="utf-8"
                ) as audit:
                    for record in records:
                        audit.write(json.dumps(record, ensure_ascii=False) + "\n")
            row["dry_run"] = dry
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            counts[row["status"]] += 1
            if row["error"]:
                print("%s: %s" % (row["doc_id"], row["error"]), file=sys.stderr)
            if i % 20 == 0 or i == len(inputs):
                print("%s %d/%d: %s" % (name, i, len(inputs), counts))
    summary = {
        "counts": counts,
        "new_requests": totals["n_calls"],
        "json_requests": totals["n_json"],
        "cache_hits": totals["n_cache_hits"],
        "parse_failures": totals["n_parse_fail"],
        "request_failures": totals["n_empty"],
    }
    (outdir / (name + ".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(str(outp))
    return 2 if counts["error"] or counts["partial"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="ANCHOR-5W1H event extraction")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="JSONL: doc_id, text, lang (no gold required)")
    source.add_argument(
        "--corpus",
        help="Corpus names from datasets/manifest.json, comma-separated, or all",
    )
    parser.add_argument(
        "--release-dir", help="Directory containing manifest.json and corpus files"
    )
    parser.add_argument("--config", default=str(CODE_DIR / "config" / "anchor.json"))
    parser.add_argument("--tag", default="anchor")
    parser.add_argument("--output-root")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be positive")
        cfg = load_config(args.config)
        apply_overrides(cfg, args.set)
        validate_config(cfg)
        if os.getenv("ANCHOR_API_KEY"):
            cfg["llm"]["api_key"] = os.environ["ANCHOR_API_KEY"]
        cache = cfg["cache"]
        cache["dir"] = str((CODE_DIR / cache["dir"]).resolve())
        from bench import release

        if args.release_dir:
            release.RELEASE = Path(args.release_dir).resolve()
        kwargs = dict(
            cfg=cfg,
            dry=args.dry_run,
            tag=args.tag,
            output_root=args.output_root,
            resume=args.resume,
            overwrite=args.overwrite,
        )
        if args.input:
            return run_documents("input", read_jsonl(args.input, args.limit), **kwargs)
        names = (
            list(release.catalog()) if args.corpus == "all" else args.corpus.split(",")
        )
        code = 0
        for name in names:
            name = name.strip()
            code = max(
                code,
                run_documents(
                    name, release.load_corpus(name, limit=args.limit), **kwargs
                ),
            )
        return code
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
