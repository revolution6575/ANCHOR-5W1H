import json
import os
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
R = Path(os.environ.get("ANCHOR_TRAIN_DIR", PROJECT / "runs" / "training")).resolve()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as f:
        tmp = Path(f.name)
        try:
            json.dump(value, f, ensure_ascii=False)
        except BaseException:
            f.close()
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(path)


def sha(path):
    from ceh.checksums import sha256_stream

    with Path(path).open("rb") as f:
        return sha256_stream(f)


def documents():
    path = R / "data" / "train.jsonl"
    if sha(path) != read(path.with_name("manifest.json"))["file_hashes"][path.name]:
        raise ValueError("Training data checksum mismatch")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def configuration():
    from ceh.config import load_config

    return load_config(PROJECT / "config" / "anchor.json")
