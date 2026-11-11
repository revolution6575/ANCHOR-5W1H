import gzip
from ceh.checksums import sha256_stream
from pathlib import Path

from .release import RELEASE, catalog, records


def digest(path, opener=open):
    with opener(path, "rb") as stream:
        return sha256_stream(stream)


def verify():
    root = Path(RELEASE)
    split_dir = root / "splits"
    totals = [0, 0]
    for name, item in catalog().items():
        path = root / item["path"]
        if (
            digest(path) != item["sha256_gzip"]
            or digest(path, gzip.open) != item["sha256_json"]
        ):
            raise ValueError(f"File checksum mismatch: {path}")
        ids = []
        events = clusters = 0
        for record in records(name):
            ids.append(record["Id"])
            clusters += len(record["Event_Clusters"])
            events += sum(
                len(cluster["Events"]) for cluster in record["Event_Clusters"]
            )
        unique = set(ids)
        if (len(ids), len(unique), clusters, events) != (
            item["records"],
            item["records"],
            item["clusters"],
            item["events"],
        ):
            raise ValueError(f"Count or duplicate ID mismatch: {name}")

        def members(prefix, split):
            p = split_dir / f"{prefix}_{name}_{split}.txt"
            return p.read_text(encoding="utf-8-sig").splitlines()

        if members("natural", "all") != ids:
            raise ValueError(f"Record order mismatch: {name}")
        splits = []
        for split in ("all", "train", "dev", "test"):
            if not (split_dir / f"natural_{name}_{split}.txt").exists():
                continue
            natural = members("natural", split)
            allowed = set(natural)
            if len(allowed) != len(natural) or not allowed <= unique:
                raise ValueError(f"Invalid IDs in {name}/{split}")
            if split != "all":
                splits.extend(natural)
            for prefix in ("enriched", "enriched_event", "enriched_record_legacy"):
                subset = members(prefix, split)
                if len(set(subset)) != len(subset) or not set(subset) <= allowed:
                    raise ValueError(f"Invalid IDs in {prefix}/{name}/{split}")
            if members("enriched", split) != members("enriched_record_legacy", split):
                raise ValueError(f"Legacy split alias mismatch: {name}/{split}")
        if splits and (len(splits) != len(ids) or set(splits) != unique):
            raise ValueError(f"Incomplete or overlapping partitions: {name}")
        totals[0] += len(ids)
        totals[1] += events
        print(f"{name:14} {len(ids):6} documents  {events:6} events")
    print(f"Total          {totals[0]:6} documents  {totals[1]:6} events")


if __name__ == "__main__":
    verify()
