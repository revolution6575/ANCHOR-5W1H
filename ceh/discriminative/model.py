import math
import os
from pathlib import Path
from ..checksums import sha256_stream
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from .policy import token_span, window_ranges
from .offsets import content_offsets, validate_mapping_options, OFFSET_POLICY


class SharedDiscriminator(nn.Module):
    def __init__(self, model_path, cfg):
        super().__init__()
        validate_mapping_options(cfg)
        self.options = dict(cfg)
        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, local_files_only=True
        )
        if not self.tokenizer.is_fast:
            raise ValueError("Exact offset mapping requires fast tokenizer")
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        config.reference_compile = False
        try:
            self.encoder = AutoModel.from_pretrained(
                model_path,
                config=config,
                local_files_only=True,
                dtype=torch.float32,
                attn_implementation="sdpa",
            )
        except ValueError as exc:
            if not any(
                s in str(exc).lower() for s in ("sdpa", "scaled_dot_product_attention")
            ):
                raise
            self.encoder = AutoModel.from_pretrained(
                model_path,
                config=config,
                local_files_only=True,
                dtype=torch.float32,
                attn_implementation="eager",
            )
        self.encoder.float()
        hidden = config.hidden_size
        d = cfg.get("projection", 256)
        rank = cfg.get("rank", 64)
        drop = cfg.get("dropout", 0.1)

        def mlp(a, b):
            return nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Dropout(drop))

        self.event_project = mlp(
            5 * hidden + 4 if cfg.get("use_4w", True) else hidden, d
        )
        self.length_embedding = nn.Embedding(257, 32)
        self.candidate_project = mlp(3 * hidden + 32, d)
        self.role_left = nn.ModuleList(
            [nn.Linear(d, rank, bias=False) for _ in range(2)]
        )
        self.role_right = nn.ModuleList(
            [nn.Linear(d, rank, bias=False) for _ in range(2)]
        )
        self.feature_dims = (4 if cfg.get("use_4w", True) else 2, 4)
        self.role_mlp = nn.ModuleList(
            [
                nn.Sequential(mlp(4 * d + n, d), nn.Linear(d, 1))
                for n in self.feature_dims
            ]
        )
        self.boundary_token = nn.ModuleList([nn.Linear(hidden, d) for _ in range(4)])
        self.boundary_event = nn.ModuleList(
            [nn.Linear(d, d, bias=False) for _ in range(4)]
        )
        self.boundary_out = nn.ModuleList(
            [nn.Linear(d, 1, bias=False) for _ in range(4)]
        )
        self.hidden = hidden
        self.rank = rank
        self.encoding_calls = 0
        self._token_cache = {}
        import hashlib

        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        self.tokenizer_fingerprint = hashlib.sha256(
            (backend.to_str() if backend else str(model_path)).encode()
        ).hexdigest()

    @property
    def device(self):
        return next(self.parameters()).device

    def tokenize(self, text):
        cache_key = (
            OFFSET_POLICY,
            self.tokenizer_fingerprint,
            self.options.get("window", 512),
            self.options.get("overlap", 128),
            text,
        )
        if cache_key in self._token_cache:
            return self._token_cache[cache_key]
        window = self.options.get("window", 512)
        overlap = self.options.get("overlap", 128)
        if window <= overlap + self.tokenizer.num_special_tokens_to_add(False):
            raise ValueError("Invalid encoder window")
        raw = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        probe = self.tokenizer("Boundary", add_special_tokens=False)["input_ids"]
        wrapped = self.tokenizer("Boundary", add_special_tokens=True)["input_ids"]
        location = next(
            (
                i
                for i in range(len(wrapped) - len(probe) + 1)
                if wrapped[i : i + len(probe)] == probe
            ),
            None,
        )
        if location is None:
            raise ValueError("Cannot determine tokenizer special-token wrapper")
        prefix, suffix = wrapped[:location], wrapped[location + len(probe) :]
        capacity = window - len(prefix) - len(suffix)
        windows = []
        for a, b in window_ranges(len(raw["input_ids"]), capacity, overlap):
            ids = prefix + raw["input_ids"][a:b] + suffix
            windows.append(
                {
                    "ids": ids,
                    "body": list(range(len(prefix), len(prefix) + b - a)),
                    "start": a,
                }
            )
        if not raw["input_ids"]:
            raise ValueError("Document has no tokens")
        original = [tuple(x) for x in raw["offset_mapping"]]
        content = content_offsets(text, original)
        value = {
            "raw_offsets": original,
            "content_offsets": content,
            "offsets": content,
            "valid_boundaries": [a < b for a, b in content],
            "offset_policy": OFFSET_POLICY,
            "windows": windows,
            "n": len(raw["input_ids"]),
        }
        if len(self._token_cache) < 8000:
            self._token_cache[cache_key] = value
        return value

    def encode(self, text):
        data = self.tokenize(text)
        windows = data["windows"]
        pieces = []
        locations = []
        batch_size = self.options.get("window_batch_size", 4)
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            length = max(len(w["ids"]) for w in batch)
            ids = torch.full(
                (len(batch), length),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            attention = torch.zeros_like(ids)
            for i, w in enumerate(batch):
                ids[i, : len(w["ids"])] = torch.tensor(w["ids"], device=self.device)
                attention[i, : len(w["ids"])] = 1
            h = self.encoder(input_ids=ids, attention_mask=attention).last_hidden_state
            self.encoding_calls += len(batch)
            for i, w in enumerate(batch):
                pieces.append(h[i, w["body"]].float())
                locations.append(
                    torch.arange(
                        w["start"], w["start"] + len(w["body"]), device=self.device
                    )
                )
        index = torch.cat(locations)
        source = torch.cat(pieces)
        total = source.new_zeros((data["n"], self.hidden)).index_add(0, index, source)
        count = source.new_zeros((data["n"],)).index_add(
            0, index, source.new_ones(len(index))
        )
        if bool((count == 0).any()):
            raise AssertionError("Unencoded document tail")
        return total / count[:, None], data["offsets"], len(windows)

    def pool(self, h, offsets, span):
        indices = token_span(offsets, span)
        return (
            h[indices[0] : indices[1] + 1].mean(0)
            if indices
            else h.new_zeros(self.hidden)
        )

    def event_vector(self, h, offsets, event):
        if not self.options.get("use_4w", True):
            return self.event_project(self.pool(h, offsets, event.span))
        parts = [self.pool(h, offsets, event.span)]
        mask = []
        for role in ("who", "what", "when", "where"):
            el = event.elements.get(role)
            present = bool(el and el.is_span() and self.options.get("use_4w", True))
            if present:
                spans = [(v.start, v.end) for v in el.evidence] or [el.span]
                indices = sorted(
                    {
                        i
                        for span in spans
                        for ids in [token_span(offsets, span)]
                        if ids is not None
                        for i in range(ids[0], ids[1] + 1)
                    }
                )
                parts.append(
                    h[indices].mean(0) if indices else h.new_zeros(self.hidden)
                )
            else:
                parts.append(h.new_zeros(self.hidden))
            mask.append(float(present))
        return self.event_project(torch.cat(parts + [h.new_tensor(mask)]))

    def boundaries(self, h, u):
        if not self.options.get("boundary_head", True):
            return h.new_zeros((2, 2, len(h)))
        return torch.stack(
            [
                out(torch.tanh(tok(h) + ev(u))).squeeze(-1)
                for tok, ev, out in zip(
                    self.boundary_token, self.boundary_event, self.boundary_out
                )
            ]
        ).reshape(2, 2, len(h))

    def candidates(self, h, offsets, clauses):
        kept = []
        ranges = []
        for c in clauses:
            span = token_span(offsets, c.span)
            if span:
                kept.append(c)
                ranges.append(span)
        if not kept:
            return kept, h.new_empty((0, self.options.get("projection", 256))), ranges
        starts = torch.tensor([a for a, b in ranges], device=h.device)
        ends = torch.tensor([b for a, b in ranges], device=h.device)
        prefix = torch.cat([h.new_zeros((1, h.shape[-1])), h.cumsum(0)], 0)
        means = (prefix[ends + 1] - prefix[starts]) / (ends - starts + 1)[:, None]
        lengths = (ends - starts + 1).clamp(max=256)
        return (
            kept,
            self.candidate_project(
                torch.cat(
                    [h[starts], h[ends], means, self.length_embedding(lengths)], -1
                )
            ),
            ranges,
        )

    def role_logits(self, u, v, features):
        if not len(v):
            return v.new_empty((0, 2))
        e = u.expand(len(v), -1)
        out = []
        for r in range(2):
            pair = torch.cat(
                [e, v, e * v, (e - v).abs(), features[:, r, : self.feature_dims[r]]], -1
            )
            low_rank = (self.role_left[r](e) * self.role_right[r](v)).sum(
                -1
            ) / math.sqrt(self.rank)
            out.append(low_rank + self.role_mlp[r](pair).squeeze(-1))
        return torch.stack(out, -1)

    def save(self, path, metadata):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        import json

        torch.save(self.state_dict(), path / "weights.pt")
        with (path / "weights.pt").open("rb") as f:
            metadata = {**metadata, "weights_sha256": sha256_stream(f)}
        provenance = Path(self.model_path) / "download_manifest.json"
        if provenance.exists():
            metadata["encoder_provenance"] = json.loads(
                provenance.read_text(encoding="utf-8")
            )
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "model_path": self.model_path,
                    "options": self.options,
                    "offset_policy": OFFSET_POLICY,
                    "tokenizer_fingerprint": self.tokenizer_fingerprint,
                    **metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path, device="cuda"):
        import json

        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        with (path / "weights.pt").open("rb") as f:
            if sha256_stream(f) != meta["weights_sha256"]:
                raise ValueError("Discriminator weights/metadata checksum mismatch")
        validate_mapping_options(meta)
        validate_mapping_options(meta["options"])
        model = cls(
            os.environ.get("ANCHOR_ENCODER_DIR") or meta["model_path"], meta["options"]
        )
        model.load_state_dict(
            torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        )
        if model.tokenizer_fingerprint != meta["tokenizer_fingerprint"]:
            raise ValueError("Tokenizer fingerprint mismatch")
        model.to(device).eval()
        return model, meta
