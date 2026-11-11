import json
from pathlib import Path
from ..checksums import sha256_stream
import torch
from torch import nn
from ..discriminative.offsets import OFFSET_POLICY, boundary_targets
from ..discriminative.losses import balanced_bce

ROLES = ("who", "what", "when", "where")
HEAD_VERSION = "fourw-boundaries-v1"


class FourWHead(nn.Module):
    def __init__(self, hidden, projection=256):
        super().__init__()
        self.hidden = hidden
        self.projection = projection
        self.event = nn.Sequential(
            nn.Linear(2 * hidden + 1, projection), nn.GELU(), nn.Dropout(0.1)
        )
        self.token = nn.ModuleList([nn.Linear(hidden, projection) for _ in range(8)])
        self.condition = nn.ModuleList(
            [nn.Linear(projection, projection, bias=False) for _ in range(8)]
        )
        self.out = nn.ModuleList([nn.Linear(projection, 1) for _ in range(8)])

    def forward(self, h, offsets, event, parsed, base):
        sentence = parsed.sentences[event.sent_idx]
        u = self.event(
            torch.cat(
                [
                    base.pool(h, offsets, event.span),
                    base.pool(h, offsets, (sentence.start, sentence.end)),
                    h.new_tensor([float(event.is_central)]),
                ]
            )
        )
        return torch.stack(
            [
                out(torch.tanh(tok(h) + cond(u))).squeeze(-1)
                for tok, cond, out in zip(self.token, self.condition, self.out)
            ]
        ).reshape(4, 2, len(h))

    def save(self, path, metadata):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        torch.save(self.state_dict(), path / "head.pt")
        with (path / "head.pt").open("rb") as f:
            digest = sha256_stream(f)
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    **metadata,
                    "head_version": HEAD_VERSION,
                    "offset_policy": OFFSET_POLICY,
                    "hidden": self.hidden,
                    "projection": self.projection,
                    "head_sha256": digest,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path, base_metadata, device):
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if (
            meta["head_version"] != HEAD_VERSION
            or meta["offset_policy"] != OFFSET_POLICY
        ):
            raise ValueError("4W head version mismatch")
        if meta["base_weights_sha256"] != base_metadata["weights_sha256"]:
            raise ValueError("4W head belongs to a different frozen encoder")
        with (path / "head.pt").open("rb") as f:
            if sha256_stream(f) != meta["head_sha256"]:
                raise ValueError("4W head checksum mismatch")
        head = cls(meta["hidden"], meta["projection"])
        head.load_state_dict(
            torch.load(path / "head.pt", map_location="cpu", weights_only=True)
        )
        head.to(device).eval()
        return head, meta


def boundary_loss(logits, offsets, gold):
    losses = []
    audit = {"aligned": 0, "unalignable": 0, "unknown_roles": 0}
    for r, role in enumerate(ROLES):
        if not gold["known"][role]:
            audit["unknown_roles"] += 1
            continue
        target, mask, known, unknown = boundary_targets(offsets, gold["slots"][role])
        audit["aligned"] += len(known)
        audit["unalignable"] += len(unknown)
        y = logits.new_tensor(target)
        valid = torch.tensor(mask, device=logits.device, dtype=torch.bool)
        for side in (0, 1):
            if valid[side].any():
                losses.append(balanced_bce(logits[r, side], y[side], valid[side]))
    return (torch.stack(losses).mean() if losses else logits.sum() * 0), audit
