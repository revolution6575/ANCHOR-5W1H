import time
import torch
from ..cluster import mark_candidate_pool
from ..discriminative.model import SharedDiscriminator
from ..discriminative.engine import event_forward
from ..discriminative.policy import RoleScore
from .heads import FourWHead
from .candidates import propose
from .refine import refine_events


def role_rows_from_encoding(model, h, offsets, parsed, events, cfg):
    clauses = mark_candidate_pool(parsed, events, cfg)
    rows = []
    with torch.autocast(
        device_type=model.device.type,
        dtype=torch.bfloat16,
        enabled=model.device.type == "cuda",
    ):
        for event in events:
            cs, logits, boundary, _, fs = event_forward(
                model, h, offsets, event, events, clauses, parsed, cfg
            )
            for i, c in enumerate(cs):
                for j, r in enumerate(("why", "how")):
                    rows.append(
                        RoleScore(
                            event.eid,
                            c.cid,
                            c.unit_id,
                            r,
                            c.start,
                            c.end,
                            c.text,
                            c.kind,
                            float(logits[i, j]),
                            float(boundary[i, j]),
                            fs[i][j],
                            c.source_event_id,
                        )
                    )
    return rows, clauses


class CooperativeBackend:
    def __init__(self, cfg):
        self.model, self.metadata = SharedDiscriminator.load(
            cfg["discriminator"]["checkpoint"],
            cfg["discriminator"].get("device", "cuda"),
        )
        self.head, self.head_metadata = FourWHead.load(
            cfg["cooperative"]["head_checkpoint"], self.metadata, self.model.device
        )

    @torch.inference_mode()
    def predict(self, parsed, events, clauses, cfg, llm):
        before = self.model.encoding_calls
        t = time.perf_counter()
        with torch.autocast(
            device_type=self.model.device.type,
            dtype=torch.bfloat16,
            enabled=self.model.device.type == "cuda",
        ):
            h, offsets, n = self.model.encode(parsed.text)
            candidate_sets = {}
            for event in events:
                logits = self.head(h, offsets, event, parsed, self.model)
                candidate_sets[event.eid] = propose(
                    parsed,
                    event,
                    offsets,
                    logits.float().sigmoid().cpu().tolist(),
                    cfg["cooperative"],
                )
        times = {"s2_encode_candidates_s": time.perf_counter() - t}
        t = time.perf_counter()
        refinement = refine_events(parsed, events, candidate_sets, llm)
        times["s3_4w_refine_s"] = time.perf_counter() - t
        t = time.perf_counter()
        before_roles = getattr(llm, "n_calls", 0)
        rows, clauses = role_rows_from_encoding(
            self.model, h, offsets, parsed, events, cfg
        )
        if getattr(llm, "n_calls", 0) != before_roles:
            raise AssertionError("LLM call in Why/How stage")
        if self.model.encoding_calls - before != n:
            raise AssertionError("Document was encoded more than once per window")
        times["s4_role_heads_s"] = time.perf_counter() - t
        return rows, {
            "version": "coop-4w-v1",
            "encoder_windows": n,
            "encoder_forward_windows": self.model.encoding_calls - before,
            "reused_document_encoding": True,
            "refinement": refinement,
            "candidates": candidate_sets,
            "n_clauses": len(clauses),
            "stage_times": times,
            "s4_llm_calls": 0,
            "offset_policy": self.model.options["offset_policy"],
            "head_metadata": self.head_metadata,
        }


_BACKENDS = {}


def get_backend(cfg):
    key = (
        cfg["discriminator"]["checkpoint"],
        cfg["cooperative"]["head_checkpoint"],
        cfg["discriminator"].get("device", "cuda"),
    )
    if not all(key):
        raise ValueError(
            "Cooperation requires explicit encoder and 4W head checkpoints"
        )
    if key not in _BACKENDS:
        _BACKENDS[key] = CooperativeBackend(cfg)
    return _BACKENDS[key]
