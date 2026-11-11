import json
import math
from pathlib import Path
from .schema import VERSION


def load_config(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        cfg = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError(
                "JSON config needs no dependency; YAML needs PyYAML."
            ) from exc
        cfg = yaml.safe_load(text)
    validate_config(cfg)
    return cfg


def _number(value, name, low=0, high=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < low
        or (high is not None and value > high)
    ):
        raise ValueError("Invalid numeric configuration: " + name)


def _integer(value, name, low=0):
    _number(value, name, low)
    if not isinstance(value, int):
        raise ValueError("Expected integer: " + name)


def validate_config(cfg):
    if not isinstance(cfg, dict) or cfg.get("method_version") != VERSION:
        raise ValueError("Configuration method_version must match " + VERSION)
    spec = cfg.get("s1_factorial", {})
    if spec.get("enabled", False):
        if spec.get("mode") not in ("joint", "parallel", "sequential") or spec.get(
            "anchor_policy"
        ) not in ("post", "early"):
            raise ValueError(
                "Unknown online S1 mode or anchor policy; fixed views are diagnostic only"
            )
    if any(key in cfg for key in ("assign", "verify", "hero")):
        raise ValueError("Removed options: assign, verify, hero.")
    if cfg.get("scoring", {}).get("backend") == "cooperative":
        o = cfg.get("cooperative", {})
        if o.get("version") != "coop-4w-v1" or not o.get("enabled"):
            raise ValueError("Cooperative version/config mismatch")
        if (
            cfg.get("s1_factorial", {}).get("anchor_policy") != "post"
            or cfg["s1_factorial"]["mode"] != "joint"
        ):
            raise ValueError("Cooperative baseline is fixed post-joint S1")
        for k in ("top_positions", "learned_candidates", "max_span_tokens"):
            _integer(o[k], "cooperative." + k, 1)
        _integer(o["syntax_candidates"], "cooperative.syntax_candidates", 0)
        if (
            o.get("refine_scope") != "all_retained"
            or not o.get("refresh_why_how")
            or not o.get("allow_outside")
        ):
            raise ValueError("Unsupported cooperation protocol")
    if cfg.get("scoring", {}).get("backend") in ("discriminator", "cooperative"):
        c = cfg["discriminator"]
        if cfg.get("runtime", {}).get("documents", 1) != 1:
            raise ValueError("Discriminator runtime is serial by document")
        if c.get("fusion") not in ("gated", "additive", "legacy_additive"):
            raise ValueError("Unknown fusion")
        if c.get("conflict") not in ("exact", "unit", "span"):
            raise ValueError("Unknown conflict")
        for r in ("why", "how"):
            _number(c["temperatures"][r], "temperature", 0.001)
            _number(c["thresholds"][r], "threshold", 0.05, 0.95)
            _number(c["boundary_lambdas"][r], "boundary_lambda", 0, 1)
    try:
        if cfg["scoring"].get("evaluation", "exhaustive") not in (
            "exhaustive",
            "certified",
        ):
            raise ValueError("scoring.evaluation must be exhaustive or certified")
        _integer(cfg["scoring"].get("query_window", 8), "query_window", 1)
        for key, default in (("documents", 1), ("max_inflight", 16)):
            _integer(cfg.get("runtime", {}).get(key, default), "runtime." + key, 1)
        for key in ("max_listed_sentences", "context_pad"):
            _integer(cfg["input"][key], "input." + key)
        _integer(cfg["cluster"]["max_events"], "cluster.max_events", 1)
        _number(cfg["cluster"]["admit_threshold"], "cluster.admit_threshold", 0, 1)
        for key in ("max_non_event_units", "max_variants_per_unit"):
            _integer(cfg["candidates"][key], "candidates." + key, 1)
        for role in ("why", "how"):
            _number(cfg["selection"]["thresholds"][role], "thresholds." + role)
            _integer(cfg["selection"]["max_fillers"][role], "max_fillers." + role)
        for section, keys in (
            (cfg["cluster"]["ev_weights"], ("pos", "trig", "cov", "cent")),
            (cfg["scoring"]["why_weights"], ("llm", "dir", "cue", "dist")),
            (cfg["scoring"]["how_weights"], ("llm", "sub", "sty", "cue")),
        ):
            if set(section) != set(keys):
                raise ValueError("Unexpected weight keys: " + str(list(section)))
            for key in keys:
                _number(section[key], "weight." + key)
        for section, key in (
            ("input", "require_syntax"),
            ("cluster", "enabled"),
            ("candidates", "group_variants"),
            ("candidates", "cross_sentence_core"),
            ("scoring", "typed"),
            ("scoring", "full_event_context"),
            ("selection", "exclusive_roles"),
            ("output", "include_trace"),
            ("cache", "enabled"),
        ):
            if not isinstance(cfg[section][key], bool):
                raise ValueError("Expected boolean: " + section + "." + key)
        if cfg["scoring"]["temporal"] not in ("explicit", "off"):
            raise ValueError("scoring.temporal must be explicit or off")
        if cfg["scoring"]["cue_scope"] not in ("candidate", "interval"):
            raise ValueError("scoring.cue_scope must be candidate or interval")
        for key in ("max_tokens", "concurrency", "max_retries"):
            _integer(cfg["llm"][key], "llm." + key, 1)
        for key in ("temperature", "retry_backoff_s"):
            _number(cfg["llm"][key], "llm." + key)
        _number(cfg["llm"]["timeout_s"], "llm.timeout_s", 0.001)
        for key in ("base_url", "model"):
            if not isinstance(cfg["llm"][key], str) or not cfg["llm"][key].strip():
                raise ValueError("Expected nonempty llm." + key)
    except KeyError as exc:
        raise ValueError("Missing configuration key: " + str(exc)) from exc
    return cfg
