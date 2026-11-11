import time
from dataclasses import asdict
from .cluster import build_cluster, mark_candidate_pool
from .config import validate_config
from .parsing import parse_document
from .schema import DocResult, ERROR
from .scoring import score_candidates
from .selection import select_evidence


def run_ours(doc_id, text, lang, llm, cfg, scorer=None):
    started = time.perf_counter()
    stage_times = {}

    def context(stage_name):
        if hasattr(llm, "context"):
            llm.context.update(doc_id=doc_id, lang=lang, stage=stage_name)

    before = getattr(llm, "n_calls", 0)
    json_before = getattr(llm, "n_json", 0)
    result = DocResult(doc_id=doc_id, lang=lang)
    stage = "input"
    try:
        validate_config(cfg)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Document text must be a nonempty string")
        if lang not in ("zh", "en", "es"):
            raise ValueError("Unsupported language: " + str(lang))
        stage = "S1"
        t = time.perf_counter()
        parsed = parse_document(text, lang)
        stage_times["parse_s"] = time.perf_counter() - t
        limit = cfg["input"]["max_listed_sentences"]
        result.stats = {
            "n_sentences": len(parsed.sentences),
            "syntax": parsed.has_complete_syntax,
            "s1_truncated": bool(limit and len(parsed.sentences) > limit),
            "s1_visible_sentences": min(
                limit or len(parsed.sentences), len(parsed.sentences)
            ),
        }
        if cfg["input"]["require_syntax"] and not parsed.has_complete_syntax:
            raise RuntimeError(
                "spaCy syntax unavailable; install the language model or "
                "explicitly set input.require_syntax=false for diagnostics"
            )
        t = time.perf_counter()
        context("S1")
        events = build_cluster(parsed, llm, cfg)
        stage_times["s1_s"] = time.perf_counter() - t
        s1_requests = getattr(llm, "n_json", 0) - json_before
        result.stats.update(parsed.diagnostics)
        result.events = events
        t = time.perf_counter()
        clauses = mark_candidate_pool(parsed, events, cfg)
        stage_times["candidate_pool_s"] = time.perf_counter() - t
        stage = "S2"
        t = time.perf_counter()
        context("S2")
        cooperative = cfg["scoring"].get("backend") == "cooperative"
        discriminative = cfg["scoring"].get("backend") in (
            "discriminator",
            "cooperative",
        )
        before_roles = getattr(llm, "n_calls", 0)
        if discriminative:
            from .discriminative.engine import get_backend, assemble

            if cooperative:
                from .cooperative.engine import get_backend as get_cooperative_backend

                backend = scorer if scorer is not None else get_cooperative_backend(cfg)
                scores, audit = backend.predict(parsed, events, clauses, cfg, llm)
                stage_times.update(audit["stage_times"])
            else:
                backend = scorer if scorer is not None else get_backend(cfg)
                scores, audit = backend.predict(parsed, events, clauses, cfg)
        else:
            table = score_candidates(events, clauses, parsed, llm, cfg)
        stage_times["s2_s"] = time.perf_counter() - t
        stage = "S4" if cooperative else "S3"
        t = time.perf_counter()
        decisions = (
            assemble(events, scores, parsed, cfg["discriminator"])
            if discriminative
            else select_evidence(events, table, parsed, cfg)
        )
        if (
            discriminative
            and not cooperative
            and getattr(llm, "n_calls", 0) != before_roles
        ):
            raise AssertionError("LLM call inside discriminator stages")
        stage_times["s3_s"] = time.perf_counter() - t
        invalid = (
            0
            if discriminative
            else sum(p.valid_why is False for p in table.pairs())
            + sum(p.valid_how is False for p in table.pairs())
        )
        entity_errors = sum(
            el.state == ERROR or el.diagnostics.get("fallback_from") == ERROR
            for ev in events
            for role, el in ev.elements.items()
            if role not in ("why", "how")
        )
        result.stats.update(
            n_events=len(events),
            n_clauses=len(clauses),
            n_units=len({c.unit_id for c in clauses}),
            n_pairs=len(scores) // 2 if discriminative else len(table),
            invalid_role_responses=invalid,
            entity_errors=entity_errors,
        )
        result.stats.update(
            s1_logical_requests=s1_requests,
            s2_logical_requests=getattr(llm, "n_json", 0) - json_before - s1_requests,
            scoring_audit=audit if discriminative else table.audit,
            prompt_language=lang,
        )
        if cooperative:
            result.stats.update(
                s2_logical_requests=0,
                s3_logical_requests=audit["refinement"]["requests"],
                s4_logical_requests=0,
            )
            result.stats["cooperative_refinement_errors"] = audit["refinement"][
                "invalid"
            ]
            stage_times.pop("s2_s", None)
            stage_times["s4_selection_s"] = stage_times.pop("s3_s")
        if (
            invalid
            or entity_errors
            or (cooperative and audit["refinement"]["invalid"])
            or result.stats["s1_truncated"]
            or not parsed.has_complete_syntax
            or parsed.diagnostics.get("unlocated_triggers")
        ):
            result.status = "partial"
        if cfg["output"]["include_trace"]:
            result.trace = {
                "candidates": [asdict(c) for c in clauses],
                "scores": [
                    asdict(p) for p in (scores if discriminative else table.pairs())
                ],
                "selection": decisions,
            }
    except Exception as exc:
        result.status = "error"
        result.error = "%s: %s: %s" % (stage, type(exc).__name__, str(exc)[:500])
    result.n_llm_calls = getattr(llm, "n_calls", 0) - before
    result.stats["logical_json_requests"] = getattr(llm, "n_json", 0) - json_before
    result.stats["stage_times"] = stage_times
    result.elapsed_s = round(time.perf_counter() - started, 3)
    return result
