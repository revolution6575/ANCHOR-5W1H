import threading
import time
from concurrent.futures import ThreadPoolExecutor
from .llm import LLM
from .pipeline import run_ours


def run_stream(documents, cfg, dry=False):
    documents = list(documents)
    if not dry:
        from .parsing import load_spacy

        for lang in sorted({item[2] for item in documents}):
            load_spacy(lang)
    rc = cfg.get("runtime", {})
    gate = threading.BoundedSemaphore(rc.get("max_inflight", 16))

    def run(item):
        did, text, lang = item
        llm = LLM(cfg, dry_run=dry, request_gate=gate)
        started = time.time()
        result = run_ours(did, text, lang, llm, cfg)
        result.stats["started_unix"] = started
        result.stats["finished_unix"] = time.time()
        counters = {
            k: getattr(llm, k)
            for k in ("n_calls", "n_json", "n_cache_hits", "n_parse_fail", "n_empty")
        }
        return result, counters, llm.request_records

    if rc.get("documents", 1) == 1:
        for document in documents:
            yield run(document)
    else:
        with ThreadPoolExecutor(max_workers=rc["documents"]) as pool:
            yield from pool.map(run, documents)
