import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
import urllib.request
import uuid
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

logger = logging.getLogger(__name__)

_UNPARSED = object()

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


class LLM:
    def __init__(self, cfg: dict, dry_run: bool = False, request_gate=None):
        c = cfg["llm"]
        self.base_url = c["base_url"].rstrip("/")
        self.api_key = c.get("api_key", "EMPTY")
        self.model = c["model"]
        self.temperature = c["temperature"]
        self.max_tokens = c["max_tokens"]
        self.seed = c["seed"]
        self.timeout = c.get("timeout_s", 120)
        self.max_retries = c.get("max_retries", 3)
        self.backoff = c.get("retry_backoff_s", 4)
        self.concurrency = max(1, int(c.get("concurrency", 1)))
        self.dry_run = dry_run

        cc = cfg.get("cache", {})
        self.cache_on = cc.get("enabled", True) and not dry_run
        self.cache_dir = cc.get("dir", "outputs/llm_cache")
        if self.cache_on:
            os.makedirs(self.cache_dir, exist_ok=True)

        self._lock = threading.Lock()
        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_json = 0
        self.n_parse_fail = 0
        self.n_empty = 0
        self.request_gate = request_gate
        self.context = {}
        self._context_local = threading.local()
        self.request_records = []


    def chat(self, prompt: str, system: str = "") -> str:
        if self.dry_run:
            self._bump("n_calls")
            return _fake_reply(prompt)
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        return self._complete(messages, self._key(prompt + "||" + system))

    def chat_messages(self, messages: list) -> str:
        if self.dry_run:
            self._bump("n_calls")
            return _fake_reply(json.dumps(messages, ensure_ascii=False)[-2000:])
        key = self._key(
            "MULTI||" + json.dumps(messages, ensure_ascii=False, sort_keys=True)
        )
        return self._complete(messages, key)

    def json_chat(self, prompt: str, system: str = "", default=None):
        text = self.chat(prompt, system)
        value = parse_json(text, _UNPARSED)
        with self._lock:
            self.n_json += 1
            if value is _UNPARSED:
                if text:
                    self.n_parse_fail += 1
                else:
                    self.n_empty += 1
        if value is _UNPARSED:
            return copy.deepcopy(default)
        return value

    def json_chat_many(
        self, prompts: List[str], system: str = "", default=None
    ) -> List[Any]:
        if self.concurrency <= 1 or len(prompts) <= 1:
            return [self.json_chat(p, system, default) for p in prompts]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(lambda p: self.json_chat(p, system, default), prompts))

    def json_chat_many_context(self, prompts, system, labels, default=None):
        if len(labels) != len(prompts):
            raise ValueError("Request context count mismatch")

        def call(item):
            prompt, label = item
            self._context_local.label = label
            try:
                return self.json_chat(prompt, system, default)
            finally:
                self._context_local.label = {}

        if self.concurrency <= 1 or len(prompts) <= 1:
            return [call(x) for x in zip(prompts, labels)]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(call, zip(prompts, labels)))

    def parse_failure_rate(self) -> float:
        return self.n_parse_fail / float(self.n_json) if self.n_json else 0.0


    def _bump(self, counter: str) -> None:
        with self._lock:
            setattr(self, counter, getattr(self, counter) + 1)

    def _key(self, raw_prompt: str) -> str:
        raw = json.dumps(
            {
                "endpoint": self.base_url,
                "m": self.model,
                "t": self.temperature,
                "mt": self.max_tokens,
                "s": self.seed,
                "p": raw_prompt,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key + ".json")

    def _cache_get(self, key: str):
        if not self.cache_on:
            return None
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)["text"]
        except Exception:
            return None

    def _cache_put(self, key: str, text: str) -> None:
        if not self.cache_on:
            return
        try:
            target = self._cache_path(key)
            temp = target + "." + uuid.uuid4().hex + ".tmp"
            with open(temp, "w", encoding="utf-8") as f:
                json.dump({"text": text}, f, ensure_ascii=False)
            os.replace(temp, target)
        except Exception as exc:
            logger.debug("缓存写入失败：%s", exc)

    def _complete(self, messages: list, key: str) -> str:
        hit = self._cache_get(key)
        if hit is not None:
            self._bump("n_cache_hits")
            return hit

        context = {**self.context, **getattr(self._context_local, "label", {})}
        request_id = uuid.uuid4().hex
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
                "metadata": {**context, "request_id": request_id},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }

        last_error = ""
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            started_unix = time.time()
            record = {
                **context,
                "request_id": request_id,
                "attempt": attempt + 1,
                "started_unix": started_unix,
                "cache_key": key,
            }
            try:
                req = urllib.request.Request(
                    self.base_url + "/chat/completions", data=body, headers=headers
                )
                with (
                    self.request_gate
                    if self.request_gate is not None
                    else nullcontext()
                ):
                    record["client_queue_s"] = time.perf_counter() - started
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        reply = json.loads(resp.read().decode("utf-8"))
                text = reply["choices"][0]["message"]["content"] or ""
                self._bump("n_calls")
                self._cache_put(key, text)
                record.update(
                    status="ok",
                    elapsed_s=time.perf_counter() - started,
                    usage=reply.get("usage", {}),
                    finish_reason=reply["choices"][0].get("finish_reason"),
                )
                with self._lock:
                    self.request_records.append(record)
                return text
            except Exception as exc:
                last_error = "%s: %s" % (type(exc).__name__, str(exc)[:120])
                record.update(
                    status="error",
                    error=last_error,
                    elapsed_s=time.perf_counter() - started,
                )
                with self._lock:
                    self.request_records.append(record)
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff * (attempt + 1))
        logger.warning("LLM 请求失败：%s", last_error)
        return ""


def parse_json(text: str, default=None):
    if not text:
        return default
    s = _THINK_BLOCK.sub("", text).strip()
    if "```" in s:
        for part in s.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith(("{", "[")):
                s = part
                break
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        pass
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    if not starts:
        return default
    try:
        value, _ = json.JSONDecoder().raw_decode(s[min(starts) :])
        return value
    except ValueError:
        return default



_NUMBERED_LINE = re.compile(r"^\[(\d+)\] (.+)$", re.M)
_CJK_WORD = re.compile(r"[一-鿿]{2}")
_LATIN_WORD = re.compile(r"[A-Za-zÀ-ÿ]{4,}")


def _pseudo(prompt: str) -> float:
    return int(hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:4], 16) % 100 / 100.0


def _first_word(sentence: str) -> str:
    m = _CJK_WORD.search(sentence) or _LATIN_WORD.search(sentence)
    return m.group(0) if m else ""


def _fake_reply(prompt: str) -> str:
    low = prompt.lower()
    if '"triggers"' in low and '"relations"' in low:
        triggers = []
        for idx, sentence in _NUMBERED_LINE.findall(prompt)[:3]:
            word = _first_word(sentence)
            if word:
                triggers.append({"text": word, "sent": int(idx), "score": 0.8})
        relations = (
            [{"source": 1, "target": 0, "type": "causes"}] if len(triggers) > 1 else []
        )
        return json.dumps(
            {"triggers": triggers, "relations": relations}, ensure_ascii=False
        )
    if '"p"' in low and '"relation"' in low:
        return json.dumps({"p": _pseudo(prompt), "relation": "direct_reason"})
    if '"q"' in low:
        return json.dumps({"q": _pseudo(prompt)})
    if '"who"' in low and '"what"' in low:
        return '{"who": "", "when": "", "where": "", "what": ""}'
    return '{"who": "", "what": "", "when": "", "where": "", "why": "", "how": ""}'
