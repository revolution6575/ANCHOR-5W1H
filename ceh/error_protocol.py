import json, re

VERSION = "c00-error-only-envelope-and-copy-v1"


def compact_copy_variants(prompt):
    begin = prompt.find("{")
    if begin < 0:
        return []
    payload, end = json.JSONDecoder().raw_decode(prompt[begin:])
    if not isinstance(payload, dict) or not isinstance(
        payload.get("document_tokens"), list
    ):
        return []
    prefix, suffix = prompt[:begin], prompt[begin + end :]
    compact = (
        prefix + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + suffix
    )
    variants = [("compact_json", compact)]
    tokens = payload["document_tokens"]
    fields = ["id", "text", "sent"]
    if all(isinstance(t, dict) and set(t) == set(fields) for t in tokens):
        packed = dict(payload)
        packed["document_token_fields"] = fields
        packed["document_tokens"] = [[t[k] for k in fields] for t in tokens]
        restored = dict(packed)
        restored["document_tokens"] = [
            dict(zip(restored["document_token_fields"], t))
            for t in restored["document_tokens"]
        ]
        del restored["document_token_fields"]
        assert restored == payload
        variants.append(
            (
                "lossless_token_table",
                prefix
                + json.dumps(packed, ensure_ascii=False, separators=(",", ":"))
                + suffix,
            )
        )
    return variants


def valid(value):
    return (
        isinstance(value, dict)
        and isinstance(value.get("triggers"), list)
        and isinstance(value.get("relations"), list)
        and all(
            isinstance(t, dict) and isinstance(t.get("text"), str) and t["text"].strip()
            for t in value["triggers"]
        )
    )


def outer_text(text):
    s = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if "```" in s:
        for part in s.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith(("{", "[")):
                s = part
                break
    return s


def numeric_key_quotes(s):
    out = []
    i = 0
    inside = False
    escaped = False
    changed = False
    while i < len(s):
        ch = s[i]
        if inside:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                inside = False
            i += 1
            continue
        if ch == '"':
            prev = s[:i].rstrip()[-1:]
            m = re.match(r'"(sent|score|source|target):(?=\s*-?\d)', s[i:])
            if m and prev in ("{", ","):
                out.append('"' + m.group(1) + '":')
                i += len(m.group(0))
                changed = True
                continue
            inside = True
        out.append(ch)
        i += 1
    return "".join(out), changed


def parse_envelope(text):
    s = outer_text(text)
    variants = [("unchanged", s)]
    fixed, changed = numeric_key_quotes(s)
    if changed:
        variants.append(("numeric_field_closing_quote", fixed))
    for prefix, s in list(variants):
        if s.startswith("{"):
            if s.endswith("]"):
                variants.append((prefix + ":missing_outer_brace", s + "}"))
            if re.search(r']\s*"\s*}$', s):
                variants.append(
                    (
                        prefix + ":extra_terminal_quote",
                        re.sub(r'(]\s*)"(\s*})$', r"\1\2", s),
                    )
                )
            if re.search(r']\s*"$', s):
                variants.append(
                    (
                        prefix + ":extra_terminal_quote_and_outer_brace",
                        re.sub(r'"$', "}", s),
                    )
                )
    found = []
    for kind, raw in variants:
        try:
            v = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict):
            v = v[0]
            kind += ":singleton_object"
        if valid(v):
            found.append((v, kind))
    if not found:
        return None, "requires_regeneration"
    canonical = {json.dumps(v, sort_keys=True, ensure_ascii=False) for v, _ in found}
    if len(canonical) != 1:
        raise ValueError("Ambiguous envelope correction")
    return found[0]


FORMAT_INSTRUCTION = {
    "zh": "上一轮回答未通过JSON格式校验。此次只修复输出协议，保持上述事件提议任务不变：返回一个完整的紧凑JSON对象，不要解释或代码块，不要重复关系。最多给出{k}个触发词，每个触发词逐字复制原文。triggers和relations都必须是数组，relations没有内容时返回[]。关系索引只引用本次触发词列表。请完整关闭所有括号。",
    "en": "The previous response failed JSON validation. Repair the output protocol while keeping the event proposal task above unchanged. Return one complete compact JSON object, without explanations or code fences; do not repeat relations. Return at most {k} triggers, copied verbatim from the text. Both triggers and relations must be arrays; use [] for no relations. Relation indices must refer to the returned trigger list. Close every bracket.",
    "es": "La respuesta anterior no pasó la validación JSON. Corrige el protocolo de salida sin cambiar la tarea de eventos anterior. Devuelve un único objeto JSON completo y compacto, sin explicaciones ni bloques de código; no repitas relaciones. Devuelve como máximo {k} desencadenantes copiados literalmente. triggers y relations deben ser listas; usa [] cuando no haya relaciones. Los índices deben referirse a la lista devuelta. Cierra todos los corchetes y llaves.",
}
