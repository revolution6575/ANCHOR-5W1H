import copy, json, re
from . import languages, prompts

VERSION = "c00-s1-relation-protocol-repair-v1"


def completed_triggers(raw, k=5):
    match = re.search(r'"triggers"\s*:\s*', raw)
    if not match:
        raise ValueError("No complete triggers field")
    value, _ = json.JSONDecoder().raw_decode(raw, match.end())
    if not isinstance(value, list) or not 0 < len(value) <= k:
        raise ValueError("Invalid completed trigger array")
    if any(
        not isinstance(t, dict)
        or not isinstance(t.get("text"), str)
        or not t["text"].strip()
        for t in value
    ):
        raise ValueError("Invalid completed trigger item")
    return copy.deepcopy(value)


def valid_relations(reply, n):
    if not isinstance(reply, dict) or not isinstance(reply.get("relations"), list):
        raise ValueError("Missing relations array")
    result = []
    seen = set()
    for rel in reply["relations"]:
        if not isinstance(rel, dict):
            raise ValueError("Malformed relation")
        a, b = rel.get("source"), rel.get("target")
        if (
            any(
                isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < n
                for v in (a, b)
            )
            or a == b
        ):
            raise ValueError("Relation index must refer to a trigger, never a sentence")
        if rel.get("type") not in prompts.RELATION_TYPES:
            raise ValueError("Unknown relation type")
        if (a, b) not in seen:
            result.append(rel)
            seen.add((a, b))
    return result


def recover(parsed, raw, llm, max_attempts=2):
    triggers = completed_triggers(raw)
    n = len(triggers)
    audit = {"version": VERSION, "triggers_preserved": triggers, "attempts": []}
    instruction = {
        "zh": "此前回答已完整给出触发词，但关系列表把句号当成事件编号并超过输出上限。触发词列表保持原样，只重新返回这些触发词之间的直接关系。source/target是下列事件index，不是原文句号，只能在0到{last}之间，不能自连，每个有序事件对最多一条。最多{maximum}条；无关系返回空数组。只返回JSON对象，唯一顶层键为relations，每条只含source/target/type。不返回触发词或解释。",
        "en": "The earlier response completed the trigger list, but its relation list confused sentence IDs with event indices and exceeded the output limit. Keep all triggers unchanged; return only direct relations between these triggers. source/target refer to event index below, NOT sentence IDs. Valid indices are 0 through {last}; no self links, at most one relation per ordered pair and at most {maximum} relations. Use an empty array if unrelated. Return only a JSON object with the sole top-level key relations; each item has source, target, type. No triggers or explanation.",
        "es": "La respuesta anterior completó los desencadenantes, pero confundió números de oración con índices de evento y excedió el límite de salida. Conserva los desencadenantes; devuelve solo sus relaciones directas. source/target indican el index del evento, NO la oración: de 0 a {last}, sin autorrelaciones, como máximo una por par ordenado y {maximum} en total. Sin relación devuelve una lista vacía. Solo JSON con la clave relations; cada elemento contiene source, target y type. Sin desencadenantes ni explicación.",
    }[parsed.lang].format(last=n - 1, maximum=n * (n - 1))
    for i in range(max_attempts):
        data = {
            "document": [{"sent": s.idx, "text": s.text} for s in parsed.sentences],
            "fixed_triggers": [{"index": j, **t} for j, t in enumerate(triggers)],
            "allowed_relation_types": sorted(prompts.RELATION_TYPES),
        }
        if audit["attempts"]:
            data["previous_invalid_response"] = audit["attempts"][-1]
        llm.context.update(stage="S1_relation_protocol_repair", repair_attempt=i + 1)
        reply = llm.json_chat(
            json.dumps(data, ensure_ascii=False) + "\n" + instruction,
            languages.system("trigger", parsed.lang),
            default=None,
        )
        try:
            rels = valid_relations(reply, n)
            audit["attempts"].append(
                {"attempt": i + 1, "response": reply, "valid": True}
            )
            parsed.diagnostics["relation_protocol_repair"] = audit
            return triggers, rels
        except ValueError as exc:
            audit["attempts"].append(
                {"attempt": i + 1, "response": reply, "valid": False, "error": str(exc)}
            )
    parsed.diagnostics["relation_protocol_repair"] = audit
    raise ValueError("Relation serialization recovery failed")
