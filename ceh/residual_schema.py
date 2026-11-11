import hashlib, json, math

VERSION = "c00-residual28-json-schema-v1"


def schema(sentences, max_events=5):
    integer = lambda top: {"type": "integer", "minimum": 0, "maximum": max(0, top)}
    trigger = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "sent": integer(sentences - 1),
            "score": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["text", "sent", "score"],
        "additionalProperties": False,
    }
    relation = {
        "type": "object",
        "properties": {
            "source": integer(max_events - 1),
            "target": integer(max_events - 1),
            "type": {
                "type": "string",
                "enum": ["causes", "enables", "motivates", "part_of", "background_of"],
            },
        },
        "required": ["source", "target", "type"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "triggers": {
                "type": "array",
                "minItems": 0,
                "maxItems": max_events,
                "items": trigger,
            },
            "relations": {
                "type": "array",
                "maxItems": max_events * (max_events - 1),
                "items": relation,
            },
        },
        "required": ["triggers", "relations"],
        "additionalProperties": False,
    }


def validate(value, sentences, max_events=5):
    if not isinstance(value, dict) or set(value) != {"triggers", "relations"}:
        raise ValueError("Wrong root schema")
    if not isinstance(value["triggers"], list) or len(value["triggers"]) > max_events:
        raise ValueError("Wrong trigger list")
    if not isinstance(value["relations"], list) or len(
        value["relations"]
    ) > max_events * (max_events - 1):
        raise ValueError("Wrong relation list")
    for t in value["triggers"]:
        if not isinstance(t, dict) or set(t) != {"text", "sent", "score"}:
            raise ValueError("Wrong trigger fields")
        if not isinstance(t["text"], str) or not t["text"].strip():
            raise ValueError("Empty trigger")
        if type(t["sent"]) is not int or not 0 <= t["sent"] < sentences:
            raise ValueError("Wrong sentence")
        if (
            type(t["score"]) not in (int, float)
            or not math.isfinite(t["score"])
            or not 0 <= t["score"] <= 1
        ):
            raise ValueError("Wrong salience")
    allowed = {"causes", "enables", "motivates", "part_of", "background_of"}
    for x in value["relations"]:
        if not isinstance(x, dict) or set(x) != {"source", "target", "type"}:
            raise ValueError("Wrong relation fields")
        if (
            any(
                type(x[k]) is not int or not 0 <= x[k] < max_events
                for k in ("source", "target")
            )
            or x["type"] not in allowed
        ):
            raise ValueError("Wrong relation value")
    return value


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


INSTRUCTIONS = {
    "zh": "这次是失败输出的结构修复。请输出一个完整JSON对象，不要重复触发词或关系。最多5个触发词、20条直接关系。触发词须为原文连续的中心动作或状态表达，保留原文词形。不要把列表扩展为所有词语；没有依据时不要制造事件。",
    "en": "This is a structural repair of a failed output. Return one complete JSON object without repeated triggers or relations. At most 5 triggers and 20 direct relations. Copy contiguous central action or state expressions verbatim, preserving original word forms. Do not expand the list to every word, and do not invent unsupported events.",
    "es": "Esta es una reparación estructural de una salida fallida. Devuelve un único objeto JSON completo, sin repetir desencadenantes ni relaciones. Máximo 5 desencadenantes y 20 relaciones directas. Copia literalmente expresiones contiguas de la acción o estado central, conservando la forma original. No amplíes la lista a todas las palabras ni inventes eventos sin fundamento.",
}
