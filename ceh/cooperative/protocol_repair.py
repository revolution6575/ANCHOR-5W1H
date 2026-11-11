import copy, json, re
from dataclasses import asdict
from .refine import resolve, prompt
from .. import languages

VERSION = "c00-4w-protocol-repair-v2"
_ID = re.compile(r"^e\d+\.")
INSTRUCTION = {
    "zh": "这是原请求的协议纠错，不是重新审核全部4W。只修复列出的failed_roles，其他角色已经固定。每个失败角色只能返回一个原候选ID、null，或原协议的原文定位对象；不要返回数组或ID对象包装。多个候选不得默认取第一项，按上面的原C00角色定义选择一项，或用连续原文表达。重复片段必须指定正确句号和occurrence。原文无证据时才能返回null；不能用null掩盖格式困难。输出只包含failed_roles的JSON，不要解释。",
    "en": "This corrects the protocol of the original request, not all 4W answers. Repair only failed_roles; other roles are fixed. Return one original candidate ID, null, or the original grounded-text object per failed role. No arrays or ID-object wrappers. Do not default to the first of multiple candidates; choose using the unchanged C00 role definitions above, or use a contiguous original span. Specify sentence and occurrence for repetitions. Use null only when evidence is absent, never to hide a formatting difficulty. Return JSON containing only failed_roles, with no explanation.",
    "es": "Esta es una corrección del protocolo original, no una revisión de todos los 4W. Corrige solo failed_roles; los demás están fijados. Devuelve un único ID original, null o el objeto original de texto localizado por rol. Sin listas ni objetos que envuelvan un ID. No elijas por defecto el primer candidato: aplica las definiciones C00 anteriores o usa un fragmento original contiguo. Para repeticiones especifica oración y occurrence. Usa null solo si falta evidencia, nunca para ocultar dificultades de formato. Devuelve únicamente JSON para failed_roles, sin explicación.",
}


def resolve_compatible(value, options, parsed):
    if isinstance(value, list):
        if len(value) != 1 or isinstance(value[0], list):
            raise ValueError("Multiple or nested role values require disambiguation")
        value = value[0]
    if isinstance(value, dict) and set(value) == {"id"}:
        if not isinstance(value["id"], str):
            raise ValueError("ID wrapper requires a string ID")
        element, source = resolve(value["id"], options, parsed)
        return element, source, "id_wrapper"
    if isinstance(value, str) and not any(c["id"] == value for c in options):
        if _ID.match(value):
            raise ValueError("Unknown or wrong-event/role ID")
        if not value or value != value.strip():
            raise ValueError("Invalid plain text boundary")
        starts = [
            m.start() for m in re.finditer("(?=" + re.escape(value) + ")", parsed.text)
        ]
        if len(starts) != 1:
            raise ValueError("Plain text is ambiguous or not grounded")
        value = {"sent_id": parsed.sentence_of(starts[0]), "text": value}
        element, source = resolve(value, options, parsed)
        return element, source, "unique_verbatim_text"
    element, source = resolve(value, options, parsed)
    return element, source, "original_protocol"


def apply_failed(event, reply, candidates, parsed, pending, audit, origin):
    if isinstance(reply, list) and len(reply) == 1 and isinstance(reply[0], dict):
        reply = reply[0]
    remaining = {}
    for role, prior in pending.items():
        old = copy.deepcopy(event.elements[role])
        record = {"role": role, "origin": origin, "before": asdict(old)}
        try:
            if not isinstance(reply, dict) or role not in reply:
                raise ValueError("Missing failed role")
            element, source, normalization = resolve_compatible(
                reply[role], candidates[role], parsed
            )
            event.elements[role] = element
            record.update(
                status="repaired",
                source=source,
                normalization=normalization,
                after=asdict(element),
            )
        except ValueError as exc:
            remaining[role] = {
                "value": reply.get(role) if isinstance(reply, dict) else reply,
                "error": str(exc),
            }
            record.update(status="unresolved", error=str(exc))
        audit.append(record)
    return remaining


def correct_event(parsed, event, events, candidates, pending, llm, max_attempts=2):
    attempts = []
    changes = []
    fixed = {
        r: asdict(v)
        for r, v in event.elements.items()
        if r not in pending and r in ("who", "what", "when", "where")
    }
    for attempt in range(max_attempts):
        if not pending:
            break
        packet = {
            "protocol": VERSION,
            "attempt": attempt + 1,
            "failed_roles": list(pending),
            "previous_invalid_values": pending,
            "fixed_roles": {
                r: {
                    "state": v["state"],
                    "text": v["text"],
                    "start": v["start"],
                    "end": v["end"],
                }
                for r, v in fixed.items()
            },
        }
        if attempts:
            packet["previous_response"] = attempts[-1]["response"]
        label = {
            "zh": "以下JSON仅为错误诊断输入，绝不是输出模板：",
            "en": "The following JSON is diagnostic INPUT only, never an output template:",
            "es": "El siguiente JSON es solo ENTRADA de diagnóstico, nunca una plantilla de salida:",
        }[parsed.lang]
        ending = {
            "zh": "最终答案的顶层键只能是下面这些角色名，值直接放在角色名下。不要输出protocol、attempt、failed_roles、fixed_roles、previous_response或诊断信息。不要解释；只输出角色JSON。所需顶层键：",
            "en": "The final answer must contain ONLY the role names below as top-level keys, with the chosen value directly beneath each. Do NOT output protocol, attempt, failed_roles, fixed_roles, previous_response, or diagnostic metadata. No explanation; output only the role JSON. Required top-level keys: ",
            "es": "La respuesta final debe contener SOLO los nombres de rol siguientes como claves de nivel superior, con el valor elegido directamente debajo. NO devuelvas protocol, attempt, failed_roles, fixed_roles, previous_response ni metadatos de diagnóstico. Sin explicación; solo el JSON de roles. Claves requeridas: ",
        }[parsed.lang]
        request = (
            prompt(parsed, event, events, candidates)
            + "\n"
            + label
            + "\n"
            + json.dumps(packet, ensure_ascii=False)
            + "\n"
            + INSTRUCTION[parsed.lang]
            + "\n"
            + ending
            + json.dumps(list(pending))
        )
        llm.context.update(
            stage="S3_protocol_repair",
            event_id=event.eid,
            protocol=VERSION,
            repair_attempt=attempt + 1,
        )
        reply = llm.json_chat(
            request, languages.system("role", parsed.lang), default=None
        )
        attempts.append(
            {"attempt": attempt + 1, "failed_roles": list(pending), "response": reply}
        )
        pending = apply_failed(
            event, reply, candidates, parsed, pending, changes, "bounded_llm_repair"
        )
        for role, value in fixed.items():
            assert asdict(event.elements[role]) == value, "Unrelated 4W slot changed"
    return pending, changes, attempts
