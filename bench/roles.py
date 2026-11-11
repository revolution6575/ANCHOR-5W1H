from collections import OrderedDict

ROLE_DEF_EN = OrderedDict(
    [
        (
            "who",
            "minimal actors, targets, affected parties, challengers, monitors, "
            "or implementers.",
        ),
        ("what", "compact central action or state plus its required object or result."),
        ("when", "the time, deadline, or as-of time belonging to the event."),
        (
            "where",
            "the physical or institutional setting, platform, affected area, "
            "or deployment site of the event.",
        ),
        (
            "why",
            "a cause, motivation, purpose, justification, enabling condition, "
            "challenge, or risk that explains the event.",
        ),
        (
            "how",
            "a means, manner, instrument, mechanism, procedure, ordered step, "
            "or implementation that realizes the event.",
        ),
    ]
)

ROLE_DEF_ZH = OrderedDict(
    [
        ("who", "最小的施事、对象、受影响方、质疑方、监督方或执行方。"),
        ("what", "紧凑的中心动作或状态，加上它必需的宾语或结果。"),
        ("when", "属于该事件的时间、截止期限或截至时点。"),
        ("where", "该事件的物理或机构场所、平台、受影响区域或部署地点。"),
        ("why", "解释该事件的原因、动机、目的、理由、使能条件、挑战或风险。"),
        ("how", "实现该事件的方式、方法、手段、机制、流程、步骤或实施途径。"),
    ]
)

ROLE_DEF_ES = OrderedDict(
    [
        (
            "who",
            "actores, objetivos, partes afectadas, cuestionadores, supervisores o ejecutores mínimos.",
        ),
        (
            "what",
            "acción o estado central compacto, junto con su objeto o resultado necesario.",
        ),
        ("when", "el momento, plazo o fecha de referencia perteneciente al evento."),
        (
            "where",
            "el entorno físico o institucional, plataforma, zona afectada o lugar de despliegue del evento.",
        ),
        (
            "why",
            "una causa, motivación, propósito, justificación, condición habilitante, desafío o riesgo que explica el evento.",
        ),
        (
            "how",
            "un medio, manera, instrumento, mecanismo, procedimiento, paso ordenado o implementación que realiza el evento.",
        ),
    ]
)


def role_defs(lang: str):
    return {"zh": ROLE_DEF_ZH, "en": ROLE_DEF_EN, "es": ROLE_DEF_ES}[lang]
