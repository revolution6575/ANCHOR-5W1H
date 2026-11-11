from typing import Any, List

from . import prompts, languages
from .llm import LLM
from .parsing import ParsedDoc
from .schema import ENTITY_SLOTS, NO_EVIDENCE, ERROR, SPAN, Element, Event
from bench.roles import role_defs


DRAFT_ORDER = ("who", "when", "where", "what")


def locate(text: str, value: str, near: int) -> int:
    best, pos = -1, text.find(value)
    while pos >= 0:
        if best < 0 or abs(pos - near) < abs(best - near):
            best = pos
        pos = text.find(value, pos + 1)
    return best


def fill_entities(
    events: List[Event], parsed: ParsedDoc, llm: LLM, cfg: dict, guidance: str = ""
) -> None:
    if not events:
        return
    limit = cfg["input"]["max_listed_sentences"]
    listing = prompts.numbered(parsed.sentences, limit=limit or len(parsed.sentences))
    defs = role_defs(parsed.lang)
    definitions = "\n".join("%s: %s" % (r, defs[r]) for r in DRAFT_ORDER)
    requests = [
        languages.template("entity", parsed.lang).format(
            trigger=ev.trigger, ti=ev.sent_idx, doc=listing, definitions=definitions
        )
        + ("\n" + guidance if guidance else "")
        for ev in events
    ]
    contextual = getattr(llm, "json_chat_many_context", None)
    replies = (
        contextual(
            requests,
            languages.system("role", parsed.lang),
            [{"stage": "S1_entity", "event_id": ev.eid} for ev in events],
            default=None,
        )
        if contextual
        else llm.json_chat_many(
            requests, languages.system("role", parsed.lang), default=None
        )
    )
    if len(replies) != len(events):
        raise ValueError("S1 entity response count mismatch")
    for ev, reply in zip(events, replies):
        ev.ensure_elements()
        for role in ENTITY_SLOTS:
            if not isinstance(reply, dict) or role not in reply:
                ev.elements[role] = Element(
                    state=ERROR, reason="invalid_entity_response"
                )
            else:
                ev.elements[role] = _ground(reply[role], ev, parsed)
        if not ev.elements["what"].is_span():
            prior = ev.elements["what"]
            ev.elements["what"] = Element(
                state=SPAN,
                text=ev.trigger,
                values=[ev.trigger],
                sent_idx=ev.sent_idx,
                start=ev.start,
                end=ev.end,
                reason="trigger_fallback",
                diagnostics={
                    "fallback_from": prior.state,
                    "fallback_reason": prior.reason,
                },
            )


def _ground(value: Any, event: Event, parsed: ParsedDoc) -> Element:
    if not isinstance(value, str):
        return Element(state=ERROR, reason="entity_value_must_be_string")
    value = value.strip()
    if not value:
        return Element(state=NO_EVIDENCE, reason="model_returned_empty")
    start = locate(parsed.text, value, event.start)
    if start < 0:
        return Element(
            state=ERROR,
            reason="entity_not_grounded",
            diagnostics={"rejected_text": value},
        )
    return Element(
        state=SPAN,
        text=value,
        sent_idx=parsed.sentence_of(start),
        start=start,
        end=start + len(value),
        values=[value],
        reason="extracted",
    )
