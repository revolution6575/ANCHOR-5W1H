import json, unittest
from pathlib import Path
from ceh.schema import Event, Element, Sentence, SPAN
from ceh.parsing import ParsedDoc
from ceh.cooperative.candidates import propose, boundary_spans
from ceh.cooperative.refine import (
    resolve,
    apply_reply,
    prompt,
    refine_events,
    INSTRUCTION,
)


def fixture(text="Alice met Bob. Alice left.", lang="en"):
    parsed = ParsedDoc(
        text=text,
        lang=lang,
        sentences=[Sentence(0, text, 0, len(text))],
        tokens=[],
        clauses=[],
    )
    ev = Event(0, "met", 0, 6, 9, is_central=True)
    ev.ensure_elements()
    ev.elements["who"] = Element(
        state=SPAN, text="Alice", start=0, end=5, sent_idx=0, values=["Alice"]
    )
    return parsed, ev


class CooperationTests(unittest.TestCase):
    def test_candidate_draft_preserved_and_what_trigger_legal(self):
        p, e = fixture()
        e.elements["what"] = Element(state=SPAN, text="met", start=6, end=9, sent_idx=0)
        cs = propose(
            p,
            e,
            [(0, 5), (5, 5), (6, 9), (10, 13)],
            [[[0.1, 0.99, 0.9, 0.1], [0.1, 0.99, 0.9, 0.1]]] * 4,
            {"learned_candidates": 1, "syntax_candidates": 0},
        )
        self.assertTrue(
            any(c["text"] == "Alice" and "llm_draft" in c["sources"] for c in cs["who"])
        )
        self.assertTrue(any(c["text"] == "met" for c in cs["what"]))
        self.assertTrue(
            all(c["start"] < c["end"] for vals in cs.values() for c in vals)
        )
        self.assertEqual(boundary_spans([[1], [1]], [(0, 1)], limit=0), [])

    def test_wrong_role_id_is_error_not_negative(self):
        p, e = fixture()
        cs = {r: [] for r in ("who", "what", "when", "where")}
        a = apply_reply(
            e, {"who": "e0.what.c0", "what": None, "when": None, "where": None}, cs, p
        )
        self.assertEqual(a["invalid"], 1)
        self.assertEqual(e.elements["who"].text, "Alice")
        self.assertIn("refinement_error", e.elements["who"].diagnostics)

    def test_null_can_remove_previous_extraction(self):
        p, e = fixture()
        a = apply_reply(
            e,
            dict.fromkeys(("who", "what", "when", "where")),
            {r: [] for r in ("who", "what", "when", "where")},
            p,
        )
        self.assertEqual(a["null"], 4)
        self.assertFalse(e.elements["who"].is_span())

    def test_outside_not_equal_missing_and_ambiguous_occurrence(self):
        p, e = fixture()
        with self.assertRaises(ValueError):
            resolve({"sent_id": 0, "text": "Alice"}, [], p)
        el, source = resolve({"sent_id": 0, "text": "Alice", "occurrence": 1}, [], p)
        self.assertEqual(el.start, 15)
        self.assertEqual(source, "outside")
        with self.assertRaises(ValueError):
            resolve({"sent_id": 0, "text": "ALICE"}, [], p)

    def test_multilingual_verbatim_and_no_probability_hints(self):
        for lang, text in [
            ("zh", "张三抵达北京。"),
            ("en", "Alice arrived."),
            ("es", "María llegó."),
        ]:
            p, e = fixture(text, lang)
            e.trigger = text[:2]
            e.start = 0
            e.end = 2
            cs = {r: [] for r in ("who", "what", "when", "where")}
            s = prompt(p, e, [e], cs)
            self.assertIn(INSTRUCTION[lang], s)
            self.assertIn(text, s)
            self.assertNotIn("boundary_support", s)
            el, _ = resolve({"sent_id": 0, "text": text[:-1]}, [], p)
            self.assertEqual(p.text[el.start : el.end], el.text)

    def test_missing_keys_record_all_errors(self):
        p, e = fixture()
        self.assertEqual(
            apply_reply(e, {}, dict.fromkeys(("who", "what", "when", "where"), []), p)[
                "invalid"
            ],
            4,
        )

    def test_one_joint_request_per_event_and_frozen_identity(self):
        p, e = fixture()
        old = (e.trigger, e.span, e.is_central)

        class Fake:
            n_calls = 0

            def json_chat_many_context(self, requests, system, labels, default=None):
                self.n_calls += len(requests)
                self.labels = labels
                return [
                    dict.fromkeys(("who", "what", "when", "where")) for _ in requests
                ]

        llm = Fake()
        a = refine_events(
            p, [e], {0: dict.fromkeys(("who", "what", "when", "where"), [])}, llm
        )
        self.assertEqual(a["actual_calls"], 1)
        self.assertEqual(llm.labels[0]["stage"], "S3_4W_refinement")
        self.assertEqual(old, (e.trigger, e.span, e.is_central))

    def test_new_config_routes_only_post_joint(self):
        from ceh.config import validate_config

        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / "config/anchor.json").read_text(encoding="utf-8"))
        validate_config(cfg)
        cfg["s1_factorial"]["anchor_policy"] = "early"
        with self.assertRaises(ValueError):
            validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
