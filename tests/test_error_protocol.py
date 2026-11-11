import json, unittest
from ceh.error_protocol import parse_envelope, compact_copy_variants


class EnvelopeTests(unittest.TestCase):
    def test_compaction_preserves_every_token_and_original_text(self):
        v = {
            "document": [{"sent": 0, "text": "a  b\n中文 español"}],
            "document_tokens": [
                {"id": 42, "text": "a  b", "sent": 0},
                {"id": 43, "text": "中\n文", "sent": 0},
            ],
            "invalid_proposals": [{"text": "changed", "sent": 0, "index": 0}],
            "previous_invalid_response": {"reply": None},
        }
        prompt = "Instruction\n" + json.dumps(v, ensure_ascii=False) + "\nOutput"
        for mode, variant in compact_copy_variants(prompt):
            self.assertTrue(variant.startswith("Instruction\n"))
            self.assertTrue(variant.endswith("\nOutput"))
            got = json.loads(variant[len("Instruction\n") : -len("\nOutput")])
            if mode == "lossless_token_table":
                got["document_tokens"] = [
                    dict(zip(got["document_token_fields"], t))
                    for t in got["document_tokens"]
                ]
                del got["document_token_fields"]
            self.assertEqual(got, v)

    def test_languages_and_exact_values(self):
        for text in ["发生", "publicó", "arrived", 'a ]"} b']:
            v = {"triggers": [{"text": text, "sent": 0, "score": 0.8}], "relations": []}
            s = json.dumps(v, ensure_ascii=False)
            for damaged in [
                s,
                s[:-1],
                s[:-1] + '"}',
                s[:-1] + '"',
                "```json\n" + s[:-1] + "\n```",
                json.dumps([v]),
            ]:
                actual, _ = parse_envelope(damaged)
                self.assertEqual(actual, v)

    def test_incomplete_content_is_not_salvaged(self):
        for s in [
            '{"triggers":[{"text":"arrived"}],"relations":[{"source":',
            '{"triggers":[{"text":"arri',
            '{"triggers":[],"relations":',
            '[{"triggers":[],"relations":[]},{"triggers":[],"relations":[]}]',
        ]:
            self.assertIsNone(parse_envelope(s)[0])

    def test_no_fields_or_relations_invented(self):
        self.assertIsNone(parse_envelope('{"triggers":[]}')[0])
        v = {
            "triggers": [{"text": "arrived"}],
            "relations": [{"source": 0, "target": 0}, {"source": 0, "target": 0}],
        }
        self.assertEqual(parse_envelope(json.dumps(v)[:-1])[0], v)

    def test_numeric_key_typo_only_outside_strings(self):
        v = {
            "triggers": [{"text": 'a "sent:3, word', "sent": 3, "score": 0.8}],
            "relations": [],
        }
        s = json.dumps(v, ensure_ascii=False).replace('"sent": 3', '"sent: 3')
        self.assertEqual(parse_envelope(s)[0], v)
        self.assertEqual(parse_envelope(json.dumps(v))[0], v)


if __name__ == "__main__":
    unittest.main()
