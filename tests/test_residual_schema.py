import copy, unittest
from ceh.residual_schema import schema, validate, fingerprint


class SchemaTests(unittest.TestCase):
    def test_language_independent_grounding_values(self):
        for text in ["爆炸", "confirmado", "arrived"]:
            v = {"triggers": [{"text": text, "sent": 0, "score": 0.8}], "relations": []}
            self.assertEqual(validate(v, 3), v)

    def test_bounded_lists_and_strict_numeric_types(self):
        v = {"triggers": [{"text": "x", "sent": 0, "score": 0.8}] * 6, "relations": []}
        with self.assertRaises(ValueError):
            validate(v, 1)
        v["triggers"] = v["triggers"][:1]
        v["triggers"][0]["sent"] = True
        with self.assertRaises(ValueError):
            validate(v, 1)
        v["triggers"][0]["sent"] = 0
        v["triggers"][0]["score"] = float("nan")
        with self.assertRaises(ValueError):
            validate(v, 1)

    def test_cache_schema_identity(self):
        self.assertNotEqual(fingerprint(schema(1)), fingerprint(schema(2)))
        self.assertEqual(fingerprint(schema(2)), fingerprint(copy.deepcopy(schema(2))))


if __name__ == "__main__":
    unittest.main()
