import json
import re
import unittest
from pathlib import Path

from ceh.cluster import build_cluster
from ceh.config import load_config
from ceh.parsing import ParsedDoc
from ceh.schema import Sentence

CONFIG = Path(__file__).resolve().parents[1] / "config" / "anchor.json"
DRAFT_ORDER = ["who", "when", "where", "what"]


class RecordingClient:
    def __init__(self, triggers, reply):
        self.context = {}
        self.triggers = triggers
        self.reply = reply
        self.requests = []

    def json_chat(self, prompt, system, default=None):
        return {
            "triggers": [
                {"text": word, "sent": 0, "score": 1} for word in self.triggers
            ],
            "relations": [],
        }

    def json_chat_many_context(self, prompts, system, labels, default=None):
        self.requests.extend(zip(prompts, labels))
        return [self.reply for _ in prompts]


class DraftTests(unittest.TestCase):
    def test_one_ordered_request_per_proposal(self):
        examples = {
            "zh": (
                "李明周一在上海开会，随后发言。",
                ["开会", "发言"],
                "李明",
                "周一",
                "上海",
            ),
            "en": (
                "Ann met Jo in Rome on Monday and left.",
                ["met", "left"],
                "Ann",
                "Monday",
                "Rome",
            ),
            "es": (
                "Ana llegó a Lima el lunes y habló.",
                ["llegó", "habló"],
                "Ana",
                "lunes",
                "Lima",
            ),
        }
        for lang, (text, triggers, who, when, where) in examples.items():
            with self.subTest(lang=lang):
                parsed = ParsedDoc(text, lang, [Sentence(0, text, 0, len(text))])
                client = RecordingClient(
                    triggers, dict(zip(DRAFT_ORDER, (who, when, where, triggers[0])))
                )
                cfg = load_config(CONFIG)
                cfg["cluster"]["max_events"] = 1
                events = build_cluster(parsed, client, cfg)
                self.assertEqual(len(client.requests), 2)
                self.assertEqual(len(events), 1)
                self.assertTrue(events[0].is_central)
                for prompt, label in client.requests:
                    schema = re.search(r'\{"who"\s*:[^}]+\}', prompt).group()
                    self.assertEqual(list(json.loads(schema)), DRAFT_ORDER)
                    self.assertEqual(label["stage"], "S1_entity")
                    self.assertIn(text, prompt)
                    positions = [prompt.index("\n" + key + ":") for key in DRAFT_ORDER]
                    self.assertEqual(positions, sorted(positions))
                self.assertEqual(events[0].elements["who"].text, who)
