import copy
import json
import unittest
from types import SimpleNamespace

from PIL import Image

from edit_topology.model_parser import ModelContractError, parse_with_model
from edit_topology.parser import parse_instruction


class _FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(self.payload))


class _FakeOpenAI:
    def __init__(self, payload):
        self.responses = _FakeResponses(payload)


class ModelParserTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (32, 32), "white")
        self.instruction = "Add a hat to the man on the left."

    def test_openai_path_uses_exactly_one_call(self):
        draft = parse_instruction(self.instruction)
        client = _FakeOpenAI(draft)
        contract, model = parse_with_model(
            self.image,
            self.instruction,
            provider="openai",
            model="test-vlm",
            client=client,
        )
        self.assertEqual(model, "test-vlm")
        self.assertEqual(contract, draft)
        self.assertEqual(len(client.responses.calls), 1)
        content = client.responses.calls[0]["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[1]["type"], "input_image")

    def test_rejects_changed_instruction(self):
        payload = parse_instruction(self.instruction)
        payload["instruction"] = "Do something else."
        with self.assertRaisesRegex(ModelContractError, "protected field"):
            parse_with_model(
                self.image,
                self.instruction,
                provider="openai",
                client=_FakeOpenAI(payload),
            )

    def test_rejects_locally_invalid_contract(self):
        payload = parse_instruction(self.instruction)
        payload["unexpected"] = True
        with self.assertRaisesRegex(ModelContractError, "local validation"):
            parse_with_model(
                self.image,
                self.instruction,
                provider="openai",
                client=_FakeOpenAI(payload),
            )

    def test_rejects_changes_to_rule_derived_fields(self):
        draft = parse_instruction(self.instruction)
        mutations = {
            "topology": "remove_entity",
            "operation": "remove",
            "edit_scope": "global",
            "referring_constraints": [
                {
                    "field": "appearance",
                    "value": "invented",
                    "source": "image",
                    "scope": "target",
                }
            ],
            "desired_state": "an invented result",
            "replacement": {
                "kind": "entity",
                "name": "invented object",
                "entity_type": "object",
                "instance_ref": "new_object",
                "count": {"value": 1, "quantifier": "exact", "exact": True},
                "existing": False,
            },
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                payload = copy.deepcopy(draft)
                payload[field] = value
                with self.assertRaisesRegex(ModelContractError, "protected field"):
                    parse_with_model(
                        self.image,
                        self.instruction,
                        provider="openai",
                        client=_FakeOpenAI(payload),
                    )

    def test_prompt_names_rule_derived_immutable_fields(self):
        draft = parse_instruction(self.instruction)
        client = _FakeOpenAI(draft)
        parse_with_model(
            self.image,
            self.instruction,
            provider="openai",
            client=client,
        )
        prompt = client.responses.calls[0]["input"][0]["content"][0]["text"]
        for field in (
            "topology",
            "operation",
            "edit_scope",
            "referring_constraints",
            "desired_state",
            "replacement",
        ):
            self.assertIn(field, prompt)

    def test_rejects_rewriting_existing_replacement_contract(self):
        instruction = "Replace the red cup with a blue ceramic mug."
        draft = parse_instruction(instruction)
        mutations = []

        changed_desired = copy.deepcopy(draft)
        changed_desired["desired_state"] = "a green bottle"
        mutations.append(changed_desired)

        changed_replacement = copy.deepcopy(draft)
        changed_replacement["replacement"]["entity_type"] = "bottle"
        mutations.append(changed_replacement)

        for payload in mutations:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ModelContractError, "protected field"):
                    parse_with_model(
                        self.image,
                        instruction,
                        provider="openai",
                        client=_FakeOpenAI(payload),
                    )

    def test_vlm_may_ground_instance_but_not_rewrite_target_semantics(self):
        instruction = "Remove the red sign."
        draft = parse_instruction(instruction)
        grounded = copy.deepcopy(draft)
        grounded["target"]["instance_ref"] = "visible_red_sign"
        grounded["region_roles"]["target"][0]["entity_ref"] = "visible_red_sign"
        contract, _ = parse_with_model(
            self.image,
            instruction,
            provider="openai",
            client=_FakeOpenAI(grounded),
        )
        self.assertEqual("visible_red_sign", contract["target"]["instance_ref"])

        rewritten = copy.deepcopy(draft)
        rewritten["target"]["entity_type"] = "cap"
        with self.assertRaisesRegex(ModelContractError, "target.entity_type"):
            parse_with_model(
                self.image,
                instruction,
                provider="openai",
                client=_FakeOpenAI(rewritten),
            )


if __name__ == "__main__":
    unittest.main()
