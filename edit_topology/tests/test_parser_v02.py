import unittest

from edit_topology.parser import (
    normalize_dataset_task,
    parse_instruction,
    validate_contract,
)


class ParserV02AcceptanceTests(unittest.TestCase):
    def assertValid(self, contract):
        self.assertEqual([], validate_contract(contract))

    def constraints(self, contract):
        return {
            (item.get("scope"), item["field"], item["value"])
            for item in contract["referring_constraints"]
        }

    def test_insert_entity_from_omniedit_addition(self):
        contract = parse_instruction(
            "Add a palm tree in the background.", task_hint="addition"
        )

        self.assertValid(contract)
        self.assertEqual("insert_entity", contract["topology"])
        self.assertEqual("add", contract["operation"])
        self.assertEqual("local", contract["edit_scope"])
        self.assertEqual("entity", contract["target"]["kind"])
        self.assertEqual("tree", contract["target"]["entity_type"])
        self.assertFalse(contract["target"]["existing"])
        self.assertEqual(
            {"value": 1, "quantifier": "exact", "exact": True},
            contract["target"]["count"],
        )
        self.assertIn(
            ("placement", "position", "background"), self.constraints(contract)
        )
        self.assertFalse(contract["clarification"]["needed"])

    def test_attach_entity_to_named_body_part(self):
        contract = parse_instruction(
            "Add exactly one silver sword with a dark hilt to the man's right hand.",
            task_hint="addition",
        )

        self.assertValid(contract)
        self.assertEqual("attach_entity", contract["topology"])
        self.assertEqual("add", contract["operation"])
        self.assertEqual("local", contract["edit_scope"])
        self.assertEqual("sword", contract["target"]["entity_type"])
        self.assertEqual(1, contract["target"]["count"]["value"])
        self.assertEqual("person", contract["anchor"]["entity_type"])
        self.assertIn("hand", contract["anchor"]["part"])
        self.assertEqual("held_by", contract["anchor"]["relation"])
        self.assertFalse(contract["clarification"]["needed"])

    def test_modify_attribute_keeps_desired_value_out_of_selectors(self):
        contract = parse_instruction(
            "turn the color of mushroom to gray",
            task_hint="attribute_modification",
        )

        self.assertValid(contract)
        self.assertEqual("modify_attribute", contract["topology"])
        self.assertEqual("modify", contract["operation"])
        self.assertEqual("local", contract["edit_scope"])
        self.assertEqual("attribute", contract["target"]["kind"])
        self.assertEqual("mushroom", contract["target"]["entity_type"])
        self.assertEqual("appearance.color", contract["target"]["attribute_name"])
        self.assertTrue(contract["target"]["existing"])
        self.assertEqual("gray", contract["desired_state"])
        self.assertFalse(
            any(item["value"] == "gray" for item in contract["referring_constraints"])
        )
        self.assertEqual("mushroom", contract["anchor"]["entity_type"])
        self.assertEqual("attribute_of", contract["anchor"]["relation"])

    def test_remove_entity_does_not_promote_descriptor_to_target(self):
        contract = parse_instruction(
            "Remove a woman with brown hair.", task_hint="removal"
        )

        self.assertValid(contract)
        self.assertEqual("remove_entity", contract["topology"])
        self.assertEqual("remove", contract["operation"])
        self.assertEqual("local", contract["edit_scope"])
        self.assertEqual("entity", contract["target"]["kind"])
        self.assertEqual("person", contract["target"]["entity_type"])
        self.assertIn("woman", contract["target"]["name"])
        self.assertNotEqual("hair", contract["target"].get("attribute_name"))
        self.assertIn(
            ("target", "appearance.color", "brown"),
            self.constraints(contract),
        )
        self.assertNotIn("anchor", contract)
        self.assertFalse(contract["clarification"]["needed"])

    def test_remove_text_attribute_grounds_its_carrier(self):
        contract = parse_instruction(
            "Remove handwritten Latvian text on a red rectangular piece of paper "
            "in the foreground.",
            task_hint="removal",
        )

        self.assertValid(contract)
        self.assertEqual("remove_attribute", contract["topology"])
        self.assertEqual("remove", contract["operation"])
        self.assertEqual("local", contract["edit_scope"])
        self.assertEqual("attribute", contract["target"]["kind"])
        self.assertEqual("surface.text", contract["target"]["attribute_name"])
        self.assertEqual("paper", contract["target"]["entity_type"])
        self.assertEqual("paper", contract["anchor"]["entity_type"])
        self.assertEqual("attribute_of", contract["anchor"]["relation"])
        self.assertIn(
            ("anchor", "appearance.color", "red"),
            self.constraints(contract),
        )
        self.assertIn(
            ("anchor", "position", "foreground"), self.constraints(contract)
        )

    def test_replace_entity_scopes_source_anchor_and_result(self):
        contract = parse_instruction(
            "Replace the red cup beside the blue plate with a green glass mug.",
            task_hint="swap",
        )

        self.assertValid(contract)
        self.assertEqual("replace_entity", contract["topology"])
        self.assertEqual("replace", contract["operation"])
        self.assertEqual("local", contract["edit_scope"])
        self.assertEqual("cup", contract["target"]["entity_type"])
        self.assertTrue(contract["target"]["existing"])
        self.assertEqual("mug", contract["replacement"]["entity_type"])
        self.assertFalse(contract["replacement"]["existing"])
        self.assertEqual("a green glass mug", contract["desired_state"])
        self.assertEqual("plate", contract["anchor"]["entity_type"])
        constraints = self.constraints(contract)
        self.assertIn(("target", "appearance.color", "red"), constraints)
        self.assertIn(("anchor", "appearance.color", "blue"), constraints)
        self.assertIn(("replacement", "appearance.color", "green"), constraints)
        self.assertIn(
            ("replacement", "appearance.material", "glass"), constraints
        )
        self.assertFalse(
            any(
                scope in {"target", "anchor"} and value in {"green", "glass"}
                for scope, _, value in constraints
            )
        )
        self.assertFalse(contract["clarification"]["needed"])

    def test_replace_background_from_omniedit_swap_pattern(self):
        contract = parse_instruction(
            "Swap the green mountainous terrain with an orange desert landscape "
            "featuring towering sandstone formations.",
            task_hint="swap",
        )

        self.assertValid(contract)
        self.assertEqual("replace_background", contract["topology"])
        self.assertEqual("replace", contract["operation"])
        self.assertEqual("regional", contract["edit_scope"])
        self.assertEqual("entity", contract["target"]["kind"])
        self.assertTrue(contract["target"]["existing"])
        self.assertIn("terrain", contract["target"]["name"])
        self.assertEqual("entity", contract["replacement"]["kind"])
        self.assertFalse(contract["replacement"]["existing"])
        self.assertIn("desert landscape", contract["replacement"]["name"])
        self.assertIn("desert landscape", contract["desired_state"])
        self.assertIn(
            ("target", "appearance.color", "green"),
            self.constraints(contract),
        )
        self.assertIn(
            ("replacement", "appearance.color", "orange"),
            self.constraints(contract),
        )

    def test_modify_environment_from_omniedit_env_pattern(self):
        contract = parse_instruction(
            "change the setting to a foggy atmosphere", task_hint="env"
        )

        self.assertValid(contract)
        self.assertEqual("modify_environment", contract["topology"])
        self.assertEqual("modify", contract["operation"])
        self.assertEqual("global", contract["edit_scope"])
        self.assertEqual("attribute", contract["target"]["kind"])
        self.assertEqual("scene", contract["target"]["entity_type"])
        self.assertEqual(
            "scene.environment", contract["target"]["attribute_name"]
        )
        self.assertEqual("a foggy atmosphere", contract["desired_state"])
        self.assertNotIn("anchor", contract)
        self.assertFalse(contract["clarification"]["needed"])

    def test_apply_style_from_omniedit_style_pattern(self):
        contract = parse_instruction(
            "Make it look like a cubist painting.", task_hint="style"
        )

        self.assertValid(contract)
        self.assertEqual("apply_style", contract["topology"])
        self.assertEqual("modify", contract["operation"])
        self.assertEqual("global", contract["edit_scope"])
        self.assertEqual("attribute", contract["target"]["kind"])
        self.assertEqual("scene", contract["target"]["entity_type"])
        self.assertEqual("scene.style", contract["target"]["attribute_name"])
        self.assertIn("cubist painting", contract["desired_state"])
        self.assertNotIn("anchor", contract)
        self.assertFalse(contract["clarification"]["needed"])

    def test_task_normalization_and_multi_target_safeguards(self):
        self.assertEqual("addition", normalize_dataset_task("Object Addition"))
        self.assertEqual(
            "attribute_modification", normalize_dataset_task("attribute-change")
        )
        self.assertEqual("env", normalize_dataset_task("environment_change"))
        self.assertEqual("unknown", normalize_dataset_task("brand_new_task"))
        self.assertEqual("unknown", normalize_dataset_task(None))

        blocked = parse_instruction(
            "Remove a large rock formation in the background and cloudy skies.",
            task_hint="removal",
        )
        self.assertValid(blocked)
        self.assertTrue(blocked["clarification"]["needed"])
        self.assertIn(
            "instruction.targets",
            {item["field"] for item in blocked["ambiguities"]},
        )

        descriptor_conjunction = parse_instruction(
            "Remove a red and white striped umbrella over a market stall.",
            task_hint="removal",
        )
        self.assertValid(descriptor_conjunction)
        self.assertEqual("remove_entity", descriptor_conjunction["topology"])
        self.assertEqual(
            "umbrella", descriptor_conjunction["target"]["entity_type"]
        )
        self.assertFalse(descriptor_conjunction["clarification"]["needed"])


if __name__ == "__main__":
    unittest.main()
