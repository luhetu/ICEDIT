import unittest

from jsonschema import Draft202012Validator

from edit_topology.parser import SCHEMA_PATH, parse_instruction
import json


class ParserTrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.validator = Draft202012Validator(schema)

    def assertValid(self, contract):
        errors = sorted(self.validator.iter_errors(contract), key=lambda error: list(error.path))
        self.assertEqual([], [f"{list(error.path)}: {error.message}" for error in errors])

    def test_add_hat_to_man_is_attachment(self):
        contract = parse_instruction("Add a hat to the man on the left.", "hat_people.png")
        self.assertValid(contract)
        self.assertEqual("attach_entity", contract["topology"])
        self.assertEqual("add", contract["operation"])
        self.assertEqual("hat", contract["target"]["entity_type"])
        self.assertEqual("person", contract["anchor"]["entity_type"])

    def test_add_long_hair_is_attribute_modification(self):
        contract = parse_instruction("Add long hair.", "person.png")
        self.assertValid(contract)
        self.assertEqual("modify_attribute", contract["topology"])
        self.assertEqual("modify", contract["operation"])
        self.assertEqual("hair", contract["anchor"]["part"])

    def test_add_bottle_on_table_keeps_bottle_as_target(self):
        contract = parse_instruction("Add one glass bottle standing on the empty part of the table.", "table.png")
        self.assertValid(contract)
        self.assertEqual("insert_entity", contract["topology"])
        self.assertEqual("bottle", contract["target"]["entity_type"])
        self.assertEqual("table", contract["anchor"]["entity_type"])
        self.assertFalse(contract["clarification"]["needed"])

    def test_deictic_anchor_requires_clarification(self):
        contract = parse_instruction("add a man on it")
        self.assertEqual(contract["topology"], "attach_entity")
        self.assertEqual(contract["anchor"]["entity_type"], "object")
        self.assertIsNone(contract["anchor"]["instance_ref"])
        self.assertTrue(contract["clarification"]["needed"])
        self.assertIn("'it'", contract["clarification"]["questions"][0])
        self.assertEqual(contract["required_relations"][0]["object"], "selected_object")

    def test_explicit_stone_anchor_is_supported_by(self):
        contract = parse_instruction(
            "Add a small man standing on the flat gray stone on the right."
        )
        self.assertEqual(contract["topology"], "insert_entity")
        self.assertEqual(contract["anchor"]["entity_type"], "stone")
        self.assertEqual(contract["anchor"]["relation"], "supported_by")
        self.assertFalse(contract["clarification"]["needed"])

    def test_possessive_hand_anchor_is_grounded(self):
        contract = parse_instruction(
            "Add exactly one silver sword with a dark hilt to the man's right hand."
        )
        self.assertValid(contract)
        self.assertEqual("attach_entity", contract["topology"])
        self.assertEqual("person", contract["anchor"]["entity_type"])
        self.assertEqual("hand", contract["anchor"]["part"])
        self.assertEqual("held_by", contract["anchor"]["relation"])
        self.assertFalse(contract["clarification"]["needed"])

    def test_remove_handwriting_from_paper_is_attribute_removal(self):
        contract = parse_instruction("Remove the handwriting from the red paper.", "red_paper.png")
        self.assertValid(contract)
        self.assertEqual("remove_attribute", contract["topology"])
        self.assertEqual("remove", contract["operation"])
        self.assertEqual("surface.handwriting", contract["target"]["attribute_name"])

    def test_remove_two_men_is_entity_removal(self):
        contract = parse_instruction("Remove two men in white shirts.", "street.png")
        self.assertValid(contract)
        self.assertEqual("remove_entity", contract["topology"])
        self.assertEqual(2, contract["target"]["count"]["value"])
        self.assertEqual("person", contract["target"]["entity_type"])

    def test_protection_clause_does_not_replace_sign_target(self):
        contract = parse_instruction(
            "Remove only the Coca-Cola signs. Keep the building and all other text unchanged."
        )
        self.assertValid(contract)
        self.assertEqual("remove_entity", contract["topology"])
        self.assertEqual("sign", contract["target"]["entity_type"])
        self.assertIn("coca-cola signs", contract["target"]["name"])

    def test_leading_protection_clause_does_not_hide_edit(self):
        contract = parse_instruction(
            "Keep the people unchanged and remove the sign."
        )
        self.assertValid(contract)
        self.assertEqual("remove_entity", contract["topology"])
        self.assertEqual("sign", contract["target"]["entity_type"])
        self.assertIn(
            "the people",
            [region["description"] for region in contract["region_roles"]["protected"]],
        )
        self.assertFalse(contract["clarification"]["needed"])

    def test_negated_edit_is_a_protection_not_the_requested_action(self):
        contract = parse_instruction(
            "Do not remove the bird; remove only the cage. Keep the bird unchanged."
        )
        self.assertValid(contract)
        self.assertEqual("remove_entity", contract["topology"])
        self.assertEqual("cage", contract["target"]["entity_type"])
        self.assertIn(
            "the bird",
            [region["description"] for region in contract["region_roles"]["protected"]],
        )
        self.assertFalse(contract["clarification"]["needed"])

    def test_leave_untouched_clause_becomes_protection(self):
        contract = parse_instruction(
            "Remove the bird, but leave the two logos and all text untouched."
        )
        self.assertValid(contract)
        self.assertEqual("bird", contract["target"]["entity_type"])
        self.assertIn(
            "the two logos and all text",
            [region["description"] for region in contract["region_roles"]["protected"]],
        )
        self.assertFalse(contract["clarification"]["needed"])

    def test_two_positive_operations_require_separate_contracts(self):
        contract = parse_instruction("Remove the sign and add a tree.")
        self.assertValid(contract)
        self.assertTrue(contract["clarification"]["needed"])
        self.assertIn(
            "instruction.atomicity",
            [ambiguity["field"] for ambiguity in contract["ambiguities"]],
        )

    def test_replace_operation_is_supported_in_v02(self):
        contract = parse_instruction("Replace the red cup with a blue mug.")
        self.assertValid(contract)
        self.assertEqual("replace_entity", contract["topology"])
        self.assertEqual("replace", contract["operation"])
        self.assertEqual("mug", contract["replacement"]["entity_type"])
        self.assertFalse(contract["clarification"]["needed"])

    def test_negated_only_instruction_requires_a_positive_edit(self):
        contract = parse_instruction("Do not remove the sign.")
        self.assertValid(contract)
        self.assertTrue(contract["clarification"]["needed"])
        self.assertIn(
            "the sign",
            [region["description"] for region in contract["region_roles"]["protected"]],
        )

    def test_two_named_targets_require_clarification(self):
        contract = parse_instruction("Remove the bird and the cage.")
        self.assertValid(contract)
        self.assertTrue(contract["clarification"]["needed"])
        self.assertIn(
            "instruction.targets",
            [ambiguity["field"] for ambiguity in contract["ambiguities"]],
        )

    def test_unspecified_count_requires_clarification(self):
        contract = parse_instruction("Remove some signs.")
        self.assertValid(contract)
        self.assertEqual("unspecified", contract["target"]["count"]["quantifier"])
        self.assertTrue(contract["clarification"]["needed"])
        self.assertIn(
            "target.count",
            [ambiguity["field"] for ambiguity in contract["ambiguities"]],
        )

    def test_except_clause_protects_the_excluded_target(self):
        contract = parse_instruction(
            "Remove all signs except the small blue sign."
        )
        self.assertValid(contract)
        self.assertEqual("all", contract["target"]["count"]["quantifier"])
        self.assertIn(
            "the small blue sign",
            [region["description"] for region in contract["region_roles"]["protected"]],
        )
        self.assertFalse(contract["clarification"]["needed"])

    def test_anchor_count_does_not_change_target_count(self):
        contract = parse_instruction("Remove the bird from the two cages.")
        self.assertValid(contract)
        self.assertEqual(1, contract["target"]["count"]["value"])
        self.assertEqual("cage", contract["anchor"]["entity_type"])


if __name__ == "__main__":
    unittest.main()
