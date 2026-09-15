import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from edit_topology.parser import parse_instruction, validate_contract


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema" / "edit_topology_contract.schema.json"
EXAMPLES_PATH = ROOT / "examples" / "contracts.json"


class EditTopologyContractSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.contracts = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def assertInvalid(self, contract):
        with self.assertRaises(ValidationError):
            self.validator.validate(contract)

    def test_schema_is_valid_draft_2020_12(self):
        try:
            Draft202012Validator.check_schema(self.schema)
        except SchemaError as error:  # pragma: no cover - improves failure output
            self.fail(str(error))

    def test_exactly_twenty_unique_review_contracts(self):
        self.assertEqual(20, len(self.contracts))
        ids = [contract["contract_id"] for contract in self.contracts]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_examples_validate(self):
        failures = {}
        for contract in self.contracts:
            errors = sorted(
                self.validator.iter_errors(contract), key=lambda error: list(error.path)
            )
            if errors:
                failures[contract.get("contract_id", "<missing>")] = [
                    f"{list(error.path)}: {error.message}" for error in errors
                ]
        self.assertEqual({}, failures)

    def test_examples_cover_initial_five_topologies(self):
        observed = {contract["topology"] for contract in self.contracts}
        self.assertEqual(
            {
                "insert_entity",
                "attach_entity",
                "modify_attribute",
                "remove_entity",
                "remove_attribute",
            },
            observed,
        )

    def test_schema_declares_all_nine_v02_topologies(self):
        self.assertEqual(
            {
                "insert_entity",
                "attach_entity",
                "modify_attribute",
                "remove_entity",
                "remove_attribute",
                "replace_entity",
                "replace_background",
                "modify_environment",
                "apply_style",
            },
            set(self.schema["properties"]["topology"]["enum"]),
        )

    def test_operation_must_match_topology(self):
        invalid = copy.deepcopy(self.contracts[0])
        invalid["operation"] = "remove"
        self.assertInvalid(invalid)

    def test_attachment_requires_anchor(self):
        invalid = copy.deepcopy(
            next(c for c in self.contracts if c["topology"] == "attach_entity")
        )
        del invalid["anchor"]
        self.assertInvalid(invalid)

    def test_unresolved_clarification_requires_question(self):
        invalid = copy.deepcopy(self.contracts[0])
        invalid["clarification"] = {"needed": True, "questions": []}
        self.assertInvalid(invalid)

    def test_exact_count_cannot_be_null(self):
        invalid = copy.deepcopy(self.contracts[0])
        invalid["target"]["count"]["value"] = None
        self.assertInvalid(invalid)

    def test_unknown_fields_are_rejected(self):
        invalid = copy.deepcopy(self.contracts[0])
        invalid["model_guess"] = "hidden parser state"
        self.assertInvalid(invalid)

    def test_referring_constraint_requires_scope(self):
        invalid = copy.deepcopy(
            next(c for c in self.contracts if c["referring_constraints"])
        )
        del invalid["referring_constraints"][0]["scope"]
        self.assertInvalid(invalid)

    def test_semantics_require_clarification_for_unresolved_ambiguity(self):
        invalid = parse_instruction("Remove the red sign.")
        invalid["ambiguities"].append(
            {
                "field": "target.instance_ref",
                "status": "unresolved",
                "confidence": 0.1,
                "description": "More than one matching sign may be visible.",
                "assumption": None,
            }
        )
        self.assertIn(
            "semantic: unresolved ambiguities require clarification",
            validate_contract(invalid),
        )

    def test_grounded_region_requires_mask_uri(self):
        invalid = parse_instruction("Remove the red sign.")
        invalid["region_roles"]["target"][0]["mask_status"] = "grounded"
        self.assertTrue(
            any("grounded without mask_uri" in error for error in validate_contract(invalid))
        )


if __name__ == "__main__":
    unittest.main()
