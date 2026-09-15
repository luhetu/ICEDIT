import unittest

from edit_topology.executor import (
    ContractExecutionError,
    compile_contract_prompt,
    load_contract,
    validate_execution_inputs,
)
from edit_topology.parser import parse_instruction


class ContractExecutorTests(unittest.TestCase):
    @staticmethod
    def _v02_contract(topology, desired_state):
        operation = "replace" if topology.startswith("replace_") else "modify"
        edit_scope = {
            "replace_entity": "local",
            "replace_background": "regional",
            "modify_environment": "global",
            "apply_style": "global",
        }[topology]
        target_kind = "entity" if topology.startswith("replace_") else "attribute"
        target_type = {
            "replace_entity": "cup",
            "replace_background": "background",
            "modify_environment": "scene",
            "apply_style": "scene",
        }[topology]
        target = {
            "kind": target_kind,
            "name": "red cup" if topology == "replace_entity" else target_type,
            "entity_type": target_type,
            "instance_ref": target_type,
            "count": {"value": 1, "quantifier": "exact", "exact": True},
            "existing": True,
        }
        if target_kind == "attribute":
            target["attribute_name"] = (
                "scene.environment"
                if topology == "modify_environment"
                else "scene.style"
            )
        contract = {
            "schema_version": "0.2.0",
            "contract_id": f"test_{topology}",
            "instruction": {
                "replace_entity": "Replace the red cup with a blue ceramic mug.",
                "replace_background": "Replace the background with snowy mountains.",
                "modify_environment": "Change the scene to golden-hour lighting.",
                "apply_style": "Apply watercolor style to the whole image.",
            }[topology],
            "topology": topology,
            "operation": operation,
            "edit_scope": edit_scope,
            "target": target,
            "source_state": f"current {target_type}",
            "desired_state": desired_state,
            "referring_constraints": [],
            "required_relations": [],
            "forbidden_outcomes": ["change protected content"],
            "invariants": [
                {
                    "subject": "protected scene content",
                    "property": "identity, geometry, layout, pose, and readable text",
                    "tolerance": "strict",
                }
            ],
            "region_roles": {
                "target": [{"description": target_type, "mask_status": "pending"}],
                "dependent": [],
                "context": [],
                "protected": [
                    {"description": "all non-target content", "mask_status": "pending"}
                ],
            },
            "ambiguities": [],
            "clarification": {"needed": False, "questions": []},
            "provenance": [
                {"claim": f"topology is {topology}", "source": "annotation_rule"}
            ],
        }
        if topology.startswith("replace_"):
            replacement_type = "mug" if topology == "replace_entity" else "background"
            contract["replacement"] = {
                "kind": "entity",
                "name": desired_state,
                "entity_type": replacement_type,
                "instance_ref": f"new_{replacement_type}",
                "count": {"value": 1, "quantifier": "exact", "exact": True},
                "existing": False,
            }
        return contract

    def test_compiles_topology_and_preservation_for_icedit(self):
        contract = parse_instruction(
            "Add a hat to the man on the left. Keep the other people unchanged."
        )
        prompt = compile_contract_prompt(contract)
        self.assertIn("local contact area", prompt)
        self.assertIn("hat", prompt)
        self.assertNotIn("other people", prompt)
        self.assertNotIn("Required relation", prompt)
        self.assertEqual(1, prompt.lower().count("hat"))

    def test_removal_prompt_does_not_repeat_protected_text(self):
        contract = parse_instruction(
            "Remove only the Coca-Cola signs. Keep the building and all other text unchanged."
        )
        prompt = compile_contract_prompt(contract)
        self.assertEqual(
            "A photorealistic seamless continuation of the surrounding "
            "background and visible surfaces, matching nearby geometry, "
            "texture, lighting, and perspective. Everything outside the "
            "masked regions is exactly unchanged.",
            prompt,
        )
        self.assertNotIn("Coca-Cola", prompt)
        self.assertNotIn("other text", prompt)

    def test_attribute_removal_prompt_does_not_repeat_removed_text(self):
        contract = parse_instruction("Remove the handwriting from the red paper.")
        prompt = compile_contract_prompt(contract)
        self.assertNotIn("handwriting", prompt.lower())
        self.assertIn("surrounding paper surface", prompt.lower())

    def test_rejects_contract_that_needs_clarification(self):
        contract = parse_instruction("Add a handle to it.")
        with self.assertRaisesRegex(ContractExecutionError, "needs clarification"):
            compile_contract_prompt(contract)

    def test_rejects_invalid_json(self):
        with self.assertRaisesRegex(ContractExecutionError, "JSON is invalid"):
            load_contract("{not-json}")

    def test_replace_entity_prompt_uses_result_not_source_selector(self):
        contract = self._v02_contract("replace_entity", "a blue ceramic mug")
        prompt = compile_contract_prompt(contract)
        self.assertIn("blue ceramic mug", prompt)
        self.assertNotIn("red cup", prompt)
        self.assertIn("outside the masked region", prompt)

    def test_replace_background_prompt_protects_foreground(self):
        contract = self._v02_contract(
            "replace_background", "a snowy mountain landscape"
        )
        prompt = compile_contract_prompt(contract)
        self.assertIn("snowy mountain landscape", prompt)
        self.assertIn("foreground entity", prompt)
        self.assertIn("readable text", prompt)

    def test_global_topology_prompts_preserve_structure_not_appearance(self):
        cases = {
            "modify_environment": "golden-hour lighting",
            "apply_style": "a watercolor painting style",
        }
        for topology, desired_state in cases.items():
            with self.subTest(topology=topology):
                prompt = compile_contract_prompt(
                    self._v02_contract(topology, desired_state)
                )
                self.assertIn(desired_state, prompt)
                self.assertIn("identity", prompt)
                self.assertIn("geometry", prompt)
                self.assertIn("readable text", prompt)
                self.assertNotIn("appearance exactly unchanged", prompt)
                self.assertNotIn("texture exactly unchanged", prompt)

    def test_replace_topologies_require_runtime_mask(self):
        for topology in ("replace_entity", "replace_background"):
            with self.subTest(topology=topology):
                contract = self._v02_contract(topology, "a valid desired result")
                with self.assertRaisesRegex(ContractExecutionError, "target mask"):
                    validate_execution_inputs(contract)
                self.assertEqual(
                    contract,
                    validate_execution_inputs(contract, target_mask=object()),
                )

    def test_global_topologies_do_not_require_runtime_mask(self):
        for topology in ("modify_environment", "apply_style"):
            with self.subTest(topology=topology):
                contract = self._v02_contract(topology, "a valid desired result")
                self.assertEqual(contract, validate_execution_inputs(contract))

    def test_regional_style_requires_runtime_mask_and_compiles_local_prompt(self):
        contract = self._v02_contract("apply_style", "a watercolor painting style")
        contract["edit_scope"] = "regional"
        with self.assertRaisesRegex(ContractExecutionError, "target mask"):
            validate_execution_inputs(contract)
        prompt = compile_contract_prompt(contract)
        self.assertIn("masked region", prompt)
        self.assertIn("outside the masked region", prompt)


if __name__ == "__main__":
    unittest.main()
