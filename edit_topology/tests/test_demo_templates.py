import unittest

from edit_topology.demo_templates import PROMPT_TEMPLATES, template_prompt
from edit_topology.parser import parse_instruction, validate_contract


class DemoTemplateTests(unittest.TestCase):
    def test_named_templates_are_nonempty_and_schema_valid(self):
        for label, prompt in PROMPT_TEMPLATES.items():
            if label == "Choose a template...":
                self.assertEqual(prompt, "")
                continue
            with self.subTest(label=label):
                self.assertTrue(prompt)
                self.assertEqual(validate_contract(parse_instruction(prompt)), [])

    def test_unknown_template_is_safe(self):
        self.assertEqual(template_prompt("missing"), "")

    def test_flower_template_populates_explicit_protection(self):
        prompt = PROMPT_TEMPLATES["Flower image · man on right stone"]
        contract = parse_instruction(prompt)
        protected = [region["description"] for region in contract["region_roles"]["protected"]]
        self.assertTrue(any("pink flower" in region for region in protected))
        self.assertTrue(any("all leaves" in region for region in protected))
        self.assertTrue(any("pink flower" in item["subject"] for item in contract["invariants"]))


if __name__ == "__main__":
    unittest.main()
