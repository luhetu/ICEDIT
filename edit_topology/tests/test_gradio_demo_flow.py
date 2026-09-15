import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
from PIL import Image

with patch.object(sys, "argv", ["gradio_demo.py"]):
    import scripts.gradio_demo as gradio_demo

from edit_topology.executor import ContractExecutionError


class GradioDemoFlowTests(unittest.TestCase):
    def test_clarification_is_reported_as_blocked(self):
        payload, status = gradio_demo.plan_edit_contract(
            None,
            "Add a handle to it.",
            "Local rules (no image/API)",
            "",
        )

        self.assertTrue(payload)
        self.assertIn("blocked_by_plan", status)
        self.assertIn("Clarification required", status)
        self.assertNotIn("Use Proceed", status)

    def test_current_instruction_must_match_planned_contract(self):
        instruction = "Add a tree in the background."
        payload, _ = gradio_demo.plan_edit_contract(
            None,
            instruction,
            "Local rules (no image/API)",
            "",
        )

        contract = gradio_demo.load_contract_for_instruction(payload, instruction)
        self.assertEqual(instruction, contract["instruction"])
        with self.assertRaisesRegex(ContractExecutionError, "instruction changed"):
            gradio_demo.load_contract_for_instruction(
                payload, "Add a cabin in the background."
            )

    def test_input_invalidation_clears_contract_and_outputs(self):
        contract, planner_status, result, gallery, executor_status = (
            gradio_demo.invalidate_plan_state("Instruction changed.")
        )

        self.assertEqual("", contract)
        self.assertIn("Instruction changed", planner_status)
        self.assertIsNone(result)
        self.assertEqual([], gallery)
        self.assertIn("not_requested", executor_status)

        reset = gradio_demo.reset_after_image_change(
            Image.new("RGB", (16, 16), "white")
        )
        self.assertEqual(6, len(reset))
        self.assertIsInstance(reset[0], dict)
        self.assertEqual("", reset[1])
        self.assertIsNone(reset[3])
        self.assertEqual([], reset[4])

    def test_masked_icedit_restores_every_unmasked_pixel(self):
        class FakePipe:
            def __call__(self, **kwargs):
                generated = Image.new(
                    "RGB", (kwargs["width"], kwargs["height"]), "blue"
                )
                return SimpleNamespace(images=[generated])

        source = Image.new("RGB", (512, 64), "red")
        mask = Image.new("L", source.size, 0)
        mask.putpixel((10, 10), 255)

        with tempfile.TemporaryDirectory() as output_dir:
            with (
                patch.object(gradio_demo, "pipe", FakePipe()),
                patch.object(gradio_demo.args, "output_dir", output_dir),
            ):
                result, actual_seed = gradio_demo.infer(
                    source,
                    "change the masked pixel",
                    seed=7,
                    randomize_seed=False,
                    guidance_scale=1,
                    num_inference_steps=1,
                    lora_scale=1.0,
                    edit_mask=mask,
                    native_fill=False,
                )

        self.assertEqual(7, actual_seed)
        self.assertEqual((255, 0, 0), result.getpixel((0, 0)))
        self.assertEqual((0, 0, 255), result.getpixel((10, 10)))

    def test_raw_run_reports_generated_but_unverified(self):
        expected = Image.new("RGB", (8, 8), "white")
        with patch.object(gradio_demo, "infer", return_value=(expected, 19)):
            result, seed, status, gallery = (
                gradio_demo.infer_original_with_status(expected, "Make it blue.")
            )

        self.assertIs(expected, result)
        self.assertEqual(19, seed)
        self.assertEqual([], gallery)
        self.assertIn("generated_unverified", status)
        self.assertIn("not_run", status)

    def test_prompt_submit_does_not_silently_run_raw(self):
        submit_functions = [
            block_function.fn
            for block_function in gradio_demo.demo.fns.values()
            if any(event == "submit" for _, event in block_function.targets)
        ]

        self.assertEqual([], submit_functions)
        self.assertNotIn(gradio_demo.infer, submit_functions)
        self.assertNotIn(gradio_demo.infer_original_with_status, submit_functions)


if __name__ == "__main__":
    unittest.main()
