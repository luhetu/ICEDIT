import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.inference_dataset import (
    normalize_task_filter,
    normalize_target_mask,
    plan_contract,
    prepare_execution,
    run_generation,
    task_compatibility,
)


class DatasetRunnerPlanningTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(planner="rule", planner_model=None)

    @staticmethod
    def row(task):
        return {"task": task, "src_img": None}

    def test_task_filter_normalizes_aliases_and_rejects_unknown(self):
        self.assertEqual(
            {"addition", "attribute_modification", "env"},
            normalize_task_filter(
                ["Object Addition", "attribute-change", "environment_change"]
            ),
        )
        with self.assertRaisesRegex(ValueError, "Unknown task filter"):
            normalize_task_filter(["future_task"])

    def test_rule_plan_records_ready_compatible_contract(self):
        planning = plan_contract(
            self.row("removal"), "Remove a woman with brown hair.", self.args
        )
        self.assertEqual("ready", planning["status"])
        self.assertTrue(planning["ready"])
        self.assertEqual("0.2.0", planning["contract_schema_version"])
        self.assertEqual("remove_entity", planning["contract"]["topology"])
        self.assertTrue(planning["task_compatibility"]["matches"])
        self.assertGreaterEqual(planning["latency_ms"], 0)

    def test_mismatch_and_clarification_fail_closed(self):
        mismatch = plan_contract(
            self.row("removal"),
            "Replace the red cup with a blue mug.",
            self.args,
        )
        self.assertEqual("task_topology_mismatch", mismatch["status"])
        self.assertFalse(mismatch["ready"])

        ambiguous = plan_contract(
            self.row("addition"), "Add a small man on a stone.", self.args
        )
        self.assertEqual("clarification_required", ambiguous["status"])
        self.assertFalse(ambiguous["ready"])

    def test_task_compatibility_splits_swap_and_global_tasks(self):
        self.assertTrue(task_compatibility("swap", "replace_entity")["matches"])
        self.assertTrue(
            task_compatibility("swap", "replace_background")["matches"]
        )
        self.assertTrue(
            task_compatibility("env", "modify_environment")["matches"]
        )
        self.assertFalse(task_compatibility("env", "apply_style")["matches"])

    def test_plan_only_explicitly_records_that_generation_was_not_requested(self):
        args = SimpleNamespace(pipeline_mode="plan-only")
        execution = prepare_execution("Add a cup.", args)
        self.assertEqual("not_requested", execution["status"])
        self.assertEqual("not_run", execution["verification_status"])
        self.assertIsNone(execution["result_path"])

    def test_plan_only_reports_future_mask_requirement(self):
        planning = plan_contract(
            self.row("removal"), "Remove the red sign.", self.args
        )
        execution = prepare_execution(
            "Remove the red sign.",
            SimpleNamespace(pipeline_mode="plan-only"),
            planning,
        )
        self.assertEqual("not_requested", execution["status"])
        self.assertTrue(execution["requires_mask"])

    def test_planned_addition_compiles_contract_prompt(self):
        planning = plan_contract(
            self.row("addition"),
            "Add exactly one blue cup on the table.",
            self.args,
        )
        args = SimpleNamespace(pipeline_mode="planned")
        execution = prepare_execution(
            "Add exactly one blue cup on the table.", args, planning
        )
        self.assertEqual("queued", execution["status"])
        self.assertEqual("icedit", execution["backend"])
        self.assertIn("Keep everything else exactly unchanged", execution["executed_prompt"])

    def test_planned_removal_without_mask_is_blocked(self):
        planning = plan_contract(
            self.row("removal"), "Remove the red sign.", self.args
        )
        args = SimpleNamespace(pipeline_mode="planned")
        execution = prepare_execution("Remove the red sign.", args, planning)
        self.assertEqual("blocked_missing_mask", execution["status"])
        self.assertTrue(execution["requires_mask"])
        self.assertEqual("fluxfill_native", execution["backend"])
        self.assertEqual("not_run", execution["verification_status"])

    def test_planned_row_with_blocked_contract_cannot_execute(self):
        planning = plan_contract(
            self.row("addition"), "Add a small man on a stone.", self.args
        )
        args = SimpleNamespace(pipeline_mode="planned")
        execution = prepare_execution(
            "Add a small man on a stone.", args, planning
        )
        self.assertEqual("blocked_by_plan", execution["status"])
        self.assertIsNone(execution["executed_prompt"])

    def test_mask_free_plan_ignores_coincidental_mask_file(self):
        planning = plan_contract(
            self.row("addition"),
            "Add exactly one blue cup on the table.",
            self.args,
        )
        execution = prepare_execution(
            "Add exactly one blue cup on the table.",
            SimpleNamespace(pipeline_mode="planned"),
            planning,
            Path("does-not-exist.png"),
        )
        self.assertEqual("queued", execution["status"])
        self.assertFalse(execution["requires_mask"])
        self.assertIsNone(execution["mask_path"])

    def test_transparent_png_mask_uses_alpha(self):
        mask = Image.new("RGBA", (4, 4), (255, 255, 255, 0))
        mask.putpixel((2, 1), (255, 255, 255, 255))
        normalized = np.asarray(normalize_target_mask(mask))
        self.assertEqual(1, int(np.count_nonzero(normalized)))

    def test_empty_and_whole_image_masks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            normalize_target_mask(Image.new("L", (4, 4), 0))
        with self.assertRaisesRegex(ValueError, "entire image"):
            normalize_target_mask(Image.new("L", (4, 4), 255))

    def test_invalid_mask_blocks_before_generation(self):
        planning = plan_contract(
            self.row("removal"), "Remove the red sign.", self.args
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_path = Path(temp_dir) / "mask.png"
            Image.new("L", (4, 4), 0).save(mask_path)
            execution = prepare_execution(
                "Remove the red sign.",
                SimpleNamespace(pipeline_mode="planned"),
                planning,
                mask_path,
            )
        self.assertEqual("blocked_invalid_mask", execution["status"])

    def test_generation_receives_compiled_prompt(self):
        class FakePipe:
            def __init__(self):
                self.prompt = None

            def __call__(self, **kwargs):
                self.prompt = kwargs["prompt"]
                return SimpleNamespace(images=[kwargs["image"].copy()])

        planning = plan_contract(
            self.row("addition"),
            "Add exactly one blue cup on the table.",
            self.args,
        )
        execution = prepare_execution(
            "Add exactly one blue cup on the table.",
            SimpleNamespace(pipeline_mode="planned"),
            planning,
        )
        generation_args = SimpleNamespace(
            guidance_scale=50.0,
            num_inference_steps=1,
        )
        pipe = FakePipe()
        result, width, height = run_generation(
            pipe,
            Image.new("RGB", (512, 320), "white"),
            execution,
            7,
            generation_args,
        )
        self.assertIn(execution["executed_prompt"], pipe.prompt)
        self.assertEqual((width, height), result.size)


if __name__ == "__main__":
    unittest.main()
