"""Lightweight Gradio demo for inspecting Edit Topology Contracts."""

from __future__ import annotations

import argparse
import json
from typing import Any

import gradio as gr

from .dev_validation import contract_summary, run_dev_cases
from .demo_templates import PROMPT_TEMPLATES, template_prompt
from .model_parser import ModelContractError, parse_with_model
from .parser import parse_instruction, validate_contract


PROVIDERS = {
    "Local rules (no image/API)": None,
    "OpenAI VLM": "openai",
    "Gemini VLM": "gemini",
}


def _region_rows(contract: dict[str, Any]) -> list[list[str]]:
    rows = []
    for role, entries in contract["region_roles"].items():
        for entry in entries:
            rows.append(
                [
                    role,
                    entry["description"],
                    entry.get("entity_ref") or "-",
                    entry["mask_status"],
                ]
            )
    return rows


def _summary_markdown(contract: dict[str, Any], planner: str) -> str:
    summary = contract_summary(contract)
    clarification = contract["clarification"]
    questions = " / ".join(clarification["questions"]) or "None"
    execution_status = (
        "blocked_by_plan"
        if clarification["needed"]
        else "not_available_in_parser_demo"
    )
    return "\n".join(
        [
            "### Parser result",
            f"- Planner: `{planner}`",
            f"- Topology: `{summary['topology']}`",
            f"- Operation: `{summary['operation']}`",
            f"- Target: `{summary['target']}`",
            f"- Anchor: `{summary['anchor']}`",
            f"- Clarification needed: `{summary['clarification']}`",
            f"- Questions: {questions}",
            f"- Execution status: `{execution_status}`",
            "- Verification status: `not_run`",
            "- Masks: semantic roles are planned; pixel grounding is not connected yet.",
        ]
    )


def parse_for_demo(image, instruction: str, provider_label: str, model: str):
    instruction = (instruction or "").strip()
    if not instruction:
        return {}, [], "Enter an edit instruction.", "FAIL: empty instruction"

    try:
        provider = PROVIDERS[provider_label]
        if provider is None:
            contract = parse_instruction(instruction)
            planner = "rule-based baseline"
        else:
            if image is None:
                raise ValueError("Upload a source image for VLM parsing.")
            contract, planner = parse_with_model(
                image, instruction, provider=provider, model=model
            )
        errors = validate_contract(contract)
        if errors:
            validation = "FAIL\n\n" + "\n".join(f"- {error}" for error in errors)
        else:
            validation = "PASS: contract matches schema and semantics v0.2.0"
        return (
            contract,
            _region_rows(contract),
            _summary_markdown(contract, planner),
            validation,
        )
    except (KeyError, ValueError, ModelContractError) as exc:
        return {}, [], "Parser could not produce a contract.", f"FAIL: {exc}"
    except Exception as exc:
        return {}, [], "Unexpected parser failure.", f"FAIL: {type(exc).__name__}: {exc}"


def run_smoke_table():
    results = run_dev_cases()
    passed = sum(row["schema"] == "PASS" and row["semantic"] == "PASS" for row in results)
    status = f"Development smoke cases: **{passed}/{len(results)} passed**"
    headers = [
        "case",
        "instruction",
        "expected",
        "actual",
        "target",
        "anchor",
        "schema",
        "semantic",
        "details",
    ]
    return [[row[key] for key in headers] for row in results], status


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Edit Topology Parser") as demo:
        gr.Markdown("# Edit Topology Parser")
        with gr.Tabs():
            with gr.Tab("Parse one edit"):
                with gr.Row():
                    image = gr.Image(type="pil", label="Source image")
                    with gr.Column():
                        template = gr.Dropdown(
                            choices=list(PROMPT_TEMPLATES),
                            value="Choose a template...",
                            label="Instruction template",
                        )
                        instruction = gr.Textbox(
                            label="Edit instruction",
                            lines=4,
                            placeholder="Add a hat to the person on the left.",
                        )
                        provider = gr.Dropdown(
                            choices=list(PROVIDERS),
                            value="Local rules (no image/API)",
                            label="Parser",
                        )
                        model = gr.Textbox(
                            label="Optional model override",
                            placeholder="Leave blank to use the provider default",
                        )
                        parse_button = gr.Button("Parse instruction", variant="primary")
                summary = gr.Markdown("No contract parsed yet.")
                with gr.Tabs():
                    with gr.Tab("Contract JSON"):
                        contract_json = gr.JSON(label="Schema-valid contract")
                    with gr.Tab("Region roles"):
                        regions = gr.Dataframe(
                            headers=["role", "description", "entity_ref", "mask_status"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False,
                            label="Planned regions",
                        )
                    with gr.Tab("Validation"):
                        validation = gr.Textbox(label="Validation result", lines=8)

                template.change(
                    fn=template_prompt,
                    inputs=template,
                    outputs=instruction,
                )
                parse_button.click(
                    fn=parse_for_demo,
                    inputs=[image, instruction, provider, model],
                    outputs=[contract_json, regions, summary, validation],
                )

            with gr.Tab("Development checks"):
                gr.Markdown(
                    "These deterministic cases are development smoke tests, not a benchmark."
                )
                smoke_button = gr.Button("Run smoke cases", variant="primary")
                smoke_status = gr.Markdown("Checks have not run yet.")
                smoke_table = gr.Dataframe(
                    headers=[
                        "case",
                        "instruction",
                        "expected",
                        "actual",
                        "target",
                        "anchor",
                        "schema",
                        "semantic",
                        "details",
                    ],
                    datatype=["str"] * 9,
                    interactive=False,
                    label="Parser checks",
                )
                smoke_button.click(
                    fn=run_smoke_table,
                    outputs=[smoke_table, smoke_status],
                )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight Edit Topology Parser demo")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_demo().launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
