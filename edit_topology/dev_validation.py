"""Small development checks for the edit-topology parser.

These cases are smoke tests, not a benchmark.  They make parser regressions
visible while the contract and grounding pipeline are still being developed.
"""

from __future__ import annotations

from typing import Any

from .parser import parse_instruction, validate_contract


DEV_CASES = [
    {
        "name": "attach hat",
        "instruction": "Add a hat to the man on the left.",
        "topology": "attach_entity",
        "operation": "add",
        "target": "hat",
        "anchor_type": "person",
    },
    {
        "name": "insert bottle",
        "instruction": "Add one glass bottle standing on the empty part of the table.",
        "topology": "insert_entity",
        "operation": "add",
        "target": "glass bottle",
        "anchor_type": "table",
    },
    {
        "name": "remove writing",
        "instruction": "Remove the handwriting from the red paper.",
        "topology": "remove_attribute",
        "operation": "remove",
        "target": "handwriting",
        "anchor_type": "paper",
    },
    {
        "name": "remove people",
        "instruction": "Remove two men in white shirts.",
        "topology": "remove_entity",
        "operation": "remove",
        "target": "men",
        "anchor_type": None,
    },
    {
        "name": "modify hair",
        "instruction": "Add long hair.",
        "topology": "modify_attribute",
        "operation": "modify",
        "target": "long hair",
        "anchor_type": "person",
    },
    {
        "name": "deictic ambiguity",
        "instruction": "Add a handle to it.",
        "topology": "attach_entity",
        "operation": "add",
        "target": "handle",
        "anchor_type": "object",
        "clarification": True,
    },
    {
        "name": "remove flower",
        "instruction": "Remove only the pink flower.",
        "topology": "remove_entity",
        "operation": "remove",
        "target": "pink flower",
        "anchor_type": None,
    },
    {
        "name": "replace object",
        "instruction": "Replace the red cup with a blue ceramic mug.",
        "topology": "replace_entity",
        "operation": "replace",
        "target": "red cup",
        "anchor_type": None,
    },
    {
        "name": "replace background",
        "instruction": "Replace the background with a snowy mountain landscape.",
        "topology": "replace_background",
        "operation": "replace",
        "target": "background",
        "anchor_type": None,
    },
    {
        "name": "modify environment",
        "instruction": "Make the whole scene rainy.",
        "topology": "modify_environment",
        "operation": "modify",
        "target": "scene environment",
        "anchor_type": None,
    },
    {
        "name": "apply style",
        "instruction": "Apply watercolor style to the whole image.",
        "topology": "apply_style",
        "operation": "modify",
        "target": "scene style",
        "anchor_type": None,
    },
]


def contract_summary(contract: dict[str, Any]) -> dict[str, Any]:
    """Return the fields a developer needs for quick visual inspection."""

    anchor = contract.get("anchor") or {}
    roles = contract["region_roles"]
    return {
        "topology": contract["topology"],
        "operation": contract["operation"],
        "target": contract["target"]["name"],
        "anchor": anchor.get("entity_type") or "-",
        "relations": len(contract["required_relations"]),
        "target_regions": len(roles["target"]),
        "dependent_regions": len(roles["dependent"]),
        "context_regions": len(roles["context"]),
        "protected_regions": len(roles["protected"]),
        "clarification": contract["clarification"]["needed"],
    }


def check_expected(contract: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Check semantic fields that JSON Schema cannot validate."""

    failures = []
    actual_anchor = (contract.get("anchor") or {}).get("entity_type")
    checks = {
        "topology": contract["topology"],
        "operation": contract["operation"],
        "target": contract["target"]["name"],
        "anchor_type": actual_anchor,
    }
    if "clarification" in expected:
        checks["clarification"] = contract["clarification"]["needed"]
    for field, actual in checks.items():
        if actual != expected.get(field):
            failures.append(
                f"{field}: expected {expected.get(field)!r}, got {actual!r}"
            )
    return failures


def run_dev_cases() -> list[dict[str, Any]]:
    """Run the deterministic parser over the development smoke cases."""

    results = []
    for case in DEV_CASES:
        contract = parse_instruction(case["instruction"])
        schema_errors = validate_contract(contract)
        semantic_errors = check_expected(contract, case)
        errors = schema_errors + semantic_errors
        results.append(
            {
                "case": case["name"],
                "instruction": case["instruction"],
                "expected": case["topology"],
                "actual": contract["topology"],
                "target": contract["target"]["name"],
                "anchor": (contract.get("anchor") or {}).get("entity_type") or "-",
                "schema": "PASS" if not schema_errors else "FAIL",
                "semantic": "PASS" if not semantic_errors else "FAIL",
                "details": "OK" if not errors else " | ".join(errors),
            }
        )
    return results
