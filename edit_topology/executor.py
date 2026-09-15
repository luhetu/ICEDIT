"""Compile a validated Edit Topology Contract for the current ICEdit backend."""

from __future__ import annotations

import json
import re
from typing import Any

from .parser import validate_contract


class ContractExecutionError(ValueError):
    """Raised when a contract is unsafe or incomplete for execution."""


TOPOLOGY_SUFFIXES = {
    "insert_entity": "Keep everything else exactly unchanged.",
    "attach_entity": (
        "Change only the necessary local contact area and keep everything else "
        "exactly unchanged."
    ),
    "modify_attribute": "Change nothing else.",
    "remove_attribute": "Preserve the carrier and change nothing else.",
    "replace_entity": (
        "Match the local scale, geometry, perspective, lighting, shadows, and "
        "occlusion. Preserve everything outside the masked region."
    ),
    "replace_background": (
        "Preserve every foreground entity, including identity, geometry, pose, "
        "and readable text."
    ),
    "modify_environment": (
        "Preserve entity identity, geometry, layout, pose, and readable text."
    ),
    "apply_style": (
        "Preserve scene content, identity, geometry, composition, pose, and "
        "readable text."
    ),
}

MASK_REQUIRED_TOPOLOGIES = frozenset(
    {"remove_entity", "remove_attribute", "replace_entity", "replace_background"}
)


def _removal_fill_prompt(contract: dict[str, Any]) -> str:
    surface = "surrounding background and visible surfaces"
    anchor = contract.get("anchor") or {}
    anchor_type = anchor.get("entity_type")
    target_type = contract["target"].get("entity_type")
    if anchor_type and contract["topology"] == "remove_attribute":
        surface = f"surrounding {anchor_type} surface"
    elif anchor_type and target_type in {"sign", "text", "logo", "writing"}:
        surface = f"surrounding {anchor_type} surface"
    return (
        f"A photorealistic seamless continuation of the {surface}, matching "
        "nearby geometry, texture, lighting, and perspective. Everything "
        "outside the masked regions is exactly unchanged."
    )


def load_contract(payload: str | dict[str, Any]) -> dict[str, Any]:
    """Decode and validate an editable contract from the Gradio interface."""

    if isinstance(payload, str):
        if not payload.strip():
            raise ContractExecutionError("Plan an edit contract before execution.")
        try:
            contract = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractExecutionError(f"Contract JSON is invalid: {exc}") from exc
    elif isinstance(payload, dict):
        contract = payload
    else:
        raise ContractExecutionError("Contract must be a JSON object.")

    if not isinstance(contract, dict):
        raise ContractExecutionError("Contract JSON must contain one object.")
    errors = validate_contract(contract)
    if errors:
        raise ContractExecutionError(
            "Contract failed local validation: " + " | ".join(errors[:3])
        )
    if contract["clarification"]["needed"]:
        questions = " | ".join(contract["clarification"]["questions"])
        raise ContractExecutionError(
            "Contract still needs clarification before execution: " + questions
        )
    return contract


def validate_execution_inputs(
    payload: str | dict[str, Any], target_mask: Any | None = None
) -> dict[str, Any]:
    """Validate runtime inputs that do not belong in a planning contract."""

    contract = load_contract(payload)
    requires_mask = contract["topology"] in MASK_REQUIRED_TOPOLOGIES or (
        contract["topology"] == "apply_style"
        and contract["edit_scope"] == "regional"
    )
    if requires_mask and target_mask is None:
        raise ContractExecutionError(
            f"{contract['topology']} requires a target mask before execution."
        )
    return contract


def _desired_state(contract: dict[str, Any]) -> str:
    desired_state = str(contract.get("desired_state") or "").strip().rstrip(".")
    if not desired_state:
        raise ContractExecutionError(
            f"{contract['topology']} requires a non-empty desired_state."
        )
    return desired_state


def compile_contract_prompt(contract: dict[str, Any]) -> str:
    """Create a concise semantic instruction for the selected editing backend.

    Runtime spatial requirements are checked separately by
    :func:`validate_execution_inputs`.
    """

    contract = load_contract(contract)
    topology = contract["topology"]
    if topology in {"remove_entity", "remove_attribute"}:
        # Naming the removed object often acts as a positive generation token.
        # The spatial mask identifies it; the diffusion prompt describes only
        # the desired visible end state.
        return _removal_fill_prompt(contract)
    if topology == "replace_entity":
        return (
            f"Create {_desired_state(contract)} within the masked region. "
            f"{TOPOLOGY_SUFFIXES[topology]}"
        )
    if topology == "replace_background":
        return (
            f"Replace only the masked background with {_desired_state(contract)}. "
            f"{TOPOLOGY_SUFFIXES[topology]}"
        )
    if topology == "modify_environment":
        return (
            f"Change only the scene environment to {_desired_state(contract)}. "
            f"{TOPOLOGY_SUFFIXES[topology]}"
        )
    if topology == "apply_style":
        subject = (
            "the masked region"
            if contract["edit_scope"] == "regional"
            else "the scene"
        )
        spatial_suffix = (
            " Preserve everything outside the masked region."
            if contract["edit_scope"] == "regional"
            else ""
        )
        return (
            f"Render {subject} in {_desired_state(contract)}. "
            f"{TOPOLOGY_SUFFIXES[topology]}{spatial_suffix}"
        )
    instruction = re.split(
        r"\b(?:keep|without changing)\b",
        contract["instruction"],
        maxsplit=1,
        flags=re.I,
    )[0].strip().rstrip(".")
    return f"{instruction}. {TOPOLOGY_SUFFIXES[topology]}"
