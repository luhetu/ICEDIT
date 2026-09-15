"""Optional VLM-backed parser for the Edit Topology Contract demo.

The rule parser creates a conservative, schema-valid draft.  A single
multimodal model call may then ground and correct that draft using the source
image.  The returned contract is never trusted implicitly: it must pass the
same local JSON Schema validator used by the hand-authored examples.
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any

from PIL import Image

from .parser import parse_instruction, validate_contract


DEFAULT_MODELS = {
    "openai": os.environ.get("OPENAI_PLANNER_MODEL") or "gpt-5.6-luna",
    "gemini": os.environ.get("GEMINI_PLANNER_MODEL") or "gemini-3.5-flash",
}


class ModelContractError(RuntimeError):
    """Raised when a provider response cannot be used as a valid contract."""


IMMUTABLE_DRAFT_FIELDS = (
    "schema_version",
    "contract_id",
    "instruction",
    "topology",
    "operation",
    "edit_scope",
    "referring_constraints",
    "source_state",
    "desired_state",
    "replacement",
)

IMMUTABLE_TARGET_FIELDS = (
    "kind",
    "name",
    "entity_type",
    "attribute_name",
    "count",
    "existing",
)
IMMUTABLE_ANCHOR_FIELDS = ("entity_type", "part", "relation")


def _image_data_url(image: Image.Image, max_side: int = 1536) -> str:
    image = image.convert("RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _planner_prompt(instruction: str, draft: dict[str, Any]) -> str:
    return f"""You are the visual grounding stage of a minimal-change image editor.
Return exactly one JSON object: a complete Edit Topology Contract. Start from
the schema-valid draft below and change a field only when the instruction or
visible image provides evidence.

Rules:
- Encode the smallest edit that satisfies the instruction.
- Separate target, dependent, context, and protected regions.
- Preserve identity, layout, color, lighting, and texture outside the required
  edit footprint unless the instruction explicitly changes them.
- Do not claim a mask exists: keep mask_status as pending or not_requested and
  mask_uri null/omitted.
- If a target, anchor, count, or placement is uncertain, record an ambiguity
  and a clarification question instead of guessing.
- Keep schema_version, contract_id, instruction, topology, operation, edit_scope,
  referring_constraints, source_state, desired_state, replacement, target
  semantics/count, anchor semantics, and relation predicates unchanged. These
  fields encode the rule parser's language decision. You may ground instance_ref,
  regions, ambiguities, and provenance, but must not rewrite the requested edit.
- Do not add unknown fields. Output JSON only, without Markdown.
- Add image-grounded claims to provenance with source set to image and concise
  visible evidence. Do not say you observed anything that is not visible.

User instruction:
{instruction}

Schema-valid draft JSON:
{json.dumps(draft, ensure_ascii=False, indent=2)}
"""


def _decode_contract(payload: str) -> dict[str, Any]:
    payload = payload.strip()
    if payload.startswith("```"):
        lines = payload.splitlines()
        payload = "\n".join(lines[1:-1]).strip()
    try:
        contract = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ModelContractError(f"provider returned invalid JSON: {exc}") from exc
    if not isinstance(contract, dict):
        raise ModelContractError("provider returned JSON that is not an object")
    return contract


def _validate_model_contract(
    contract: dict[str, Any], draft: dict[str, Any]
) -> dict[str, Any]:
    for field in IMMUTABLE_DRAFT_FIELDS:
        changed_presence = (field in contract) != (field in draft)
        if changed_presence or contract.get(field) != draft.get(field):
            raise ModelContractError(
                f"provider changed protected field {field!r}: "
                f"expected {draft.get(field)!r}, got {contract.get(field)!r}"
            )
    target_fields = IMMUTABLE_TARGET_FIELDS + (
        ("instance_ref",) if not draft.get("target", {}).get("existing") else ()
    )
    for container, fields in (
        ("target", target_fields),
        ("anchor", IMMUTABLE_ANCHOR_FIELDS),
    ):
        expected = draft.get(container) or {}
        actual = contract.get(container) or {}
        for field in fields:
            changed_presence = (field in actual) != (field in expected)
            if changed_presence or actual.get(field) != expected.get(field):
                raise ModelContractError(
                    f"provider changed protected field {container}.{field}: "
                    f"expected {expected.get(field)!r}, got {actual.get(field)!r}"
                )
    expected_relations = [
        (item.get("predicate"), item.get("required", True))
        for item in draft.get("required_relations", [])
    ]
    actual_relations = [
        (item.get("predicate"), item.get("required", True))
        for item in contract.get("required_relations", [])
    ]
    if actual_relations != expected_relations:
        raise ModelContractError(
            "provider changed protected relation predicates or cardinality"
        )
    errors = validate_contract(contract)
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:12])
        raise ModelContractError(f"provider contract failed local validation:\n{preview}")
    return contract


def _parse_openai(
    image: Image.Image, prompt: str, model: str, client: Any | None = None
) -> str:
    if not os.environ.get("OPENAI_API_KEY") and client is None:
        raise ModelContractError("OPENAI_API_KEY is not set")
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelContractError("install the optional 'openai' package") from exc
        client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(image),
                        "detail": "high",
                    },
                ],
            }
        ],
        text={"format": {"type": "json_object"}},
        reasoning={"effort": "low"},
    )
    if not getattr(response, "output_text", None):
        raise ModelContractError("OpenAI returned no text output")
    return response.output_text


def _parse_gemini(
    image: Image.Image, prompt: str, model: str, client: Any | None = None
) -> str:
    if not os.environ.get("GEMINI_API_KEY") and client is None:
        raise ModelContractError("GEMINI_API_KEY is not set")
    if client is None:
        try:
            from google import genai
        except ImportError as exc:
            raise ModelContractError("install the optional 'google-genai' package") from exc
        client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=[prompt, image.convert("RGB")],
        config={"response_mime_type": "application/json"},
    )
    if not getattr(response, "text", None):
        raise ModelContractError("Gemini returned no text output")
    return response.text


def parse_with_model(
    image: Image.Image,
    instruction: str,
    provider: str,
    model: str | None = None,
    client: Any | None = None,
    task_hint: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Create and validate one contract, using at most one provider call."""

    provider = provider.strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"unknown provider {provider!r}; choose openai or gemini")
    if image is None:
        raise ValueError("an image is required for VLM planning")
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("an edit instruction is required")

    draft = parse_instruction(instruction, task_hint=task_hint)
    prompt = _planner_prompt(instruction, draft)
    selected_model = (model or "").strip() or DEFAULT_MODELS[provider]
    if provider == "openai":
        payload = _parse_openai(image, prompt, selected_model, client=client)
    else:
        payload = _parse_gemini(image, prompt, selected_model, client=client)
    contract = _validate_model_contract(_decode_contract(payload), draft)
    return contract, selected_model
