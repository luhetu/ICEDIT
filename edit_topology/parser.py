"""Rule-based trial parser for Edit Topology Contracts.

This is intentionally conservative: it parses the edit instruction, leaves
image-dependent grounding unresolved, and emits a schema-shaped contract that a
later VLM/LLM parser can replace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.2.0"
ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "schema" / "edit_topology_contract.schema.json"

ADD_WORDS = {"add", "insert", "put", "place"}
REMOVE_WORDS = {"remove", "erase", "delete"}
MODIFY_WORDS = {
    "apply",
    "change",
    "convert",
    "make",
    "restyle",
    "set",
    "stylize",
    "transform",
    "turn",
}
REPLACE_WORDS = {"exchange", "replace", "swap"}
UNSUPPORTED_EDIT_WORDS = {
    "crop",
    "detach",
    "duplicate",
    "move",
    "relocate",
    "reposition",
    "resize",
    "rotate",
}
COMMAND_WORDS = (
    ADD_WORDS | REMOVE_WORDS | MODIFY_WORDS | REPLACE_WORDS | UNSUPPORTED_EDIT_WORDS
)

DATASET_TASK_ALIASES = {
    "addition": "addition",
    "object_addition": "addition",
    "removal": "removal",
    "object_removal": "removal",
    "attribute_modification": "attribute_modification",
    "attribute_change": "attribute_modification",
    "swap": "swap",
    "swapping": "swap",
    "environment": "env",
    "environment_change": "env",
    "env": "env",
    "style": "style",
    "style_transfer": "style",
}

NUMBER_WORDS = {
    "one": 1,
    "a": 1,
    "an": 1,
    "single": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

PERSON_WORDS = {
    "man": "person",
    "men": "person",
    "woman": "person",
    "women": "person",
    "boy": "person",
    "girl": "person",
    "person": "person",
    "people": "person",
}

ENTITY_WORDS = {
    "aircraft",
    "artwork",
    "bag",
    "baby",
    "background",
    "bird",
    "bottle",
    "building",
    "cage",
    "cap",
    "car",
    "church",
    "cup",
    "desk",
    "glasses",
    "hat",
    "key",
    "landscape",
    "logo",
    "man",
    "mug",
    "mountain",
    "paper",
    "person",
    "plate",
    "plaque",
    "rabbit",
    "rock",
    "shoe",
    "shirt",
    "sign",
    "sky",
    "stone",
    "sword",
    "table",
    "terrain",
    "tree",
    "umbrella",
    "woman",
}

ATTRIBUTE_PATTERNS = [
    (re.compile(r"\b(?:(?:long|short|curly|straight|blond|black|brown|white|red|blue|green)\s+)?hair\b"), "person", "hair", "hair"),
    (re.compile(r"\bhandwriting\b"), "paper", "handwriting", "surface.handwriting"),
    (re.compile(r"\b(text|lettering|writing)\b"), "surface", "text", "surface.text"),
]

COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}
ENVIRONMENT_WORDS = {
    "day",
    "dawn",
    "dusk",
    "fog",
    "foggy",
    "golden-hour",
    "night",
    "rain",
    "rainy",
    "season",
    "snow",
    "snowy",
    "sunrise",
    "sunset",
    "weather",
}
STYLE_PATTERN = re.compile(
    r"\b(?:anime|cartoon|cinematic|cubist|illustration|oil painting|painting|pencil sketch|"
    r"photorealistic|sketch|style|watercolor)\b",
    flags=re.I,
)

ATTACH_DEFAULTS = {
    "hat": ("person", "head", "worn_on"),
    "cap": ("person", "head", "worn_on"),
    "sword": ("person", "hand", "held_by"),
    "bag": ("person", None, "carried_by"),
    "glasses": ("person", "face", "worn_on"),
}

SUPPORTED_BY_WORDS = {"table", "desk", "bench", "plate", "shelf", "ground", "floor"}
DEICTIC_ANCHORS = {"it", "this", "that", "this object", "that object"}
STOPWORDS = {
    "the",
    "a",
    "an",
    "one",
    "two",
    "three",
    "four",
    "five",
    "all",
    "exactly",
    "visible",
    "selected",
    "left",
    "right",
    "foreground",
    "background",
    "empty",
    "part",
    "of",
    "in",
    "on",
    "to",
    "from",
}

_CLAUSE_START = (
    r"(?:add|insert|put|place|remove|erase|delete|apply|change|convert|exchange|"
    r"make|restyle|set|stylize|swap|transform|turn|replace|crop|detach|duplicate|"
    r"move|relocate|reposition|resize|rotate|"
    r"keep|leave|preserve|preserving|do\s+not|don't|never|without\s+changing)"
)
_CLAUSE_BOUNDARY_RE = re.compile(
    rf"(?:[.;!?]+\s*|"
    rf",\s*(?:(?:and\s+then|and|but|then|while)\s+)?(?={_CLAUSE_START}\b)|"
    rf"\s+(?:and\s+then|and|but|then|while)\s+(?={_CLAUSE_START}\b)|"
    rf"\s+(?=without\s+changing\b))",
    flags=re.I,
)


def _slug(text: str, max_length: int = 56) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug) or "edit"
    return slug[:max_length].strip("_")


def _contract_id(instruction: str) -> str:
    digest = hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:8]
    return f"trial_{_slug(instruction, 45)}_{digest}"[:80]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9-]+", text.lower())


def _first_verb(tokens: list[str]) -> str | None:
    for token in tokens:
        if token in ADD_WORDS:
            return "add"
        if token in REMOVE_WORDS:
            return "remove"
        if token in MODIFY_WORDS:
            return "modify"
        if token in REPLACE_WORDS:
            return "replace"
    return None


def _first_unsupported_verb(tokens: list[str]) -> str | None:
    for token in tokens:
        if token in UNSUPPORTED_EDIT_WORDS:
            return token
    return None


def normalize_dataset_task(task: str | None) -> str:
    """Normalize an OmniEdit task label without guessing unknown categories."""
    if not task:
        return "unknown"
    normalized = re.sub(r"[^a-z0-9]+", "_", task.strip().lower()).strip("_")
    return DATASET_TASK_ALIASES.get(normalized, "unknown")


def _split_instruction_clauses(instruction: str) -> list[str]:
    """Split only at punctuation or a connector that starts a new command."""
    return [
        re.sub(r"\s+", " ", clause).strip(" ,")
        for clause in _CLAUSE_BOUNDARY_RE.split(instruction)
        if clause.strip(" ,")
    ]


def _protection_subject(clause: str) -> str | None:
    """Return the protected subject for an imperative or negated edit clause."""
    clause = re.sub(r"^(?:please\s+)?", "", clause.strip(), flags=re.I)
    prefixes = (
        r"keep\s+",
        r"leave\s+",
        r"preserve\s+",
        r"preserving\s+",
        r"without\s+(?:changing|modifying|altering|editing)\s+",
        r"(?:do\s+not|don't|never)\s+"
        r"(?:change|modify|alter|edit|remove|erase|delete|add|move|replace|swap)\s+",
    )
    for prefix in prefixes:
        match = re.match(prefix, clause, flags=re.I)
        if not match:
            continue
        subject = clause[match.end():]
        subject = re.sub(
            r"\s+(?:unchanged|untouched|intact|unmodified|as\s+is)$",
            "",
            subject,
            flags=re.I,
        )
        subject = re.sub(r"\s+", " ", subject).strip(" ,.!")
        return subject or None
    return None


def _instruction_parts(
    instruction: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return positive edits, protected subjects, unsupported, and other clauses."""
    edits: list[str] = []
    protections: list[str] = []
    unsupported: list[str] = []
    other: list[str] = []

    for clause in _split_instruction_clauses(instruction):
        protected = _protection_subject(clause)
        if protected:
            protections.append(protected)
            continue

        tokens = _tokens(clause)
        if _first_verb(tokens):
            edits.append(clause)
        if _first_unsupported_verb(tokens):
            unsupported.append(clause)
        if not _first_verb(tokens) and not _first_unsupported_verb(tokens):
            other.append(clause)

    for clause in edits:
        exception = re.search(r"\bexcept(?:\s+for)?\s+(.+)$", clause, flags=re.I)
        if exception:
            protections.append(
                exception.group(1).strip(" ,.!")
            )

    deduplicated = []
    seen = set()
    for protected in protections:
        key = protected.casefold()
        if key not in seen:
            deduplicated.append(protected)
            seen.add(key)
    return edits, deduplicated, unsupported, other


def _replace_parts(
    instruction: str, task_hint: str = "unknown"
) -> tuple[str, str] | None:
    """Extract source and replacement phrases from a replacement instruction."""
    command_pattern = "|".join(
        re.escape(word)
        for word in sorted(REPLACE_WORDS | MODIFY_WORDS, key=len, reverse=True)
    )
    match = re.search(rf"\b(?:{command_pattern})\b", instruction, flags=re.I)
    body = instruction[match.end():].strip(" ,.") if match else instruction.strip(" ,.")
    verb = _first_verb(_tokens(instruction))
    separators = r"\b(?:with|for|by)\b"
    if verb == "modify":
        separators = r"\binto\b"
    if task_hint == "swap":
        separators = r"\b(?:with|for|by|to|into)\b"
    parts = re.split(separators, body, maxsplit=1, flags=re.I)
    if len(parts) != 2 or not all(part.strip(" ,.") for part in parts):
        return None
    return parts[0].strip(" ,."), parts[1].strip(" ,.")


def _target_segment(instruction: str, task_hint: str = "unknown") -> str:
    """Return the command's target span before a placement or carrier phrase."""
    if _first_verb(_tokens(instruction)) == "replace" or task_hint == "swap":
        replacement = _replace_parts(instruction, task_hint)
        if replacement:
            return replacement[0]
    command_pattern = "|".join(
        re.escape(word) for word in sorted(COMMAND_WORDS, key=len, reverse=True)
    )
    match = re.search(rf"\b(?:{command_pattern})\b", instruction, flags=re.I)
    segment = instruction[match.end():] if match else instruction
    return re.split(
        r"\b(?:to|on|onto|from|in|into|at|beside|behind|near)\b",
        segment,
        maxsplit=1,
        flags=re.I,
    )[0].strip()


def _known_entity_types(phrase: str) -> set[str]:
    known = set()
    for token in _tokens(phrase):
        if token in PERSON_WORDS:
            known.add(PERSON_WORDS[token])
        elif token == "skies":
            known.add("sky")
        elif token in ENTITY_WORDS:
            known.add(token)
        elif token.endswith("s") and token[:-1] in ENTITY_WORDS:
            known.add(token[:-1])
    attribute = _attribute_match(phrase)
    if attribute:
        known.add(attribute[2])
    return known


def _has_coordinated_targets(
    instruction: str, task_hint: str = "unknown"
) -> bool:
    """Detect two explicit object heads that one atomic contract cannot represent."""
    replacement = _replace_parts(instruction, task_hint)
    target_description = replacement[0] if replacement else _command_body(instruction)
    parts = re.split(r"\band\b", target_description, flags=re.I)
    return sum(bool(_known_entity_types(part)) for part in parts) > 1


def _count(tokens: list[str], plural_hint: bool) -> dict[str, Any]:
    if "all" in tokens:
        return {"value": None, "quantifier": "all", "exact": False}
    if "some" in tokens or "several" in tokens:
        return {"value": None, "quantifier": "unspecified", "exact": False}
    for token in tokens:
        if token.isdigit():
            return {"value": int(token), "quantifier": "exact", "exact": True}
        if token in NUMBER_WORDS:
            return {"value": NUMBER_WORDS[token], "quantifier": "exact", "exact": True}
    if plural_hint:
        return {"value": None, "quantifier": "all", "exact": False}
    return {"value": 1, "quantifier": "exact", "exact": True}


def _entity_type(phrase: str) -> str:
    phrase_tokens = _tokens(phrase)
    for token in phrase_tokens:
        if token in PERSON_WORDS:
            return PERSON_WORDS[token]
    for token in phrase_tokens:
        if token == "skies":
            return "sky"
        singular = token[:-1] if token.endswith("s") and token[:-1] in ENTITY_WORDS else token
        if singular in ENTITY_WORDS:
            return singular
    for token in reversed(phrase_tokens):
        if token not in STOPWORDS:
            return token[:-1] if token.endswith("s") and token[:-1] in ENTITY_WORDS else token
    return "object"


def _clean_phrase(phrase: str) -> str:
    command_pattern = "|".join(
        re.escape(word) for word in sorted(COMMAND_WORDS, key=len, reverse=True)
    )
    phrase = re.sub(rf"\b(?:{command_pattern})\b", "", phrase, flags=re.I)
    phrase = re.split(r"\b(to|on|onto|from|in|into|at|beside|behind|near)\b", phrase, maxsplit=1, flags=re.I)[0]
    phrase = re.sub(
        r"\b(exactly|all|only|some|several|visible|selected)\b",
        "",
        phrase,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", phrase).strip(" .")


def _target_phrase(instruction: str) -> str:
    cleaned = _clean_phrase(instruction)
    cleaned = re.sub(r"^(a|an|the|one|two|three|four|five)\s+", "", cleaned, flags=re.I)
    words = _tokens(cleaned)
    for index, token in enumerate(words):
        singular = token[:-1] if token.endswith("s") and token[:-1] in ENTITY_WORDS else token
        if singular in PERSON_WORDS or singular in ENTITY_WORDS:
            return " ".join(words[: index + 1]) or singular
    return cleaned or "object"


def _anchor_from_instruction(instruction: str, target_entity: str) -> dict[str, Any] | None:
    match = re.search(r"\b(?:to|on|onto|from)\s+(?:the\s+|a\s+|an\s+)?(?P<anchor>[a-z0-9' -]+?)(?:[.?!]|$)", instruction.lower())
    if not match:
        return None
    anchor_phrase = match.group("anchor").strip()
    if anchor_phrase in DEICTIC_ANCHORS:
        return {
            "entity_type": "object",
            "instance_ref": None,
            "part": None,
            "relation": "related_to",
        }
    anchor_type = _entity_type(anchor_phrase)
    part = None
    relation = "related_to"
    if anchor_type in SUPPORTED_BY_WORDS | {"stone", "rock", "pot"}:
        part = "surface"
        relation = "supported_by"
    elif target_entity in ATTACH_DEFAULTS:
        default_anchor, part, relation = ATTACH_DEFAULTS[target_entity]
        anchor_type = anchor_type if anchor_type != "object" else default_anchor
    elif "hand" in anchor_phrase:
        part = "hand"
        relation = "held_by"
        anchor_type = "person"
    return {
        "entity_type": anchor_type,
        "instance_ref": _slug(anchor_phrase),
        "part": part,
        "relation": relation,
    }


def _attribute_match_details(
    instruction: str,
) -> tuple[str, str, str, tuple[int, int]] | None:
    for pattern, entity_type, name, attribute_name in ATTRIBUTE_PATTERNS:
        match = pattern.search(instruction.lower())
        if match:
            if instruction.lower()[match.end():].startswith("-based"):
                continue
            return entity_type, match.group(0), attribute_name, match.span()
    return None


def _attribute_match(instruction: str) -> tuple[str, str, str] | None:
    details = _attribute_match_details(instruction)
    return details[:3] if details else None


def _scoped_attribute_match(
    instruction: str, operation: str, task_hint: str = "unknown"
) -> tuple[str, str, str] | None:
    """Match an attribute only when it is the edit target, not a descriptor."""
    segment = _target_segment(instruction, task_hint)
    details = _attribute_match_details(segment)
    if not details:
        return None

    entity_type, name, attribute_name, (start, _) = details
    prefix = segment[:start].lower()
    descriptor_marker = re.search(
        r"\b(?:bearing|featuring|having|holding|showing|wearing|with|whose)\b",
        prefix,
    )
    entity_before_attribute = bool(_known_entity_types(prefix))
    if operation in {"add", "remove"} and (
        descriptor_marker or entity_before_attribute
    ):
        return None
    return entity_type, name, attribute_name


def _desired_state(
    instruction: str,
    operation: str,
    task_hint: str = "unknown",
    attribute: tuple[str, str, str] | None = None,
) -> str:
    if operation == "replace" or task_hint == "swap":
        replacement = _replace_parts(instruction, task_hint)
        if replacement:
            return replacement[1]

    if operation == "add" and attribute:
        return _target_segment(instruction, task_hint).strip(" ,.")

    match = re.search(r"\b(?:to|into|as)\s+(.+?)(?:[.!]|$)", instruction, re.I)
    if match:
        return match.group(1).strip(" ,.")

    colors = [token for token in _tokens(instruction) if token in COLOR_WORDS]
    if colors and operation == "modify":
        return colors[-1]

    command_pattern = "|".join(
        re.escape(word) for word in sorted(COMMAND_WORDS, key=len, reverse=True)
    )
    state = re.sub(rf"^.*?\b(?:{command_pattern})\b", "", instruction, flags=re.I)
    state = re.split(
        r"\b(?:keep|leave|without\s+changing)\b", state, maxsplit=1, flags=re.I
    )[0]
    return re.sub(r"\s+", " ", state).strip(" ,.") or instruction.strip(" ,.")


def _inferred_attribute(
    target_name: str, entity_type: str, desired_state: str
) -> tuple[str, str, str]:
    desired_tokens = set(_tokens(desired_state))
    if desired_tokens & COLOR_WORDS:
        return entity_type, f"{target_name} color", "appearance.color"
    if desired_tokens & {"large", "larger", "small", "smaller", "tall", "short"}:
        return entity_type, f"{target_name} size", "geometry.size"
    if desired_tokens & {"glass", "metal", "stone", "wood", "wooden"}:
        return entity_type, f"{target_name} material", "appearance.material"
    return entity_type, target_name, "appearance"


def _global_topology(
    instruction: str, operation: str, task_hint: str
) -> str | None:
    if operation != "modify":
        return None
    lowered = instruction.lower()
    if re.search(r"\b(?:stylize|restyle)\b", lowered) or (
        STYLE_PATTERN.search(lowered)
        and re.search(r"\b(?:image|photo|scene|style|look)\b", lowered)
    ):
        return "apply_style"
    if set(_tokens(lowered)) & ENVIRONMENT_WORDS and re.search(
        r"\b(?:atmosphere|environment|image|lighting|scene|sky|weather)\b",
        lowered,
    ):
        return "modify_environment"
    local_entities = _known_entity_types(_target_segment(instruction, task_hint)) - {
        "background",
        "sky",
    }
    if not local_entities and task_hint == "style":
        return "apply_style"
    if not local_entities and task_hint == "env":
        return "modify_environment"
    return None


def _referring_constraints(
    instruction: str, scope: str = "target"
) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    lowered = instruction.lower()
    for position in ("left", "right", "foreground", "background"):
        if re.search(rf"\b{position}\b", lowered):
            constraints.append(
                {
                    "field": "position",
                    "value": position,
                    "source": "instruction",
                    "scope": scope,
                }
            )
    if re.search(r"\b(?:mid-?ground|middle\s+ground)\b", lowered):
        constraints.append(
            {
                "field": "position",
                "value": "middle_ground",
                "source": "instruction",
                "scope": scope,
            }
        )
    for color in sorted(COLOR_WORDS):
        if re.search(rf"\b{color}\b", lowered):
            constraints.append(
                {
                    "field": "appearance.color",
                    "value": color,
                    "source": "instruction",
                    "scope": scope,
                }
            )
    for material in ("antique", "ceramic", "glass", "metal", "stone", "wood", "wooden"):
        if re.search(rf"\b{material}\b", lowered):
            constraints.append(
                {
                    "field": "appearance.material",
                    "value": material,
                    "source": "instruction",
                    "scope": scope,
                }
            )
    if "shirt" in lowered:
        constraints.append({"field": "clothing", "value": "shirt", "source": "instruction", "scope": scope})
    return constraints


def _command_body(instruction: str) -> str:
    command_pattern = "|".join(
        re.escape(word) for word in sorted(COMMAND_WORDS, key=len, reverse=True)
    )
    match = re.search(rf"\b(?:{command_pattern})\b", instruction, flags=re.I)
    body = instruction[match.end():] if match else instruction
    return re.split(r"\bexcept(?:\s+for)?\b", body, maxsplit=1, flags=re.I)[0].strip(" ,.")


def _selector_spans(phrase: str) -> tuple[str, str | None, str | None]:
    """Split a target description from a spatial/carrier anchor description."""
    relation = re.search(
        r"\b(?P<relation>in\s+front\s+of|"
        r"in(?=\s+(?:the\s+)?(?:foreground|background|left|right|"
        r"mid-?ground|middle\s+ground)\b)|"
        r"next\s+to|beside|behind|near|onto|on|to|from|at)\b",
        phrase,
        flags=re.I,
    )
    if not relation:
        return phrase.strip(" ,."), None, None
    return (
        phrase[: relation.start()].strip(" ,."),
        phrase[relation.end():].strip(" ,.") or None,
        re.sub(r"\s+", "_", relation.group("relation").lower()),
    )


def _spatial_anchor(
    phrase: str | None, relation: str | None
) -> dict[str, Any] | None:
    if not phrase or not relation:
        return None
    is_deictic = phrase.lower() in DEICTIC_ANCHORS
    if not _known_entity_types(phrase) and not is_deictic:
        return None
    entity_type = _entity_type(phrase)
    if relation == "in" and entity_type == "background":
        return None
    relation_map = {
        "beside": "beside",
        "behind": "behind",
        "near": "near",
        "next_to": "beside",
        "in_front_of": "in_front_of",
    }
    return {
        "entity_type": entity_type,
        "instance_ref": None
        if is_deictic
        else _slug(phrase),
        "part": None,
        "relation": relation_map.get(relation, "related_to"),
    }


def _deduplicate_constraints(
    constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for constraint in constraints:
        key = (
            constraint["scope"],
            constraint["field"],
            constraint["value"],
            constraint["source"],
        )
        if key not in seen:
            result.append(constraint)
            seen.add(key)
    return result


def _explicit_protections(instruction: str) -> list[str]:
    """Extract user-stated protections without inventing image regions."""
    _, protections, _, _ = _instruction_parts(instruction)
    return protections


def parse_instruction(
    instruction: str,
    source_image: str | None = None,
    task_hint: str | None = None,
) -> dict[str, Any]:
    """Parse one atomic instruction into a conservative v0.2 contract.

    ``task_hint`` may disambiguate an OmniEdit label, but explicit instruction
    semantics always win. Any incomplete or unsupported request remains schema
    valid and is blocked through ``clarification``.
    """
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("instruction must not be empty")

    normalized_task = normalize_dataset_task(task_hint)
    edit_clauses, explicit_protections, unsupported_clauses, other_clauses = (
        _instruction_parts(instruction)
    )
    if edit_clauses:
        edit_clause = edit_clauses[0]
    elif unsupported_clauses:
        edit_clause = unsupported_clauses[0]
    elif explicit_protections:
        edit_clause = explicit_protections[0]
    else:
        edit_clause = other_clauses[0] if other_clauses else instruction

    parsed_operation = _first_verb(_tokens(edit_clause)) if edit_clauses else None
    hinted_operations = {
        "addition": "add",
        "removal": "remove",
        "attribute_modification": "modify",
        "swap": "replace",
        "env": "modify",
        "style": "modify",
    }
    operation = parsed_operation or hinted_operations.get(normalized_task, "modify")
    global_topology = _global_topology(edit_clause, operation, normalized_task)
    replacement_parts = (
        _replace_parts(edit_clause, normalized_task)
        if operation in {"modify", "replace"}
        else None
    )
    replacement_requested = parsed_operation == "replace"
    if operation == "modify" and global_topology is None and replacement_parts:
        source_entities = _known_entity_types(replacement_parts[0])
        desired_entities = _known_entity_types(replacement_parts[1])
        replacement_requested = bool(
            desired_entities
            and source_entities
            and ("into" in edit_clause.lower() or normalized_task == "swap")
        ) or "background" in source_entities
    if parsed_operation is None and normalized_task == "swap":
        replacement_requested = True
    if replacement_requested:
        operation = "replace"

    target: dict[str, Any]
    replacement: dict[str, Any] | None = None
    anchor: dict[str, Any] | None = None
    constraints: list[dict[str, Any]] = []
    desired_state: str | None = None
    source_state: str | None = None

    if global_topology:
        topology = global_topology
        desired_state = _desired_state(edit_clause, "modify", normalized_task)
        if topology == "apply_style":
            style_match = STYLE_PATTERN.search(edit_clause)
            descriptive_style = re.search(
                r"\b(?:look\s+like|into|as)\s+(.+?)(?:[.!]|$)",
                edit_clause,
                flags=re.I,
            )
            if descriptive_style:
                desired_state = descriptive_style.group(1).strip(" ,.")
            elif style_match and style_match.group(0).lower() != "style":
                desired_state = f"{style_match.group(0).lower()} style"
            whole_scene = bool(
                re.search(
                    r"\b(?:entire|whole)\s+(?:image|photo|picture|scene)\b",
                    edit_clause,
                    flags=re.I,
                )
            )
            regional_match = re.search(
                r"\b(?:only\s+)?(?:to|on)\s+(?:the\s+)?(.+?)(?:[.!]|$)",
                edit_clause,
                flags=re.I,
            )
            if regional_match and not whole_scene:
                region_phrase = regional_match.group(1).strip(" ,.")
                region_entity = _entity_type(region_phrase)
                edit_scope = "regional"
                target = {
                    "kind": "attribute",
                    "name": f"{region_phrase} style",
                    "entity_type": region_entity,
                    "attribute_name": "appearance.style",
                    "instance_ref": _slug(region_phrase),
                    "count": {"value": 1, "quantifier": "exact", "exact": True},
                    "existing": True,
                }
                constraints.extend(_referring_constraints(region_phrase, "target"))
            else:
                edit_scope = "global"
                target = {
                    "kind": "attribute",
                    "name": "scene style",
                    "entity_type": "scene",
                    "attribute_name": "scene.style",
                    "instance_ref": "scene",
                    "count": {"value": 1, "quantifier": "exact", "exact": True},
                    "existing": True,
                }
        else:
            edit_scope = "global"
            target = {
                "kind": "attribute",
                "name": "scene environment",
                "entity_type": "scene",
                "attribute_name": "scene.environment",
                "instance_ref": "scene",
                "count": {"value": 1, "quantifier": "exact", "exact": True},
                "existing": True,
            }
    elif operation == "replace":
        source_phrase, result_phrase = replacement_parts or (
            _command_body(edit_clause) or "object",
            "unspecified replacement",
        )
        target_span, anchor_span, relation = _selector_spans(source_phrase)
        target_name = _target_phrase(target_span)
        entity_type = _entity_type(target_name)
        is_background = entity_type == "background" or bool(
            re.search(
                r"\b(?:background|backdrop|landscape|scenery|setting|sky|skies|terrain)\b",
                source_phrase,
                flags=re.I,
            )
        )
        topology = "replace_background" if is_background else "replace_entity"
        edit_scope = "regional" if is_background else "local"
        count = _count(_tokens(target_span), plural_hint=target_name.endswith("s"))
        target = {
            "kind": "entity",
            "name": target_name,
            "entity_type": "background" if is_background else entity_type,
            "instance_ref": None
            if count["quantifier"] == "all"
            else _slug(target_name),
            "count": count,
            "existing": True,
        }
        result_name = result_phrase if is_background else _target_phrase(result_phrase)
        result_entity = "background" if is_background else _entity_type(result_name)
        result_count = (
            {"value": 1, "quantifier": "exact", "exact": True}
            if is_background
            else _count(_tokens(result_phrase), plural_hint=result_name.endswith("s"))
        )
        replacement = {
            "kind": "entity",
            "name": result_name,
            "entity_type": result_entity,
            "instance_ref": f"new_{result_entity}",
            "count": result_count,
            "existing": False,
        }
        desired_state = result_phrase
        source_state = target_name
        anchor = _spatial_anchor(anchor_span, relation)
        constraints.extend(_referring_constraints(target_span, "target"))
        if anchor_span:
            constraints.extend(_referring_constraints(anchor_span, "anchor"))
        constraints.extend(_referring_constraints(result_phrase, "replacement"))
    else:
        requested_operation = operation
        body = _command_body(edit_clause)
        if requested_operation == "modify":
            target_span = re.split(
                r"\b(?:to|into|as)\b", body, maxsplit=1, flags=re.I
            )[0].strip(" ,.")
            anchor_span = None
            relation = None
        else:
            target_span, anchor_span, relation = _selector_spans(body)
        target_name = _target_phrase(target_span)
        entity_type = _entity_type(target_name)
        count = _count(_tokens(target_span), plural_hint=target_name.endswith("s"))
        attribute = _scoped_attribute_match(
            edit_clause, requested_operation, normalized_task
        )
        candidate_anchor = (
            _anchor_from_instruction(edit_clause, entity_type)
            if requested_operation in {"add", "remove"}
            else None
        ) or _spatial_anchor(anchor_span, relation)

        if requested_operation == "add" and attribute:
            topology = "modify_attribute"
            operation = "modify"
        elif requested_operation == "remove" and attribute:
            topology = "remove_attribute"
        elif requested_operation == "modify":
            topology = "modify_attribute"
        elif requested_operation == "add" and candidate_anchor and candidate_anchor.get("relation") == "supported_by":
            topology = "insert_entity"
        elif requested_operation == "add" and (
            candidate_anchor or entity_type in ATTACH_DEFAULTS
        ):
            topology = "attach_entity"
        elif requested_operation == "add":
            topology = "insert_entity"
        else:
            topology = "remove_entity"
        edit_scope = "local"

        if topology in {"modify_attribute", "remove_attribute"}:
            if requested_operation == "modify":
                desired_state = _desired_state(
                    edit_clause, requested_operation, normalized_task, attribute
                )
            elif requested_operation == "add":
                desired_state = _desired_state(
                    edit_clause, requested_operation, normalized_task, attribute
                )
            if attribute:
                attr_entity, attr_name, attribute_name = attribute
                anchor = candidate_anchor or {
                    "entity_type": attr_entity,
                    "instance_ref": None,
                    "part": attribute_name.split(".")[-1],
                    "relation": "attribute_of",
                }
            else:
                desired_state = desired_state or _desired_state(
                    edit_clause, "modify", normalized_task
                )
                attr_entity, attr_name, attribute_name = _inferred_attribute(
                    target_name, entity_type, desired_state
                )
                anchor = {
                    "entity_type": entity_type,
                    "instance_ref": _slug(target_name),
                    "part": attribute_name.split(".")[-1],
                    "relation": "attribute_of",
                }
            anchor["part"] = anchor.get("part") or attribute_name.split(".")[-1]
            anchor["relation"] = "attribute_of"
            target = {
                "kind": "attribute",
                "name": attr_name,
                "entity_type": anchor["entity_type"],
                "attribute_name": attribute_name,
                "instance_ref": anchor["instance_ref"],
                "count": count
                if topology == "remove_attribute"
                else {"value": 1, "quantifier": "exact", "exact": True},
                "existing": True,
            }
            source_state = attr_name if topology == "remove_attribute" else target_name
        else:
            anchor = candidate_anchor
            target = {
                "kind": "entity",
                "name": target_name,
                "entity_type": entity_type,
                "instance_ref": (
                    f"new_{entity_type}"
                    if operation == "add"
                    else None
                    if operation == "remove" and count["quantifier"] == "all"
                    else _slug(target_name)
                ),
                "count": count,
                "existing": operation != "add",
            }
            if topology == "attach_entity" and anchor is None:
                default_anchor, part, relation_name = ATTACH_DEFAULTS.get(
                    entity_type, ("object", None, "attached_to")
                )
                anchor = {
                    "entity_type": default_anchor,
                    "instance_ref": None,
                    "part": part,
                    "relation": relation_name,
                }
            if operation == "remove":
                source_state = target_name

        constraints.extend(_referring_constraints(target_span, "target"))
        if anchor_span:
            if anchor:
                constraints.extend(_referring_constraints(anchor_span, "anchor"))
            elif relation == "in":
                placement = re.match(
                    r"(?:the\s+)?(?:foreground|background|left|right|"
                    r"mid-?ground|middle\s+ground)\b",
                    anchor_span,
                    flags=re.I,
                )
                placement_span = placement.group(0) if placement else anchor_span
                constraints.extend(
                    _referring_constraints(placement_span, "placement")
                )
                if placement:
                    descriptor_span = re.split(
                        r"\bwhile\b",
                        anchor_span[placement.end():],
                        maxsplit=1,
                        flags=re.I,
                    )[0]
                    constraints.extend(
                        _referring_constraints(descriptor_span, "target")
                    )
            else:
                constraints.extend(
                    _referring_constraints(anchor_span, "placement")
                )

    constraints = _deduplicate_constraints(constraints)
    count = target["count"]
    deictic_anchor = re.search(
        r"\b(?:to|on|onto|from|beside|behind|near)\s+(?:the\s+)?"
        r"(it|this|that|this object|that object)\b",
        edit_clause.lower(),
    )

    ambiguities: list[dict[str, Any]] = []
    questions: list[str] = []

    def add_ambiguity(
        field: str, confidence: float, description: str, question: str
    ) -> None:
        ambiguities.append(
            {
                "field": field,
                "status": "unresolved",
                "confidence": confidence,
                "description": description,
                "assumption": None,
            }
        )
        questions.append(question)

    if len(edit_clauses) > 1:
        add_ambiguity(
            "instruction.atomicity",
            1.0,
            "The instruction contains more than one positive edit command.",
            "The v0.2 contract supports one atomic edit at a time. Please split the commands.",
        )
    if unsupported_clauses:
        unsupported_verbs = sorted(
            {
                verb
                for clause in unsupported_clauses
                if (verb := _first_unsupported_verb(_tokens(clause)))
            }
        )
        add_ambiguity(
            "instruction.operation",
            1.0,
            "Unsupported edit operation(s): " + ", ".join(unsupported_verbs),
            "Use one supported add, remove, attribute, replacement, environment, or style edit.",
        )
    if not edit_clauses and not unsupported_clauses:
        add_ambiguity(
            "instruction.operation",
            0.0,
            "No positive supported edit command was found.",
            "What single edit operation should be performed?",
        )
    if operation == "replace" and replacement_parts is None:
        add_ambiguity(
            "replacement",
            0.0,
            "The replacement source and desired result could not both be parsed.",
            "What should replace the selected target?",
        )
    if edit_clauses and topology not in {"modify_environment", "apply_style"} and _has_coordinated_targets(
        edit_clause, normalized_task
    ):
        add_ambiguity(
            "instruction.targets",
            0.9,
            "The instruction names multiple target objects in one edit clause.",
            "Please split the targets or describe them as one unambiguous target group.",
        )
    if count["quantifier"] == "unspecified":
        add_ambiguity(
            "target.count",
            1.0,
            "The requested target count is not exact.",
            f"How many {target['name']} should be edited?",
        )
    if topology == "replace_entity" and count["quantifier"] == "all":
        add_ambiguity(
            "replacement.cardinality",
            0.5,
            "A plural replacement does not state whether mapping is one-to-one.",
            "Should every target be replaced one-for-one with a new object?",
        )
    if (
        operation == "add"
        and anchor
        and re.search(
            r"\b(?:to|on|onto|beside|near)\s+(?:a|an)\s+",
            edit_clause,
            flags=re.I,
        )
    ):
        add_ambiguity(
            "anchor.definiteness",
            0.4,
            "An indefinite anchor may describe another new entity rather than an existing one.",
            "Is the anchor already visible, or should it be added as part of the new content?",
        )

    preflight_blocked = bool(questions)
    has_placement = any(
        item["scope"] == "placement" or item["field"] == "position"
        for item in constraints
    )
    if (
        not preflight_blocked
        and topology == "insert_entity"
        and anchor is None
        and not has_placement
    ):
        add_ambiguity(
            "placement",
            0.25,
            "The parser found no explicit insertion placement.",
            f"Where should the {target['name']} be placed?",
        )
    if not preflight_blocked and deictic_anchor and anchor and anchor["instance_ref"] is None:
        reference = deictic_anchor.group(1)
        add_ambiguity(
            "anchor.instance_ref",
            0.0,
            f"The pronoun '{reference}' does not identify a unique visible anchor.",
            f"Which visible object does '{reference}' refer to?",
        )
    elif (
        not preflight_blocked
        and topology in {"attach_entity", "modify_attribute", "remove_attribute"}
        and anchor
        and anchor["instance_ref"] is None
    ):
        add_ambiguity(
            "anchor.instance_ref",
            0.2,
            "The anchor object must be grounded in the image.",
            f"Which {anchor['entity_type']} is the anchor?",
        )

    relation_subject = target["instance_ref"] or _slug(target["name"])
    required_relations: list[dict[str, Any]] = []
    if anchor:
        object_ref = anchor["instance_ref"] or f"selected_{anchor['entity_type']}"
        if anchor.get("part"):
            object_ref = f"{object_ref}.{anchor['part']}"
        required_relations.append(
            {
                "subject": relation_subject,
                "predicate": anchor.get("relation") or "related_to",
                "object": object_ref,
                "required": True,
            }
        )

    if topology == "modify_environment":
        base_invariants = [
            {
                "subject": "scene entities and text",
                "property": "identity, geometry, layout, and legibility",
                "tolerance": "strict",
            }
        ]
        default_protected = "scene entities, geometry, layout, and text"
        dependent_description = "lighting, shadows, reflections, and atmospheric effects"
        context_description = "whole-scene environmental coherence"
        explicit_protection_property = "identity, geometry, layout, pose, and legibility"
    elif topology == "apply_style":
        base_invariants = [
            {
                "subject": "semantic scene content and text",
                "property": "identity, geometry, layout, and legibility",
                "tolerance": "strict",
            }
        ]
        default_protected = "semantic identities, geometry, layout, and readable text"
        dependent_description = "style-consistent edges, shading, and fine texture"
        context_description = "source composition and semantic structure"
        explicit_protection_property = "identity, geometry, layout, pose, and legibility"
    elif topology == "replace_background":
        base_invariants = [
            {
                "subject": "foreground subjects and objects",
                "property": "identity, geometry, pose, and appearance",
                "tolerance": "strict",
            }
        ]
        default_protected = "all foreground subjects and objects"
        dependent_description = "subject boundaries, contact edges, and scene lighting"
        context_description = "foreground geometry, depth, perspective, and lighting"
        explicit_protection_property = "identity, geometry, pose, appearance, and texture"
    else:
        base_invariants = [
            {
                "subject": "unrelated scene content",
                "property": "identity, layout, and appearance",
                "tolerance": "strict",
            }
        ]
        default_protected = "all other non-target scene content"
        dependent_description = "local occlusion, boundary, contact, and shadow"
        context_description = "nearby geometry, lighting, scale, and source style"
        explicit_protection_property = "identity, layout, appearance, and texture"

    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": _contract_id(instruction),
        "instruction": instruction,
        "topology": topology,
        "operation": operation,
        "edit_scope": edit_scope,
        "target": target,
        "referring_constraints": constraints,
        "required_relations": required_relations,
        "forbidden_outcomes": [
            "change unrelated entities",
            "alter protected content outside the edit footprint",
            "produce a different edit topology than requested",
        ],
        "invariants": [
            *base_invariants,
            *[
                {
                    "subject": protected,
                    "property": explicit_protection_property,
                    "tolerance": "strict",
                }
                for protected in explicit_protections
            ],
        ],
        "region_roles": {
            "target": [
                {
                    "description": target["name"],
                    "entity_ref": target.get("instance_ref"),
                    "mask_status": "pending",
                }
            ],
            "dependent": [
                {"description": dependent_description, "mask_status": "pending"}
            ],
            "context": [
                {"description": context_description, "mask_status": "not_requested"}
            ],
            "protected": [
                *[
                    {"description": protected, "mask_status": "pending"}
                    for protected in explicit_protections
                ],
                {"description": default_protected, "mask_status": "pending"},
            ],
        },
        "ambiguities": ambiguities,
        "clarification": {"needed": bool(questions), "questions": questions},
        "provenance": [
            {
                "claim": f"operation represented as {operation}",
                "source": "annotation_rule",
                "evidence": "supported keyword rule"
                if parsed_operation
                else "task hint or non-executable placeholder",
            },
            {
                "claim": f"topology parsed as {topology}",
                "source": "annotation_rule",
                "evidence": "rule-based v0.2 parser",
            },
        ],
    }
    if source_image:
        contract["source_image"] = source_image
        contract["provenance"].append(
            {
                "claim": "source image path recorded but not visually parsed",
                "source": "default",
                "evidence": source_image,
            }
        )
    if source_state:
        contract["source_state"] = source_state
    if desired_state:
        contract["desired_state"] = desired_state
    if replacement:
        contract["replacement"] = replacement
    if anchor:
        contract["anchor"] = anchor
    return contract


def validate_contract_schema(contract: dict[str, Any]) -> list[str]:
    """Return JSON Schema errors without applying execution semantics."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return ["jsonschema is not installed; install edit_topology/requirements-test.txt to validate"]
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    return [
        f"{list(error.path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(contract), key=lambda error: list(error.path)
        )
    ]


def validate_contract_semantics(contract: dict[str, Any]) -> list[str]:
    """Validate cross-field rules that are clearer in code than JSON Schema."""
    errors: list[str] = []
    topology = contract.get("topology")
    operation = contract.get("operation")
    edit_scope = contract.get("edit_scope")
    expected: dict[str, tuple[str, set[str]]] = {
        "insert_entity": ("add", {"local"}),
        "attach_entity": ("add", {"local"}),
        "modify_attribute": ("modify", {"local"}),
        "remove_entity": ("remove", {"local"}),
        "remove_attribute": ("remove", {"local"}),
        "replace_entity": ("replace", {"local"}),
        "replace_background": ("replace", {"regional"}),
        "modify_environment": ("modify", {"global"}),
        "apply_style": ("modify", {"regional", "global"}),
    }
    if topology in expected:
        expected_operation, expected_scopes = expected[topology]
        if operation != expected_operation:
            errors.append(
                f"semantic: {topology} requires operation={expected_operation}"
            )
        if edit_scope not in expected_scopes:
            errors.append(
                f"semantic: {topology} requires edit_scope in {sorted(expected_scopes)}"
            )

    for field in ("target", "replacement"):
        item = contract.get(field)
        if not isinstance(item, dict):
            continue
        count = item.get("count", {})
        if (
            count.get("quantifier") == "exact"
            and isinstance(count.get("value"), int)
            and count["value"] < 1
        ):
            errors.append(f"semantic: {field}.count.value must be at least 1")

    unresolved = [
        item
        for item in contract.get("ambiguities", [])
        if item.get("status") == "unresolved"
    ]
    clarification = contract.get("clarification", {})
    if unresolved and not clarification.get("needed"):
        errors.append("semantic: unresolved ambiguities require clarification")
    if clarification.get("needed") and not unresolved:
        errors.append("semantic: clarification requires an unresolved ambiguity")
    if not clarification.get("needed") and clarification.get("questions"):
        errors.append("semantic: clarification questions must be empty when not needed")

    anchor = contract.get("anchor")
    replacement = contract.get("replacement")
    for index, constraint in enumerate(contract.get("referring_constraints", [])):
        scope = constraint.get("scope")
        if scope == "anchor" and not anchor:
            errors.append(
                f"semantic: referring_constraints[{index}] scopes an absent anchor"
            )
        if scope == "replacement" and not replacement:
            errors.append(
                f"semantic: referring_constraints[{index}] scopes an absent replacement"
            )

    if topology == "attach_entity" and not contract.get("required_relations"):
        errors.append("semantic: attach_entity requires a positive relation")
    if topology in {"replace_entity", "replace_background"}:
        if not replacement:
            errors.append(f"semantic: {topology} requires replacement")
        if not contract.get("desired_state"):
            errors.append(f"semantic: {topology} requires desired_state")
    if topology in {"modify_attribute", "modify_environment", "apply_style"} and not contract.get(
        "desired_state"
    ):
        errors.append(f"semantic: {topology} requires desired_state")

    for role, regions in contract.get("region_roles", {}).items():
        for index, region in enumerate(regions):
            if region.get("mask_status") == "grounded" and not region.get("mask_uri"):
                errors.append(
                    f"semantic: region_roles.{role}[{index}] is grounded without mask_uri"
                )

    for invariant in contract.get("invariants", []):
        if invariant.get("tolerance") != "strict":
            continue
        changed_properties = (
            f"{invariant.get('subject', '')} {invariant.get('property', '')}"
        ).lower()
        if topology == "modify_environment" and re.search(
            r"\b(?:atmosphere|environment|lighting|shadow|weather)\b",
            changed_properties,
        ):
            errors.append(
                "semantic: environment edits cannot strictly preserve environmental appearance"
            )
        if topology == "apply_style" and re.search(
            r"\b(?:appearance|color|palette|style|texture)\b", changed_properties
        ):
            errors.append(
                "semantic: style edits cannot strictly preserve the changed visual style"
            )
    return errors


def validate_contract(contract: dict[str, Any]) -> list[str]:
    schema_errors = validate_contract_schema(contract)
    if schema_errors:
        return schema_errors
    return validate_contract_semantics(contract)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def _write_review_csv(path: Path, contracts: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "contract_id",
                "source_image",
                "instruction",
                "topology",
                "operation",
                "edit_scope",
                "target",
                "replacement",
                "desired_state",
                "anchor",
                "clarification_needed",
                "questions",
            ],
        )
        writer.writeheader()
        for contract in contracts:
            writer.writerow({
                "contract_id": contract["contract_id"],
                "source_image": contract.get("source_image", ""),
                "instruction": contract["instruction"],
                "topology": contract["topology"],
                "operation": contract["operation"],
                "edit_scope": contract["edit_scope"],
                "target": contract["target"]["name"],
                "replacement": (contract.get("replacement") or {}).get("name", ""),
                "desired_state": contract.get("desired_state", ""),
                "anchor": json.dumps(contract.get("anchor", {}), ensure_ascii=False),
                "clarification_needed": contract["clarification"]["needed"],
                "questions": " | ".join(contract["clarification"]["questions"]),
            })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trial rule-based Edit Topology Contract parser")
    parser.add_argument("--instruction", help="single edit instruction")
    parser.add_argument("--image", help="optional source image path")
    parser.add_argument(
        "--task-hint",
        help="optional OmniEdit task label used only to disambiguate the instruction",
    )
    parser.add_argument("--input-jsonl", type=Path, help="batch JSONL with instruction and optional image/source_image")
    parser.add_argument("--output", type=Path, help="output JSON for single mode, JSONL for batch mode")
    parser.add_argument("--review-csv", type=Path, help="optional batch review CSV")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip schema and semantic validation",
    )
    args = parser.parse_args(argv)

    if not args.instruction and not args.input_jsonl:
        parser.error("provide --instruction or --input-jsonl")

    if args.input_jsonl:
        contracts = []
        for row in _load_jsonl(args.input_jsonl):
            image = row.get("image") or row.get("source_image")
            contracts.append(
                parse_instruction(
                    row["instruction"],
                    image,
                    row.get("task") or row.get("task_hint"),
                )
            )
        if args.output:
            args.output.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in contracts) + "\n", encoding="utf-8")
        else:
            for contract in contracts:
                print(json.dumps(contract, ensure_ascii=False))
        if args.review_csv:
            _write_review_csv(args.review_csv, contracts)
        if not args.no_validate:
            errors = {c["contract_id"]: validate_contract(c) for c in contracts}
            errors = {k: v for k, v in errors.items() if v}
            if errors:
                print(json.dumps(errors, indent=2, ensure_ascii=False), file=sys.stderr)
                return 1
        return 0

    contract = parse_instruction(args.instruction, args.image, args.task_hint)
    if not args.no_validate:
        errors = validate_contract(contract)
        if errors:
            print(json.dumps(errors, indent=2, ensure_ascii=False), file=sys.stderr)
            return 1
    payload = json.dumps(contract, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
