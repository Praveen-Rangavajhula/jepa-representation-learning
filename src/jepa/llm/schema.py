"""Strict schema and validators for grounded LLM commentary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

COMMENTARY_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "selected_candidate_index",
        "anticipation_summary",
        "why_selected",
        "uncertainty_note",
        "baseline_note",
    ],
    "properties": {
        "selected_candidate_index": {"type": "integer"},
        "anticipation_summary": {"type": "string"},
        "why_selected": {"type": "string"},
        "uncertainty_note": {"type": "string"},
        "baseline_note": {"type": "string"},
    },
}


class CommentarySchemaError(ValueError):
    """Raised when the LLM output does not satisfy the commentary schema."""


@dataclass(slots=True)
class ValidatedCommentaryPayload:
    """Normalized JSON payload for grounded commentary."""

    selected_candidate_index: int
    anticipation_summary: str
    why_selected: str
    uncertainty_note: str
    baseline_note: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_candidate_index": self.selected_candidate_index,
            "anticipation_summary": self.anticipation_summary,
            "why_selected": self.why_selected,
            "uncertainty_note": self.uncertainty_note,
            "baseline_note": self.baseline_note,
        }


def validate_commentary_payload(
    payload: Any,
    *,
    expected_selected_index: int | None = None,
    valid_candidate_indices: Sequence[int] | None = None,
) -> ValidatedCommentaryPayload:
    """Validate and normalize an LLM commentary payload."""

    normalized = _validate_schema(payload, COMMENTARY_JSON_SCHEMA, path="$")

    selected_index = int(normalized["selected_candidate_index"])
    if expected_selected_index is not None and selected_index != int(expected_selected_index):
        raise CommentarySchemaError(
            "LLM commentary selected a different candidate than the scorer: "
            f"{selected_index} != {int(expected_selected_index)}"
        )

    if valid_candidate_indices is not None:
        allowed = {int(index) for index in valid_candidate_indices}
        if selected_index not in allowed:
            raise CommentarySchemaError(
                f"LLM commentary selected candidate {selected_index}, which is not in the candidate set."
            )
        _validate_candidate_mentions(
            normalized,
            allowed_indices=allowed,
        )

    return ValidatedCommentaryPayload(
        selected_candidate_index=selected_index,
        anticipation_summary=str(normalized["anticipation_summary"]).strip(),
        why_selected=str(normalized["why_selected"]).strip(),
        uncertainty_note=str(normalized["uncertainty_note"]).strip(),
        baseline_note=str(normalized["baseline_note"]).strip(),
    )


def _validate_schema(value: Any, schema: Mapping[str, Any], *, path: str) -> Any:
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, Mapping):
            raise CommentarySchemaError(f"{path} must be an object.")

        required = list(schema.get("required", []))
        for key in required:
            if key not in value:
                raise CommentarySchemaError(f"{path} is missing required property {key!r}.")

        properties = dict(schema.get("properties", {}))
        additional_properties = bool(schema.get("additionalProperties", True))

        validated: Dict[str, Any] = {}
        for key, item in value.items():
            if key in properties:
                validated[key] = _validate_schema(item, properties[key], path=f"{path}.{key}")
            elif additional_properties:
                validated[key] = item
            else:
                raise CommentarySchemaError(f"{path} contains unsupported property {key!r}.")

        return validated

    if schema_type == "array":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise CommentarySchemaError(f"{path} must be an array.")
        items_schema = schema.get("items", {})
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < int(min_items):
            raise CommentarySchemaError(f"{path} must contain at least {int(min_items)} items.")
        if max_items is not None and len(value) > int(max_items):
            raise CommentarySchemaError(f"{path} must contain at most {int(max_items)} items.")
        return [
            _validate_schema(item, items_schema, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if schema_type == "string":
        if not isinstance(value, str):
            raise CommentarySchemaError(f"{path} must be a string.")
        enum = schema.get("enum")
        if enum is not None and value not in enum:
            raise CommentarySchemaError(f"{path} must be one of {list(enum)!r}.")
        return value

    if schema_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise CommentarySchemaError(f"{path} must be an integer.")
        return value

    if schema_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise CommentarySchemaError(f"{path} must be a number.")
        return float(value)

    if schema_type == "boolean":
        if not isinstance(value, bool):
            raise CommentarySchemaError(f"{path} must be a boolean.")
        return value

    raise CommentarySchemaError(f"Unsupported schema type at {path}: {schema_type!r}")


def _validate_candidate_mentions(
    payload: Mapping[str, Any],
    *,
    allowed_indices: Iterable[int],
) -> None:
    allowed = {int(index) for index in allowed_indices}
    candidate_pattern = re.compile(r"\bcandidate\s+(\d+)\b", re.IGNORECASE)
    texts = [
        str(payload.get("anticipation_summary", "")),
        str(payload.get("why_selected", "")),
        str(payload.get("uncertainty_note", "")),
        str(payload.get("baseline_note", "")),
    ]
    for text in texts:
        for match in candidate_pattern.finditer(text):
            index = int(match.group(1))
            if index not in allowed:
                raise CommentarySchemaError(
                    f"LLM commentary mentions candidate {index}, which is outside the candidate set."
                )
