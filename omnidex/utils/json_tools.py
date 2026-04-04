"""Helpers for parsing JSON-like model output."""

from __future__ import annotations

import json
import re


def strip_code_fences(text: str) -> str:
    """Remove a surrounding fenced code block when present."""
    return re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def load_json_object(text: str) -> dict[str, object] | None:
    """Parse a JSON object from raw model output when possible."""
    cleaned = strip_code_fences(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}")
        if json_start == -1 or json_end == -1 or json_end <= json_start:
            return None
        try:
            payload = json.loads(cleaned[json_start:json_end + 1])
        except json.JSONDecodeError:
            return None

    if isinstance(payload, dict):
        return payload
    return None

