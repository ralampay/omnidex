"""Shared filename and path extraction helpers."""

from __future__ import annotations

import re


def find_path_with_suffix(query: str, suffix: str) -> str | None:
    """Extract the first quoted or unquoted path with the given suffix."""
    escaped_suffix = re.escape(suffix)
    pattern = (
        rf'(?P<path>"[^"]+{escaped_suffix}"|'
        rf"'[^']+{escaped_suffix}'|"
        rf'[^\s"\']+{escaped_suffix})'
    )
    match = re.search(pattern, query, re.IGNORECASE)
    if not match:
        return None
    return match.group("path").strip("\"'")


def extract_output_filename(query: str) -> str | None:
    """Extract an explicit output filename from a user request."""
    patterns = (
        r'(?:save|write|export)\s+(?:the\s+)?(?:output|result|summary|report|answer)?\s*(?:to|as|into)\s+(?P<path>"[^"]+"|\'[^\']+\'|[^\s,;:]+)',
        r'(?:filename|file name|output file)\s*(?:is|=|:)?\s*(?P<path>"[^"]+"|\'[^\']+\'|[^\s,;:]+)',
        r'(?:call|name)\s+(?:the\s+)?(?:output|file)\s+(?P<path>"[^"]+"|\'[^\']+\'|[^\s,;:]+)',
    )
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if not match:
            continue
        candidate = match.group("path").strip("\"'").strip()
        if candidate:
            return candidate
    return None


def has_explicit_output_request(query: str) -> bool:
    """Return whether the user explicitly asked to persist or name output."""
    patterns = (
        r'(?:save|write|export)\s+(?:the\s+)?(?:output|result|summary|report|answer)?\s*(?:to|as|into)\s+(?:"[^"]+"|\'[^\']+\'|[^\s,;:]+)',
        r'(?:filename|file name|output file)\s*(?:is|=|:)?\s*(?:"[^"]+"|\'[^\']+\'|[^\s,;:]+)',
        r'(?:call|name)\s+(?:the\s+)?(?:output|file)\s+(?:"[^"]+"|\'[^\']+\'|[^\s,;:]+)',
    )
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns)
