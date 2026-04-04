"""Shared text-processing helpers."""

from __future__ import annotations

import re

COMMON_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "about",
    "into",
    "their",
    "there",
    "have",
    "using",
    "used",
    "such",
    "than",
    "then",
    "they",
    "them",
    "were",
    "been",
    "also",
    "can",
    "could",
    "would",
}

MATCHING_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "about",
}


def chunk_text(text: str, chunk_size: int = 2000) -> list[str]:
    """Split long text into manageable chunks."""
    normalized = text.strip()
    if not normalized:
        return []
    return [
        normalized[index:index + chunk_size]
        for index in range(0, len(normalized), chunk_size)
    ]


def compute_input_char_budget(ctx_size: int, max_tokens: int) -> int:
    """Estimate a safe input size for local prompts."""
    available_tokens = max(1024, ctx_size - max_tokens - 1024)
    return max(2000, min(6000, available_tokens * 2))


def preview_text(text: str, *, limit: int = 80) -> str:
    """Return a compact one-line preview."""
    normalized = " ".join(text.strip().split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit - 3]}..."


def tokenize_for_matching(text: str) -> set[str]:
    """Extract simple lowercase keyword tokens for heuristic matching."""
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
        if token not in MATCHING_STOP_WORDS
    }


def truncate_text(text: str, *, limit: int) -> str:
    """Trim arbitrary context to a bounded size."""
    normalized = text.strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}\n\n[Context truncated]"


def extract_keywords(text: str, *, limit: int = 5) -> list[str]:
    """Extract simple frequent keywords from document text."""
    counts: dict[str, int] = {}
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower()):
        if token in COMMON_STOP_WORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [keyword for keyword, _ in ranked[:limit]]


def extract_title(text: str, *, default: str = "Untitled Report") -> str:
    """Extract a best-effort title from the first non-empty line."""
    for line in text.splitlines():
        normalized = line.strip()
        if normalized:
            return normalized[:120]
    return default

