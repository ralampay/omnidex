"""PDF reading tool for local agent workflows."""

from __future__ import annotations

import math
from pathlib import Path
import re

from pypdf import PdfReader

from omnidex.tools.base import BaseTool

HEADER_FOOTER_WINDOW = 3
HEADER_FOOTER_MAX_LENGTH = 140
PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s+)?(?:\d{1,4}|[ivxlcdm]{1,8})(?:\s*(?:/|of|-)\s*(?:\d{1,4}|[ivxlcdm]{1,8}))?$",
    re.IGNORECASE,
)
LIST_ITEM_RE = re.compile(r"^(?:[-*•]|\d+[.)]|[a-z][.)])\s+")


def _normalize_pdf_line(line: str) -> str:
    """Normalize whitespace within a PDF-extracted line."""
    return " ".join(line.replace("\u00a0", " ").split()).strip()


def _is_page_number_line(line: str) -> bool:
    """Detect standalone page-number noise."""
    normalized = _normalize_pdf_line(line)
    return bool(normalized and PAGE_NUMBER_RE.fullmatch(normalized))


def _is_header_footer_candidate(line: str) -> bool:
    """Return whether a line is a plausible repeated header/footer."""
    normalized = _normalize_pdf_line(line)
    if not normalized or len(normalized) > HEADER_FOOTER_MAX_LENGTH:
        return False
    if _is_page_number_line(normalized):
        return True
    alpha_count = sum(character.isalpha() for character in normalized)
    return alpha_count > 0


def _prepare_page_lines(page_text: str) -> list[str]:
    """Normalize one PDF page into stripped lines with blank separators."""
    normalized_lines: list[str] = []
    for raw_line in page_text.splitlines():
        normalized = _normalize_pdf_line(raw_line)
        if normalized:
            normalized_lines.append(normalized)
            continue
        if normalized_lines and normalized_lines[-1] != "":
            normalized_lines.append("")
    while normalized_lines and normalized_lines[0] == "":
        normalized_lines.pop(0)
    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()
    return normalized_lines


def _collect_repeated_margin_lines(pages: list[list[str]]) -> set[str]:
    """Detect repeated headers and footers appearing on many pages."""
    non_empty_pages = [page for page in pages if page]
    if len(non_empty_pages) < 2:
        return set()

    threshold = max(2, math.ceil(len(non_empty_pages) * 0.5))
    counts: dict[str, int] = {}
    for page in non_empty_pages:
        meaningful = [line for line in page if line]
        candidates = meaningful[:HEADER_FOOTER_WINDOW] + meaningful[-HEADER_FOOTER_WINDOW:]
        seen_on_page: set[str] = set()
        for candidate in candidates:
            if not _is_header_footer_candidate(candidate) or candidate in seen_on_page:
                continue
            counts[candidate] = counts.get(candidate, 0) + 1
            seen_on_page.add(candidate)
    return {line for line, count in counts.items() if count >= threshold}


def _trim_page_margins(lines: list[str], repeated_margin_lines: set[str]) -> list[str]:
    """Remove repeated headers, footers, and page-number-only lines."""
    trimmed = list(lines)
    while trimmed and (
        trimmed[0] == ""
        or trimmed[0] in repeated_margin_lines
        or _is_page_number_line(trimmed[0])
    ):
        trimmed.pop(0)
    while trimmed and (
        trimmed[-1] == ""
        or trimmed[-1] in repeated_margin_lines
        or _is_page_number_line(trimmed[-1])
    ):
        trimmed.pop()
    return [
        line
        for line in trimmed
        if line == "" or not _is_page_number_line(line)
    ]


def _join_paragraph_lines(lines: list[str]) -> str:
    """Reflow line-broken PDF text into a cleaner paragraph."""
    if not lines:
        return ""

    merged = lines[0]
    for line in lines[1:]:
        if LIST_ITEM_RE.match(line):
            merged = f"{merged}\n{line}"
            continue
        if merged.endswith("-") and line[:1].islower():
            merged = f"{merged[:-1]}{line.lstrip()}"
            continue
        if merged.endswith(":"):
            merged = f"{merged}\n{line}"
            continue
        merged = f"{merged} {line.lstrip()}"
    return merged.strip()


def _render_clean_page(lines: list[str]) -> str:
    """Render cleaned page lines back into readable text."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraph = _join_paragraph_lines(current)
                if paragraph:
                    paragraphs.append(paragraph)
                current = []
            continue
        current.append(line)
    if current:
        paragraph = _join_paragraph_lines(current)
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs).strip()


def clean_pdf_text_pages(pages: list[str]) -> list[str]:
    """Strip repeated PDF chrome and normalize extracted page text."""
    prepared_pages = [_prepare_page_lines(page) for page in pages]
    repeated_margin_lines = _collect_repeated_margin_lines(prepared_pages)
    cleaned_pages: list[str] = []
    for page in prepared_pages:
        trimmed = _trim_page_margins(page, repeated_margin_lines)
        rendered = _render_clean_page(trimmed)
        if rendered:
            cleaned_pages.append(rendered)
    return cleaned_pages


class PDFReaderTool(BaseTool):
    """Read text content from a PDF file on disk."""

    name = "pdf_reader"
    output_fields = ("file_path", "text", "content")

    def run(self, file_path: str) -> dict[str, object]:
        """Extract and return text from every page in the target PDF."""
        pdf_path = Path(file_path).expanduser()
        if not pdf_path.exists():
            return {
                "status": "error",
                "file_path": str(pdf_path),
                "message": "Error: File not found.",
                "content": "Error: File not found.",
            }

        try:
            reader = PdfReader(str(pdf_path))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception:
            return {
                "status": "error",
                "file_path": str(pdf_path),
                "message": "Error: Unable to read PDF.",
                "content": "Error: Unable to read PDF.",
            }

        cleaned_pages = clean_pdf_text_pages(pages)
        content = "\n\n".join(cleaned_pages).strip()
        if not content:
            return {
                "status": "warning",
                "file_path": str(pdf_path),
                "message": "Warning: PDF contains no extractable text.",
                "text": "",
                "content": "Warning: PDF contains no extractable text.",
            }
        return {
            "status": "ok",
            "file_path": str(pdf_path),
            "text": content,
            "content": content,
        }
