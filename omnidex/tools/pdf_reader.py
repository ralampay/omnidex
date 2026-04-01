"""PDF reading tool for local agent workflows."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from omnidex.tools.base import BaseTool


class PDFReaderTool(BaseTool):
    """Read text content from a PDF file on disk."""

    name = "pdf_reader"

    def run(self, file_path: str) -> str:
        """Extract and return text from every page in the target PDF."""
        pdf_path = Path(file_path).expanduser()
        if not pdf_path.exists():
            return "Error: File not found."

        try:
            reader = PdfReader(str(pdf_path))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception:
            return "Error: Unable to read PDF."

        content = "\n\n".join(page.strip() for page in pages if page.strip()).strip()
        if not content:
            return "Warning: PDF contains no extractable text."
        return content
