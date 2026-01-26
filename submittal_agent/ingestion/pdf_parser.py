"""PDF parsing module for extracting text from construction submittals."""

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, BinaryIO

import fitz  # pymupdf


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""

    page_num: int
    text: str
    source_file: str = ""
    has_tables: bool = False


@dataclass
class ParsedDocument:
    """Represents a fully parsed PDF document."""

    filename: str
    pages: list[PageContent]
    total_pages: int
    metadata: dict = field(default_factory=dict)

    def get_full_text(self) -> str:
        """Concatenate all page text."""
        return "\n\n".join(f"[Page {p.page_num}]\n{p.text}" for p in self.pages if p.text)

    def get_text_by_page(self, page_num: int) -> str | None:
        """Get text for a specific page."""
        for page in self.pages:
            if page.page_num == page_num:
                return page.text
        return None

    def get_pages_with_tables(self) -> list[PageContent]:
        """Return pages that likely contain tables."""
        return [p for p in self.pages if p.has_tables]


class PDFParser:
    """
    Extracts text from PDF documents.

    Uses PyMuPDF (fitz) for efficient extraction without OCR dependency.
    Optimized for construction submittals (shop drawings, product data, specs).
    """

    def __init__(self, max_pages: int | None = None):
        """
        Initialize the PDF parser.

        Args:
            max_pages: Maximum number of pages to parse (None for all)
        """
        self.max_pages = max_pages

    def parse(self, pdf_source: str | Path | BinaryIO) -> ParsedDocument:
        """
        Parse a PDF file and extract text content.

        Args:
            pdf_source: File path or file-like object

        Returns:
            ParsedDocument with extracted content
        """
        if isinstance(pdf_source, (str, Path)):
            pdf_path = Path(pdf_source)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            doc = fitz.open(pdf_path)
            filename = pdf_path.name
        else:
            # File-like object (e.g., from FastAPI upload)
            doc = fitz.open(stream=pdf_source.read(), filetype="pdf")
            filename = getattr(pdf_source, 'filename', 'uploaded.pdf')

        pages = list(self._extract_pages(doc, filename))

        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "page_count": len(doc),
        }

        doc.close()

        return ParsedDocument(
            filename=filename,
            pages=pages,
            total_pages=len(pages),
            metadata=metadata,
        )

    def _extract_pages(
        self, doc: fitz.Document, filename: str
    ) -> Iterator[PageContent]:
        """Extract text content from each page."""
        max_pages = self.max_pages or len(doc)
        
        for page_num, page in enumerate(doc, start=1):
            if page_num > max_pages:
                break
                
            text = page.get_text("text")

            # Detect if page likely has tables
            has_tables = self._detect_tables(page, text)

            yield PageContent(
                page_num=page_num,
                text=text.strip(),
                source_file=filename,
                has_tables=has_tables,
            )

    def _detect_tables(self, page: fitz.Page, text: str) -> bool:
        """
        Heuristic table detection based on:
        1. Presence of drawing paths (grid lines)
        2. Text alignment patterns
        """
        # Check for vector drawings (table borders)
        drawings = page.get_drawings()
        has_grid_lines = len(drawings) > 10

        # Check for tab-separated or consistently spaced content
        lines = text.split("\n")
        tabular_lines = sum(1 for line in lines if "\t" in line or "  " in line)
        has_tabular_text = tabular_lines > 3

        return has_grid_lines or has_tabular_text

    def parse_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> ParsedDocument:
        """
        Parse PDF from bytes.

        Args:
            pdf_bytes: PDF file content as bytes
            filename: Name to associate with the document

        Returns:
            ParsedDocument with extracted content
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = list(self._extract_pages(doc, filename))

        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "page_count": len(doc),
        }

        doc.close()

        return ParsedDocument(
            filename=filename,
            pages=pages,
            total_pages=len(pages),
            metadata=metadata,
        )
