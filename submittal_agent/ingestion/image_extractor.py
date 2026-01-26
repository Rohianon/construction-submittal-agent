"""Image extraction module for extracting drawings and images from PDFs."""

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO
from io import BytesIO

import fitz  # pymupdf
from PIL import Image


@dataclass
class ExtractedImage:
    """Represents an extracted image from a PDF."""

    page_num: int
    image_index: int
    width: int
    height: int
    format: str
    image_bytes: bytes
    base64_data: str
    source_file: str = ""
    
    @property
    def media_type(self) -> str:
        """Get the media type for API calls."""
        format_map = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "webp": "image/webp",
        }
        return format_map.get(self.format.lower(), "image/png")


@dataclass
class ImageExtractionResult:
    """Result of image extraction from a document."""

    filename: str
    images: list[ExtractedImage]
    total_pages: int
    pages_with_images: list[int] = field(default_factory=list)

    def get_images_for_page(self, page_num: int) -> list[ExtractedImage]:
        """Get all images from a specific page."""
        return [img for img in self.images if img.page_num == page_num]

    def get_base64_images(self) -> list[dict]:
        """Get images formatted for vision API calls."""
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type,
                    "data": img.base64_data,
                }
            }
            for img in self.images
        ]


class ImageExtractor:
    """
    Extracts images and renders pages from PDF documents.

    Designed for construction submittals where shop drawings and
    diagrams need to be analyzed by vision models.
    """

    # Minimum image size to extract (skip logos, icons)
    MIN_WIDTH = 100
    MIN_HEIGHT = 100
    
    # Maximum images to extract per document (for cost control)
    MAX_IMAGES = 10
    
    # DPI for page rendering
    RENDER_DPI = 150

    def __init__(
        self,
        min_width: int = MIN_WIDTH,
        min_height: int = MIN_HEIGHT,
        max_images: int = MAX_IMAGES,
        render_dpi: int = RENDER_DPI,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.max_images = max_images
        self.render_dpi = render_dpi

    def extract(self, pdf_source: str | Path | BinaryIO) -> ImageExtractionResult:
        """
        Extract images from a PDF document.

        Args:
            pdf_source: File path or file-like object

        Returns:
            ImageExtractionResult with extracted images
        """
        if isinstance(pdf_source, (str, Path)):
            pdf_path = Path(pdf_source)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            doc = fitz.open(pdf_path)
            filename = pdf_path.name
        else:
            doc = fitz.open(stream=pdf_source.read(), filetype="pdf")
            filename = getattr(pdf_source, 'filename', 'uploaded.pdf')

        images = []
        pages_with_images = []
        image_count = 0

        for page_num, page in enumerate(doc, start=1):
            if image_count >= self.max_images:
                break

            page_images = self._extract_page_images(page, page_num, filename)
            
            if page_images:
                pages_with_images.append(page_num)
                for img in page_images:
                    if image_count >= self.max_images:
                        break
                    images.append(img)
                    image_count += 1

        # If no embedded images found, render key pages
        if not images:
            images = self._render_pages(doc, filename)
            pages_with_images = [img.page_num for img in images]

        doc.close()

        return ImageExtractionResult(
            filename=filename,
            images=images,
            total_pages=len(doc) if hasattr(doc, '__len__') else 0,
            pages_with_images=pages_with_images,
        )

    def _extract_page_images(
        self, page: fitz.Page, page_num: int, filename: str
    ) -> list[ExtractedImage]:
        """Extract embedded images from a page."""
        images = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)
                
                if not base_image:
                    continue

                width = base_image["width"]
                height = base_image["height"]

                # Skip small images (likely logos/icons)
                if width < self.min_width or height < self.min_height:
                    continue

                image_bytes = base_image["image"]
                ext = base_image["ext"]
                
                # Convert to PNG if needed for consistency
                if ext not in ("png", "jpeg", "jpg"):
                    image_bytes, ext = self._convert_to_png(image_bytes)

                base64_data = base64.b64encode(image_bytes).decode("utf-8")

                images.append(ExtractedImage(
                    page_num=page_num,
                    image_index=img_index,
                    width=width,
                    height=height,
                    format=ext,
                    image_bytes=image_bytes,
                    base64_data=base64_data,
                    source_file=filename,
                ))

            except Exception:
                # Skip problematic images
                continue

        return images

    def _render_pages(
        self, doc: fitz.Document, filename: str
    ) -> list[ExtractedImage]:
        """
        Render pages as images when no embedded images are found.

        Useful for shop drawings that are vector-based.
        """
        images = []
        
        # Render first few pages (most important for submittals)
        pages_to_render = min(3, len(doc))
        
        for page_num in range(1, pages_to_render + 1):
            page = doc[page_num - 1]
            
            # Render at specified DPI
            mat = fitz.Matrix(self.render_dpi / 72, self.render_dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            
            image_bytes = pix.tobytes("png")
            base64_data = base64.b64encode(image_bytes).decode("utf-8")

            images.append(ExtractedImage(
                page_num=page_num,
                image_index=0,
                width=pix.width,
                height=pix.height,
                format="png",
                image_bytes=image_bytes,
                base64_data=base64_data,
                source_file=filename,
            ))

        return images

    def _convert_to_png(self, image_bytes: bytes) -> tuple[bytes, str]:
        """Convert image to PNG format."""
        try:
            img = Image.open(BytesIO(image_bytes))
            output = BytesIO()
            img.save(output, format="PNG")
            return output.getvalue(), "png"
        except Exception:
            return image_bytes, "png"

    def extract_from_bytes(
        self, pdf_bytes: bytes, filename: str = "document.pdf"
    ) -> ImageExtractionResult:
        """
        Extract images from PDF bytes.

        Args:
            pdf_bytes: PDF file content as bytes
            filename: Name to associate with the document

        Returns:
            ImageExtractionResult with extracted images
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        images = []
        pages_with_images = []
        image_count = 0

        for page_num, page in enumerate(doc, start=1):
            if image_count >= self.max_images:
                break

            page_images = self._extract_page_images(page, page_num, filename)
            
            if page_images:
                pages_with_images.append(page_num)
                for img in page_images:
                    if image_count >= self.max_images:
                        break
                    images.append(img)
                    image_count += 1

        if not images:
            images = self._render_pages(doc, filename)
            pages_with_images = [img.page_num for img in images]

        total_pages = len(doc)
        doc.close()

        return ImageExtractionResult(
            filename=filename,
            images=images,
            total_pages=total_pages,
            pages_with_images=pages_with_images,
        )
