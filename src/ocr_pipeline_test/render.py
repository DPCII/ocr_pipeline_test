"""Utility module to render PDF pages to high-resolution images and extract layout metadata."""

import base64
from pathlib import Path
from typing import Any
import pymupdf


def get_pdf_page_count(pdf_path: Path | str) -> int:
    """Return total number of pages in the PDF."""
    doc = pymupdf.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def render_page_to_png_bytes(pdf_path: Path | str, page_number: int = 0, dpi: int = 200) -> bytes:
    """Render a specific page of a PDF file to PNG bytes at the specified DPI."""
    doc = pymupdf.open(str(pdf_path))
    if page_number < 0 or page_number >= len(doc):
        raise ValueError(f"Page number {page_number} out of range (0 to {len(doc) - 1})")

    page = doc[page_number]
    # Standard PDF resolution is 72 DPI; zoom factor scales to desired DPI
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


def render_page_to_base64(pdf_path: Path | str, page_number: int = 0, dpi: int = 200) -> str:
    """Render a specific page of a PDF file to a base64-encoded string."""
    png_bytes = render_page_to_png_bytes(pdf_path, page_number=page_number, dpi=dpi)
    return base64.b64encode(png_bytes).decode("utf-8")


def save_rendered_page(pdf_path: Path | str, output_path: Path | str, page_number: int = 0, dpi: int = 200) -> Path:
    """Render a page and save it directly to a file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    png_bytes = render_page_to_png_bytes(pdf_path, page_number=page_number, dpi=dpi)
    output.write_bytes(png_bytes)
    return output


def get_reference_layout_blocks(pdf_path: Path | str, page_number: int = 0) -> list[dict[str, Any]]:
    """Extract ground-truth visual bounding boxes and text blocks from the PDF for structural reference."""
    doc = pymupdf.open(str(pdf_path))
    page = doc[page_number]
    raw_blocks: Any = page.get_text("blocks")
    doc.close()

    blocks: list[dict[str, Any]] = []
    for b in raw_blocks:
        if isinstance(b, (list, tuple)) and len(b) >= 7:
            x0, y0, x1, y1, text, _block_no, block_type = b[:7]
            blocks.append({
                "bbox": (
                    round(float(x0), 1),
                    round(float(y0), 1),
                    round(float(x1), 1),
                    round(float(y1), 1),
                ),
                "text": str(text).strip(),
                "block_type": int(block_type),
            })
    return blocks
