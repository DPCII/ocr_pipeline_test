"""Stage 2: Scribe - High-Precision Spatial Text Extraction via PyMuPDF / OCR."""

from pathlib import Path
from typing import Any
import pymupdf
from spatial_guided_ocr.architect import SectionBox



def extract_text_from_section_box(
    doc: pymupdf.Document,
    box: SectionBox,
    padding_pts: float = 3.0,
) -> str:
    """Extract verbatim text enclosed inside the 2D bounding box from the specified PDF page."""
    page = doc[box.page_number]
    pw = page.rect.width
    ph = page.rect.height

    # Convert 0-1000 normalized coordinates [ymin, xmin, ymax, xmax] to PDF points
    ymin, xmin, ymax, xmax = box.box_2d
    x0 = max(0.0, (xmin / 1000.0) * pw - padding_pts)
    y0 = max(0.0, (ymin / 1000.0) * ph - padding_pts)
    x1 = min(pw, (xmax / 1000.0) * pw + padding_pts)
    y1 = min(ph, (ymax / 1000.0) * ph + padding_pts)

    clip_rect = pymupdf.Rect(x0, y0, x1, y1)
    text = page.get_text("text", clip=clip_rect).strip()

    # If digital text is empty (e.g. scanned image box), check if RapidOCR is available
    if not text:
        try:
            from rapidocr import RapidOCR
            # Render clipped region to pixmap and run RapidOCR
            mat = pymupdf.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
            img_bytes = pix.tobytes("png")
            engine = RapidOCR()
            result, _ = engine(img_bytes)
            if result:
                text = "\n".join([line[1] for line in result]).strip()
        except Exception:
            pass

    return text


def extract_all_sections(
    pdf_path: Path | str,
    boxes: list[SectionBox],
) -> list[dict[str, Any]]:
    """Extract verbatim text for all detected section boxes in the document."""
    doc = pymupdf.open(str(pdf_path))
    extracted_sections: list[dict[str, Any]] = []

    for box in boxes:
        text = extract_text_from_section_box(doc, box)
        extracted_sections.append({
            "category": box.category,
            "label": box.label,
            "thread_id": box.thread_id,
            "order": box.order,
            "page_number": box.page_number + 1,
            "box_2d": box.box_2d,
            "text": text,
        })

    doc.close()
    return extracted_sections
