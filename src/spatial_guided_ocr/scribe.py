"""Stage 2: Scribe - High-Precision Spatial Text Extraction (PyMuPDF) & Image OCR (Gemma via Ollama)."""

import base64
import json
from pathlib import Path
import sys
from typing import Any
import httpx
import pymupdf
from spatial_guided_ocr.architect import SectionBox


def ocr_image_with_gemma(
    doc: pymupdf.Document,
    box: SectionBox,
    model_name: str = "gemma4:12b",
    ollama_url: str = "http://localhost:11434",
    padding_pts: float = 4.0,
    timeout_seconds: float = 120.0,
) -> str:
    """Perform image OCR on a cropped section/graphic using Gemma vision served via Ollama."""
    page = doc[box.page_number]
    pw = page.rect.width
    ph = page.rect.height

    ymin, xmin, ymax, xmax = box.box_2d
    x0 = max(0.0, (xmin / 1000.0) * pw - padding_pts)
    y0 = max(0.0, (ymin / 1000.0) * ph - padding_pts)
    x1 = min(pw, (xmax / 1000.0) * pw + padding_pts)
    y1 = min(ph, (ymax / 1000.0) * ph + padding_pts)

    clip_rect = pymupdf.Rect(x0, y0, x1, y1)
    # Render cropped region at 2x resolution for crisp character recognition
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0), clip=clip_rect)
    img_bytes = pix.tobytes("png")
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt = (
        "Transcribe all text from this cropped document image/graphic verbatim. "
        "Preserve exact wording, numbers, line breaks, and labels. "
        "Do not describe the image, do not add introductory remarks, and do not invent text. "
        "Output ONLY the transcribed text, or leave blank if there is no text in the image."
    )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [img_b64],
        "stream": True,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.1,
            "num_ctx": 16384,
            "num_predict": 2048,
        },
    }

    content_chunks: list[str] = []
    print(f"\n[Image OCR Gemma] Transcribing [{box.category}] (Page {box.page_number + 1}): ", end="", flush=True)

    timeout = httpx.Timeout(timeout_seconds, connect=30.0)
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", f"{ollama_url}/api/generate", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("response", "")
                    if token:
                        content_chunks.append(token)
                        sys.stdout.write(token)
                        sys.stdout.flush()
                    if data.get("done", False):
                        break
                print()
    except Exception as exc:
        print(f" (Image OCR skipped: {exc})")
        return ""

    return "".join(content_chunks).strip()


# Backwards-compatibility alias
extract_text_with_gemma_vision = ocr_image_with_gemma


def extract_text_from_section_box(
    doc: pymupdf.Document,
    box: SectionBox,
    padding_pts: float = 3.0,
    enable_image_ocr: bool = True,
    ocr_model: str = "gemma4:12b",
    ollama_url: str = "http://localhost:11434",
) -> str:
    """Extract text from a section box: uses PyMuPDF for digital text, and Gemma for image OCR jobs."""
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

    # If digital text is empty or category is Graphic/Image, delegate to Gemma for image OCR
    is_graphic = box.category.lower() in ("graphic", "image", "picture", "figure")
    if (not text or is_graphic) and enable_image_ocr:
        ocr_text = ocr_image_with_gemma(
            doc,
            box,
            model_name=ocr_model,
            ollama_url=ollama_url,
        )
        if ocr_text:
            text = ocr_text

    return text


def extract_all_sections(
    pdf_path: Path | str,
    boxes: list[SectionBox],
    enable_image_ocr: bool = True,
    ocr_model: str = "gemma4:12b",
    ollama_url: str = "http://localhost:11434",
    scribe_engine: str | None = None,
    model_name: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Extract text for all detected section boxes.

    Uses PyMuPDF for digital text extraction and Gemma (served via Ollama) for image OCR jobs.
    """
    effective_ocr_model = model_name or ocr_model
    doc = pymupdf.open(str(pdf_path))
    extracted_sections: list[dict[str, Any]] = []

    for box in boxes:
        text = extract_text_from_section_box(
            doc,
            box,
            enable_image_ocr=enable_image_ocr,
            ocr_model=effective_ocr_model,
            ollama_url=ollama_url,
        )

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
