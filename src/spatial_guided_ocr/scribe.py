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


def extract_text_with_gemma_vision(
    doc: pymupdf.Document,
    box: SectionBox,
    model_name: str = "gemma4:12b",
    ollama_url: str = "http://localhost:11434",
    padding_pts: float = 4.0,
    timeout_seconds: float = 120.0,
) -> str:
    """Extract and OCR text from the bounded region using Gemma 4 vision via Ollama."""
    import base64
    import json
    import sys
    import httpx

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
        "Transcribe all text from this cropped document section verbatim. "
        "Preserve exact wording, line breaks, and paragraph structure. "
        "Do not describe the image, do not add introductory remarks, and do not invent text. "
        "Output ONLY the transcribed text."
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
    print(f"\n[Scribe Gemma 4] Transcribing [{box.category}] (Page {box.page_number + 1}):\n", flush=True)

    timeout = httpx.Timeout(timeout_seconds, connect=30.0)
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

    return "".join(content_chunks).strip()


def extract_all_sections(
    pdf_path: Path | str,
    boxes: list[SectionBox],
    scribe_engine: str = "gemma",
    model_name: str = "gemma4:12b",
) -> list[dict[str, Any]]:
    """Extract text for all detected section boxes via Gemma 4 vision OCR or PyMuPDF."""
    doc = pymupdf.open(str(pdf_path))
    extracted_sections: list[dict[str, Any]] = []

    for box in boxes:
        if scribe_engine.lower() in ("gemma", "vlm", "ollama", "ocr"):
            text = extract_text_with_gemma_vision(doc, box, model_name=model_name)
        else:
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
