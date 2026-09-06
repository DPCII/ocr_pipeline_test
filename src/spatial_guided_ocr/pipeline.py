"""End-to-end Pipeline: VLM Architect + Spatial Scribe Extractor."""

import json
from pathlib import Path
import time
from typing import Any

from ocr_pipeline_test.render import render_page_to_base64, get_document_page_count
from ocr_pipeline_test.output_utils import resolve_output_path, prepend_tools_header
from spatial_guided_ocr.architect import SectionBox, detect_page_sections
from spatial_guided_ocr.scribe import extract_all_sections
from spatial_guided_ocr.assembler import assemble_categorized_markdown


def run_spatial_guided_pipeline(
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    model_name: str = "docling-layout",
    backend: str = "docling",
    pages: list[int] | None = None,
    dpi: int = 150,
    show_thinking: bool = False,
    enable_image_ocr: bool = True,
    ocr_model: str = "gemma4:12b",
    ollama_url: str = "http://localhost:11434",
    pdf_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute Spatial Extraction:

    Stage 1: Architect (Docling or VLM: Gemini, Muse) detects visual layout bounding boxes & thread IDs.
    Stage 2: Scribe (PyMuPDF for digital text; Gemma Vision via Ollama for image OCR) extracts verbatim text.
    Stage 3: Assembler groups text by Category & Thread into structured Markdown.
    """
    resolved_path = input_path or pdf_path
    if not resolved_path:
        raise ValueError("input_path is required for run_spatial_guided_pipeline")
    doc_file = Path(resolved_path)
    out_file = resolve_output_path(output_path, default_dir="outputs")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    total_pages = get_document_page_count(doc_file)
    target_pages = pages if pages is not None else list(range(total_pages))

    # Normalize backend and model names
    if backend.lower() in ("docling", "docling-layout", "local") or model_name.lower() in ("docling", "docling-layout"):
        backend = "docling"
        model_name = "docling-layout"
    elif backend.lower() in ("muse", "meta") or "muse" in model_name.lower():
        backend = "muse"
        if model_name.lower() in ("muse", "auto", ""):
            model_name = "muse-spark-1.3-contributor"
    elif backend.lower() in ("gemini", "google") or "gemini" in model_name.lower():
        backend = "gemini"
        if model_name.lower() in ("gemini", "auto", ""):
            model_name = "gemini-3.8-flash"

    ocr_info = f" | Image OCR: {ocr_model}" if enable_image_ocr else " | Image OCR: disabled"
    print("=" * 65)
    print(f"▶ Running Spatial-Guided Pipeline (Architect: {model_name}{ocr_info})")
    print(f"  Input: {doc_file.name} ({len(target_pages)} pages) | Architect: {model_name} ({backend})")
    print("=" * 65)

    start_total = time.time()
    all_boxes: list[SectionBox] = []

    # Stage 1: Architect (Layout detection per page)
    vlm_start = time.time()
    for p in target_pages:
        img_b64 = render_page_to_base64(doc_file, page_number=p, dpi=dpi)
        page_boxes = detect_page_sections(
            img_b64=img_b64,
            page_number=p,
            model_name=model_name,
            backend=backend,
            show_thinking=show_thinking,
            input_path=doc_file,
        )
        all_boxes.extend(page_boxes)
    vlm_elapsed = time.time() - vlm_start

    # Save layout schema to JSON for inspection
    layout_json_file = out_file.with_name(f"{out_file.stem}_layout.json")
    boxes_data = [
        {
            "category": b.category,
            "label": b.label,
            "thread_id": b.thread_id,
            "order": b.order,
            "page_number": b.page_number + 1,
            "box_2d": b.box_2d,
        }
        for b in all_boxes
    ]
    layout_json_file.write_text(json.dumps(boxes_data, indent=2), encoding="utf-8")
    print(f"[Architect] Layout schema saved -> {layout_json_file} ({vlm_elapsed:.2f}s)")

    # Stage 2: Scribe (Digital text extraction via PyMuPDF + Image OCR via Gemma)
    scribe_start = time.time()
    print(f"[Scribe] Extracting text from {len(all_boxes)} bounding boxes...")
    extracted_sections = extract_all_sections(
        doc_file,
        all_boxes,
        enable_image_ocr=enable_image_ocr,
        ocr_model=ocr_model,
        ollama_url=ollama_url,
    )
    scribe_elapsed = time.time() - scribe_start
    print(f"[Scribe] Extraction completed in {scribe_elapsed:.2f}s.")

    # Stage 3: Assembler (Markdown generation by category & thread)
    markdown_content = assemble_categorized_markdown(extracted_sections)

    # Determine tools used
    tools_used = ["PyMuPDF"]
    if backend == "docling":
        tools_used.append("IBM Docling")
    elif backend == "gemini":
        tools_used.append("Gemini 3.8 Flash")
    elif backend == "muse":
        tools_used.append("Muse Spark 1.3")
    else:
        tools_used.append(model_name)

    if enable_image_ocr:
        ocr_label = "Gemma 4 (12B via Ollama)" if "gemma" in ocr_model.lower() else f"{ocr_model} (via Ollama)"
        tools_used.append(ocr_label)

    markdown_with_header = prepend_tools_header(markdown_content, tools_used, input_path=doc_file)
    out_file.write_text(markdown_with_header, encoding="utf-8")

    total_elapsed = time.time() - start_total
    print(f"\n[Success] Generated Categorized Markdown -> {out_file}")
    print(f"  Total time: {total_elapsed:.2f}s (VLM Architect: {vlm_elapsed:.2f}s, Scribe: {scribe_elapsed:.3f}s)")

    return {
        "pipeline": f"Spatial Guided ({model_name} + Scribe)",
        "output_file": str(out_file),
        "layout_json": str(layout_json_file),
        "sections_count": len(all_boxes),
        "vlm_elapsed": vlm_elapsed,
        "scribe_elapsed": scribe_elapsed,
        "total_elapsed": total_elapsed,
        "content": markdown_content,
    }
