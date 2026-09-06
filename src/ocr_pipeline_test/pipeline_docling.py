"""Pipeline B: Local Layout Intelligence via IBM Docling."""

from pathlib import Path
import time
from typing import Any
from docling.document_converter import DocumentConverter
from ocr_pipeline_test.output_utils import generate_output_path, prepend_tools_header


def run_docling_pipeline(
    input_path: Path | str,
    output_dir: Path | str,
    pages: list[int] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Execute Pipeline B: IBM Docling document intelligence layout parsing."""
    in_file = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Pipeline B] Initializing IBM Docling for {in_file.name}...")
    start_time = time.time()

    # Initialize DocumentConverter with natural default pipeline
    # Docling automatically detects digital text streams, layout regions, and applies OCR where needed
    converter = DocumentConverter()

    print(f"[Pipeline B] Converting document via Docling...")
    conv_result = converter.convert(str(in_file))
    elapsed = time.time() - start_time

    # Export markdown and prepend standardized Tools & Input header
    raw_markdown = conv_result.document.export_to_markdown()
    markdown_content = prepend_tools_header(raw_markdown, ["IBM Docling"], input_path=in_file)

    # Determine output suffix based on targeted pages
    is_page1_only = pages is not None and pages == [0]
    out_suffix = "docling_page1" if is_page1_only else "docling_full"

    out_file = generate_output_path(out_dir, suffix=out_suffix, timestamp=timestamp)
    out_file.write_text(markdown_content, encoding="utf-8")

    print(f"[Pipeline B] Completed in {elapsed:.2f}s -> Saved to {out_file}")
    return {
        "pipeline": "Pipeline B (IBM Docling)",
        "output_file": str(out_file),
        "elapsed_seconds": elapsed,
        "content": markdown_content,
    }
