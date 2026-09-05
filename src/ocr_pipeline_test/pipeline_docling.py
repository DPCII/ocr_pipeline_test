"""Pipeline B: Local Layout Intelligence via IBM Docling."""

from pathlib import Path
import time
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat


def run_docling_pipeline(
    input_path: Path | str,
    output_path: Path | str,
    page_number: int | None = 0,
) -> dict[str, str | float]:
    """Execute Pipeline B: IBM Docling document intelligence layout parsing."""
    in_file = Path(input_path)
    print(f"[Pipeline B] Initializing IBM Docling for {in_file.name}...")
    start_time = time.time()

    # Configure pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # PDF has digital layout layer; use DocLayNet layout parsing
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"[Pipeline B] Converting document via Docling...")
    conv_result = converter.convert(str(in_file))
    elapsed = time.time() - start_time

    # Export markdown
    markdown_content = conv_result.document.export_to_markdown()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(markdown_content, encoding="utf-8")

    print(f"[Pipeline B] Completed in {elapsed:.2f}s -> Saved to {out_file}")
    return {
        "pipeline": "Pipeline B (IBM Docling)",
        "output_file": str(out_file),
        "elapsed_seconds": elapsed,
        "content": markdown_content,
    }
