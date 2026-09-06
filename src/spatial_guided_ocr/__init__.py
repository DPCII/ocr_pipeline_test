"""CLI Entrypoint for VLM-Guided Spatial Extraction."""

import argparse
from pathlib import Path
import sys

from spatial_guided_ocr.pipeline import run_spatial_guided_pipeline


def main() -> None:
    """CLI dispatcher for the spatial guided OCR pipeline."""
    parser = argparse.ArgumentParser(
        description="Spatial Document Intelligence: Layout analysis (Docling / VLM) + Scribe extraction (PyMuPDF & Gemma Image OCR)."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="multipage_newsletter.pdf",
        help="Path to PDF document (default: multipage_newsletter.pdf)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="docling-layout",
        help="Model / engine for layout detection ('docling-layout', 'gemini-3.8-flash', 'muse-spark-1.3-contributor'; default: docling-layout)",
    )
    parser.add_argument(
        "--backend",
        choices=["docling", "gemini", "muse", "auto"],
        default="auto",
        help="Backend for layout detection (docling, gemini, muse; default: auto)",
    )
    parser.add_argument(
        "--ocr-model",
        type=str,
        default="gemma4:12b",
        help="Vision model target for image OCR jobs served via Ollama (default: gemma4:12b)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama host URL for serving local models (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--no-image-ocr",
        action="store_true",
        help="Disable Gemma image OCR for graphics and scanned regions",
    )
    parser.add_argument(
        "--page",
        type=str,
        default="all",
        help="Page to process ('all' for entire document, or 1-indexed number like '1'; default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination markdown file (default: outputs/output_<ddMMMyyyy-hhmm-ss>.md)",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        default=True,
        help="Stream the VLM's live layout reasoning trace directly to stdout (default: True)",
    )
    parser.add_argument(
        "--no-thinking",
        dest="show_thinking",
        action="store_false",
        help="Suppress streaming of the live reasoning trace (prints compact progress markers instead)",
    )

    args = parser.parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: File {pdf_path} not found.")
        sys.exit(1)

    # Auto-detect backend and normalize model names
    backend = args.backend
    model = args.model

    if backend == "auto":
        if "gemini" in model.lower():
            backend = "gemini"
        elif "muse" in model.lower():
            backend = "muse"
        else:
            backend = "docling"
            model = "docling-layout"

    if backend == "docling" or model.lower() in ("docling", "docling-layout"):
        backend = "docling"
        model = "docling-layout"
    elif backend == "muse" or "muse" in model.lower():
        backend = "muse"
        if model.lower() in ("muse", "auto", ""):
            model = "muse-spark-1.3-contributor"
    elif backend == "gemini" or "gemini" in model.lower():
        backend = "gemini"
        if model.lower() in ("gemini", "auto", ""):
            model = "gemini-3.8-flash"

    # Determine pages
    if args.page.lower() == "all":
        target_pages = None
    else:
        try:
            p_num = int(args.page)
            target_pages = [p_num - 1 if p_num > 0 else 0]
        except ValueError:
            print(f"Error: Invalid --page argument '{args.page}'. Use 'all' or an integer.")
            sys.exit(1)

    try:
        run_spatial_guided_pipeline(
            pdf_path=pdf_path,
            output_path=args.output,
            model_name=model,
            backend=backend,
            pages=target_pages,
            show_thinking=args.show_thinking,
            enable_image_ocr=not args.no_image_ocr,
            ocr_model=args.ocr_model,
            ollama_url=args.ollama_url,
        )
    except Exception as err:
        print(f"\n[ERROR] Pipeline failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
