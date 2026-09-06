"""CLI Entrypoint for VLM-Guided Spatial Extraction."""

import argparse
from pathlib import Path
import sys

from spatial_guided_ocr.pipeline import run_spatial_guided_pipeline


def main() -> None:
    """CLI dispatcher for the spatial guided OCR pipeline."""
    parser = argparse.ArgumentParser(
        description="VLM-Guided Spatial Extraction: VLM detects visual section bounding boxes; Scribe extracts verbatim text."
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
        default="gemma4:12b",
        help="VLM model to use for layout detection (default: gemma4:12b, or 'docling')",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "gemini", "muse", "docling", "auto"],
        default="auto",
        help="Backend to use for layout detection (ollama, gemini, muse, or docling; default: auto)",
    )
    parser.add_argument(
        "--scribe",
        choices=["gemma", "pymupdf"],
        default="gemma",
        help="Scribe engine for text extraction (gemma: Gemma 4 Vision OCR, pymupdf: digital extraction; default: gemma)",
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
        help="Suppress streaming of the live reasoning trace",
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
        if "docling" in model.lower():
            backend = "docling"
        elif "gemini" in model.lower():
            backend = "gemini"
        elif "muse" in model.lower():
            backend = "muse"
        else:
            backend = "ollama"

    if backend == "docling" or model.lower() == "docling":
        backend = "docling"
        model = "docling-layout"
    elif backend == "muse" and model.lower() in ("muse", "auto", "", "gemma4:12b"):
        model = "muse-spark-1.3-contributor"
    elif backend == "gemini" and model.lower() in ("gemini", "auto", "", "gemma4:12b"):
        model = "gemini-3.8-flash"
    elif model.lower() == "muse":
        model = "muse-spark-1.3-contributor"
        backend = "muse"

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
            scribe_engine=args.scribe,
        )
    except Exception as err:
        print(f"\n[ERROR] Pipeline failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
