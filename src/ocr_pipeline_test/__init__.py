"""Main CLI entrypoint for ocr-pipeline-test."""

import argparse
from pathlib import Path
import sys

from ocr_pipeline_test.pipeline_gemma import run_gemma_pipeline
from ocr_pipeline_test.pipeline_docling import run_docling_pipeline
from ocr_pipeline_test.pipeline_gemini import run_gemini_pipeline
from ocr_pipeline_test.evaluate import evaluate_markdown_fidelity, print_comparison_table


def main() -> None:
    """CLI dispatcher to run document intelligence pipelines and evaluate results."""
    parser = argparse.ArgumentParser(
        description="Run and compare OCR / Document Intelligence pipelines (Gemma 4 vision, Docling, Gemini 3.8 Flash)."
    )
    parser.add_argument(
        "--pipeline",
        choices=["all", "gemma", "docling", "gemini", "evaluate"],
        default="all",
        help="Which pipeline to execute (default: all)",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="multipage_newsletter.pdf",
        help="Path to the PDF input document (default: multipage_newsletter.pdf)",
    )
    parser.add_argument(
        "--docx",
        type=str,
        default="multipage_newsletter.docx",
        help="Path to the DOCX input document for Docling (default: multipage_newsletter.docx)",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=0,
        help="Page index to test (0-indexed, default: 0 for Page 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to store markdown outputs (default: outputs/)",
    )

    args = parser.parse_args()
    pdf_path = Path(args.pdf)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        print(f"Error: Input PDF {pdf_path} not found.")
        sys.exit(1)

    outputs = {
        "gemma": output_dir / f"pipeline_a_gemma4_page{args.page + 1}.md",
        "docling": output_dir / f"pipeline_b_docling_page{args.page + 1}.md",
        "gemini": output_dir / f"pipeline_c_gemini38_page{args.page + 1}.md",
    }

    # Pipeline A: Local Gemma 4
    if args.pipeline in ("all", "gemma"):
        print("\n" + "=" * 60)
        print("▶ Running Pipeline A: Local Gemma 4:12b Vision (Ollama)")
        print("=" * 60)
        try:
            run_gemma_pipeline(pdf_path, outputs["gemma"], page_number=args.page)
        except Exception as err:
            print(f"[Pipeline A ERROR]: {err}")

    # Pipeline B: Local IBM Docling
    if args.pipeline in ("all", "docling"):
        print("\n" + "=" * 60)
        print("▶ Running Pipeline B: Local IBM Docling Layout Intelligence")
        print("=" * 60)
        try:
            run_docling_pipeline(pdf_path, outputs["docling"], page_number=args.page)
        except Exception as err:
            print(f"[Pipeline B ERROR]: {err}")

    # Pipeline C: Remote Gemini 3.8 Flash
    if args.pipeline in ("all", "gemini"):
        print("\n" + "=" * 60)
        print("▶ Running Pipeline C: Remote Gemini 3.8 Flash (HTTP API)")
        print("=" * 60)
        try:
            run_gemini_pipeline(pdf_path, outputs["gemini"], page_number=args.page)
        except Exception as err:
            print(f"[Pipeline C ERROR]: {err}")

    # Evaluation phase
    if args.pipeline in ("all", "evaluate"):
        print("\n" + "=" * 60)
        print("📊 Evaluation & Comparison Summary")
        print("=" * 60)
        eval_results = []
        for name, path in outputs.items():
            eval_results.append(evaluate_markdown_fidelity(path))

        table = print_comparison_table(eval_results)
        print(table)


if __name__ == "__main__":
    main()
