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
        choices=["all", "gemma", "docling", "gemini", "muse", "evaluate"],
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
        type=str,
        default="all",
        help="Page to process ('all' for entire document, or 1-indexed number like '1'; default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to store markdown outputs (default: outputs/)",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Stream the model's live reasoning trace directly to stdout",
    )

    args = parser.parse_args()
    pdf_path = Path(args.pdf)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        print(f"Error: Input PDF {pdf_path} not found.")
        sys.exit(1)

    # Determine pages to process
    if args.page.lower() == "all":
        target_pages = None  # Process all pages
        eval_suffix = "full"
    else:
        try:
            p_num = int(args.page)
            target_pages = [p_num - 1 if p_num > 0 else 0]
            eval_suffix = f"page{target_pages[0] + 1}"
        except ValueError:
            print(f"Error: Invalid --page argument '{args.page}'. Use 'all' or an integer.")
            sys.exit(1)

    # Pipeline A: Local Gemma 4
    if args.pipeline in ("all", "gemma"):
        print("\n" + "=" * 60)
        print("▶ Running Pipeline A: Local Gemma 4:12b Vision (Ollama)")
        print("=" * 60)
        try:
            run_gemma_pipeline(
                pdf_path,
                output_dir,
                pages=target_pages,
                show_thinking=args.show_thinking,
            )
        except Exception as err:
            print(f"[Pipeline A ERROR]: {err}")

    # Pipeline B: Local IBM Docling
    if args.pipeline in ("all", "docling"):
        print("\n" + "=" * 60)
        print("▶ Running Pipeline B: Local IBM Docling Layout Intelligence")
        print("=" * 60)
        try:
            run_docling_pipeline(pdf_path, output_dir, pages=target_pages)
        except Exception as err:
            print(f"[Pipeline B ERROR]: {err}")

    # Pipeline C: Remote Gemini 3.8 Flash
    if args.pipeline in ("all", "gemini"):
        print("\n" + "=" * 60)
        print("▶ Running Pipeline C: Remote Gemini 3.8 Flash (HTTP API)")
        print("=" * 60)
        try:
            run_gemini_pipeline(pdf_path, output_dir, pages=target_pages)
        except Exception as err:
            print(f"[Pipeline C ERROR]: {err}")

    # Pipeline C2: Remote Muse Spark 1.3
    if args.pipeline in ("all", "muse"):
        print("\n" + "=" * 60)
        print("▶ Running Pipeline C2: Remote Muse Spark 1.3 (Meta API)")
        print("=" * 60)
        try:
            from ocr_pipeline_test.pipeline_muse import run_muse_pipeline
            run_muse_pipeline(pdf_path, output_dir, pages=target_pages)
        except Exception as err:
            print(f"[Pipeline Muse ERROR]: {err}")

    # Evaluation phase
    if args.pipeline in ("all", "evaluate"):
        print("\n" + "=" * 60)
        print(f"📊 Evaluation & Comparison Summary ({eval_suffix.upper()})")
        print("=" * 60)
        eval_files = [
            output_dir / f"pipeline_a_gemma4_{eval_suffix}.md",
            output_dir / f"pipeline_b_docling_{eval_suffix}.md",
            output_dir / f"pipeline_c_gemini38_{eval_suffix}.md",
            output_dir / f"pipeline_c2_muse_{eval_suffix}.md",
        ]
        # Also include page 1 files if full not yet evaluated
        eval_results = []
        for path in eval_files:
            if not path.exists() and eval_suffix == "full":
                alt = path.with_name(path.name.replace("_full.md", "_page1.md"))
                eval_results.append(evaluate_markdown_fidelity(alt))
            else:
                eval_results.append(evaluate_markdown_fidelity(path))

        table = print_comparison_table(eval_results)
        print(table)


if __name__ == "__main__":
    main()
