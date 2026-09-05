"""Evaluation module to compare Markdown outputs against layout and section bounding criteria."""

from pathlib import Path
from typing import Any
import re


def evaluate_markdown_fidelity(markdown_path: Path | str) -> dict[str, Any]:
    """Evaluate a markdown file against the expected multi-column newsletter layout structure."""
    path = Path(markdown_path)
    if not path.exists():
        return {
            "file": str(path),
            "status": "Missing",
            "error": "File does not exist",
        }

    text = path.read_text(encoding="utf-8")

    # 1. Check Banner / Title elements
    has_title = bool(re.search(r"today['’]s news", text, re.IGNORECASE))
    has_topics = bool(re.search(r"key topic 1", text, re.IGNORECASE))
    has_date = bool(re.search(r"september 5, 2026", text, re.IGNORECASE))

    # 2. Check Left Sidebar elements
    has_heading1 = bool(re.search(r"heading 1", text, re.IGNORECASE))
    has_heading2 = bool(re.search(r"heading 2", text, re.IGNORECASE))
    has_heading3 = bool(re.search(r"heading 3", text, re.IGNORECASE))
    sidebar_complete = has_heading1 and has_heading2 and has_heading3

    # 3. Check Main Article elements
    has_main_heading = bool(re.search(r"main multipage text", text, re.IGNORECASE))
    has_main_body = bool(re.search(r"to get started.*just tap or click", text, re.IGNORECASE | re.DOTALL))
    main_article_complete = has_main_heading and has_main_body

    # 4. Check Column Separation (No Interleaving / Cross-column bleeding)
    # If the text correctly processes column-by-column, either all sidebar headings appear
    # before the main text, or after it—never interleaved line-by-line.
    h1_pos = text.lower().find("heading 1")
    h2_pos = text.lower().find("heading 2")
    h3_pos = text.lower().find("heading 3")
    main_pos = text.lower().find("main multipage text")

    column_separation_preserved = False
    if sidebar_complete and main_pos != -1:
        # Either Sidebar comes before Main, or Main comes before Sidebar
        sidebar_together = (h1_pos < h2_pos < h3_pos)
        no_interleaving = (h3_pos < main_pos) or (main_pos < h1_pos)
        column_separation_preserved = sidebar_together and no_interleaving

    # 5. Header counts
    h1_count = len(re.findall(r"^#\s+", text, re.MULTILINE))
    h2_count = len(re.findall(r"^##\s+", text, re.MULTILINE))
    h3_count = len(re.findall(r"^###\s+", text, re.MULTILINE))

    return {
        "file": path.name,
        "status": "Evaluated",
        "has_banner_title": has_title,
        "has_topic_ribbon": has_topics,
        "has_date_stamp": has_date,
        "sidebar_captured": sidebar_complete,
        "main_article_captured": main_article_complete,
        "column_separation_preserved": column_separation_preserved,
        "header_counts": {"#": h1_count, "##": h2_count, "###": h3_count},
        "char_count": len(text),
    }


def print_comparison_table(results: list[dict[str, Any]]) -> str:
    """Format and print evaluation results as a clean Markdown table."""
    headers = [
        "Pipeline / Output",
        "Banner Title",
        "Sidebar Complete",
        "Main Story Complete",
        "Column Separation",
        "Headers (#/##/###)",
    ]
    rows = []
    for r in results:
        if r.get("status") != "Evaluated":
            rows.append(f"| {r.get('file', 'Unknown')} | N/A | N/A | N/A | N/A (Missing) | N/A |")
            continue

        banner_check = "✅ Pass" if r["has_banner_title"] else "❌ Fail"
        sidebar_check = "✅ Pass" if r["sidebar_captured"] else "❌ Fail"
        main_check = "✅ Pass" if r["main_article_captured"] else "❌ Fail"
        col_check = "✅ Preserved" if r["column_separation_preserved"] else "❌ Bleed/Failed"
        h = r["header_counts"]
        headers_summary = f"{h['#']} / {h['##']} / {h['###']}"

        rows.append(
            f"| {r['file']} | {banner_check} | {sidebar_check} | {main_check} | {col_check} | {headers_summary} |"
        )

    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    table += "\n".join(rows)
    return table
