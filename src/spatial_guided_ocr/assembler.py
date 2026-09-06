"""Stage 3: Assembler - Deterministic Markdown Assembly by Category & Thread."""

from typing import Any
from collections import defaultdict


def assemble_categorized_markdown(extracted_sections: list[dict[str, Any]]) -> str:
    """Group extracted sections by thread_id/category and assemble clean, structured Markdown."""
    # Filter out empty text sections
    valid_sections = [s for s in extracted_sections if s.get("text", "").strip()]

    # Group by logical thread_id
    threads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in valid_sections:
        threads[s["thread_id"]].append(s)

    # Sort items within each thread by page number then sequential order
    for tid in threads:
        threads[tid].sort(key=lambda item: (item.get("page_number", 1), item.get("order", 1)))

    markdown_parts: list[str] = []

    # Priority order for presentation: Banner -> Main Story -> Sidebars -> Others -> Footer
    def category_priority(cat: str) -> int:
        c = cat.lower()
        if "banner" in c or "header" in c:
            return 1
        if "main" in c or "story" in c or "article" in c:
            return 2
        if "sidebar" in c:
            return 3
        if "footer" in c:
            return 99
        return 4

    # Sort thread keys based on category priority of the first element
    sorted_threads = sorted(
        threads.keys(),
        key=lambda tid: category_priority(threads[tid][0].get("category", "")),
    )

    for tid in sorted_threads:
        items = threads[tid]
        first = items[0]
        category = first.get("category", "Section")
        label = first.get("label", tid)

        header_banner = f"# [Category: {category}] {label}".strip()
        markdown_parts.append(header_banner)

        # Concatenate text blocks belonging to this thread
        thread_texts: list[str] = []
        for it in items:
            page_info = f"*(Page {it['page_number']})*" if len(items) > 1 else ""
            t = it["text"].strip()
            if t:
                if page_info:
                    thread_texts.append(f"{page_info}\n{t}")
                else:
                    thread_texts.append(t)

        markdown_parts.append("\n\n".join(thread_texts))
        markdown_parts.append("\n---\n")

    return "\n\n".join(markdown_parts).strip() + "\n"
