"""Pipeline C: Remote Frontier Multimodal API (Gemini 3.8 Flash) via HTTP."""

import os
from pathlib import Path
import time
from dotenv import load_dotenv
import httpx
from typing import Any
from ocr_pipeline_test.render import render_page_to_base64

# Ensure .env is loaded
load_dotenv()

PROMPT_TEMPLATE = """You are a state-of-the-art document layout intelligence and transcription engine.
Analyze the visual layout of this document page and convert its full content into clean, semantic Markdown.

CRITICAL LAYOUT & STRUCTURAL REQUIREMENTS:
1. Multi-Column Reading Order: Do NOT merge text horizontally across columns. For multi-column areas, transcribe top-to-bottom within column 1 before proceeding to column 2.
2. Section & Bounding Box Isolation: Keep visually distinct sections (e.g. sidebars, callouts, headers, footers) strictly isolated in their own Markdown sections.
3. Hierarchy: Use appropriate markdown headers (# for main banner/title, ## for section headlines, ### for sub-headers).
4. Fidelity: Transcribe every text element accurately. Do not summarize or skip text.
5. Format: Return ONLY raw Markdown. Do not include markdown code block backticks (```markdown), preambles, or conversational commentary.
"""


def transcribe_page_gemini(
    pdf_path: Path | str,
    page_number: int,
    model_name: str = "gemini-3.8-flash",
    api_key: str | None = None,
    timeout_seconds: float = 60.0,
    dpi: int = 200,
) -> tuple[str, float]:
    """Transcribe a single page using Gemini 3.8 Flash via HTTP. Returns (content, elapsed_seconds)."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")

    print(f"\n[Pipeline C] Rendering page {page_number + 1} of {pdf_path} at {dpi} DPI...")
    img_b64 = render_page_to_base64(pdf_path, page_number=page_number, dpi=dpi)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img_b64,
                        }
                    },
                    {
                        "text": PROMPT_TEMPLATE,
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
        },
    }

    print(f"[Pipeline C] Sending HTTP request to {model_name}...")
    start_time = time.time()

    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    elapsed = time.time() - start_time

    # Extract text response from Gemini format
    try:
        content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as err:
        raise RuntimeError(f"Unexpected response structure from Gemini API: {data}") from err

    # Strip markdown fences if present
    if content.startswith("```markdown"):
        content = content[len("```markdown"):].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    return content, elapsed


def run_gemini_pipeline(
    pdf_path: Path | str,
    output_dir: Path | str,
    pages: list[int] | None = None,
    model_name: str = "gemini-3.8-flash",
    api_key: str | None = None,
    timeout_seconds: float = 60.0,
    dpi: int = 200,
) -> dict[str, Any]:
    """Execute Pipeline C: Remote Gemini 3.8 Flash multimodal API across all or specified pages."""
    from ocr_pipeline_test.render import get_pdf_page_count

    pdf_file = Path(pdf_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_pages = get_pdf_page_count(pdf_file)
    target_pages = pages if pages is not None else list(range(total_pages))

    print(f"[Pipeline C] Starting Gemini 3.8 Flash transcription for {len(target_pages)} page(s) of {pdf_file.name}...")
    page_contents: list[str] = []
    total_elapsed = 0.0

    for p in target_pages:
        content, elapsed = transcribe_page_gemini(
            pdf_path=pdf_file,
            page_number=p,
            model_name=model_name,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            dpi=dpi,
        )
        total_elapsed += elapsed
        page_contents.append(content)

        # Save per-page output
        page_file = out_dir / f"pipeline_c_gemini38_page{p + 1}.md"
        page_file.write_text(content, encoding="utf-8")
        print(f"[Pipeline C] Page {p + 1} saved -> {page_file} ({elapsed:.1f}s)")

    # Combine all pages if multi-page
    full_output_file = out_dir / "pipeline_c_gemini38_full.md"
    merged_markdown = "\n\n---\n\n".join(
        f"<!-- Page {p + 1} -->\n\n{text}" for p, text in zip(target_pages, page_contents)
    )
    full_output_file.write_text(merged_markdown, encoding="utf-8")
    print(f"[Pipeline C] Full document saved -> {full_output_file} (Total time: {total_elapsed:.1f}s)")

    return {
        "pipeline": f"Pipeline C (Remote {model_name})",
        "output_file": str(full_output_file),
        "pages_processed": len(target_pages),
        "total_elapsed_seconds": total_elapsed,
    }
