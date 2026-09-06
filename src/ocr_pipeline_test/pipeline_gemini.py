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
    input_path: Path | str,
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

    print(f"\n[Pipeline C] Rendering page {page_number + 1} of {input_path} at {dpi} DPI...")
    img_b64 = render_page_to_base64(input_path, page_number=page_number, dpi=dpi)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?key={key}&alt=sse"
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

    print(f"[Pipeline C] Streaming reasoning & transcription from {model_name}...")
    start_time = time.time()

    thought_chunks: list[str] = []
    content_chunks: list[str] = []
    in_thinking = True
    import sys

    timeout = httpx.Timeout(timeout_seconds, connect=30.0)
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, json=payload) as response:
            if response.is_error:
                response.read()
                try:
                    err_data = response.json()
                    err_msg = err_data.get("error", {}).get("message", response.text)
                except Exception:
                    err_msg = response.text
                raise RuntimeError(f"Gemini API error ({response.status_code}): {err_msg}")

            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                candidates = data.get("candidates", [])
                if not candidates:
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                for p in parts:
                    txt = p.get("text", "")
                    is_thought = p.get("thought", False)
                    if is_thought:
                        thought_chunks.append(txt)
                        sys.stdout.write(txt)
                        sys.stdout.flush()
                    else:
                        if in_thinking and thought_chunks:
                            in_thinking = False
                            print("\n=== [End Gemini Reasoning - Begin Markdown Output] ===\n", flush=True)
                        content_chunks.append(txt)
                        sys.stdout.write(txt)
                        sys.stdout.flush()

            print()

    elapsed = time.time() - start_time
    content = "".join(content_chunks).strip()

    # Strip markdown fences if present
    if content.startswith("```markdown"):
        content = content[len("```markdown"):].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    return content, elapsed


def run_gemini_pipeline(
    input_path: Path | str,
    output_dir: Path | str,
    pages: list[int] | None = None,
    model_name: str = "gemini-3.8-flash",
    api_key: str | None = None,
    timeout_seconds: float = 60.0,
    dpi: int = 200,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Execute Pipeline C: Remote Gemini 3.8 Flash multimodal API across all or specified pages."""
    from ocr_pipeline_test.render import get_document_page_count
    from ocr_pipeline_test.output_utils import generate_output_path

    doc_file = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_pages = get_document_page_count(doc_file)
    target_pages = pages if pages is not None else list(range(total_pages))

    print(f"[Pipeline C] Starting Gemini 3.8 Flash transcription for {len(target_pages)} page(s) of {doc_file.name}...")
    page_contents: list[str] = []
    total_elapsed = 0.0

    for p in target_pages:
        content, elapsed = transcribe_page_gemini(
            input_path=doc_file,
            page_number=p,
            model_name=model_name,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            dpi=dpi,
        )
        total_elapsed += elapsed
        page_contents.append(content)

        # Save per-page output
        page_file = generate_output_path(out_dir, suffix=f"gemini38_page{p + 1}", timestamp=timestamp)
        page_file.write_text(content, encoding="utf-8")
        print(f"[Pipeline C] Page {p + 1} saved -> {page_file} ({elapsed:.1f}s)")

    # Combine all pages if multi-page
    full_output_file = generate_output_path(out_dir, suffix="gemini38_full", timestamp=timestamp)
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
