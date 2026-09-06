"""Pipeline C2: Remote Frontier Multimodal API (Muse Spark 1.3) via Meta API."""

import json
import os
from pathlib import Path
import time
from typing import Any
from dotenv import load_dotenv
import httpx
from ocr_pipeline_test.render import render_page_to_base64

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


def transcribe_page_muse(
    pdf_path: Path | str,
    page_number: int,
    model_name: str = "muse-spark-1.3-contributor",
    base_url: str = "https://api.meta.ai/v1",
    api_key: str | None = None,
    timeout_seconds: float = 60.0,
    dpi: int = 200,
) -> tuple[str, float]:
    """Transcribe a single page using Muse Spark 1.3 via Meta API over HTTP."""
    if model_name.lower() in ("muse", "auto", ""):
        model_name = "muse-spark-1.3-contributor"

    key = api_key or os.getenv("MUSE_API_KEY") or os.getenv("MODEL_API_KEY")
    if not key:
        raise ValueError("MUSE_API_KEY (or MODEL_API_KEY) is not set in environment or .env file.")

    print(f"\n[Pipeline Muse] Rendering page {page_number + 1} of {pdf_path} at {dpi} DPI...")
    img_b64 = render_page_to_base64(pdf_path, page_number=page_number, dpi=dpi)

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_TEMPLATE,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                        },
                    },
                ],
            }
        ],
        "temperature": 0.1,
        "stream": True,
    }

    url = f"{base_url.rstrip('/')}/chat/completions"
    print(f"[Pipeline Muse] Streaming reasoning & transcription from {model_name}...")
    start_time = time.time()

    thought_chunks: list[str] = []
    content_chunks: list[str] = []
    in_thinking = True
    import sys

    timeout = httpx.Timeout(timeout_seconds, connect=30.0)
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, json=payload, headers=headers) as response:
            if response.is_error:
                response.read()
                try:
                    err_data = response.json()
                    err_msg = err_data.get("error", {}).get("message", response.text)
                except Exception:
                    err_msg = response.text
                raise RuntimeError(f"Meta API error ({response.status_code}): {err_msg}")

            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                thought = delta.get("reasoning_content") or delta.get("thinking") or delta.get("reasoning")
                content_part = delta.get("content", "")

                if thought:
                    thought_chunks.append(thought)
                    sys.stdout.write(thought)
                    sys.stdout.flush()

                if content_part:
                    if in_thinking and thought_chunks:
                        in_thinking = False
                        print("\n=== [End Muse Reasoning - Begin Markdown Output] ===\n", flush=True)
                    content_chunks.append(content_part)
                    sys.stdout.write(content_part)
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


def run_muse_pipeline(
    pdf_path: Path | str,
    output_dir: Path | str,
    pages: list[int] | None = None,
    model_name: str = "muse-spark-1.3-contributor",
    base_url: str = "https://api.meta.ai/v1",
    api_key: str | None = None,
    timeout_seconds: float = 60.0,
    dpi: int = 200,
) -> dict[str, Any]:
    """Execute Pipeline: Remote Muse Spark 1.3 multimodal API across all or specified pages."""
    from ocr_pipeline_test.render import get_pdf_page_count

    pdf_file = Path(pdf_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_name.lower() in ("muse", "auto", ""):
        model_name = "muse-spark-1.3-contributor"

    total_pages = get_pdf_page_count(pdf_file)
    target_pages = pages if pages is not None else list(range(total_pages))

    print(f"[Pipeline Muse] Starting Muse Spark 1.3 transcription for {len(target_pages)} page(s) of {pdf_file.name}...")
    page_contents: list[str] = []
    total_elapsed = 0.0

    for p in target_pages:
        content, elapsed = transcribe_page_muse(
            pdf_path=pdf_file,
            page_number=p,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            dpi=dpi,
        )
        total_elapsed += elapsed
        page_contents.append(content)

        page_file = out_dir / f"pipeline_c2_muse_page{p + 1}.md"
        page_file.write_text(content, encoding="utf-8")
        print(f"[Pipeline Muse] Page {p + 1} saved -> {page_file} ({elapsed:.1f}s)")

    full_output_file = out_dir / "pipeline_c2_muse_full.md"
    merged_markdown = "\n\n---\n\n".join(
        f"<!-- Page {p + 1} -->\n\n{text}" for p, text in zip(target_pages, page_contents)
    )
    full_output_file.write_text(merged_markdown, encoding="utf-8")
    print(f"[Pipeline Muse] Full document saved -> {full_output_file} (Total time: {total_elapsed:.1f}s)")

    return {
        "pipeline": f"Pipeline C2 (Remote {model_name})",
        "output_file": str(full_output_file),
        "pages_processed": len(target_pages),
        "total_elapsed_seconds": total_elapsed,
    }
