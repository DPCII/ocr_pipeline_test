"""Pipeline C: Remote Frontier Multimodal API (Gemini 3.8 Flash) via HTTP."""

import os
from pathlib import Path
import time
from dotenv import load_dotenv
import httpx
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


def run_gemini_pipeline(
    pdf_path: Path | str,
    output_path: Path | str,
    page_number: int = 0,
    model_name: str = "gemini-3.8-flash",
    api_key: str | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, str | float]:
    """Execute Pipeline C: Remote Gemini 3.8 Flash multimodal API over HTTP."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")

    print(f"[Pipeline C] Rendering page {page_number + 1} of {pdf_path}...")
    img_b64 = render_page_to_base64(pdf_path, page_number=page_number, dpi=200)

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

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(content, encoding="utf-8")

    print(f"[Pipeline C] Completed in {elapsed:.2f}s -> Saved to {out_file}")
    return {
        "pipeline": f"Pipeline C (Remote {model_name})",
        "output_file": str(out_file),
        "elapsed_seconds": elapsed,
        "content": content,
    }
