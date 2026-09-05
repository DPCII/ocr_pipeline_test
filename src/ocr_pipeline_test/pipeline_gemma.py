"""Pipeline A: Local Gemma 4:12b Vision via Ollama API."""

from pathlib import Path
import time
import httpx
from ocr_pipeline_test.render import render_page_to_base64

PROMPT_TEMPLATE = """You are a state-of-the-art document layout intelligence and transcription engine.
Analyze the visual layout of this document page and convert its full content into clean, semantic Markdown.

CRITICAL LAYOUT & STRUCTURAL REQUIREMENTS:
1. Multi-Column Reading Order: Do NOT merge text horizontally across columns. For multi-column areas, transcribe top-to-bottom within column 1 before proceeding to column 2.
2. Section & Bounding Box Isolation: Keep visually distinct sections (e.g. sidebars, callouts, headers, footers) strictly isolated in their own Markdown sections.
3. Hierarchy: Use appropriate markdown headers (# for main banner/title, ## for section headlines, ### for sub-headers).
4. Fidelity: Transcribe every text element accurately. Do not summarize or skip text.
5. Format: Return ONLY raw Markdown. Do not include markdown code block backticks (```markdown), preambles, or conversational commentary.
"""


def run_gemma_pipeline(
    pdf_path: Path | str,
    output_path: Path | str,
    page_number: int = 0,
    model_name: str = "gemma4:12b",
    ollama_url: str = "http://localhost:11434",
    timeout_seconds: float = 120.0,
) -> dict[str, str | float]:
    """Execute Pipeline A: Local Gemma 4:12b vision via Ollama HTTP API."""
    print(f"[Pipeline A] Rendering page {page_number + 1} of {pdf_path}...")
    img_b64 = render_page_to_base64(pdf_path, page_number=page_number, dpi=200)

    print(f"[Pipeline A] Sending vision request to Ollama ({model_name})...")
    start_time = time.time()

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE,
                "images": [img_b64],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(f"{ollama_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

    elapsed = time.time() - start_time
    content = data.get("message", {}).get("content", "").strip()

    # Clean any accidental outer markdown fences if present
    if content.startswith("```markdown"):
        content = content[len("```markdown"):].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(content, encoding="utf-8")

    print(f"[Pipeline A] Completed in {elapsed:.2f}s -> Saved to {out_file}")
    return {
        "pipeline": "Pipeline A (Local Gemma 4:12b)",
        "output_file": str(out_file),
        "elapsed_seconds": elapsed,
        "content": content,
    }
