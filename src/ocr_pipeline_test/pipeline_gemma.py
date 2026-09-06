"""Pipeline A: Local Gemma 4:12b Vision via Ollama API."""

from pathlib import Path
import time
from typing import Any
import httpx
from ocr_pipeline_test.render import render_page_to_base64

PROMPT_TEMPLATE = """You are a state-of-the-art document layout intelligence and transcription engine.
Analyze the visual layout of this document page and convert its full content into clean, semantic Markdown.

CRITICAL LAYOUT & STRUCTURAL REQUIREMENTS:
1. Multi-Column Reading Order: Do NOT merge text horizontally across columns. For multi-column areas, transcribe top-to-bottom within column 1 before proceeding to column 2.
2. Section & Bounding Box Isolation: Keep visually distinct sections (e.g. sidebars, callouts, headers, footers) strictly isolated in their own Markdown sections.
3. Hierarchy: Use appropriate markdown headers (# for main banner/title, ## for section headlines, ### for sub-headers).
4. Fidelity: Transcribe every text element accurately. Do not summarize or skip text.
5. Format: Return ONLY raw Markdown in your final answer. Do not include markdown code block backticks (```markdown), preambles, or conversational commentary.
"""


def transcribe_page_gemma(
    pdf_path: Path | str,
    page_number: int,
    model_name: str = "gemma4:12b",
    ollama_url: str = "http://localhost:11434",
    timeout_seconds: float = 900.0,
    dpi: int = 150,
    show_thinking: bool = False,
) -> tuple[str, str, float]:
    """Transcribe a single page using Gemma 4 vision. Returns (content, thinking, elapsed_seconds)."""
    import json
    import re
    import sys

    print(f"\n[Pipeline A] Rendering page {page_number + 1} of {pdf_path} at {dpi} DPI...")
    img_b64 = render_page_to_base64(pdf_path, page_number=page_number, dpi=dpi)

    print(f"[Pipeline A] Sending vision request to Ollama ({model_name}) with 32k context...")
    start_time = time.time()

    from ocr_pipeline_test.gemma_tuning import GemmaThoughtWatchdog, clean_gemma_chat_history

    messages = [
        {
            "role": "user",
            "content": PROMPT_TEMPLATE,
            "images": [img_b64],
        }
    ]
    sanitized_messages = clean_gemma_chat_history(messages)

    payload = {
        "model": model_name,
        "messages": sanitized_messages,
        "stream": True,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 40,
            "repeat_penalty": 1.18,
            "num_ctx": 32768,
            "num_predict": 4096,  # Cap generation horizon to mitigate infinite self-doubt
        },
    }

    watchdog = GemmaThoughtWatchdog(max_thought_tokens=2500, warning_threshold=2000)
    thought_chunks: list[str] = []
    content_chunks: list[str] = []
    in_thinking_phase = True

    timeout = httpx.Timeout(timeout_seconds, connect=30.0)
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", f"{ollama_url}/api/chat", json=payload) as response:
            response.raise_for_status()
            if show_thinking:
                print("\n=== [Gemma 4 Live Reasoning Trace] ===", flush=True)
            else:
                print("[Pipeline A] Model thinking (progress: T per 25 tokens): ", end="", flush=True)

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = data.get("message", {})
                thought = msg.get("thinking", "")
                content_part = msg.get("content", "")

                if thought:
                    watchdog.record_thought_chunk(thought)
                    thought_chunks.append(thought)
                    if show_thinking:
                        sys.stdout.write(thought)
                        sys.stdout.flush()
                    else:
                        if len(thought_chunks) % 25 == 0:
                            print("T", end="", flush=True)

                if content_part:
                    watchdog.record_content_chunk(content_part)
                    if in_thinking_phase:
                        in_thinking_phase = False
                        if show_thinking:
                            print("\n=== [End Reasoning Trace] ===\n", flush=True)
                        print("\n[Pipeline A] Generating markdown content: ", end="", flush=True)
                    content_chunks.append(content_part)
                    if len(content_chunks) % 25 == 0:
                        print(".", end="", flush=True)

                if data.get("done", False):
                    break
            print(" done.")

    elapsed = time.time() - start_time
    content = "".join(content_chunks).strip()
    thinking = "".join(thought_chunks).strip()

    # Fallback: if model was truncated or placed markdown in thinking
    if not content and thinking:
        print("[Pipeline A Warning] Content was empty; extracting from thinking trace...")
        match = re.search(r"(#\s+Today['’]s News.*)", thinking, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            content = thinking.strip()

    # Clean any accidental outer markdown fences if present
    if content.startswith("```markdown"):
        content = content[len("```markdown"):].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    return content, thinking, elapsed


def run_gemma_pipeline(
    pdf_path: Path | str,
    output_dir: Path | str,
    pages: list[int] | None = None,
    model_name: str = "gemma4:12b",
    ollama_url: str = "http://localhost:11434",
    timeout_seconds: float = 900.0,
    dpi: int = 150,
    show_thinking: bool = False,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Execute Pipeline A: Local Gemma 4:12b vision across all or specified pages."""
    from ocr_pipeline_test.render import get_pdf_page_count
    from ocr_pipeline_test.output_utils import generate_output_path

    pdf_file = Path(pdf_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_pages = get_pdf_page_count(pdf_file)
    target_pages = pages if pages is not None else list(range(total_pages))

    print(f"[Pipeline A] Starting Gemma 4 transcription for {len(target_pages)} page(s) of {pdf_file.name}...")
    page_contents: list[str] = []
    total_elapsed = 0.0

    for p in target_pages:
        content, thinking, elapsed = transcribe_page_gemma(
            pdf_path=pdf_file,
            page_number=p,
            model_name=model_name,
            ollama_url=ollama_url,
            timeout_seconds=timeout_seconds,
            dpi=dpi,
            show_thinking=show_thinking,
        )
        total_elapsed += elapsed
        page_contents.append(content)

        # Save per-page output
        page_file = generate_output_path(out_dir, suffix=f"gemma4_page{p + 1}", timestamp=timestamp)
        page_file.write_text(content, encoding="utf-8")
        print(f"[Pipeline A] Page {p + 1} saved -> {page_file} ({elapsed:.1f}s)")

        # Save per-page reasoning trace
        if thinking:
            thought_file = generate_output_path(out_dir, suffix=f"gemma4_page{p + 1}_thinking", timestamp=timestamp)
            thought_file.write_text(thinking, encoding="utf-8")

    # Combine all pages if multi-page
    full_output_file = generate_output_path(out_dir, suffix="gemma4_full", timestamp=timestamp)
    merged_markdown = "\n\n---\n\n".join(
        f"<!-- Page {p + 1} -->\n\n{text}" for p, text in zip(target_pages, page_contents)
    )
    full_output_file.write_text(merged_markdown, encoding="utf-8")
    print(f"[Pipeline A] Full document saved -> {full_output_file} (Total time: {total_elapsed:.1f}s)")

    return {
        "pipeline": "Pipeline A (Local Gemma 4:12b)",
        "output_file": str(full_output_file),
        "pages_processed": len(target_pages),
        "total_elapsed_seconds": total_elapsed,
    }



