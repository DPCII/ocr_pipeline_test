"""Stage 1: Architect - Visual Layout & Section Bounding Box Detection via VLM."""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import httpx

load_dotenv()


@dataclass
class SectionBox:
    """Represents a semantically labeled visual bounding box detected by the VLM."""
    category: str  # "Banner" | "Main Story" | "Sidebar" | "Footer" | "Graphic Caption"
    box_2d: list[int]  # [ymin, xmin, ymax, xmax] normalized on a 0-1000 scale
    thread_id: str  # Logical thread ID across columns/pages (e.g. "main_multipage_text", "sidebar_tips")
    order: int  # Sequential order within the thread
    label: str  # Short title or heading inside the box
    page_number: int  # 0-indexed page number


LAYOUT_DETECTION_PROMPT = """Analyze the visual layout of this document page.
Identify all distinct semantic section bounding boxes and their logical reading threads.

DECISION PROTOCOL:
1. Make no more than 3 visual passes to identify major blocks (Banner, Columns, Sidebars, Footer).
2. Once you determine the boxes, immediately emit the JSON and terminate.

For each section box, return:
1. "category": Choose one of ["Banner", "Main Story", "Sidebar", "Footer", "Caption"].
2. "box_2d": Bounding box coordinates [ymin, xmin, ymax, xmax] normalized on a 0 to 1000 scale.
   - ymin: top edge (0 = top of page, 1000 = bottom of page)
   - xmin: left edge (0 = left of page, 1000 = right of page)
   - ymax: bottom edge
   - xmax: right edge
3. "thread_id": Logical story identifier. If a section continues an article from another column or page (e.g. "main_multipage_text"), use the SAME thread_id.
4. "order": Reading order number within this thread (1, 2, ...).
5. "label": Brief description or heading title for this section.

Return ONLY a valid JSON object with the format:
{
  "sections": [
    {
      "category": "Banner",
      "box_2d": [ymin, xmin, ymax, xmax],
      "thread_id": "banner",
      "order": 1,
      "label": "Newsletter Title Banner"
    }
  ]
}
"""


def detect_sections_with_muse(
    img_b64: str,
    model_name: str = "muse-spark-1.3-contributor",
    base_url: str = "https://api.meta.ai/v1",
    api_key: str | None = None,
    timeout_seconds: float = 60.0,
) -> list[dict[str, Any]]:
    """Detect section bounding boxes using Muse Spark 1.3 via Meta API over HTTP."""
    if model_name.lower() in ("muse", "auto", ""):
        model_name = "muse-spark-1.3-contributor"

    key = api_key or os.getenv("MUSE_API_KEY") or os.getenv("MODEL_API_KEY")
    if not key:
        raise ValueError("MUSE_API_KEY (or MODEL_API_KEY) is not set in environment or .env")

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
                        "text": LAYOUT_DETECTION_PROMPT,
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
    }

    url = f"{base_url.rstrip('/')}/chat/completions"
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(url, json=payload, headers=headers)
        if response.is_error:
            try:
                err_data = response.json()
                err_msg = err_data.get("error", {}).get("message", response.text)
            except Exception:
                err_msg = response.text
            raise RuntimeError(f"Meta API error ({response.status_code}): {err_msg}")
        data = response.json()

    content = data["choices"][0]["message"]["content"].strip()
    return _parse_json_sections(content)


def detect_sections_with_ollama(
    img_b64: str,
    model_name: str = "gemma4:12b",
    ollama_url: str = "http://localhost:11434",
    timeout_seconds: float = 600.0,
    show_thinking: bool = False,
) -> list[dict[str, Any]]:
    """Detect section bounding boxes using a local Ollama vision model (e.g. Gemma 4:12b) with streaming."""
    import sys

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": LAYOUT_DETECTION_PROMPT,
                "images": [img_b64],
            }
        ],
        "format": "json",
        "stream": True,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 40,
            "repeat_penalty": 1.18,
            "num_ctx": 32768,
        },
    }

    thought_chunks: list[str] = []
    content_chunks: list[str] = []
    in_thinking_phase = True

    timeout = httpx.Timeout(timeout_seconds, connect=30.0)
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", f"{ollama_url}/api/chat", json=payload) as response:
            response.raise_for_status()
            if show_thinking:
                print("\n=== [Architect Live Layout Reasoning] ===", flush=True)
            else:
                print("[Architect] Thinking layout geometry (T per 25 tokens): ", end="", flush=True)

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
                    thought_chunks.append(thought)
                    if show_thinking:
                        sys.stdout.write(thought)
                        sys.stdout.flush()
                    else:
                        if len(thought_chunks) % 25 == 0:
                            print("T", end="", flush=True)

                if content_part:
                    if in_thinking_phase:
                        in_thinking_phase = False
                        if show_thinking:
                            print("\n=== [End Layout Reasoning] ===\n", flush=True)
                        print("\n[Architect] Generating bounding boxes JSON: ", end="", flush=True)
                    content_chunks.append(content_part)
                    if len(content_chunks) % 25 == 0:
                        print(".", end="", flush=True)

                if data.get("done", False):
                    break
            print(" done.")

    content = "".join(content_chunks).strip()
    return _parse_json_sections(content)


def detect_sections_with_gemini(
    img_b64: str,
    model_name: str = "gemini-3.8-flash",
    api_key: str | None = None,
    timeout_seconds: float = 60.0,
) -> list[dict[str, Any]]:
    """Detect section bounding boxes using Google Gemini 3.8 Flash via HTTP."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set in environment or .env")

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
                        "text": LAYOUT_DETECTION_PROMPT,
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    }

    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(url, json=payload)
        if response.is_error:
            try:
                err_data = response.json()
                err_msg = err_data.get("error", {}).get("message", response.text)
            except Exception:
                err_msg = response.text
            raise RuntimeError(f"Gemini API error ({response.status_code}): {err_msg}")
        data = response.json()

    content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    return _parse_json_sections(content)


def _parse_json_sections(raw_text: str) -> list[dict[str, Any]]:
    """Parse JSON string into a list of section dictionaries."""
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and "sections" in parsed:
            return parsed["sections"]
        if isinstance(parsed, list):
            return parsed
        return []
    except json.JSONDecodeError:
        # Fallback regex search for JSON array or object
        match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        if match:
            try:
                res = json.loads(match.group(1))
                return res.get("sections", []) if isinstance(res, dict) else res
            except Exception:
                pass
        return []


def detect_page_sections(
    img_b64: str,
    page_number: int,
    model_name: str = "gemma4:12b",
    backend: str = "ollama",
    show_thinking: bool = False,
) -> list[SectionBox]:
    """Unified entrypoint to detect layout bounding boxes for a single page."""
    if backend.lower() == "muse" and model_name.lower() in ("muse", "auto", ""):
        model_name = "muse-spark-1.3-contributor"
    elif backend.lower() == "gemini" and model_name.lower() in ("gemini", "auto", ""):
        model_name = "gemini-3.8-flash"
    elif model_name.lower() == "muse":
        model_name = "muse-spark-1.3-contributor"
        backend = "muse"

    print(f"[Architect] Detecting visual sections on Page {page_number + 1} with {model_name} ({backend})...")
    if backend.lower() == "gemini" or "gemini" in model_name.lower():
        raw_sections = detect_sections_with_gemini(img_b64, model_name=model_name)
    elif backend.lower() == "muse" or "muse" in model_name.lower():
        raw_sections = detect_sections_with_muse(img_b64, model_name=model_name)
    else:
        raw_sections = detect_sections_with_ollama(
            img_b64,
            model_name=model_name,
            show_thinking=show_thinking,
        )

    boxes: list[SectionBox] = []
    for s in raw_sections:
        category = s.get("category", "General Content")
        box_2d = s.get("box_2d", [0, 0, 1000, 1000])
        thread_id = s.get("thread_id", f"section_{len(boxes)}")
        order = s.get("order", 1)
        label = s.get("label", category)

        # Validate box_2d bounds
        if isinstance(box_2d, list) and len(box_2d) == 4:
            boxes.append(
                SectionBox(
                    category=category,
                    box_2d=[int(c) for c in box_2d],
                    thread_id=thread_id,
                    order=int(order),
                    label=label,
                    page_number=page_number,
                )
            )

    print(f"[Architect] Identified {len(boxes)} semantic section bounding boxes on Page {page_number + 1}.")
    return boxes
