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
3. "thread_id": Logical story/article identifier. If a section continues content from another column or page, use the SAME thread_id.
4. "order": Reading order number within this thread (1, 2, ...).
5. "label": Brief description or heading title for this section.

Return ONLY a valid JSON object with the format:
{
  "sections": [
    {
      "category": "Banner",
      "box_2d": [ymin, xmin, ymax, xmax],
      "thread_id": "header_thread",
      "order": 1,
      "label": "Document Header / Title"
    }
  ]
}
"""


def detect_sections_with_muse(
    img_b64: str,
    model_name: str = "muse-spark-1.3-contributor",
    base_url: str = "https://api.meta.ai/v1",
    api_key: str | None = None,
    timeout_seconds: float = 90.0,
    show_thinking: bool = True,
) -> list[dict[str, Any]]:
    """Detect section bounding boxes using Muse Spark 1.3 via Meta API with streaming reasoning/content."""
    import sys

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
        "stream": True,
    }

    url = f"{base_url.rstrip('/')}/chat/completions"
    thought_chunks: list[str] = []
    content_chunks: list[str] = []
    in_thinking_phase = True

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

            if show_thinking:
                print(f"\n=== [Architect Live Layout Reasoning ({model_name})] ===", flush=True)
            else:
                print(f"[Architect] Thinking layout ({model_name}): ", end="", flush=True)

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
                    if show_thinking:
                        sys.stdout.write(thought)
                        sys.stdout.flush()
                    else:
                        if len(thought_chunks) % 20 == 0:
                            print("T", end="", flush=True)

                if content_part:
                    if in_thinking_phase and thought_chunks:
                        in_thinking_phase = False
                        if show_thinking:
                            print("\n=== [End Layout Reasoning] ===\n", flush=True)
                        print("[Architect] Generating bounding boxes JSON: ", end="", flush=True)
                    content_chunks.append(content_part)
                    if show_thinking:
                        sys.stdout.write(content_part)
                        sys.stdout.flush()
                    else:
                        if len(content_chunks) % 20 == 0:
                            print(".", end="", flush=True)

            print()

    content = "".join(content_chunks).strip()
    return _parse_json_sections(content)



def detect_sections_with_gemini(
    img_b64: str,
    model_name: str = "gemini-3.8-flash",
    api_key: str | None = None,
    timeout_seconds: float = 90.0,
    show_thinking: bool = True,
) -> list[dict[str, Any]]:
    """Detect section bounding boxes using Google Gemini 3.8 Flash via SSE streaming."""
    import sys

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set in environment or .env")

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

    thought_chunks: list[str] = []
    content_chunks: list[str] = []
    in_thinking_phase = True

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

            if show_thinking:
                print(f"\n=== [Architect Live Layout Reasoning ({model_name})] ===", flush=True)
            else:
                print(f"[Architect] Thinking layout ({model_name}): ", end="", flush=True)

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
                        if show_thinking:
                            sys.stdout.write(txt)
                            sys.stdout.flush()
                        else:
                            if len(thought_chunks) % 20 == 0:
                                print("T", end="", flush=True)
                    else:
                        if in_thinking_phase and thought_chunks:
                            in_thinking_phase = False
                            if show_thinking:
                                print("\n=== [End Layout Reasoning] ===\n", flush=True)
                            print("[Architect] Generating bounding boxes JSON: ", end="", flush=True)
                        content_chunks.append(txt)
                        if show_thinking:
                            sys.stdout.write(txt)
                            sys.stdout.flush()
                        else:
                            if len(content_chunks) % 20 == 0:
                                print(".", end="", flush=True)

            print()

    content = "".join(content_chunks).strip()
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


# Cache Docling document to avoid re-converting the same PDF for each page
_DOCLING_CACHE: dict[str, Any] = {}


def detect_sections_with_docling(
    input_path: Path | str,
    page_number: int,
) -> list[dict[str, Any]]:
    """Detect section bounding boxes using IBM Docling layout analysis engine."""
    from docling.document_converter import DocumentConverter

    doc_key = str(Path(input_path).resolve())
    if doc_key not in _DOCLING_CACHE:
        print(f"[Architect] Initializing IBM Docling layout engine for {Path(input_path).name}...", flush=True)
        converter = DocumentConverter()
        _DOCLING_CACHE[doc_key] = converter.convert(doc_key).document

    doc = _DOCLING_CACHE[doc_key]
    doc_page_no = page_number + 1
    if doc_page_no not in doc.pages:
        # Fallback to PyMuPDF visual layout block analysis for document formats
        # without native Docling page geometry (e.g., DOCX, PPTX reflowable pages)
        import pymupdf

        try:
            p_doc = pymupdf.open(str(input_path))
            if page_number < len(p_doc):
                page = p_doc[page_number]
                pw, ph = page.rect.width, page.rect.height
                raw_blocks = page.get_text("blocks")
                p_doc.close()

                result: list[dict[str, Any]] = []
                section_counter = 0
                current_thread_id = f"section_p{doc_page_no}"
                current_section_label = "General Content"
                thread_order_counters: dict[str, int] = {}

                for b in raw_blocks:
                    if len(b) >= 7:
                        x0, y0, x1, y1, text, _block_no, block_type = b[:7]
                        text = str(text).strip()
                        if not text and block_type == 0:
                            continue
                        ymin = max(0, min(1000, int((y0 / ph) * 1000)))
                        ymax = max(0, min(1000, int((y1 / ph) * 1000)))
                        xmin = max(0, min(1000, int((x0 / pw) * 1000)))
                        xmax = max(0, min(1000, int((x1 / pw) * 1000)))
                        if ymin >= ymax or xmin >= xmax:
                            continue

                        if block_type == 1:
                            category = "Graphic"
                            tid = f"figure_p{doc_page_no}_{len(result) + 1}"
                            lbl = "Figure / Image"
                        elif len(text.splitlines()) == 1 and len(text) < 60 and not text.endswith((".", ",")):
                            category = "Section Header"
                            section_counter += 1
                            current_thread_id = f"section_{doc_page_no}_{section_counter}"
                            current_section_label = text
                            tid = current_thread_id
                            lbl = text
                        else:
                            category = "Main Story"
                            tid = current_thread_id
                            lbl = current_section_label

                        thread_order_counters[tid] = thread_order_counters.get(tid, 0) + 1
                        result.append({
                            "category": category,
                            "box_2d": [ymin, xmin, ymax, xmax],
                            "thread_id": tid,
                            "order": thread_order_counters[tid],
                            "label": lbl,
                        })
                return result
        except Exception as exc:
            print(f"[Architect] Docling layout fallback error: {exc}", flush=True)
        return []

    page = doc.pages[doc_page_no]
    pw, ph = page.size.width, page.size.height

    p_items = [
        it for it, _ in doc.iterate_items()
        if getattr(it, "prov", None) and it.prov[0].page_no == doc_page_no
    ]

    result: list[dict[str, Any]] = []
    section_counter = 0
    current_thread_id = "intro"
    current_section_label = "General Content"
    thread_order_counters: dict[str, int] = {}

    for it in p_items:
        b = it.prov[0].bbox
        ymin = max(0, min(1000, int(((ph - b.t) / ph) * 1000)))
        ymax = max(0, min(1000, int(((ph - b.b) / ph) * 1000)))
        xmin = max(0, min(1000, int((b.l / pw) * 1000)))
        xmax = max(0, min(1000, int((b.r / pw) * 1000)))

        # Ensure valid coordinates
        if ymin >= ymax or xmin >= xmax:
            continue

        raw_label = str(getattr(it, "label", "text")).lower()
        item_text = getattr(it, "text", "").strip()

        # Semantic classification based entirely on Docling's native layout model
        if "title" in raw_label or "page_header" in raw_label:
            category = "Banner"
            tid = f"header_p{doc_page_no}"
            lbl = item_text if item_text else "Document Header / Banner"
        elif "page_footer" in raw_label:
            category = "Footer"
            tid = f"footer_p{doc_page_no}"
            lbl = item_text if item_text else "Page Footer"
        elif "section_header" in raw_label:
            category = "Section Header"
            section_counter += 1
            current_thread_id = f"section_{doc_page_no}_{section_counter}"
            current_section_label = item_text if item_text else f"Section {section_counter}"
            tid = current_thread_id
            lbl = current_section_label
        elif "list_item" in raw_label:
            category = "List"
            tid = current_thread_id
            lbl = current_section_label
        elif "table" in raw_label:
            category = "Table"
            tid = current_thread_id
            lbl = f"Table in {current_section_label}"
        elif "picture" in raw_label or "chart" in raw_label:
            category = "Graphic"
            tid = f"figure_p{doc_page_no}_{len(result) + 1}"
            lbl = "Figure / Image"
        elif "caption" in raw_label:
            category = "Caption"
            tid = current_thread_id
            lbl = "Caption"
        elif "footnote" in raw_label:
            category = "Footnote"
            tid = f"footnote_p{doc_page_no}"
            lbl = "Footnote"
        else:
            category = "Main Story"
            tid = current_thread_id
            lbl = current_section_label

        thread_order_counters[tid] = thread_order_counters.get(tid, 0) + 1

        result.append({
            "category": category,
            "box_2d": [ymin, xmin, ymax, xmax],
            "thread_id": tid,
            "order": thread_order_counters[tid],
            "label": lbl,
        })

    return result


def detect_page_sections(
    img_b64: str,
    page_number: int,
    model_name: str = "docling-layout",
    backend: str = "docling",
    show_thinking: bool = False,
    input_path: Path | str | None = None,
    pdf_path: Path | str | None = None,
) -> list[SectionBox]:
    """Unified entrypoint to detect layout bounding boxes for a single page."""
    doc_input = input_path or pdf_path
    if backend.lower() in ("docling", "docling-layout", "local") or model_name.lower() in ("docling", "docling-layout"):
        backend = "docling"
        model_name = "docling-layout"
    elif backend.lower() in ("gemini", "google") or "gemini" in model_name.lower():
        backend = "gemini"
        if model_name.lower() in ("gemini", "auto", ""):
            model_name = "gemini-3.8-flash"
    elif backend.lower() in ("muse", "meta") or "muse" in model_name.lower():
        backend = "muse"
        if model_name.lower() in ("muse", "auto", ""):
            model_name = "muse-spark-1.3-contributor"
    else:
        raise ValueError(
            f"Unsupported layout backend: '{backend}'. "
            f"Supported layout engines: 'docling' (local), 'gemini' (remote VLM), 'muse' (remote VLM)."
        )

    print(f"[Architect] Detecting visual sections on Page {page_number + 1} with {model_name} ({backend})...")
    if backend == "docling":
        if not doc_input:
            raise ValueError("input_path is required for Docling layout detection")
        raw_sections = detect_sections_with_docling(doc_input, page_number)
    elif backend == "gemini":
        raw_sections = detect_sections_with_gemini(img_b64, model_name=model_name, show_thinking=show_thinking)
    elif backend == "muse":
        raw_sections = detect_sections_with_muse(img_b64, model_name=model_name, show_thinking=show_thinking)

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
