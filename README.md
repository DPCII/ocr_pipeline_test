# Document Intelligence & OCR Pipeline Benchmark

A reproducible, high-accuracy document intelligence testbed on macOS (Apple Silicon M3, 18GB unified memory) comparing local vision models, dedicated layout engines, and frontier multimodal APIs.

## Objectives
- Accurately convert complex multi-column documents (e.g. `multipage_newsletter.pdf` and `multipage_newsletter.docx`) into semantic Markdown.
- Strict reading order preservation (column 1 followed by column 2, no cross-column text bleed).
- Section bounding box isolation (keeping sidebars, banners, and article bodies isolated).
- Zero reliance on legacy OCR tools (e.g., Tesseract).

---

## Pipelines

| Pipeline | Model / Tool | Execution | Key Strengths |
| :--- | :--- | :--- | :--- |
| **Pipeline A** | **Gemma 4:12b Vision** | Local (Ollama) | 100% private, native vision/CLIP projector, fits within 18GB unified RAM. |
| **Pipeline B** | **IBM Docling** | Local (`docling`) | Layout segmentation, DocLayNet reading order, bounding box tracking, PDF/DOCX support. |
| **Pipeline C** | **Gemini 3.8 Flash** | Remote (HTTP API) | Frontier multimodal vision accuracy, rich formatting, zero local compute overhead. |

---

## Quickstart & Replay

### 1. Requirements & Environment
The project is managed via `mise` and `uv`:
```zsh
# Verify tools
uv --version
pixi --version

# Install dependencies (already pinned in uv.lock)
uv sync
```

Ensure `.env` contains your Gemini API key:
```bash
GEMINI_API_KEY=your_key_here
```

### 2. Running Pipelines

To execute all three pipelines and run the evaluation:
```zsh
uv run ocr-pipeline-test --pipeline all
```

To run individual pipelines:
```zsh
# Run local Gemma 4:12b Vision
uv run ocr-pipeline-test --pipeline gemma

# Run local IBM Docling
uv run ocr-pipeline-test --pipeline docling

# Run remote Gemini 3.8 Flash
uv run ocr-pipeline-test --pipeline gemini

# Re-run evaluation on existing outputs
uv run ocr-pipeline-test --pipeline evaluate
```

### 3. Output Artifacts
All generated Markdown files are saved to `outputs/`:
- `outputs/pipeline_a_gemma4_page1.md`
- `outputs/pipeline_b_docling_page1.md`
- `outputs/pipeline_c_gemini38_page1.md`
