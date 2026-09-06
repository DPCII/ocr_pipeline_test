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
| **Spatial-Guided** | **Docling + PyMuPDF (+ Gemma OCR)** | Local (`spatial-ocr`) | Hybrid architecture: Docling layout detection, verbatim PyMuPDF extraction, Gemma 4 for image OCR. |
| **Pipeline B** | **IBM Docling** | Local (`docling`) | Layout segmentation, DocLayNet reading order, bounding box tracking, PDF/DOCX support. |
| **Pipeline C** | **Gemini 3.8 Flash** | Remote (HTTP API) | Frontier multimodal vision accuracy, rich formatting, zero local compute overhead. |
| **Pipeline C2** | **Muse Spark 1.3** | Remote (Meta API) | Frontier multimodal vision with live reasoning traces via Meta API. |

---

## Quickstart & Replay

### 1. Requirements & Environment
The project is managed via `mise` and `uv`:
```zsh
# Verify tools
uv --version

# Install dependencies (already pinned in uv.lock)
uv sync
```

Ensure `.env` contains your API keys:
```bash
GEMINI_API_KEY=your_key_here
MUSE_API_KEY=your_key_here
```

### 2. Running Pipelines

#### Spatial-Guided Hybrid Extraction:
```zsh
# Run spatial-guided OCR (Docling layout + PyMuPDF digital text + Gemma image OCR)
uv run spatial-ocr --input multipage_newsletter.pdf

# Run on MS Word or other document formats
uv run spatial-ocr --input multipage_newsletter.docx

# Quiet mode (suppress live thought streaming from VLMs)
uv run spatial-ocr --no-thinking
```

#### Benchmark Comparison Testbed:
```zsh
# Run all benchmark pipelines on PDF, DOCX, etc.
uv run ocr-pipeline-test --pipeline all --input multipage_newsletter.docx

# Run individual pipelines:
uv run ocr-pipeline-test --pipeline docling --input multipage_newsletter.docx
uv run ocr-pipeline-test --pipeline gemini --input multipage_newsletter.pdf
uv run ocr-pipeline-test --pipeline muse --input multipage_newsletter.pdf

# Re-run evaluation on existing outputs
uv run ocr-pipeline-test --pipeline evaluate
```

### 3. Output Artifacts
All generated Markdown files are saved to `outputs/`:
- `outputs/output_<timestamp>.md` (Spatial-guided output)
- `outputs/output_<timestamp>_layout.json` (Spatial layout schema)
- `outputs/output_<timestamp>_docling_page1.md`
- `outputs/output_<timestamp>_gemini38_page1.md`
- `outputs/output_<timestamp>_muse_page1.md`
