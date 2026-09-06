# Workspace Rules: System Clock & Recency Validation

## 1. System Clock & Temporal Context Verification
- **System Clock Authority**: Always anchor reasoning to the current system clock / execution timestamp provided in the environment (Current year: 2026).
- **Recency Check on Output**: Before generating any recommendations, technical architectures, model suggestions, or benchmark plans, run a validation check against the current date to ensure that all referenced technologies, APIs, and model families are state-of-the-art and actively maintained.

## 2. Model Family & Technology Recency Guidelines
- **Remote Frontier Vision & Document Intelligence**:
  - Reference current generation models: **Gemini 3.8 Flash** and **Gemini 3 Pro**.
  - **Strictly Prohibited**: Never reference obsolete generations (e.g., Gemini 1.0 / 1.5, GPT-3.5, Claude 1/2) as current or recommended systems.
- **Local Multimodal & Open Weights Models**:
  - Reference current generation open-source models: **Gemma 4** (e.g., `gemma4:12b` with native vision/multimodal capabilities) and current high-efficiency vision-language models.
- **Document OCR & Layout Intelligence**:
  - **Strictly Prohibited**: Never recommend or use legacy OCR tools like Tesseract.
  - Rely exclusively on modern layout intelligence frameworks (e.g., IBM Docling, Marker) or modern multimodal vision LLMs (Gemma 4 vision, Gemini 3.8 Flash) that preserve reading order, bounding boxes, and column hierarchy.

## 3. Pre-Output Validation Checklist
Before returning a response to the user:
1. Did I verify the current system date/year?
2. Are all proposed model names and tool versions current (2026 standard)?
3. Are any legacy or deprecated technologies accidentally referenced? If so, replace them with current alternatives immediately.

## 4. Reasoning & Token Streaming Observability
- **Universal Streaming Requirement**: All model executions—both local models (Gemma 4 via Ollama) and remote APIs (Gemini 3.8 Flash, Muse Spark 1.3)—must implement real-time streaming of thinking/reasoning logic and generated tokens to stdout.
- **Logic Flow Transparency**: Never run remote or local model invocations as silent blocking HTTP calls. Expose live reasoning/thought deltas and generation progress so the user can observe the reasoning process and verify progress continuously.
