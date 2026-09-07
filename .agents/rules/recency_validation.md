# Workspace Rules: System Clock & Recency Validation

## 1. System Clock & Temporal Context Verification
- **System Clock Authority**: Always anchor reasoning to the current system clock / execution timestamp provided in the environment (Current year: 2026).
- **Recency Check on Output**: Before generating any recommendations, technical architectures, model suggestions, or benchmark plans, run a validation check against the current date to ensure that all referenced technologies, APIs, and model families are state-of-the-art and actively maintained.

## 2. Model Family & Technology Recency Guidelines
- **Remote Frontier Vision & Document Intelligence**:
  - **Strictly Prohibited**: Never reference obsolete generations (e.g., Gemini 1.0 / 1.5, GPT-3.5, Claude 1/2) as current or recommended systems.
- **Local Multimodal & Open Weights Models**:
  - Reference current generation models.
- **Document OCR & Layout Intelligence**:
  - **Strictly Prohibited**: Never recommend or use legacy OCR tools like Tesseract.
  - Rely exclusively on modern layout intelligence frameworks or modern multimodal vision LLMs (VLMs) that preserve reading order, bounding boxes, and column hierarchy.

## 3. Pre-Output Validation Checklist
Before returning a response to the user:
1. Did I verify the current system date/year?
2. Are all proposed model names and tool versions current?
3. Are any legacy or deprecated technologies accidentally referenced? If so, replace them with current alternatives immediately.
