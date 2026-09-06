"""Output path and naming utilities for OCR pipelines."""

from datetime import datetime
from pathlib import Path


def get_timestamp_str(dt: datetime | None = None) -> str:
    """Return the current timestamp in 'ddMMMyyyy-hhmm-ss' format (e.g. '06Sep2026-1638-56')."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%d%b%Y-%H%M-%S")


def generate_output_path(
    output_dir: Path | str = "outputs",
    suffix: str = "",
    extension: str = ".md",
    timestamp: str | None = None,
) -> Path:
    """Generate an output Path following the 'output_<ddMMMyyyy-hhmm-ss>[_suffix].ext' pattern.

    Examples:
        generate_output_path("outputs")
        -> Path("outputs/output_06Sep2026-1638-56.md")

        generate_output_path("outputs", suffix="layout", extension=".json")
        -> Path("outputs/output_06Sep2026-1638-56_layout.json")

        generate_output_path("outputs", suffix="gemma4_page1", extension=".md")
        -> Path("outputs/output_06Sep2026-1638-56_gemma4_page1.md")
    """
    ts = timestamp or get_timestamp_str()
    clean_suffix = f"_{suffix.lstrip('_')}" if suffix else ""
    ext = extension if extension.startswith(".") else f".{extension}"
    filename = f"output_{ts}{clean_suffix}{ext}"
    out_dir = Path(output_dir)
    return out_dir / filename


def resolve_output_path(
    output_arg: str | Path | None,
    default_dir: Path | str = "outputs",
    suffix: str = "",
    extension: str = ".md",
    timestamp: str | None = None,
) -> Path:
    """Resolve an output path from user CLI input or fall back to the standardized timestamped name."""
    if not output_arg:
        return generate_output_path(default_dir, suffix=suffix, extension=extension, timestamp=timestamp)

    p = Path(output_arg)
    # If the user passed an existing directory or a path without an extension, treat as directory
    if p.is_dir() or str(output_arg).endswith(("/", "\\")) or p.suffix == "":
        return generate_output_path(p, suffix=suffix, extension=extension, timestamp=timestamp)

    # User provided an explicit file path
    return p


def prepend_tools_header(
    content: str,
    tools: list[str],
    input_path: Path | str | None = None,
) -> str:
    """Ensure the output markdown starts with standardized 'Tools: ...' and 'Input: ...' lines."""
    tools_str = ", ".join(tools)
    header_lines = [f"Tools: {tools_str}"]
    if input_path:
        header_lines.append(f"Input: {Path(input_path).name}")
    header = "\n".join(header_lines)

    lines = content.splitlines()
    idx = 0
    while idx < len(lines) and (lines[idx].startswith("Tools:") or lines[idx].startswith("Input:")):
        idx += 1

    clean_remaining = "\n".join(lines[idx:]).lstrip("\r\n")
    return f"{header}\n\n{clean_remaining}"
