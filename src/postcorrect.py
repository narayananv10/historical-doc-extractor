"""Claude vision post-correction.

One API call per document: the full scan image plus all TrOCR-transcribed lines
(prefixed with [LINE_N]) are sent together so the model has cross-line context.
The model returns per-line corrections and a self-reported confidence score.

Pass no_api=True to skip the API call and return raw TrOCR text unchanged
(useful for the --no-api CLI flag and offline testing).
"""

from __future__ import annotations

import base64
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import anthropic
from dotenv import load_dotenv
from PIL import Image, ImageOps

from src.ocr_trocr import Line

MODEL_ID = "claude-haiku-4-5-20251001"

_CORRECTION_TOOL: dict = {
    "name": "submit_corrections",
    "description": "Submit per-line corrections for the transcribed document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "line_id": {"type": "integer"},
                        "original": {"type": "string"},
                        "corrected": {"type": "string"},
                        "changed": {"type": "boolean"},
                        "llm_confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["line_id", "original", "corrected", "changed", "llm_confidence"],
                },
            }
        },
        "required": ["lines"],
    },
}


@dataclass
class CorrectedLine:
    """A post-corrected line with provenance from both TrOCR and Claude."""

    line_id: int
    original: str
    corrected: str
    changed: bool
    llm_confidence: float | None
    bbox: tuple[int, int, int, int] | None = None


@lru_cache(maxsize=1)
def _get_client() -> anthropic.Anthropic:
    load_dotenv()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to .env or export it as an environment variable. "
            "See .env.example for the required format."
        )
    return anthropic.Anthropic(api_key=key)


@lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / "v1" / "postcorrect.md"
    return prompt_path.read_text(encoding="utf-8")


def _encode_image(image_path: str | Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for the given image file.

    Always re-encodes through PIL as JPEG. Two reasons:
      1. Anthropic accepts JPEG/PNG/GIF/WebP only — HEIC, TIFF, and other
         formats common on iPhone / scanner exports must be converted first.
         Reading the file as raw bytes and labelling it `image/jpeg` (the
         previous behaviour) trips a 400 "Could not process image" when the
         contents don't actually match the MIME type.
      2. PIL's HEIF plugin (registered in src.preprocess) handles HEIC files
         that are mislabelled with a `.jpg` / `.jpeg` extension by macOS
         Photos / iPhone exports.
    """
    from io import BytesIO

    # Importing src.preprocess registers the HEIF opener as a side effect, so
    # PIL.Image.open() handles HEIC files even when their extension lies.
    import src.preprocess  # noqa: F401

    pil = Image.open(image_path)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def _build_transcription_block(trocr_lines: Sequence[Line]) -> str:
    return "\n".join(f"[LINE_{i}] {line.text}" for i, line in enumerate(trocr_lines))


def post_correct(
    image_path: str | Path,
    trocr_lines: Sequence[Line],
    *,
    no_api: bool = False,
    model: str = MODEL_ID,
) -> list[CorrectedLine]:
    """Post-correct TrOCR output with Claude vision (one call per document).

    Args:
        image_path: Path to the original scan.
        trocr_lines: Transcribed lines from ocr_trocr.transcribe().
        no_api: If True, skip the API call and return raw TrOCR text unchanged.
        model: Claude model ID to use.

    Returns:
        List of CorrectedLine, one per input line, in order.
    """
    if no_api or not trocr_lines:
        return [
            CorrectedLine(
                line_id=i,
                original=line.text,
                corrected=line.text,
                changed=False,
                llm_confidence=None,
                bbox=line.bbox,
            )
            for i, line in enumerate(trocr_lines)
        ]

    client = _get_client()
    system_prompt = _load_system_prompt()
    image_data, media_type = _encode_image(image_path)
    transcription_block = _build_transcription_block(trocr_lines)

    user_text = (
        f"Please correct the following OCR transcription. "
        f"There are {len(trocr_lines)} lines.\n\n"
        f"{transcription_block}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                # Cache the system prompt across calls in the same batch run.
                # Saves ~90% of input token cost on the system prompt for docs 2+.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[_CORRECTION_TOOL],
        tool_choice={"type": "tool", "name": "submit_corrections"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    )

    tool_use_block = next(
        (b for b in response.content if b.type == "tool_use"),
        None,
    )
    if tool_use_block is None:
        print(
            "[postcorrect] no tool_use block in response; returning raw TrOCR lines",
            file=sys.stderr,
        )
        return [
            CorrectedLine(
                line_id=i,
                original=line.text,
                corrected=line.text,
                changed=False,
                llm_confidence=None,
                bbox=line.bbox,
            )
            for i, line in enumerate(trocr_lines)
        ]

    corrections: dict[int, dict] = {
        item["line_id"]: item for item in tool_use_block.input["lines"]
    }

    result: list[CorrectedLine] = []
    for i, line in enumerate(trocr_lines):
        if i in corrections:
            c = corrections[i]
            result.append(
                CorrectedLine(
                    line_id=i,
                    original=line.text,
                    corrected=c["corrected"],
                    changed=c["changed"],
                    llm_confidence=c["llm_confidence"],
                    bbox=line.bbox,
                )
            )
        else:
            # Model didn't return this line; keep original
            print(
                f"[postcorrect] LINE_{i} missing from model response; keeping original",
                file=sys.stderr,
            )
            result.append(
                CorrectedLine(
                    line_id=i,
                    original=line.text,
                    corrected=line.text,
                    changed=False,
                    llm_confidence=None,
                    bbox=line.bbox,
                )
            )

    return result
