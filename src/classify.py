"""Document type classification via Claude vision tool-use.

One API call per document. System prompt cached via cache_control so a batch
run pays for the prompt tokens once. Returns ClassifyResult(doc_type,
confidence, reasoning).

`--no-api` returns doc_type="unknown" with a note. The original plan called
for a `facebook/bart-large-mnli` zero-shot fallback; dropped to avoid a
1.6 GB extra HF download for a feature that's only useful in offline mode.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.postcorrect import _get_client

MODEL_ID = "claude-haiku-4-5-20251001"
DOC_TYPES = ["letter", "receipt", "ledger", "deed"]
MAX_TEXT_CHARS = 16000  # rough cap; haiku handles 200k tokens but no need to send a book

_CLASSIFY_TOOL: dict = {
    "name": "classify_document",
    "description": "Submit a document classification with reasoning.",
    "input_schema": {
        "type": "object",
        "properties": {
            "doc_type": {
                "type": "string",
                "enum": DOC_TYPES,
                "description": "Best-matching document type",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "reasoning": {
                "type": "string",
                "description": "1-2 sentences citing structural cues",
            },
        },
        "required": ["doc_type", "confidence", "reasoning"],
    },
}


@dataclass
class ClassifyResult:
    doc_type: str
    confidence: float
    reasoning: str


@lru_cache(maxsize=1)
def _load_prompt() -> str:
    p = Path(__file__).parent.parent / "prompts" / "v1" / "classify.md"
    return p.read_text(encoding="utf-8")


def _truncate(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[:MAX_TEXT_CHARS] + "\n\n[TRUNCATED]"


def classify(
    text: str,
    *,
    no_api: bool = False,
    model: str = MODEL_ID,
) -> ClassifyResult:
    if no_api:
        return ClassifyResult(
            doc_type="unknown", confidence=0.0, reasoning="--no-api mode"
        )
    if not text.strip():
        return ClassifyResult(
            doc_type="unknown", confidence=0.0, reasoning="empty input text"
        )

    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": _load_prompt(),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[_CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "classify_document"},
        messages=[{"role": "user", "content": _truncate(text)}],
    )

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        print("[classify] no tool_use in response; returning unknown", file=sys.stderr)
        return ClassifyResult(
            doc_type="unknown", confidence=0.0, reasoning="no tool response"
        )

    return ClassifyResult(
        doc_type=str(tool_block.input["doc_type"]),
        confidence=float(tool_block.input["confidence"]),
        reasoning=str(tool_block.input["reasoning"]),
    )
