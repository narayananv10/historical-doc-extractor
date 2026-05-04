"""Named-entity extraction.

spaCy (`en_core_web_sm` by default) always runs and produces baseline
PERSON / DATE / GPE / ORG entities. Claude adds doc-type-aware custom
entities (sender, recipient, amount, signed_date, etc.) when the API is
available and `--no-api` isn't set. Each entity is tagged with `source` so
downstream consumers know whether to trust the label.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.postcorrect import _get_client

MODEL_ID = "claude-haiku-4-5-20251001"
SPACY_MODEL = "en_core_web_sm"
SPACY_LABELS = {"PERSON", "DATE", "GPE", "LOC", "ORG"}
MAX_TEXT_CHARS = 16000

_EXTRACT_TOOL: dict = {
    "name": "extract_entities",
    "description": "Extract structured entities from a transcribed document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "label": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["text", "label"],
                },
            }
        },
        "required": ["entities"],
    },
}


@dataclass
class Entity:
    text: str
    label: str
    source: str  # "spacy" | "claude"
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "source": self.source,
            "confidence": self.confidence,
        }


@lru_cache(maxsize=1)
def _load_spacy():
    import spacy

    try:
        return spacy.load(SPACY_MODEL)
    except OSError as exc:
        sys.exit(
            f"spaCy model {SPACY_MODEL!r} not found. Install with:\n"
            f"  python -m spacy download {SPACY_MODEL}\n\nDetails: {exc}"
        )


@lru_cache(maxsize=1)
def _load_prompt() -> str:
    p = Path(__file__).parent.parent / "prompts" / "v1" / "extract.md"
    return p.read_text(encoding="utf-8")


def _truncate(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[:MAX_TEXT_CHARS] + "\n\n[TRUNCATED]"


def _extract_spacy(text: str) -> list[Entity]:
    nlp = _load_spacy()
    doc = nlp(text)
    return [
        Entity(text=ent.text, label=ent.label_, source="spacy")
        for ent in doc.ents
        if ent.label_ in SPACY_LABELS
    ]


def _extract_claude(text: str, doc_type: str, model: str) -> list[Entity]:
    client = _get_client()
    user_msg = f"Document type: {doc_type}\n\nText:\n{_truncate(text)}"

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": _load_prompt(),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[_EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_entities"},
        messages=[{"role": "user", "content": user_msg}],
    )

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        print("[ner] no tool_use in response; returning empty", file=sys.stderr)
        return []

    return [
        Entity(
            text=str(item["text"]),
            label=str(item["label"]),
            source="claude",
            confidence=float(item["confidence"]) if "confidence" in item else None,
        )
        for item in tool_block.input.get("entities", [])
    ]


def extract_entities(
    text: str,
    *,
    doc_type: str = "unknown",
    no_api: bool = False,
    model: str = MODEL_ID,
) -> list[Entity]:
    """spaCy always runs; Claude runs when available and not in --no-api mode.
    Returns the union with each entity tagged by source."""
    if not text.strip():
        return []
    entities = _extract_spacy(text)
    if not no_api:
        try:
            entities.extend(_extract_claude(text, doc_type, model))
        except Exception as exc:
            print(
                f"[ner] Claude extraction failed ({exc!r}); spaCy-only output",
                file=sys.stderr,
            )
    return entities
