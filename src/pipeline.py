"""End-to-end pipeline for one document.

preprocess -> ocr_trocr -> postcorrect -> flagger -> classify -> ner
-> DocumentResult.

CLI:
    python -m src.pipeline data/samples/letter1.jpg
    python -m src.pipeline data/samples/letter1.jpg --no-api
    python -m src.pipeline data/samples/letter1.jpg --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.classify import ClassifyResult, classify
from src.flagger import compute_features, describe, flag
from src.ner import Entity, extract_entities
from src.ocr_trocr import transcribe
from src.postcorrect import post_correct
from src.preprocess import preprocess


@dataclass
class PipelineLine:
    line_id: int
    bbox: tuple[int, int, int, int]
    trocr_text: str
    corrected_text: str
    changed: bool
    llm_confidence: float | None
    prob_wrong: float
    flagged: bool
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "line_id": self.line_id,
            "bbox": list(self.bbox),
            "trocr_text": self.trocr_text,
            "corrected_text": self.corrected_text,
            "changed": self.changed,
            "llm_confidence": self.llm_confidence,
            "prob_wrong": round(self.prob_wrong, 3),
            "flagged": self.flagged,
            "reasons": self.reasons,
            "reason_descriptions": [describe(r) for r in self.reasons],
        }


@dataclass
class DocumentResult:
    image_path: Path
    lines: list[PipelineLine]
    classification: ClassifyResult
    entities: list[Entity] = field(default_factory=list)

    @property
    def n_review_lines(self) -> int:
        return sum(1 for line in self.lines if line.flagged)

    @property
    def mean_prob_wrong(self) -> float:
        if not self.lines:
            return 0.0
        return sum(line.prob_wrong for line in self.lines) / len(self.lines)

    @property
    def full_text(self) -> str:
        return "\n".join(line.corrected_text for line in self.lines)

    def to_dict(self) -> dict:
        return {
            "image_path": str(self.image_path),
            "n_lines": len(self.lines),
            "n_review_lines": self.n_review_lines,
            "mean_prob_wrong": round(self.mean_prob_wrong, 3),
            "doc_type": self.classification.doc_type,
            "doc_type_confidence": round(self.classification.confidence, 3),
            "doc_type_reasoning": self.classification.reasoning,
            "entities": [e.to_dict() for e in self.entities],
            "lines": [line.to_dict() for line in self.lines],
        }


def process(image_path: str | Path, *, no_api: bool = False) -> DocumentResult:
    image_path = Path(image_path)
    crops = preprocess(image_path)
    trocr_lines = transcribe(crops)
    corrected_lines = post_correct(image_path, trocr_lines, no_api=no_api)

    pipeline_lines: list[PipelineLine] = []
    for i, (trocr_line, corrected) in enumerate(zip(trocr_lines, corrected_lines)):
        features = compute_features(trocr_line, corrected)
        flagger_out = flag(features)
        pipeline_lines.append(
            PipelineLine(
                line_id=i,
                bbox=trocr_line.bbox,
                trocr_text=trocr_line.text,
                corrected_text=corrected.corrected,
                changed=corrected.changed,
                llm_confidence=corrected.llm_confidence,
                prob_wrong=flagger_out.prob_wrong,
                flagged=flagger_out.flagged,
                reasons=flagger_out.reasons,
            )
        )

    full_text = "\n".join(line.corrected_text for line in pipeline_lines)
    classification = classify(full_text, no_api=no_api)
    entities = extract_entities(
        full_text, doc_type=classification.doc_type, no_api=no_api
    )

    return DocumentResult(
        image_path=image_path,
        lines=pipeline_lines,
        classification=classification,
        entities=entities,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path)
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Skip Claude post-correction (TrOCR-only output)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON; otherwise prints a human-readable summary",
    )
    args = parser.parse_args()

    if not args.image_path.exists():
        sys.exit(f"image not found: {args.image_path}")

    result = process(args.image_path, no_api=args.no_api)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"=== {args.image_path} ===")
        print(
            f"doc_type: {result.classification.doc_type} "
            f"(confidence {result.classification.confidence:.2f})"
        )
        if result.classification.reasoning:
            print(f"  why: {result.classification.reasoning}")
        print(
            f"{len(result.lines)} lines | "
            f"{result.n_review_lines} flagged for review | "
            f"mean prob_wrong = {result.mean_prob_wrong:.3f}"
        )
        if result.entities:
            print()
            print(f"entities ({len(result.entities)}):")
            for e in result.entities:
                conf = f" [{e.confidence:.2f}]" if e.confidence is not None else ""
                print(f"  {e.label:<20} {e.text!r}  (source: {e.source}{conf})")
        print()
        for line in result.lines:
            marker = "!!" if line.flagged else "  "
            print(f"{marker} [{line.line_id:>3}] p={line.prob_wrong:.2f}  {line.corrected_text}")
            if line.flagged:
                for r in line.reasons:
                    print(f"        reason: {r} - {describe(r)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
