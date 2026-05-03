"""Flagger: predicts which post-corrected lines still need human review.

Two implementations behind a single flag(features) -> FlaggerOutput interface:
  - learned (Phase 6+): sklearn classifier loaded from models/flagger_v1.pkl
  - rule-based: hand-tuned thresholds on the same feature set

Phase 5 ships rule-based only; the learned path is wired in Phase 6 with the
same FlaggerOutput contract so the pipeline / Streamlit UI don't need to
change. A failing or missing model file always degrades to rule-based,
logged loudly to stderr, so the review queue surface never disappears.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from rapidfuzz.distance import Levenshtein

from src.ocr_trocr import Line
from src.postcorrect import CorrectedLine

MODEL_PATH = Path(__file__).parent.parent / "models" / "flagger_v1.pkl"
DEFAULT_THRESHOLD = 0.5


@dataclass
class FlaggerOutput:
    prob_wrong: float
    flagged: bool
    reasons: list[str] = field(default_factory=list)


REASON_DESCRIPTIONS: dict[str, str] = {
    "LOW_MIN_LOGPROB_TOKEN": "TrOCR was very uncertain about at least one token in this line.",
    "LOW_MEAN_LOGPROB": "TrOCR was uncertain across the whole line.",
    "HIGH_CORRECTION_DELTA": "Claude vision changed many characters relative to the TrOCR output.",
    "LOW_LLM_CONFIDENCE": "Claude reported low confidence in the corrected text.",
    "VERY_SHORT_LINE": "Very short line; the OCR had little context to work with.",
}


def describe(reason_code: str) -> str:
    return REASON_DESCRIPTIONS.get(reason_code, reason_code)


def compute_features(trocr_line: Line, corrected_line: CorrectedLine) -> dict:
    """Per-line features shared by the rule-based and learned flaggers."""
    logprobs = np.asarray(trocr_line.token_logprobs, dtype=np.float64)
    n_tokens = int(logprobs.size)
    if n_tokens > 0:
        mean_lp = float(logprobs.mean())
        min_lp = float(logprobs.min())
        std_lp = float(logprobs.std())
        length_norm = float(logprobs.sum() / max(n_tokens, 1))
    else:
        mean_lp = min_lp = std_lp = length_norm = 0.0

    distance = Levenshtein.distance(trocr_line.text, corrected_line.corrected)
    base_len = max(len(trocr_line.text), 1)

    return {
        "n_tokens": n_tokens,
        "mean_logprob": mean_lp,
        "min_logprob": min_lp,
        "std_logprob": std_lp,
        "length_normalized_logprob": length_norm,
        "edit_distance_trocr_vs_corrected": int(distance),
        "n_chars_changed": int(distance),
        "frac_chars_changed": float(distance / base_len),
        "llm_confidence": corrected_line.llm_confidence,
        "line_height_px": int(trocr_line.bbox[3]),
        "line_width_px": int(trocr_line.bbox[2]),
    }


def _flag_rule_based(features: dict, threshold: float) -> FlaggerOutput:
    """Hand-tuned thresholds. Each rule contributes a candidate score; we
    take the max. Crude on purpose — Phase 6's learned model replaces this."""
    reasons: list[str] = []
    score = 0.0

    n_tokens = features["n_tokens"]

    if n_tokens > 0 and features["min_logprob"] < -3.0:
        reasons.append("LOW_MIN_LOGPROB_TOKEN")
        score = max(score, 0.6)

    if n_tokens > 0 and features["mean_logprob"] < -1.0:
        reasons.append("LOW_MEAN_LOGPROB")
        score = max(score, 0.5)

    if features["frac_chars_changed"] > 0.3:
        reasons.append("HIGH_CORRECTION_DELTA")
        score = max(score, 0.7)

    llm_conf = features["llm_confidence"]
    if llm_conf is not None and llm_conf < 0.6:
        reasons.append("LOW_LLM_CONFIDENCE")
        score = max(score, 1.0 - float(llm_conf))

    if 0 < n_tokens <= 2:
        reasons.append("VERY_SHORT_LINE")
        score = max(score, 0.4)

    return FlaggerOutput(
        prob_wrong=float(min(score, 1.0)),
        flagged=bool(score >= threshold),
        reasons=reasons,
    )


def _try_load_model():
    """Returns the loaded model bundle dict, or None to fall back."""
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib

        return joblib.load(MODEL_PATH)
    except Exception as exc:
        print(
            f"[flagger] failed to load {MODEL_PATH}: {exc!r}; "
            f"falling back to rule-based",
            file=sys.stderr,
        )
        return None


def flag(features: dict, *, threshold: float | None = None) -> FlaggerOutput:
    """Flag a single line. Uses the learned model if available, otherwise
    rule-based. Reason codes are identical across both paths."""
    bundle = _try_load_model()
    if bundle is None:
        # Phase 5: rule-based only. Phase 6 will replace this branch with
        # a learned-model path that emits the same FlaggerOutput shape.
        return _flag_rule_based(features, threshold or DEFAULT_THRESHOLD)
    # Learned path wired in Phase 6.
    return _flag_rule_based(features, threshold or bundle.get("threshold", DEFAULT_THRESHOLD))


def flag_many(
    features_iter: Iterable[dict], *, threshold: float | None = None
) -> list[FlaggerOutput]:
    return [flag(f, threshold=threshold) for f in features_iter]
