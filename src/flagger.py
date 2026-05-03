"""Learned residual-error flagger.

Predicts whether a Claude-vision-corrected line is *still* wrong, using
TrOCR logprob features, TrOCR-vs-corrected agreement features, and
linguistic-plausibility features on the corrected text.

Loads models/flagger_v1.pkl (sklearn classifier + scaler). If the model
file is missing or fails to load, falls back to a rule-based flagger using
the same feature set with hand-tuned thresholds.

Each flagged line carries human-readable reason codes derived from which
features fired most strongly.
"""

# TODO: implement extract_features(line) -> dict
# TODO: implement flag(line) -> {prob_wrong, flagged, reasons}
