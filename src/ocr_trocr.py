"""TrOCR with per-token logprobs.

Uses microsoft/trocr-base-handwritten. Returns transcribed text plus the raw
per-token logprobs needed by the flagger's feature extraction.
"""

# TODO: implement transcribe(line_crops) -> list[Line]
# Line = {text, bbox, token_logprobs}
