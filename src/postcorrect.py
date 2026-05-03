"""Claude vision post-correction.

One call per document: full scan image + TrOCR transcription with line IDs.
The model returns per-line corrections plus self-reported confidence.
"""

# TODO: implement post_correct(image_path, trocr_lines) -> list[CorrectedLine]
# CorrectedLine = {line_id, original, corrected, changed, llm_confidence}
