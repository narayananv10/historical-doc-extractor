<!-- System prompt for Claude vision post-correction.

Input shape:
- The full document scan as an image.
- The TrOCR transcription with one line per row, each prefixed by [LINE_ID].

Required tool-call output: per-line corrections, with `changed` flag and
`llm_confidence` in [0, 1] (the model's self-reported confidence in the
corrected text). Lines that look correct should be returned unchanged with
`changed: false`.

Final prompt body to be drafted alongside the postcorrect.py implementation. -->
