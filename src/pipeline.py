"""End-to-end pipeline orchestration for a single document.

preprocess -> ocr_trocr -> postcorrect -> classify -> ner -> flagger.
Returns a DocumentResult dataclass holding raw and corrected lines, the
classifier output, the entity union, and the flagger's per-line output.
"""

# TODO: implement process(image_path, no_api=False) -> DocumentResult
