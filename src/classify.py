"""Document type classification via Claude tool-use.

Returns one of letter | receipt | ledger | deed with a confidence and
reasoning. Falls back to facebook/bart-large-mnli zero-shot under --no-api.
"""

# TODO: implement classify(text) -> {doc_type, confidence, reasoning}
