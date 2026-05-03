"""Named-entity extraction.

spaCy baseline (en_core_web_sm by default; en_core_web_trf where memory
allows) for PERSON / DATE / GPE / ORG. Claude tool-use call extracts
doc-type-specific fields (sender/recipient/amount/signed_date/places).
Output union is tagged by source for transparency.
"""

# TODO: implement extract(text, doc_type) -> list[Entity]
# Entity = {label, value, source: 'spacy' | 'claude'}
