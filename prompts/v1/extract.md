<!-- System prompt for doc-type-specific entity extraction.

Schema branches on doc_type:
  letter  -> sender, recipient, signed_date, referenced_places
  receipt -> amount, currency, date, payer, payee
  ledger  -> entries (date, description, amount), account_holder
  deed    -> parties, jurisdiction, signed_date, property_description

Final prompt body to be drafted alongside the ner.py implementation. -->
