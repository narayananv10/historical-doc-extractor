"""One-shot: run TrOCR + Claude vision post-correction over IAM-GW.

Produces data/parquet_cache/iam_gw_pipeline.parquet with columns:
    line_id, trocr_text, trocr_logprobs, corrected_text,
    llm_confidence, gt, is_still_wrong

This file is committed to the repo so the flagger notebook can iterate
without re-running the heavy steps.
"""

# TODO: implement main()
