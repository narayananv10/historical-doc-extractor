# Historical Document Extractor

End-to-end ingestion of historical handwritten archives into structured, human-reviewable output. Pairs a local OCR model with a frontier vision LLM for post-correction, then uses a learned classifier to flag the lines that *still* need a human after the LLM has had a look.

The same pipeline shape generalizes to anything a fast-moving organization needs to digest reliably from images: archived correspondence, leaked documents, court filings, scanned press releases.

## What the pipeline does

```
scan.jpg
  → preprocess (deskew, line-segment)
  → TrOCR (per-token logprobs)
  → Claude vision post-correction (one call per doc; per-line corrections)
  → learned residual-error flagger (P(line still wrong) + reason codes)
  → classify (letter | receipt | ledger | deed)
  → entity extraction (spaCy + Claude)
  → structured JSON / catalogue.csv
  → Streamlit review UI
```

## Headline result *(populated after evaluation runs)*

| OCR | CER ↓ (IAM-GW) | CER ↓ (LoC holdout) |
|---|---|---|
| TrOCR raw | _TBD_ | _TBD_ |
| TrOCR + Claude vision post-correction | _TBD_ | _TBD_ |

**Flagger on residual errors:** ROC AUC _TBD_, Brier _TBD_. At the chosen threshold, flagging _TBD_% of lines catches _TBD_% of remaining errors.

## Install

```bash
git clone https://github.com/narayananv10/historical-doc-extractor.git
cd historical-doc-extractor

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # use _trf instead if you have ≥16 GB RAM

cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

# Pre-download the model weights (doctr + TrOCR)
python scripts/setup_models.py
```

`scripts/setup_models.py` works around a known issue where the doctr library hits an HTTP 308 redirect that `urllib` doesn't follow, silently leaving a 0-byte file in `~/.cache/doctr/models/`. Run it once and you're set.

No system-level dependencies required. Tested on macOS (Apple Silicon) and Linux; runs on CPU/MPS without a GPU.

## Run

```bash
# 1. Pull a curated LoC sample set
python scripts/download_loc.py

# 2. Cache the IAM-GW pipeline run (one-shot; produces flagger training data)
python scripts/cache_iam_gw.py

# 3. Single-document end-to-end
python -m src.pipeline data/samples/letter1.jpg

# 4. Batch over a folder → catalogue.csv
python -m src.batch data/raw/loc/abraham-lincoln-papers -o catalogue.csv

# 5. Streamlit demo
streamlit run app.py
```

Pass `--no-api` to any pipeline command to skip the Claude vision post-correction step (TrOCR-only mode).

## Repo layout

```
src/         pipeline modules (preprocess, ocr, postcorrect, classify, ner, flagger, batch)
scripts/     download_loc, cache_iam_gw, evaluate
notebooks/   flagger.ipynb, calibration.ipynb, errors.ipynb
prompts/v1/  versioned Claude system prompts
models/      flagger_v1.pkl (trained sklearn classifier)
data/        samples/ (checked in), parquet_cache/ (committed flagger data)
app.py       Streamlit demo
WRITEUP.md   narrative deep-dive: framing, eval methodology, calibration, limits
```

See [tentative_plan.md](tentative_plan.md) for the full architecture and design rationale.

## Responsible AI

This project takes a *fail-loudly* posture: every line that gets emitted carries a probability of error and, when flagged, a human-readable explanation. Specifically:

- **Accuracy** — published CER/WER on a public benchmark and a hand-labeled holdout; flagger evaluated with ROC, Brier, reliability diagram.
- **Transparency** — flagger feature importances are published; entities tagged by source (`spacy` vs `claude`); Claude prompts versioned in `prompts/v1/`.
- **Human oversight** — the Review queue tab surfaces every flagged line with its crop, both the raw and corrected text, the probability, and the reasons.
- **Accountability** — per-line probability and reasons are persisted to `catalogue.csv` so every downstream record can be audited.
