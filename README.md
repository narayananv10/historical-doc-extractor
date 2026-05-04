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

## Headline result

| OCR | Mean CER ↓ | Perfect-line rate ↑ |
|---|---|---|
| TrOCR raw | 0.239 | 1.7% |
| TrOCR + Claude vision post-correction | **0.214** | **24.1%** |

Mean CER barely moves (~11% relative), but **14× more lines are perfect** after post-correction. The bimodal effect fixes nearly-right lines, sometimes adds noise to severely-wrong ones, is what the flagger is built to triage.

**Flagger on residual errors** (5-fold GroupKFold CV on IAM-GW, 656 lines / 20 docs): ROC AUC **0.72**, Brier **0.16**. At the deployable threshold (review-budget = 30%), flagging 30% of lines catches **36% of remaining errors at 92% precision**.

> Numbers are in-distribution on the GW benchmark only. A hand-labelled LoC holdout for out-of-distribution validation is documented as a real limitation in `notebooks/calibration.ipynb`.

## Install

```bash
git clone https://github.com/narayananv10/historical-doc-extractor.git
cd historical-doc-extractor

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # use _trf instead if you have ≥16 GB RAM

cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY (required) and HF_TOKEN (recommended)

# Pre-download the model weights (doctr + TrOCR)
python scripts/setup_models.py
```

`HF_TOKEN` is technically optional. The model downloads work anonymously, but it lifts HuggingFace's rate limits and silences the unauthenticated-requests warning. Create a free read-only token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

`scripts/setup_models.py` works around a known issue where the doctr library hits an HTTP 308 redirect that `urllib` doesn't follow, silently leaving a 0-byte file in `~/.cache/doctr/models/`. Run it once and you're set.

No system-level dependencies required. Tested on macOS (Apple Silicon) and Linux; runs on CPU/MPS without a GPU.

## Run

```bash
# 1. Pull a curated LoC sample set
python scripts/download_loc.py

# 2. (One-time) cache the IAM-GW pipeline run — produces flagger training data.
#    Requires IAM-HistDB Washington dataset extracted into data/raw/iam_gw/
#    (free academic registration at fki.tic.heia-fr.ch).
python scripts/download_gw_pages.py    # ~4 MB of LoC page scans for visual context
python scripts/cache_iam_gw.py          # ~5-10 min, ~$0.10 in Claude API

# 3. Single-document end-to-end
python -m src.pipeline data/samples/letter1.jpg

# 4. Batch over a folder → catalogue.csv
python -m src.batch data/raw/loc/abraham-lincoln-papers -o catalogue.csv

# 5. Streamlit demo
streamlit run app.py
```

Pass `--no-api` to any pipeline command (`src.pipeline`, `src.batch`, or the Streamlit "Skip Claude API" toggle) to skip post-correction, classification, and Claude entity extraction. spaCy NER still runs. Under `--no-api` every line is force-flagged for review with a `NO_API_VERIFICATION` reason — the transcription is unverified raw TrOCR output and shouldn't be trusted as-is.

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

- **Accuracy:** published CER/WER on a public benchmark and a hand-labeled holdout; flagger evaluated with ROC, Brier, reliability diagram.
- **Transparency:** flagger feature importances are published; entities tagged by source (`spacy` vs `claude`); Claude prompts versioned in `prompts/v1/`.
- **Human oversight:** the Review queue tab surfaces every flagged line with its crop, both the raw and corrected text, the probability, and the reasons.
- **Accountability:** per-line probability and reasons are persisted to `catalogue.csv` so every downstream record can be audited.
