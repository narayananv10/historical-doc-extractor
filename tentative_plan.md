# Library Handwritten Document Digitizer — Implementation Plan

## Context

A portfolio project demonstrating end-to-end ingestion of historical handwritten archives into structured, human-reviewable output. The same pipeline shape applies to any document a newsroom or research org needs to digest at speed: archived correspondence, leaked documents, court filings, scanned press releases.

The pipeline does four things end to end:
1. **OCR** the handwriting (TrOCR with per-token logprob signals).
2. **Post-correct** the OCR output via Claude vision — the multimodal model sees the original image alongside the TrOCR transcription and corrects per-line errors. This is what turns the system from "honest about being wrong" into a system that actually digitizes well.
3. **Classify** the document type (`letter | receipt | ledger | deed`).
4. **Extract entities** (PERSON, DATE, GPE/PLACE, plus document-specific fields).

Headline differentiator: a **learned residual-error flagger** — even after frontier-model post-correction, some lines remain wrong. A small classifier trained on TrOCR confidence features + TrOCR/Claude-vision agreement signals decides which post-corrected lines *still* need human review, with **human-readable reasons** layered on top. The flagger is evaluated with a reliability diagram, ROC curve, and Brier score, not asserted from a hand-tuned threshold. The story is concrete: post-correction reduces CER from X to Y, and the flagger catches Z% of the remaining errors when reviewing only W% of lines.

Outcome: a `git clone && streamlit run app.py` repo, a flagger calibration notebook, a `catalogue.csv` artifact from a batch run, and a narrative writeup.

## Datasets

- **Primary demo set**: Library of Congress *By the People* (crowd.loc.gov) — download ~30–50 images spanning Lincoln Papers (letters), Civil War diaries, Mary Church Terrell papers (letters/diaries), and at least one ledger campaign. Skew toward documents with recognizable structure: letters with named senders/recipients, ledgers with amounts. Volunteer transcriptions ship as ground truth.
- **Benchmark + flagger training set**: IAM-HistDB **George Washington** subset (20 pages, line-level ground truth) — used to (a) compute CER/WER for both TrOCR raw and TrOCR + Claude vision post-correction, and (b) generate `(line_features, is_correct_after_post_correction)` training pairs for the learned flagger. The FKI release (`washingtondb-v1.0`) ships line images and word images only, no full page scans, so we pull the original page scans from the Library of Congress (Series 2, Letterbook 1) via `scripts/download_gw_pages.py`. FKI page IDs map 1:1 to LoC `sp=N` with no offset (verified against FKI's ground truth header). Pages used: 270-279 and 300-309 (the FKI dataset's two non-contiguous blocks).
- **Hand-labeled holdout**: 8 LoC documents transcribed manually into `data/hand_labeled/`, used as a held-out set for the flagger calibration analysis (since IAM-GW likely overlaps TrOCR pretraining).
- All raw files live under `data/raw/{loc,iam_gw}/`; transcriptions under `data/ground_truth/`. The IAM-GW tree contains both the FKI distribution (`data/raw/iam_gw/washingtondb-v1.0/`) and the LoC page scans (`data/raw/iam_gw/loc_pages/`). `data/raw/` is gitignored; `data/samples/` (3–5 small images for the live demo) and `data/parquet_cache/` (committed flagger training data from the IAM-GW pipeline run) are tracked.

## Architecture

```
scan.jpg
  │
  ▼
[preprocess]  ── deskew, binarize, line-segment
  │
  ▼
[ocr_trocr] ──────────► lines + per-token logprobs
  │                              │
  ▼                              │
[postcorrect_claude_vision]      │  ── one call per doc with full image +
  │                              │     TrOCR text inline; returns per-line
  │                              │     corrections + per-line LLM confidence
  ▼                              │
corrected_lines ─────────────────┤
  │                              ▼
  │              [feature_extract] ── TrOCR logprob stats, char perplexity,
  │                              │     edit_distance(trocr, corrected),
  │                              │     n_chars_changed, llm_confidence, etc.
  │                              ▼
  │              [learned_flagger] ──► P(corrected line still wrong) + reason codes
  │                              │
  ▼                              │
full_text ──┬──► [classify_claude]   → doc_type
            ├──► [ner_spacy]         → baseline entities
            └──► [extract_claude]    → custom fields
                              │
                              ▼
            [review_queue]: flagged lines + human-readable reasons
                              │
                              ▼
              structured JSON  ── catalogue.csv
                              │
                              ▼
                       Streamlit UI
```

## File layout

```
historical-doc-extractor/
├── README.md                  # narrative, install, how to run, CER/WER + flagger results
├── WRITEUP.md                 # narrative deep-dive: framing, flagger model, calibration, limits
├── requirements.txt
├── .env.example               # ANTHROPIC_API_KEY
├── data/
│   ├── samples/               # checked in: 3–5 small images for demo
│   ├── raw/                   # gitignored
│   ├── ground_truth/          # gitignored
│   ├── hand_labeled/          # 8 LoC docs hand-transcribed for the flagger holdout
│   └── parquet_cache/         # cached (text, logprobs, ground_truth) from IAM-GW pass
├── prompts/
│   └── v1/
│       ├── postcorrect.md     # Claude vision: image + TrOCR text → per-line corrections
│       ├── classify.md
│       └── extract.md
├── models/
│   └── flagger_v1.pkl         # trained sklearn classifier + feature scaler
├── src/
│   ├── __init__.py
│   ├── preprocess.py          # deskew, binarize, line segmentation (doctr default)
│   ├── ocr_trocr.py           # TrOCR + per-token logprobs
│   ├── postcorrect.py         # Claude vision post-correction (one call per doc)
│   ├── classify.py            # Claude tool-use: returns doc_type + reasoning
│   ├── ner.py                 # spaCy baseline + Claude custom-entity extraction
│   ├── flagger.py             # feature extraction + load trained model + reason codes
│   ├── pipeline.py            # orchestrates the full flow on one image
│   └── batch.py               # runs pipeline over a folder, writes catalogue.csv
├── scripts/
│   ├── download_loc.py        # fetches LoC sample set (Lincoln, Terrell papers)
│   ├── download_gw_pages.py   # fetches GW page scans from LoC for the IAM-GW pipeline
│   ├── setup_models.py        # pre-downloads doctr (curl-based, works around urllib redirect bug) + TrOCR
│   ├── cache_iam_gw.py        # one-shot: TrOCR + post-correction over IAM-GW → parquet
│   └── evaluate.py            # CER/WER for TrOCR raw vs post-corrected; flagger ROC/Brier
├── app.py                     # Streamlit demo
└── notebooks/
    ├── flagger.ipynb          # feature engineering + train classifier + ROC + feature importance
    ├── calibration.ipynb      # reliability diagram + threshold-via-F1 + precision/recall
    └── errors.ipynb           # char-level confusion matrix + cherry-picked failure cases
```

## Component-by-component

### 1. Preprocessing — `src/preprocess.py`
- OpenCV grayscale → adaptive threshold → deskew (Hough lines or `deskew` package).
- Line segmentation via `doctr`'s detector by default (robust on cursive); horizontal projection profile as a fallback for clean printed pages.
- Returns a list of line crops + their bounding boxes.

### 2. OCR — `src/ocr_trocr.py`
- `microsoft/trocr-base-handwritten` via `transformers` (the **base** variant, not large — see Hardware notes for rationale).
- Run `model.generate(..., output_scores=True, return_dict_in_generate=True)`; retain per-token logits for the flagger's feature extraction.
- Returns `list[Line]` where `Line = {text, bbox, token_logprobs}`. Mean/min/std of per-token logprobs are computed downstream by the flagger, not here.
- Cache the processor/model as a module-level singleton.

### 3. Post-correction — `src/postcorrect.py`
- **One Claude vision call per document**, not per line. The full scan image is sent alongside the TrOCR transcription with line IDs inline. Tool-use schema returns `{lines: [{line_id, original, corrected, changed: bool, llm_confidence: float}]}`.
- Model: `claude-haiku-4-5-20251001` (latency + cost favored over `opus`); the prompt at `prompts/v1/postcorrect.md` is cached via `cache_control` since it's reused per document.
- Why per-document, not per-line: a single call lets the model use cross-line context (a name introduced on line 2 likely recurs on line 7), and it cuts cost by ~30x vs per-line.
- The post-corrected text becomes the canonical transcription that flows downstream. The raw TrOCR text + per-token logprobs are retained and passed to the flagger as features.
- **Skipped under `--no-api`**: the pipeline degrades to TrOCR-only output, with a banner in the UI noting the post-correction was skipped.

### 4. Classification — `src/classify.py`
- Single Claude API call (model: `claude-haiku-4-5-20251001` for cost) using **tool use** with a structured schema: `{"doc_type": "letter|receipt|ledger|deed", "confidence": float, "reasoning": str}`.
- Input: full transcribed text (truncate at ~4k tokens). Cache the system prompt with `cache_control` since it's reused per document.
- Backup: `facebook/bart-large-mnli` zero-shot pipeline. Activated by `--no-api` flag for offline reproducibility.

### 5. NER — `src/ner.py`
- **Baseline**: `spacy` for PERSON / DATE / GPE / ORG. Default to `en_core_web_sm` for portability; `en_core_web_trf` is a drop-in upgrade where memory allows (see Hardware notes). Always run.
- **Custom entities via Claude**: same call as classification or a follow-up tool — extracts `sender`, `recipient`, `amount`, `signed_date`, `referenced_places` depending on doc type. Schema branches on `doc_type`.
- Output is the union, with each entity tagged by source (`spacy` or `claude`) for transparency.

### 6. Learned residual-error flagger + review queue — `src/flagger.py` + `notebooks/flagger.ipynb`

This is the project's headline contribution. The flagger predicts whether a line is *still* wrong **after** Claude vision post-correction — these are the lines that need a human.

**Training data** (one-shot, via `scripts/cache_iam_gw.py`): run TrOCR per line image (FKI's pre-segmented line crops, perfect 1:1 alignment with ground truth) and Claude vision post-correction per page (sending the original LoC page scan downloaded by `scripts/download_gw_pages.py`, so training-time inputs match what production sees). Label `is_still_wrong = (CER(corrected, gt) > 0)`. Cache to `data/parquet_cache/iam_gw_pipeline.parquet` with columns `(doc_id, line_id, trocr_text, trocr_token_logprobs, n_tokens, mean_logprob, min_logprob, std_logprob, length_normalized_logprob, corrected_text, llm_confidence, changed, edit_distance_trocr_vs_corrected, n_chars_changed, frac_chars_changed, line_height_px, line_width_px, gt, gt_cer, is_still_wrong)`.

**Features** (per line, computed in `src/flagger.py`):
- *TrOCR confidence signals*: `mean_logprob`, `min_logprob`, `std_logprob`, `length_normalized_logprob`, `n_tokens`
- *Agreement signals between TrOCR and post-correction* (the strongest predictors, expected): `edit_distance_trocr_vs_corrected`, `n_chars_changed`, `frac_chars_changed`, `llm_confidence` (self-reported)
- *Linguistic plausibility of the corrected text*: `char_perplexity` (character-level n-gram model fit on a clean English corpus), `edit_distance_to_nearest_dict_word`, `frac_oov_tokens`, `frac_punctuation_tokens`
- *Geometry from preprocess*: `line_height_px`, `line_width_px`, `deskew_angle_residual`, `line_segmentation_confidence`

**Model**: logistic regression with standardized features (sklearn). Reported in `notebooks/flagger.ipynb` with: ROC curve + AUC, Brier score, calibration via `CalibratedClassifierCV` if needed, feature importance (coefficients × std). 5-fold CV over IAM-GW; final eval on the hand-labeled LoC holdout. The expected interesting finding: agreement features dominate logprob features once post-correction is in the loop.

**Layered human-readable reasons**: alongside the probability, the flagger emits a small set of reason codes derived from which features fired most strongly (e.g. `LOW_MIN_LOGPROB_TOKEN`, `MANY_OOV_WORDS`, `DESKEW_RESIDUAL_HIGH`). Each code maps to a one-line human-readable string surfaced in the review queue. Reasons are interpretive, not predictive — the probability is the source of truth.

**Threshold**: chosen on a held-out IAM-GW slice by maximizing F1 of error detection. The chosen value, with its precision/recall, is reported in the README and tunable in the Streamlit sidebar.

**Graceful fallback**: if `models/flagger_v1.pkl` is missing or fails to load, `src/flagger.py` falls back to a rule-based flagger that uses the same feature set with hand-tuned thresholds and emits the same reason codes — so the review queue surface degrades but never disappears.

**Output**: `flag(line_features) -> {prob_wrong: float, flagged: bool, reasons: list[str]}`.

### 7. Pipeline orchestration — `src/pipeline.py`
- `process(image_path) -> DocumentResult` runs preprocess → OCR → post-correct → classify → NER → flagger.
- `DocumentResult` is a dataclass holding both `raw_lines` (TrOCR) and `corrected_lines` (post-corrected) plus the flagger output.
- `.to_dict()` for JSON, `.to_row()` for CSV.

### 8. Batch — `src/batch.py`
- CLI: `python -m src.batch data/raw/loc/ -o catalogue.csv`.
- Sequential processing for a 30–50 doc demo set (post-correction's per-doc Claude call dominates; multiprocessing isn't worth the PyTorch fork hazards).
- Output columns: `filename, doc_type, doc_type_confidence, mean_prob_wrong, n_review_lines, persons, dates, places, signed_date, amount`.
- Screenshot the resulting CSV for the README.

### 9. Evaluation — `scripts/evaluate.py` + notebooks
- **CER/WER comparison table** (the digitization story): `jiwer` on (a) TrOCR raw, (b) TrOCR + Claude vision post-correction. Computed on IAM-GW *and* the 8-doc hand-labeled LoC holdout. Expected story in the README: *"Post-correction reduces CER from ~0.40 → ~0.10 on LoC; the flagger catches Z% of the remaining errors when reviewing W% of lines."*
- **Flagger evaluation** (`notebooks/flagger.ipynb`): ROC curve + AUC, Brier score, feature importance, 5-fold CV.
- **Flagger calibration** (`notebooks/calibration.ipynb`): reliability diagram, threshold-via-F1, precision/recall of flagger at chosen threshold, confusion-matrix-style breakdown of flagged-vs-unflagged line accuracy. *Proves* the headline differentiator.
- **Error analysis** (`notebooks/errors.ipynb`): char-level confusion matrix on the residual errors (post-correction did not fix), 3 cherry-picked failure cases with reason codes.

### 10. Streamlit demo — `app.py`
- Sidebar: file uploader, threshold slider (controls the flagger cutoff), toggle to skip post-correction (`--no-api` parity).
- Main area, four tabs:
  - **Image** — original scan with bbox overlays color-coded by `prob_wrong` (post-correction residual).
  - **Transcription** — per-line view with three columns: TrOCR raw text | Claude-vision-corrected text | corrections diff (insertions/deletions highlighted). Confidence chip per line.
  - **Structured** — doc type, entities table, JSON view.
  - **Review queue** — flagged corrected-lines side-by-side with their crops, each annotated with both the probability and the human-readable reasons.
- Caches model load via `@st.cache_resource`.

## Reused utilities (don't reinvent)

- `transformers` (TrOCR), `spacy`, `jiwer` (CER/WER), `scikit-learn` (flagger model + calibration), `opencv-python`, `streamlit`, `anthropic` SDK, `python-dotenv`, `pyarrow` (parquet cache).
- `deskew` package (one-liner deskewing), `doctr` (line detector — default for cursive, projection-profile fallback).

## Build order

Each phase has its own verification. The flagger notebook (phase 5) is the critical path — protect time for it.

1. **Scaffold** — repo skeleton, `requirements.txt`, `download_loc.py`, pull LoC. Hand-label 8 LoC docs into `data/hand_labeled/`.
2. **OCR + cache** — `preprocess.py`, `ocr_trocr.py` with per-token logprobs.
3. **Post-correction** — `src/postcorrect.py` + `prompts/v1/postcorrect.md`. End-to-end on one doc: scan → TrOCR → Claude vision corrections → corrected lines.
4. **Cache training data** — register for IAM-HistDB, extract `washingtondb-v1.0` into `data/raw/iam_gw/`, then `python scripts/download_gw_pages.py` (pulls 20 LoC page scans for the GW pages, ~4 MB), then `python scripts/cache_iam_gw.py` (TrOCR + post-correction over the FKI line images using LoC page scans for visual context) to produce `data/parquet_cache/iam_gw_pipeline.parquet`. **Commit this parquet** so the flagger notebook can iterate without re-running the heavy steps.
5. **Pipeline (rule-based flagger)** — `pipeline.py` end-to-end on one doc with the rule-based fallback flagger; produces JSON. Confirms the surface works before the learned model exists.
6. **Learned flagger (headline)** — `notebooks/flagger.ipynb`: feature engineering, train logistic regression, ROC + Brier + feature importance. Persist to `models/flagger_v1.pkl`. Wire into `src/flagger.py`.
7. **Calibration** — `notebooks/calibration.ipynb`: reliability diagram + threshold-via-F1 + precision/recall on the LoC holdout.
8. **Claude integration** — `classify.py` + `ner.py` with prompt caching; prompts versioned in `prompts/v1/`.
9. **Batch** — `batch.py` → `catalogue.csv` on the LoC set.
10. **UI** — `app.py` Streamlit with the four tabs.
11. **Error analysis** — `notebooks/errors.ipynb`: char-level confusion + cherry-picked failures.
12. **Writeup + polish** — `WRITEUP.md`, README (architecture diagram, screenshots, install steps incl. `python -m spacy download en_core_web_sm`), short demo GIF.

## Verification

End-to-end checks before declaring done:
1. `streamlit run app.py`, upload a sample LoC letter → all four tabs populated; the Transcription tab shows TrOCR-vs-corrected diffs; the Review queue contains at least one flagged line with a probability and a human-readable reason on a deliberately noisy scan.
2. `python -m src.batch data/samples/ -o /tmp/catalogue.csv` → CSV has one row per image with non-empty `doc_type` and `persons`.
3. `python scripts/evaluate.py` prints a CER/WER table comparing TrOCR raw vs TrOCR + post-correction (IAM-GW and LoC holdout) and the flagger's ROC AUC + Brier score.
4. Deleting `models/flagger_v1.pkl` and re-running the pipeline still produces a populated review queue (rule-based fallback works).
5. `python -m src.pipeline data/samples/letter1.jpg --no-api` runs offline (skips post-correction; uses BART-MNLI fallback for classify) — proves no hard API dependency.
6. Repo clones and runs cleanly in a fresh venv on macOS with the README instructions only — no undocumented step.

## Responsible AI

| Principle | How this system implements it |
|---|---|
| **Accuracy** | Published CER/WER on IAM-GW + 8-doc LoC holdout; flagger evaluated with ROC, Brier, and a reliability diagram in `notebooks/flagger.ipynb` and `notebooks/calibration.ipynb`. |
| **Transparency** | Flagger feature importances are published; every flagged line carries human-readable reason codes alongside its probability. Each entity is tagged by source (`spacy` or `claude`); Claude calls return reasoning. Prompts versioned in `prompts/v1/`. |
| **Human oversight** | Flagged lines surface in the **Review queue** tab (`app.py`) with crops, OCR text, probability, and reasons — the system fails loudly *and explains why*. |
| **Accountability** | Per-line probability and reasons are persisted to `catalogue.csv`, so every downstream record carries the provenance needed to audit it. |

## Cost

TrOCR runs locally — free at inference time. Claude API is used for three things per document: post-correction (vision call with the full scan + TrOCR text), classification, and custom-entity extraction. All on `claude-haiku-4-5-20251001` with `cache_control` enabled on system prompts. The post-correction call dominates per-document cost. The README reports the measured per-document cost from the batch run and projects to 10k documents by linear scaling. For high-volume backfills the system can be run in `--no-api` mode (TrOCR-only output, flagger surfaces the much-larger error surface) and selectively post-corrected only on flagged documents.

## Stretch

- **Fine-tune TrOCR** (LoRA) on hand-labeled LoC lines and report the CER delta vs the off-the-shelf model — concrete evidence of distribution-shift handling. Add a third row to the eval table (TrOCR raw / TrOCR fine-tuned / + post-correction).
- **Pure Claude-vision OCR arm** (no TrOCR involved) as an additional eval row — "what does cutting TrOCR out entirely cost or save?" Useful comparison for a cost-quality discussion.
- **Active-learning analysis**: with the trained flagger, plot review-budget vs residual-error-recall ("reviewing 10% of post-corrected lines catches 75% of remaining errors") on the LoC holdout.
- **Embedding search** over `catalogue.csv` (`sentence-transformers`) so the catalogue can be queried semantically; expose as a 5th Streamlit tab.
- Per-class confusion matrix from the hand-labeled subset.

## Productionization sketch (for WRITEUP.md)

Brief notes on how this would deploy beyond the laptop demo — useful for the writeup but not built:

- **Storage**: raw scans in S3 (or any object store), keyed by document ID; structured output (the `catalogue.csv` schema) in a queryable store (DynamoDB or Postgres). Per-line probabilities and reasons live alongside the corrected text so any consumer can decide whether to trust a record.
- **Worker pool**: TrOCR inference and the per-document Claude vision call are independent per document — embarrassingly parallel. Run as a Lambda or a small autoscaling worker pool reading from a queue (SQS / Pub/Sub).
- **Human review surface**: the Streamlit demo is the prototype; the production version is a thin web UI that pulls flagged lines from the catalogue store, ordered by `prob_wrong` descending, and lets a reviewer accept/correct each.
- **Throughput / cost trade-off**: the `--no-api` path (TrOCR only, larger error surface, flagger does more work) is the cheap-fast lane for high-volume backfills; selective post-correction only on flagged docs is the middle ground; full post-correction is the high-quality lane.
- **Where this generalizes**: any image-bearing document a fast-moving org needs to digest reliably — archived correspondence, leaked PDFs, court filings, scanned press releases. The "post-correct then flag what's still wrong" pattern is domain-agnostic.

## Hardware notes

- **TrOCR variant**: use `microsoft/trocr-base-handwritten` (~330M params), not `-large`. On Apple Silicon (MPS), base does ~1 sec/line vs large's ~5–8 sec/line; the full IAM-GW pass takes ~5 min vs ~30+ min. CER will be a few points worse than large but the flagger story is about *which* errors happen, not absolute accuracy.
- **spaCy model**: default to `en_core_web_sm` (15MB) rather than `en_core_web_trf` (500MB) on machines with ≤8GB RAM — TrOCR + spaCy-trf + Streamlit + browser will swap heavily. On 16GB+ machines, `_trf` is fine and gives meaningfully better NER.
- **No GPU required.** All training (the sklearn flagger) and inference (TrOCR-base) run comfortably on CPU/MPS. No fine-tuning in the core scope.
- **One-time IAM-GW cache**: the heaviest single operation. Run `scripts/download_gw_pages.py` once (~30 seconds, ~4 MB from LoC), then `scripts/cache_iam_gw.py` once (~5–10 min, ~$0.10 in Claude API). Commit the resulting parquet so the flagger notebook can be developed/iterated without re-running TrOCR or the API.
- **Disk**: ~3 GB total (HuggingFace model cache dominates).