# Deploying to Hugging Face Spaces

Step-by-step for getting the Streamlit demo running publicly on a free Hugging Face Space, linked to this GitHub repository for auto-deploy on push.

## Prerequisites

- A free Hugging Face account: [huggingface.co/join](https://huggingface.co/join)
- An Anthropic API key (to keep in the Space secrets, not in any committed file)
- A HuggingFace access token with at least Read scope: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- This repository accessible to your HF account

## 1. Set a hard ceiling on the Anthropic API key

**Do this first, before the Space is public.** Anthropic console → Settings → Limits → set a monthly spend cap on the key. Any value you're comfortable with — e.g. $5/month is well above what a few hundred visitors browsing the demo would consume, and well below "I forgot the demo was up."

The hosted app already defaults the **Skip Claude API** toggle to ON when it detects HF Spaces (no API spend per visitor by default). The cap is a backstop in case the toggle is turned off and someone abuses it.

## 2. Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Owner**: your username
   - **Space name**: `historical-doc-extractor` (or anything you prefer)
   - **License**: MIT (matches the repository)
   - **SDK**: **Streamlit** (this is the important one — HF will auto-run `streamlit run app.py`)
   - **Hardware**: **CPU basic** (free tier). The app fits in 16 GB RAM after model downloads. GPU is overkill.
   - **Visibility**: Public
3. Click **Create Space**.

## 3. Configure secrets

Space page → **Settings** (top-right) → scroll to **Variables and secrets** → **New secret** for each:

| Name | Value | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | `sk-ant-…` | Used by post-correction / classify / extract when the toggle is off |
| `HF_TOKEN` | `hf_…` | Lifts HuggingFace download rate limits during cold start |

You don't need to set `HOSTED_DEMO` manually — HF Spaces sets `SPACE_ID` automatically and the app uses that to detect hosted mode (see `HOSTED_DEMO` in `app.py`).

## 4. Connect the GitHub repo

Two paths — pick one:

### Path A — Mirror from GitHub (recommended; auto-deploys on push)

1. Space page → **Settings** → scroll to **Repository** → **Sync with a GitHub repository**
2. Authorise the Hugging Face GitHub app for the org/user that owns the repo
3. Pick `narayananv10/historical-doc-extractor` and the `main` branch
4. Save. HF will pull the repo, run `pip install -r requirements.txt`, then start `streamlit run app.py`.
5. Future commits to `main` will trigger an automatic rebuild.

### Path B — Push directly to the Space's git remote (one-shot, no auto-sync)

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/historical-doc-extractor
git push hf main
```

Path B is useful for testing changes without merging to GitHub first, but Path A is what you want for normal operation.

## 5. Wait for the first build

The first build takes 8–15 minutes because HF needs to install heavy ML dependencies (PyTorch, transformers, doctr, spaCy + the en_core_web_sm wheel). Watch the **Logs** tab on the Space page.

After the build finishes, the Space starts. The first user upload triggers another ~2 minute delay while TrOCR weights download from HuggingFace and the doctr text-detection model loads. Subsequent uploads are fast.

## 6. Verify

Open the Space URL (`https://huggingface.co/spaces/<your-username>/historical-doc-extractor`). You should see:

- The blue info banner at the top: *"Hosted demo — Skip Claude API is on by default…"*
- The sidebar with the upload widget, threshold slider, "Skip Claude API" toggle (defaulting to ON), and a **View Source on GitHub** badge at the bottom
- Upload one of the LoC sample scans from `data/raw/loc/abraham-lincoln-papers/`. The pipeline runs in `--no-api` mode by default, so every line will be flagged with `NO_API_VERIFICATION` reason. That's the intended hosted behaviour — visitors see the architecture (TrOCR + spaCy + flagger surface) without burning your API budget.

To test the full pipeline including Claude calls: toggle **Skip Claude API** off and re-process. This will charge your `ANTHROPIC_API_KEY` (~$0.02 per page).

## 7. Link from your portfolio

The Space URL is stable: `https://huggingface.co/spaces/<your-username>/historical-doc-extractor`

Add it to your portfolio with a short blurb like:

> **Historical Document Extractor** — End-to-end pipeline (TrOCR + Claude vision + learned residual-error flagger) for digitising handwritten historical archives with confidence-aware human review. [Live demo](URL) · [Source](https://github.com/narayananv10/historical-doc-extractor)

## Cost expectations

| Scenario | Cost |
|---|---|
| Visitor lands on Space, doesn't upload | $0 |
| Visitor uploads + processes with `--no-api` (default) | $0 |
| Visitor toggles API on + processes one scan | ~$0.02 |
| Visitor toggles API on + processes 10 scans | ~$0.20 |

The Anthropic spend cap from step 1 is your hard ceiling regardless.

## Troubleshooting

**Build fails on dependency install (out of memory):** Free CPU tier has 16 GB RAM during build. If pip install OOMs, upgrade to **CPU upgrade** (still free; just gives the build job more RAM headroom).

**App starts but spaCy model not found:** The `en_core_web_sm` wheel URL in `requirements.txt` should install it automatically. If pip skipped the URL line for some reason, add a `pre-build.sh` or set `PIP_EXTRA_INDEX_URL` in Variables. Easiest fix: re-trigger the build via the Space's **Factory rebuild** button.

**Streamlit shows "Connection error":** First-load model downloads timed out. Wait 2 minutes and refresh — the models cache to disk after the first successful load.

**Doctr line-detection model fails to download:** Same urllib 308-redirect bug we hit locally. Add a build-time hook that runs `python scripts/setup_models.py` before the app starts. Easiest path: add `pre-startup` script via HF Space settings.

**The bottom of every transcription is the same garbage:** A single page got too tall and TrOCR's image processor downsampled it. Increase image quality before upload, or split very tall scans into pages.
