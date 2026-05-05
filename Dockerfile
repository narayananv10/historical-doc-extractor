# Hugging Face Spaces deploy image (SDK = Docker).
#
# Pre-downloads the doctr and TrOCR weights at build time so:
#   - the doctr 308-redirect bug doesn't surface at runtime
#   - first-user cold-start doesn't pay the ~5 min model-download tax
#
# HF Spaces requires the app to listen on 0.0.0.0:7860 and to run as a
# non-root user (uid 1000). Standard pattern.

FROM python:3.10-slim

# curl is needed by scripts/setup_models.py to fetch the doctr weights
# (urllib doesn't follow the 308 redirect; we worked around it with curl).
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root user — HF Spaces convention. Created early so model caches land
# in the right home directory.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Layer 1 — Python deps. Only invalidated when requirements.txt changes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Layer 2 — pre-cache model weights. Only invalidated when the setup script
# changes (typically: never).
COPY --chown=user scripts/setup_models.py scripts/setup_models.py
RUN python scripts/setup_models.py

# Layer 3 — the app itself. Invalidated on every code change, but the
# expensive layers above stay cached.
COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]
