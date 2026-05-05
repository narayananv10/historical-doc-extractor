# Hugging Face Spaces deploy image (SDK = Docker).
#
# Pre-downloads the doctr and TrOCR weights at build time so:
#   - the doctr 308-redirect bug doesn't surface at runtime
#   - first-user cold-start doesn't pay the ~5 min model-download tax
#
# HF Spaces requires the app to listen on 0.0.0.0:7860 and to run as a
# non-root user (uid 1000). Standard pattern.

FROM python:3.12-slim

# curl is needed by scripts/setup_models.py to fetch the doctr weights
# (urllib doesn't follow the 308 redirect; we worked around it with curl).
RUN apt-get update \
 && apt-get upgrade -y \
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
RUN pip install --no-cache-dir --user -r requirements.txt \
 && pip uninstall -y opencv-python \
 && pip install --no-cache-dir --user --force-reinstall opencv-python-headless
# python-doctr[torch] declares opencv-python (the GUI variant) as a transitive
# dep. Both opencv-python and opencv-python-headless install to the same
# site-packages/cv2/ directory; whichever pip installs last wins. doctr's
# install order overwrites our headless cv2 with the GUI one, which then
# tries to load libxcb at import time and crashes on the slim base image.
# Explicitly uninstall the GUI variant and force-reinstall headless so the
# final cv2/ contents are the no-X11 build.

# Layer 2 — pre-cache model weights. Only invalidated when the setup script
# changes (typically: never).
COPY --chown=user scripts/setup_models.py scripts/setup_models.py
RUN python scripts/setup_models.py

# Layer 3 — the app itself. Invalidated on every code change, but the
# expensive layers above stay cached.
COPY --chown=user . .

EXPOSE 7860
# CORS + XSRF disabled because HF Spaces proxies Streamlit through an iframe;
# the default XSRF token check rejects the file-upload POST as cross-origin
# and the browser shows "AxiosError 403". HF's reverse proxy handles
# same-origin enforcement at the platform layer, so disabling these in
# Streamlit is safe for this deployment.
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
