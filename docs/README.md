# Demo asset

`demo.gif` should be a short (~10–20 second) screen recording of the Streamlit app in use, suitable for embedding at the top of the project README.

Suggested capture flow:
1. Start the app: `.venv/bin/streamlit run app.py`
2. In the sidebar, upload one of the LoC sample scans, e.g. `data/raw/loc/abraham-lincoln-papers/2020780882.jpg`
3. Click **Process** and let the pipeline run (~30–60 s; trim the wait out of the recording)
4. Walk through the five tabs in order: Image → Transcription → Structured → Summary → Review queue
5. End on the Review queue tab so the GIF leaves the viewer looking at the headline UX

Use any screen-capture tool that exports GIF directly (Kap on macOS, ScreenToGif on Windows, peek on Linux). Keep the file under ~5 MB so it loads quickly on GitHub.
