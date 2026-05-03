"""Load environment variables (HF_TOKEN, ANTHROPIC_API_KEY) once at import time
so downstream modules don't have to."""

from dotenv import load_dotenv

load_dotenv()
