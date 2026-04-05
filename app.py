"""Hugging Face Spaces entrypoint: `streamlit run app.py` from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.streamlit_app import main

main()
