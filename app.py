"""
Streamlit Cloud entry point.
Streamlit Cloud runs `streamlit run app.py`, so this file IS the app.
It runs the agent in-process (no separate FastAPI backend needed on the cloud).
"""
import os
import sys

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Force in-process mode (UI calls the agent directly, not via HTTP)
os.environ["QUICKSERVE_INPROCESS"] = "1"

# Now import and run the UI module.
# Since the file uses top-level Streamlit calls, importing it executes the page.
import quickserve.ui.app  # noqa: F401
