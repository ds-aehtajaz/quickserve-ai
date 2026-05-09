"""
Hugging Face Spaces entry point.
Runs the Streamlit chat UI directly (no separate FastAPI process needed on HF Spaces).
The agent is invoked in-process so we don't need uvicorn here.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Force in-process agent mode (no HTTP backend on HF Spaces)
os.environ.setdefault("QUICKSERVE_INPROCESS", "1")

# Run the Streamlit UI
import streamlit.web.bootstrap as bootstrap
import streamlit.web.cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "src/quickserve/ui/app.py",
                "--server.port=7860",
                "--server.address=0.0.0.0",
                "--server.headless=true"]
    sys.exit(stcli.main())
