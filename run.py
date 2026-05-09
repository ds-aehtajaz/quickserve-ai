"""
Single-command launcher for QuickServe.AI.
Starts the FastAPI backend and Streamlit UI together. Press Ctrl+C to stop both.

Usage:
    python run.py
"""
import os
import sys
import signal
import subprocess
import time
import urllib.request

PYTHON = sys.executable
HERE   = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def wait_for_backend(url="http://localhost:8000/health", timeout=30):
    print("Waiting for FastAPI backend to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    print("  -> Backend is up.")
                    return True
        except Exception:
            time.sleep(0.5)
    print("  -> WARNING: backend did not respond within timeout. Continuing anyway.")
    return False


def main():
    print("=" * 60)
    print(" QuickServe.AI — starting backend + UI")
    print("=" * 60)
    print(f"Python: {PYTHON}")
    print()

    # Start FastAPI (uvicorn) in a subprocess
    backend = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "src.quickserve.api.main:app",
         "--host", "0.0.0.0", "--port", "8000"],
        cwd=HERE,
    )

    try:
        wait_for_backend()

        # Start Streamlit in a subprocess
        print("\nStarting Streamlit UI on http://localhost:8501 ...\n")
        ui = subprocess.Popen(
            [PYTHON, "-m", "streamlit", "run", "src/quickserve/ui/app.py",
             "--server.headless=false"],
            cwd=HERE,
        )

        # Wait for either process to exit
        while True:
            time.sleep(1)
            if backend.poll() is not None:
                print("Backend exited. Shutting down UI...")
                ui.terminate()
                break
            if ui.poll() is not None:
                print("UI exited. Shutting down backend...")
                backend.terminate()
                break

    except KeyboardInterrupt:
        print("\n\nCtrl+C received — shutting down both services...")
        try:
            ui.terminate()
        except Exception:
            pass
        backend.terminate()

    # Wait for clean shutdown
    for proc in [backend]:
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    print("Goodbye.")


if __name__ == "__main__":
    main()
