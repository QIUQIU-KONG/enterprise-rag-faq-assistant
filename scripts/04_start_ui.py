#!/usr/bin/env python
"""Step 4: Start the Streamlit chat UI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess

if __name__ == "__main__":
    app_path = Path(__file__).parent.parent / "app" / "streamlit_app.py"
    subprocess.run([
        "streamlit", "run", str(app_path),
        "--server.port", "8501",
        "--server.address", "localhost",
    ])
