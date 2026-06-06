#!/usr/bin/env python
"""Step 3: Start the FastAPI server."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from config.settings import settings

if __name__ == "__main__":
    print(f"Starting API server at http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"API docs: http://localhost:{settings.API_PORT}/docs")
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
    )
