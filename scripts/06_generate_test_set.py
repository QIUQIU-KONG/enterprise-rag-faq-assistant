#!/usr/bin/env python
"""Generate evaluation test set without running full eval."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.test_set_generator import generate_test_set, save_test_set

if __name__ == "__main__":
    pairs = generate_test_set()
    save_test_set(pairs)
    print(f"Generated {len(pairs)} QA pairs → data/eval/test_set.json")
