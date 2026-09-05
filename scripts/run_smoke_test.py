#!/usr/bin/env python3
"""
RLFinetuneLab Local Smoke Verification Script.
Executes an ultra-fast, zero-download test on local CPU with in-memory mock architecture.
Safe for laptops with 0 GPU and constrained RAM.
"""

import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rlfinetunelab.mock.dummy_models import run_local_smoke_test

if __name__ == "__main__":
    result = run_local_smoke_test()
    print("\nSmoke Test Result:", result["status"].upper())
    sys.exit(0 if result["status"] == "success" else 1)
