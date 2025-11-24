#!/usr/bin/env python3
"""
SSVEP BCI GUI Runner (Phase 2)

Run this from the repository root:
    python run_gui.py
"""

import sys
from pathlib import Path

# Add ssvep_bci to path
sys.path.insert(0, str(Path(__file__).parent / "ssvep_bci"))

from main import run_gui

if __name__ == "__main__":
    sys.exit(run_gui())
