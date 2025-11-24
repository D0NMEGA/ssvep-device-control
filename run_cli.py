#!/usr/bin/env python3
"""
SSVEP BCI CLI Runner

Run this from the repository root:
    python run_cli.py
    python run_cli.py --target 12.0
    python run_cli.py --port COM3
"""

import sys
from pathlib import Path

# Add ssvep_bci to path
sys.path.insert(0, str(Path(__file__).parent / "ssvep_bci"))

from cli_test import main

if __name__ == "__main__":
    sys.exit(main())
