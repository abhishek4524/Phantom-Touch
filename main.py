"""
PhantomTouch: Gesture-Controlled Real-Time Webcam Invisibility System.

Usage:
    python main.py
"""

import sys
import os
from pathlib import Path

# Ensure src/ directory is in Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phantomtouch.app.application import run

if __name__ == "__main__":
    sys.exit(run())
