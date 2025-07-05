#!/usr/bin/env python3
"""
Sovereign AI Agent - Main Launcher

This script provides a convenient entry point for launching the Sovereign AI Agent
in both command-line and graphical user interface modes.

Usage:
    python run_sovereign.py --gui          # Launch GUI
    python run_sovereign.py               # Launch CLI mode
    python run_sovereign.py --help        # Show help
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Now we can import from sovereign
from sovereign.cli import main

if __name__ == "__main__":
    main() 