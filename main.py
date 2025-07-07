#!/usr/bin/env python3
"""
Sovereign AI Agent - Main Entry Point

Ultra-lightweight launcher that starts the application in < 1 second.
No heavy imports - all services are lazy-loaded on demand.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from sovereign.core.app import CoreApp


def main():
    """Main entry point for Sovereign AI Agent"""
    from time import perf_counter
    start_time = perf_counter()
    
    print("🚀 Sovereign AI Agent v2.0 - Starting...")
    
    # Create and run the core application
    app = CoreApp()
    app.run()
    
    # Report startup time
    startup_time = perf_counter() - start_time
    print(f"✅ Startup completed in {startup_time:.3f}s")


if __name__ == "__main__":
    main() 