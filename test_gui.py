#!/usr/bin/env python3
"""
Simple GUI test script for Sovereign AI Agent

This script tests the GUI functionality in isolation to ensure
it works correctly before full integration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import asyncio
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gui_test.log')
    ]
)

logger = logging.getLogger("gui_test")

def test_gui_imports():
    """Test that GUI imports work correctly"""
    logger.info("🧪 Testing GUI imports...")
    
    try:
        # Test basic imports
        from sovereign.config import config
        logger.info("✅ Config import successful")
        
        # Test GUI imports
        from sovereign.gui import SovereignGUI, run_gui
        logger.info("✅ GUI imports successful")
        
        # Test optional imports
        from sovereign import GUI_AVAILABLE
        logger.info(f"✅ GUI availability: {GUI_AVAILABLE}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def test_gui_creation():
    """Test GUI creation without showing it"""
    logger.info("🧪 Testing GUI creation...")
    
    try:
        from sovereign.gui import SovereignGUI
        
        # Create GUI instance
        gui = SovereignGUI()
        logger.info("✅ GUI instance created successfully")
        
        # Test GUI setup without showing
        gui.setup_gui()
        logger.info("✅ GUI setup completed successfully")
        
        # Test root window creation
        if gui.root:
            logger.info("✅ Root window created successfully")
            gui.root.withdraw()  # Hide the window
            gui.root.quit()
            gui.root.destroy()
            logger.info("✅ Root window destroyed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GUI creation error: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    logger.info("🧪 Testing configuration loading...")
    
    try:
        from sovereign.config import config
        
        # Test basic config access
        logger.info(f"✅ Config loaded: {config.debug}")
        logger.info(f"✅ Hardware config: {config.hardware.gpu_enabled}")
        logger.info(f"✅ Voice config: {config.voice.enabled}")
        logger.info(f"✅ Screen capture config: {config.screen_capture.enabled}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config loading error: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting GUI test suite...")
    
    # Test results
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Imports
    if test_gui_imports():
        tests_passed += 1
    
    # Test 2: Config loading
    if test_config_loading():
        tests_passed += 1
    
    # Test 3: GUI creation
    if test_gui_creation():
        tests_passed += 1
    
    # Summary
    logger.info(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        logger.info("✅ All tests passed! GUI is ready for use.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 