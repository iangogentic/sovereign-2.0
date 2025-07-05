"""
Basic tests for Sovereign AI Agent
"""

import pytest
import torch
from pathlib import Path


def test_imports():
    """Test that core modules can be imported"""
    try:
        from sovereign import Config
        from sovereign.logger import setup_logger
        from sovereign.hardware import get_device, check_system_requirements
        from sovereign.cli import create_parser
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")


def test_config_creation():
    """Test that configuration can be created"""
    from sovereign.config import Config
    
    config = Config()
    assert config is not None
    assert hasattr(config, 'models')
    assert hasattr(config, 'hardware')
    assert hasattr(config, 'voice')
    assert hasattr(config, 'screen_capture')
    assert hasattr(config, 'database')


def test_logger_setup():
    """Test that logger can be set up"""
    from sovereign.logger import setup_logger
    
    logger = setup_logger("test", log_level="INFO")
    assert logger is not None
    assert logger.name == "test"


@pytest.mark.gpu
def test_gpu_detection():
    """Test GPU detection (only runs if GPU is available)"""
    from sovereign.hardware import hardware_detector
    
    if torch.cuda.is_available():
        assert hardware_detector.gpu_info is not None
        assert hardware_detector.gpu_info.is_available is True
        assert hardware_detector.gpu_info.memory_total > 0
    else:
        assert hardware_detector.gpu_info is None


def test_cli_parser():
    """Test CLI argument parser"""
    from sovereign.cli import create_parser
    
    parser = create_parser()
    assert parser is not None
    
    # Test basic argument parsing
    args = parser.parse_args(['--help'])
    # This will raise SystemExit, which is expected for --help


def test_project_structure():
    """Test that required project files exist"""
    required_files = [
        "requirements.txt",
        "setup.py",
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "src/sovereign/__init__.py",
        "src/sovereign/config.py",
        "src/sovereign/logger.py",
        "src/sovereign/hardware.py",
        "src/sovereign/cli.py",
    ]
    
    for file_path in required_files:
        assert Path(file_path).exists(), f"Required file {file_path} is missing"


@pytest.mark.unit
def test_system_requirements_check():
    """Test system requirements checking"""
    from sovereign.hardware import check_system_requirements
    
    # This should return a boolean without crashing
    result = check_system_requirements()
    assert isinstance(result, bool)


@pytest.mark.unit
def test_device_detection():
    """Test device detection"""
    from sovereign.hardware import get_device
    
    device = get_device()
    assert device is not None
    assert str(device) in ['cuda', 'mps', 'cpu'] 