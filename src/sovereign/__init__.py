"""
Sovereign AI Agent - A private, powerful, locally-running AI assistant

This package provides a dual-model AI architecture with local execution,
privacy-first design, and advanced features including voice interface,
screen context awareness, and external tool integration.
"""

from .config import Config, config
from .logger import setup_logger, get_performance_logger, PerformanceTimer
# COMMENTED OUT TO FIX STARTUP LAG: hardware imports torch at module level
# from .hardware import check_system_requirements, hardware_detector, get_device
from .ollama_client import OllamaClient, OllamaError
# COMMENTED OUT TO FIX STARTUP LAG: these models import torch at module level
# from .talker_model import TalkerModel
# from .thinker_model import ThinkerModel, TaskType
# from .orchestrator import ModelOrchestrator, QueryContext, OrchestrationResult, QueryComplexity, ModelChoice

# Optional voice interface imports (requires audio dependencies)
try:
    from .voice_interface import VoiceInterfaceManager, VoiceState, VoiceMetrics, test_voice_interface, get_audio_devices
    VOICE_INTERFACE_AVAILABLE = True
except ImportError:
    # Voice interface dependencies not available
    VoiceInterfaceManager = None
    VoiceState = None
    VoiceMetrics = None
    test_voice_interface = None
    get_audio_devices = None
    VOICE_INTERFACE_AVAILABLE = False

# Optional GUI imports (requires GUI dependencies)
try:
    from .gui import SovereignGUI, run_gui
    GUI_AVAILABLE = True
except ImportError:
    # GUI dependencies not available
    SovereignGUI = None
    run_gui = None
    GUI_AVAILABLE = False

from .cli import main

__version__ = "0.1.0"
__author__ = "Sovereign AI Team"
__email__ = "contact@sovereign-ai.com"

# Base exports
__all__ = [
    "Config",
    "config",
    "setup_logger",
    "get_performance_logger",
    "PerformanceTimer",
    # Hardware functions commented out to fix startup lag
    # "check_system_requirements",
    # "hardware_detector", 
    # "get_device",
    "OllamaClient",
    "OllamaError",
    # Model classes commented out to fix startup lag
    # "TalkerModel",
    # "ThinkerModel", 
    # "TaskType",
    # "ModelOrchestrator",
    # "QueryContext",
    # "OrchestrationResult",
    # "QueryComplexity",
    # "ModelChoice",
    "VOICE_INTERFACE_AVAILABLE",
    "GUI_AVAILABLE",
    "main"
]

# Add voice interface exports if available
if VOICE_INTERFACE_AVAILABLE:
    __all__.extend([
        "VoiceInterfaceManager",
        "VoiceState",
        "VoiceMetrics",
        "test_voice_interface",
        "get_audio_devices"
    ])

# Add GUI exports if available
if GUI_AVAILABLE:
    __all__.extend([
        "SovereignGUI",
        "run_gui"
    ]) 