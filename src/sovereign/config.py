"""
Configuration management for Sovereign AI Agent
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    # Ollama configuration
    ollama_endpoint: str = "http://localhost:11434"
    talker_model: str = "gemma2:9b"  # Ollama model for fast responses
    thinker_model: str = "deepseek-r1:14b"  # Ollama model for complex reasoning
    
    # External routing
    external_router_url: Optional[str] = None
    
    # Performance settings
    max_sequence_length: int = 2048
    response_timeout: float = 30.0
    talker_timeout: float = 2.0  # Must respond in under 2 seconds
    thinker_timeout: float = 60.0  # More time for complex reasoning
    
    # Ollama-specific settings
    ollama_timeout: float = 10.0
    ollama_temperature: float = 0.7
    ollama_top_p: float = 0.9
    ollama_stream: bool = True
    
    # Thinker-specific settings
    thinker_temperature: float = 0.3  # Lower temperature for more focused reasoning
    thinker_max_tokens: int = 4096  # Longer responses for complex tasks
    thinker_context_window: int = 8192  # Larger context for complex reasoning
    
    # External model settings
    external_routing_enabled: bool = True
    external_routing_threshold: float = 0.3  # Minimum confidence for external routing
    external_model_provider: str = "openrouter"
    external_model_default: str = "anthropic/claude-3-sonnet"
    external_model_temperature: float = 0.7
    external_model_max_tokens: int = 2048
    external_model_timeout: float = 30.0
    external_cache_ttl_hours: int = 24
    external_require_user_consent: bool = True


@dataclass
class HardwareConfig:
    """Hardware and performance configuration"""
    gpu_enabled: bool = True
    cpu_threads: int = os.cpu_count() or 4
    memory_limit_gb: int = 16
    batch_size: int = 1
    precision: str = "float16"  # float16, float32, int8


@dataclass
class ScreenCaptureConfig:
    """Screen capture and OCR configuration"""
    enabled: bool = False  # Default disabled for privacy
    capture_interval: float = 5.0  # seconds
    ocr_enabled: bool = True
    max_screenshot_history: int = 100
    tesseract_cmd_path: Optional[str] = None  # Path to tesseract executable


@dataclass
class VoiceConfig:
    """Voice interface configuration"""
    # General settings
    enabled: bool = True
    wake_word: str = "hey sovereign"
    wake_word_enabled: bool = True
    wake_word_sensitivity: float = 0.5  # 0.0 (less sensitive) to 1.0 (more sensitive)
    
    # Audio settings
    sample_rate: int = 44100  # Hz
    chunk_duration_ms: int = 100  # milliseconds
    input_device_name: str = "default"
    output_device_name: str = "default"
    
    # Speech recognition
    stt_enabled: bool = True
    whisper_model_size: str = "base"  # tiny, base, small, medium, large
    speech_timeout_ms: int = 2000  # milliseconds
    silence_threshold: float = 0.01  # Energy threshold for silence detection
    
    # Text-to-speech
    tts_enabled: bool = True
    tts_voice_name: str = "default"
    tts_speech_rate: int = 200  # Words per minute
    tts_volume: float = 0.9  # 0.0 to 1.0
    
    # Voice Activity Detection
    vad_enabled: bool = True
    vad_aggressiveness: int = 2  # 0-3 (0 = least aggressive, 3 = most aggressive)
    
    # Audio processing
    noise_reduction_enabled: bool = True
    echo_cancellation_enabled: bool = True
    
    # Porcupine wake word detection
    porcupine_access_key: str = ""  # Required for Porcupine wake word detection
    
    # Performance settings
    max_audio_buffer_duration: float = 30.0  # seconds
    audio_processing_threads: int = 2


@dataclass
class DatabaseConfig:
    """Database and memory configuration"""
    db_path: str = "data/sovereign.db"
    max_conversation_history: int = 10000
    enable_rag: bool = True
    vector_db_path: str = "data/vectors"


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestration system"""
    # Cache settings
    cache_max_size: int = 1000
    cache_ttl_hours: int = 24
    
    # Complexity detection thresholds
    simple_threshold: float = -0.5
    moderate_threshold: float = 0.5
    complex_threshold: float = 1.5
    
    # Handoff settings
    handoff_confidence_threshold: float = 0.7
    handoff_response_length_threshold: int = 20
    
    # Performance settings
    max_processing_time: float = 30.0  # seconds
    enable_caching: bool = True
    enable_telemetry: bool = True
    
    # Notification settings
    enable_notifications: bool = True
    notification_delay: float = 0.5  # seconds


@dataclass
class Config:
    """Main configuration class"""
    models: ModelConfig
    hardware: HardwareConfig
    screen_capture: ScreenCaptureConfig
    voice: VoiceConfig
    database: DatabaseConfig
    orchestrator: OrchestratorConfig
    log_level: str = "INFO"
    debug: bool = False
    
    def __init__(self):
        self.models = ModelConfig()
        self.hardware = HardwareConfig()
        self.screen_capture = ScreenCaptureConfig()
        self.voice = VoiceConfig()
        self.database = DatabaseConfig()
        self.orchestrator = OrchestratorConfig()
        
        # Detect GPU capabilities
        self._detect_hardware()
        
        # Load from config file if exists
        self.load_config()
    
    def _detect_hardware(self):
        """Detect and configure hardware capabilities"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                logger.info(f"Detected GPU: {gpu_name}")
                logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
                
                # Configure based on GPU memory
                if gpu_memory >= 16:
                    self.hardware.memory_limit_gb = int(gpu_memory * 0.8)  # Use 80% of GPU memory
                    self.hardware.precision = "float16"
                elif gpu_memory >= 8:
                    self.hardware.memory_limit_gb = int(gpu_memory * 0.7)
                    self.hardware.precision = "int8"
                else:
                    logger.warning(f"GPU memory ({gpu_memory:.1f} GB) may be insufficient for optimal performance")
                    self.hardware.gpu_enabled = False
                    
            else:
                logger.warning("No CUDA-capable GPU detected. Falling back to CPU mode.")
                self.hardware.gpu_enabled = False
                
        except Exception as e:
            logger.error(f"Error detecting hardware: {e}")
            self.hardware.gpu_enabled = False
    
    def load_config(self, config_path: str = "config/config.json"):
        """Load configuration from file"""
        config_file = Path(config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update configuration with loaded data
                for section, values in config_data.items():
                    if hasattr(self, section):
                        section_obj = getattr(self, section)
                        for key, value in values.items():
                            if hasattr(section_obj, key):
                                setattr(section_obj, key, value)
                
                logger.info(f"Configuration loaded from {config_path}")
                
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
    
    def save_config(self, config_path: str = "config/config.json"):
        """Save current configuration to file"""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_data = {
                'models': asdict(self.models),
                'hardware': asdict(self.hardware),
                'screen_capture': asdict(self.screen_capture),
                'voice': asdict(self.voice),
                'database': asdict(self.database),
                'orchestrator': asdict(self.orchestrator),
                'log_level': self.log_level,
                'debug': self.debug
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get_env_vars(self) -> Dict[str, Any]:
        """Get configuration from environment variables"""
        env_vars = {}
        
        # Model configuration
        if os.getenv('TALKER_MODEL'):
            env_vars['talker_model'] = os.getenv('TALKER_MODEL')
        if os.getenv('THINKER_MODEL'):
            env_vars['thinker_model'] = os.getenv('THINKER_MODEL')
        
        # Hardware configuration
        if os.getenv('GPU_ENABLED'):
            env_vars['gpu_enabled'] = os.getenv('GPU_ENABLED').lower() == 'true'
        if os.getenv('CPU_THREADS'):
            env_vars['cpu_threads'] = int(os.getenv('CPU_THREADS'))
        
        # Debug mode
        if os.getenv('DEBUG'):
            env_vars['debug'] = os.getenv('DEBUG').lower() == 'true'
        if os.getenv('LOG_LEVEL'):
            env_vars['log_level'] = os.getenv('LOG_LEVEL')
        
        return env_vars


# Global configuration instance (lazy-loaded)
_config = None

def get_config():
    """Get the global config instance (lazy-loaded)"""
    global _config
    if _config is None:
        _config = Config()
    return _config

# Create a property-like object for backwards compatibility
class _ConfigProxy:
    def __getattr__(self, name):
        return getattr(get_config(), name)
    
    def __setattr__(self, name, value):
        setattr(get_config(), name, value)

config = _ConfigProxy() 