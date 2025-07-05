"""
Hardware detection and optimization for Sovereign AI Agent
"""

import logging
import platform
import psutil
import torch
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU information structure"""
    name: str
    memory_total: float  # GB
    memory_free: float   # GB
    compute_capability: Tuple[int, int]
    cuda_version: Optional[str] = None
    is_available: bool = False


@dataclass
class SystemInfo:
    """System information structure"""
    platform: str
    cpu_count: int
    memory_total: float  # GB
    python_version: str
    pytorch_version: str
    cuda_available: bool
    gpu_info: Optional[GPUInfo] = None


class HardwareDetector:
    """Hardware detection and optimization utilities"""
    
    def __init__(self):
        self.system_info = self._detect_system()
        self.gpu_info = self._detect_gpu()
        self._log_system_info()
    
    def _detect_system(self) -> SystemInfo:
        """Detect basic system information"""
        try:
            memory_bytes = psutil.virtual_memory().total
            memory_gb = memory_bytes / (1024**3)
            
            system_info = SystemInfo(
                platform=platform.platform(),
                cpu_count=psutil.cpu_count(),
                memory_total=memory_gb,
                python_version=platform.python_version(),
                pytorch_version=torch.__version__,
                cuda_available=torch.cuda.is_available()
            )
            
            return system_info
            
        except Exception as e:
            logger.error(f"Error detecting system information: {e}")
            # Return minimal info
            return SystemInfo(
                platform="Unknown",
                cpu_count=4,
                memory_total=8.0,
                python_version="Unknown",
                pytorch_version="Unknown",
                cuda_available=False
            )
    
    def _detect_gpu(self) -> Optional[GPUInfo]:
        """Detect GPU information and capabilities"""
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available")
            return None
        
        try:
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            
            memory_total = properties.total_memory / (1024**3)  # Convert to GB
            memory_free = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            
            # Get free memory more accurately
            torch.cuda.empty_cache()
            memory_free = torch.cuda.memory_reserved(device) / (1024**3)
            memory_free = memory_total - memory_free
            
            gpu_info = GPUInfo(
                name=properties.name,
                memory_total=memory_total,
                memory_free=memory_free,
                compute_capability=properties.major_minor,
                cuda_version=torch.version.cuda,
                is_available=True
            )
            
            return gpu_info
            
        except Exception as e:
            logger.error(f"Error detecting GPU information: {e}")
            return None
    
    def _log_system_info(self):
        """Log comprehensive system information"""
        logger.info("=== System Information ===")
        logger.info(f"Platform: {self.system_info.platform}")
        logger.info(f"CPU Cores: {self.system_info.cpu_count}")
        logger.info(f"System Memory: {self.system_info.memory_total:.1f} GB")
        logger.info(f"Python Version: {self.system_info.python_version}")
        logger.info(f"PyTorch Version: {self.system_info.pytorch_version}")
        logger.info(f"CUDA Available: {self.system_info.cuda_available}")
        
        if self.gpu_info:
            logger.info("=== GPU Information ===")
            logger.info(f"GPU Name: {self.gpu_info.name}")
            logger.info(f"GPU Memory Total: {self.gpu_info.memory_total:.1f} GB")
            logger.info(f"GPU Memory Free: {self.gpu_info.memory_free:.1f} GB")
            logger.info(f"Compute Capability: {self.gpu_info.compute_capability}")
            logger.info(f"CUDA Version: {self.gpu_info.cuda_version}")
        else:
            logger.warning("No GPU information available")
        
        logger.info("=" * 30)
    
    def verify_requirements(self) -> Tuple[bool, list]:
        """
        Verify system meets minimum requirements for Sovereign AI Agent
        
        Returns:
            Tuple of (meets_requirements, list_of_issues)
        """
        issues = []
        
        # Check Python version (3.10+)
        python_version = platform.python_version_tuple()
        if int(python_version[0]) < 3 or (int(python_version[0]) == 3 and int(python_version[1]) < 10):
            issues.append(f"Python 3.10+ required, found {platform.python_version()}")
        
        # Check system memory (minimum 16GB recommended)
        if self.system_info.memory_total < 16.0:
            issues.append(f"16GB+ system memory recommended, found {self.system_info.memory_total:.1f} GB")
        
        # Check GPU requirements
        if not self.system_info.cuda_available:
            issues.append("CUDA-capable GPU required for optimal performance")
        elif self.gpu_info:
            # Check GPU memory (16GB recommended)
            if self.gpu_info.memory_total < 16.0:
                issues.append(f"16GB+ GPU memory recommended (RTX 5070 Ti or equivalent), found {self.gpu_info.memory_total:.1f} GB")
            
            # Check compute capability (7.0+ recommended)
            compute_version = self.gpu_info.compute_capability[0] + self.gpu_info.compute_capability[1] * 0.1
            if compute_version < 7.0:
                issues.append(f"GPU compute capability 7.0+ recommended, found {self.gpu_info.compute_capability}")
        
        return len(issues) == 0, issues
    
    def get_optimal_settings(self) -> Dict[str, Any]:
        """
        Get optimal settings based on detected hardware
        
        Returns:
            Dictionary of recommended settings
        """
        settings = {}
        
        if self.gpu_info and self.gpu_info.memory_total >= 16.0:
            # High-end GPU settings
            settings.update({
                'batch_size': 4,
                'precision': 'float16',
                'max_sequence_length': 4096,
                'parallel_processing': True,
                'memory_efficient': False
            })
        elif self.gpu_info and self.gpu_info.memory_total >= 8.0:
            # Mid-range GPU settings
            settings.update({
                'batch_size': 2,
                'precision': 'int8',
                'max_sequence_length': 2048,
                'parallel_processing': False,
                'memory_efficient': True
            })
        else:
            # CPU or low-memory settings
            settings.update({
                'batch_size': 1,
                'precision': 'float32',
                'max_sequence_length': 1024,
                'parallel_processing': False,
                'memory_efficient': True,
                'use_cpu': True
            })
        
        # CPU settings
        settings['cpu_threads'] = min(self.system_info.cpu_count, 8)  # Cap at 8 threads
        
        return settings
    
    def optimize_pytorch(self):
        """Apply PyTorch optimizations based on hardware"""
        try:
            if torch.cuda.is_available():
                # Enable CUDA optimizations
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                
                # Set memory management
                torch.cuda.empty_cache()
                
                logger.info("PyTorch CUDA optimizations applied")
            
            # CPU optimizations
            torch.set_num_threads(min(self.system_info.cpu_count, 8))
            
            # Memory optimizations
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Metal Performance Shaders (MPS) available")
            
            logger.info("PyTorch optimizations applied")
            
        except Exception as e:
            logger.error(f"Error applying PyTorch optimizations: {e}")


# Global hardware detector instance
hardware_detector = HardwareDetector()


def get_device() -> torch.device:
    """Get the optimal PyTorch device for this system"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def check_system_requirements() -> bool:
    """
    Check if system meets requirements and log results
    
    Returns:
        True if requirements are met, False otherwise
    """
    meets_requirements, issues = hardware_detector.verify_requirements()
    
    if meets_requirements:
        logger.info("✅ System meets all requirements for Sovereign AI Agent")
    else:
        logger.warning("⚠️  System requirements check:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        logger.warning("The system may still work but performance could be suboptimal")
    
    return meets_requirements 