"""
Hardware detection and optimization for Sovereign AI Agent
"""

import logging
import platform
import psutil
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
            import torch
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
        import torch
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
            import torch
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


# Global hardware detector instance (lazy-loaded)
_hardware_detector = None

def get_hardware_detector():
    """Get the global hardware detector instance (lazy-loaded)"""
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector()
    return _hardware_detector

# Create a proxy object for backwards compatibility
class _HardwareDetectorProxy:
    def __getattr__(self, name):
        return getattr(get_hardware_detector(), name)

hardware_detector = _HardwareDetectorProxy()


def get_device():
    """Get the optimal PyTorch device for this system"""
    import torch
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


def diagnose_gpu_environment():
    """
    Comprehensive GPU diagnostic function to identify CUDA/GPU issues
    """
    print("=" * 60)
    print("🔍 SOVEREIGN AI - GPU ENVIRONMENT DIAGNOSTIC")
    print("=" * 60)
    
    # Import torch here to ensure it's available
    import torch
    
    # Step 1: Print PyTorch version
    print(f"\n1. PyTorch Version:")
    print(f"   torch.__version__ = {torch.__version__}")
    
    # Step 2: Check CUDA availability
    print(f"\n2. CUDA Availability:")
    cuda_available = torch.cuda.is_available()
    print(f"   torch.cuda.is_available() = {cuda_available}")
    
    # Step 3: Print CUDA version that PyTorch was built with
    print(f"\n3. CUDA Version (PyTorch build):")
    cuda_version = torch.version.cuda
    print(f"   torch.version.cuda = {cuda_version}")
    
    # Step 4: Additional CUDA version info
    print(f"\n4. CUDA Runtime Version:")
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            nvcc_output = result.stdout
            for line in nvcc_output.split('\n'):
                if 'release' in line.lower():
                    print(f"   NVCC: {line.strip()}")
                    break
        else:
            print("   NVCC: Not found or not accessible")
    except Exception as e:
        print(f"   NVCC: Error checking nvcc - {e}")
    
    # Step 5: Detailed GPU analysis (wrapped in try-except)
    print(f"\n5. Detailed GPU Analysis:")
    try:
        if not cuda_available:
            print("   ❌ CUDA not available - running detailed diagnostics...")
            
            # Check if NVIDIA drivers are installed
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("   ✅ NVIDIA drivers are installed (nvidia-smi works)")
                    print("   📋 NVIDIA-SMI output:")
                    lines = result.stdout.split('\n')
                    for line in lines[:10]:  # First 10 lines usually contain key info
                        if line.strip():
                            print(f"      {line}")
                else:
                    print("   ❌ NVIDIA drivers may not be installed (nvidia-smi failed)")
            except Exception as e:
                print(f"   ❌ Error checking NVIDIA drivers: {e}")
            
            # Check for AMD GPUs
            try:
                import subprocess
                result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("   ℹ️  AMD ROCm detected")
                else:
                    print("   ℹ️  No AMD ROCm detected")
            except:
                print("   ℹ️  ROCm not available")
                
        else:
            print("   ✅ CUDA is available - performing device tests...")
            
            # Get device count
            device_count = torch.cuda.device_count()
            print(f"   Device count: {device_count}")
            
            # Get current device info
            current_device = torch.cuda.current_device()
            print(f"   Current device: {current_device}")
            
            # Get device name
            device_name = torch.cuda.get_device_name(current_device)
            print(f"   Current device name: {device_name}")
            
            # Get device properties
            props = torch.cuda.get_device_properties(current_device)
            print(f"   Device properties:")
            print(f"     - Name: {props.name}")
            print(f"     - Total memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"     - Compute capability: {props.major}.{props.minor}")
            print(f"     - Multi-processor count: {props.multi_processor_count}")
            
            # Test tensor operations
            print(f"\n   Testing tensor operations:")
            
            # Create a simple tensor on CPU
            cpu_tensor = torch.randn(100, 100)
            print(f"     ✅ Created CPU tensor: {cpu_tensor.shape}")
            
            # Try to move tensor to CUDA
            try:
                cuda_tensor = cpu_tensor.to('cuda')
                print(f"     ✅ Moved tensor to CUDA: {cuda_tensor.device}")
                
                # Test a simple operation
                result = torch.matmul(cuda_tensor, cuda_tensor.t())
                print(f"     ✅ Matrix multiplication on CUDA successful: {result.shape}")
                
                # Check memory usage
                memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**2
                print(f"     📊 CUDA memory - Allocated: {memory_allocated:.1f} MB, Reserved: {memory_reserved:.1f} MB")
                
                # Clean up
                del cuda_tensor, result
                torch.cuda.empty_cache()
                print(f"     ✅ Memory cleanup successful")
                
            except Exception as tensor_error:
                print(f"     ❌ Failed to move tensor to CUDA: {tensor_error}")
                print(f"     Error type: {type(tensor_error).__name__}")
    
    except Exception as e:
        print(f"   ❌ Critical error in GPU analysis: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Full traceback:")
        traceback.print_exc()
    
    # Step 6: Environment variables check
    print(f"\n6. Environment Variables:")
    import os
    cuda_env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_HOME', 
        'CUDA_PATH',
        'CUDA_ROOT',
        'PATH'
    ]
    
    for var in cuda_env_vars:
        value = os.environ.get(var, 'Not set')
        if var == 'PATH':
            # For PATH, just check if it contains CUDA-related paths
            if 'cuda' in value.lower():
                print(f"   {var}: Contains CUDA paths")
            else:
                print(f"   {var}: No CUDA paths found")
        else:
            print(f"   {var}: {value}")
    
    # Step 7: PyTorch CUDA compilation info
    print(f"\n7. PyTorch CUDA Compilation Info:")
    try:
        print(f"   torch.version.cuda: {torch.version.cuda}")
        print(f"   torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
        print(f"   torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")
    except Exception as e:
        print(f"   Error getting CUDA compilation info: {e}")
    
    # Step 8: Final recommendations
    print(f"\n8. Diagnostic Summary & Recommendations:")
    if cuda_available:
        print("   🎉 CUDA is working correctly!")
        print("   ✅ GPU acceleration should be available for Sovereign AI")
    else:
        print("   ⚠️  CUDA is not available. Possible causes:")
        print("   1. NVIDIA GPU drivers not installed or outdated")
        print("   2. CUDA toolkit not properly installed")
        print("   3. PyTorch not compiled with CUDA support")
        print("   4. No compatible NVIDIA GPU in system")
        print("   5. GPU is being used by another process")
        print("\n   🔧 Recommended fixes:")
        print("   1. Install latest NVIDIA drivers from nvidia.com")
        print("   2. Install CUDA toolkit matching PyTorch version")
        print("   3. Reinstall PyTorch with CUDA support:")
        print("      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("   4. Verify GPU is detected in Device Manager (Windows)")
    
    print("\n" + "=" * 60)
    print("🔍 GPU DIAGNOSTIC COMPLETE")
    print("=" * 60) 