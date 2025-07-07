"""
Test Environment Validation and Health Checking

This module provides comprehensive validation and health checking for the integration
testing environment, ensuring all components are properly configured and functioning
before test execution begins.

Features:
- Pre-test environment validation
- Service health checking
- Resource availability verification
- Configuration compliance testing
- Performance baseline establishment
- Environment readiness assessment
"""

import os
import sys
import time
import psutil
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from test_environment_setup import TestEnvironmentManager, TestEnvironmentConfig
from test_environment_config import TestConfigurationManager, TestEnvironmentType, TestComplexity


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    status: bool
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    execution_time_ms: float = 0.0


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    service_name: str
    status: HealthStatus
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    response_time_ms: float = 0.0


class EnvironmentValidator:
    """Validates test environment setup and configuration"""
    
    def __init__(self, config: TestEnvironmentConfig):
        self.config = config
        self.validation_results = []
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up validation logger"""
        logger = logging.getLogger("environment_validator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def validate_complete_environment(self) -> Dict[str, Any]:
        """Run complete environment validation"""
        self.logger.info("Starting complete environment validation...")
        start_time = time.time()
        
        validation_summary = {
            "overall_status": True,
            "validation_timestamp": start_time,
            "validation_duration_ms": 0,
            "checks_performed": 0,
            "checks_passed": 0,
            "checks_failed": 0,
            "critical_issues": 0,
            "warnings": 0,
            "results": []
        }
        
        # Run all validation checks
        await self._validate_system_resources()
        await self._validate_directories()
        await self._validate_configuration()
        await self._validate_dependencies()
        await self._validate_permissions()
        await self._validate_network_connectivity()
        await self._validate_storage_capacity()
        
        # Analyze results
        validation_summary["results"] = [asdict(result) for result in self.validation_results]
        validation_summary["checks_performed"] = len(self.validation_results)
        validation_summary["checks_passed"] = sum(1 for r in self.validation_results if r.status)
        validation_summary["checks_failed"] = sum(1 for r in self.validation_results if not r.status)
        validation_summary["critical_issues"] = sum(1 for r in self.validation_results 
                                                   if not r.status and r.severity == ValidationSeverity.CRITICAL)
        validation_summary["warnings"] = sum(1 for r in self.validation_results 
                                            if r.severity == ValidationSeverity.WARNING)
        
        # Determine overall status
        validation_summary["overall_status"] = (
            validation_summary["critical_issues"] == 0 and
            validation_summary["checks_failed"] < validation_summary["checks_performed"] * 0.1
        )
        
        validation_summary["validation_duration_ms"] = (time.time() - start_time) * 1000
        
        self.logger.info(f"Validation completed in {validation_summary['validation_duration_ms']:.1f}ms")
        self.logger.info(f"Status: {'PASS' if validation_summary['overall_status'] else 'FAIL'}")
        self.logger.info(f"Checks: {validation_summary['checks_passed']}/{validation_summary['checks_performed']} passed")
        
        return validation_summary
    
    async def _validate_system_resources(self) -> None:
        """Validate system resource availability"""
        await self._run_validation_check(
            "system_memory",
            self._check_memory_availability,
            ValidationSeverity.CRITICAL,
            "Checking available system memory"
        )
        
        await self._run_validation_check(
            "system_cpu",
            self._check_cpu_availability,
            ValidationSeverity.WARNING,
            "Checking CPU availability"
        )
        
        await self._run_validation_check(
            "system_disk",
            self._check_disk_space,
            ValidationSeverity.CRITICAL,
            "Checking disk space availability"
        )
    
    async def _validate_directories(self) -> None:
        """Validate required directories exist and are writable"""
        directories = [
            ("data_dir", self.config.isolated_data_dir),
            ("cache_dir", self.config.isolated_cache_dir),
            ("log_dir", self.config.isolated_log_dir),
            ("temp_dir", self.config.isolated_temp_dir)
        ]
        
        for name, directory in directories:
            await self._run_validation_check(
                f"directory_{name}",
                lambda d=directory: self._check_directory_access(d),
                ValidationSeverity.CRITICAL,
                f"Checking directory access: {directory}"
            )
    
    async def _validate_configuration(self) -> None:
        """Validate configuration settings"""
        await self._run_validation_check(
            "config_values",
            self._check_configuration_values,
            ValidationSeverity.ERROR,
            "Validating configuration values"
        )
        
        await self._run_validation_check(
            "config_consistency",
            self._check_configuration_consistency,
            ValidationSeverity.WARNING,
            "Checking configuration consistency"
        )
    
    async def _validate_dependencies(self) -> None:
        """Validate system dependencies"""
        await self._run_validation_check(
            "python_version",
            self._check_python_version,
            ValidationSeverity.CRITICAL,
            "Checking Python version compatibility"
        )
        
        await self._run_validation_check(
            "required_packages",
            self._check_required_packages,
            ValidationSeverity.CRITICAL,
            "Checking required Python packages"
        )
        
        if self.config.enable_gpu:
            await self._run_validation_check(
                "gpu_availability",
                self._check_gpu_availability,
                ValidationSeverity.WARNING,
                "Checking GPU availability"
            )
    
    async def _validate_permissions(self) -> None:
        """Validate file system permissions"""
        await self._run_validation_check(
            "file_permissions",
            self._check_file_permissions,
            ValidationSeverity.ERROR,
            "Checking file system permissions"
        )
    
    async def _validate_network_connectivity(self) -> None:
        """Validate network connectivity if needed"""
        if self.config.enable_external_services and not self.config.mock_external_apis:
            await self._run_validation_check(
                "network_connectivity",
                self._check_network_connectivity,
                ValidationSeverity.WARNING,
                "Checking network connectivity"
            )
    
    async def _validate_storage_capacity(self) -> None:
        """Validate storage capacity for test data"""
        await self._run_validation_check(
            "storage_capacity",
            self._check_storage_capacity,
            ValidationSeverity.CRITICAL,
            "Checking storage capacity for test data"
        )
    
    async def _run_validation_check(self, 
                                  check_name: str, 
                                  check_func: Callable, 
                                  severity: ValidationSeverity,
                                  description: str) -> None:
        """Run a single validation check"""
        start_time = time.time()
        
        try:
            result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
            status = result.get("status", False) if isinstance(result, dict) else bool(result)
            message = result.get("message", "Check completed") if isinstance(result, dict) else description
            details = result.get("details", {}) if isinstance(result, dict) else {}
            
        except Exception as e:
            status = False
            message = f"Check failed with exception: {str(e)}"
            details = {"exception": str(e)}
        
        execution_time = (time.time() - start_time) * 1000
        
        validation_result = ValidationResult(
            check_name=check_name,
            status=status,
            severity=severity,
            message=message,
            details=details,
            execution_time_ms=execution_time
        )
        
        self.validation_results.append(validation_result)
        
        level = logging.ERROR if not status and severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] else logging.INFO
        self.logger.log(level, f"{check_name}: {message}")
    
    def _check_memory_availability(self) -> Dict[str, Any]:
        """Check if sufficient memory is available"""
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        required_mb = self.config.max_memory_mb
        
        status = available_mb >= required_mb
        
        return {
            "status": status,
            "message": f"Memory: {available_mb:.0f}MB available, {required_mb}MB required",
            "details": {
                "available_mb": available_mb,
                "required_mb": required_mb,
                "total_mb": memory.total / (1024 * 1024),
                "usage_percent": memory.percent
            }
        }
    
    def _check_cpu_availability(self) -> Dict[str, Any]:
        """Check CPU availability"""
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Consider CPU available if usage is below 80%
        status = cpu_percent < 80.0
        
        return {
            "status": status,
            "message": f"CPU: {cpu_count} cores, {cpu_percent:.1f}% current usage",
            "details": {
                "cpu_count": cpu_count,
                "cpu_percent": cpu_percent,
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
        }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        data_dir = Path(self.config.isolated_data_dir)
        if not data_dir.exists():
            data_dir = data_dir.parent
        
        disk_usage = psutil.disk_usage(str(data_dir))
        available_mb = disk_usage.free / (1024 * 1024)
        required_mb = self.config.max_disk_mb
        
        status = available_mb >= required_mb
        
        return {
            "status": status,
            "message": f"Disk: {available_mb:.0f}MB available, {required_mb}MB required",
            "details": {
                "available_mb": available_mb,
                "required_mb": required_mb,
                "total_mb": disk_usage.total / (1024 * 1024),
                "used_mb": disk_usage.used / (1024 * 1024)
            }
        }
    
    def _check_directory_access(self, directory: str) -> Dict[str, Any]:
        """Check directory exists and is writable"""
        path = Path(directory)
        
        # Create directory if it doesn't exist
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return {
                "status": False,
                "message": f"Cannot create directory: {e}",
                "details": {"path": str(path), "error": str(e)}
            }
        
        # Test write access
        test_file = path / "test_write_access.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()
            status = True
            message = f"Directory accessible: {directory}"
        except Exception as e:
            status = False
            message = f"Directory not writable: {e}"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "path": str(path),
                "exists": path.exists(),
                "is_dir": path.is_dir(),
                "writable": status
            }
        }
    
    def _check_configuration_values(self) -> Dict[str, Any]:
        """Check configuration values are valid"""
        issues = []
        
        # Check memory limits
        if self.config.max_memory_mb < 256:
            issues.append("Memory limit too low (< 256MB)")
        elif self.config.max_memory_mb > 32768:
            issues.append("Memory limit very high (> 32GB)")
        
        # Check CPU limits
        if self.config.max_cpu_percent > 100:
            issues.append("CPU limit exceeds 100%")
        
        # Check disk limits
        if self.config.max_disk_mb < 1024:
            issues.append("Disk limit too low (< 1GB)")
        
        # Check environment ID is valid
        if not self.config.environment_id or not self.config.environment_id.strip():
            issues.append("Environment ID is empty")
        
        status = len(issues) == 0
        message = "Configuration valid" if status else f"Configuration issues: {'; '.join(issues)}"
        
        return {
            "status": status,
            "message": message,
            "details": {"issues": issues}
        }
    
    def _check_configuration_consistency(self) -> Dict[str, Any]:
        """Check configuration consistency"""
        warnings = []
        
        # Check GPU/CPU consistency
        if self.config.enable_gpu and self.config.max_memory_mb < 1024:
            warnings.append("GPU enabled but memory limit may be insufficient")
        
        # Check external services consistency
        if self.config.enable_external_services and self.config.mock_external_apis:
            warnings.append("External services enabled but APIs are mocked")
        
        # Check monitoring consistency
        if self.config.enable_performance_monitoring and not self.config.collect_metrics:
            warnings.append("Performance monitoring enabled but metrics collection disabled")
        
        status = True  # Warnings don't fail validation
        message = "Configuration consistent" if not warnings else f"Configuration warnings: {'; '.join(warnings)}"
        
        return {
            "status": status,
            "message": message,
            "details": {"warnings": warnings}
        }
    
    def _check_python_version(self) -> Dict[str, Any]:
        """Check Python version compatibility"""
        version = sys.version_info
        min_version = (3, 8)
        
        status = version >= min_version
        message = f"Python {version.major}.{version.minor}.{version.micro}, minimum {min_version[0]}.{min_version[1]} required"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "current_version": f"{version.major}.{version.minor}.{version.micro}",
                "minimum_version": f"{min_version[0]}.{min_version[1]}",
                "python_executable": sys.executable
            }
        }
    
    def _check_required_packages(self) -> Dict[str, Any]:
        """Check required Python packages are available"""
        required_packages = [
            "torch", "transformers", "sentence_transformers",
            "numpy", "psutil", "pytest", "asyncio"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        status = len(missing_packages) == 0
        message = "All packages available" if status else f"Missing packages: {', '.join(missing_packages)}"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "required_packages": required_packages,
                "missing_packages": missing_packages
            }
        }
    
    def _check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability if enabled"""
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            
            if gpu_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                message = f"GPU available: {gpu_name} ({gpu_memory:.1f}GB)"
                details = {
                    "gpu_count": gpu_count,
                    "gpu_name": gpu_name,
                    "gpu_memory_gb": gpu_memory
                }
            else:
                message = "GPU not available"
                details = {"gpu_count": 0}
            
            return {
                "status": gpu_available,
                "message": message,
                "details": details
            }
            
        except ImportError:
            return {
                "status": False,
                "message": "PyTorch not available for GPU check",
                "details": {"error": "torch not importable"}
            }
    
    def _check_file_permissions(self) -> Dict[str, Any]:
        """Check file system permissions"""
        directories_to_check = [
            self.config.isolated_data_dir,
            self.config.isolated_cache_dir,
            self.config.isolated_log_dir,
            self.config.isolated_temp_dir
        ]
        
        permission_issues = []
        
        for directory in directories_to_check:
            path = Path(directory)
            if path.exists():
                # Check read permission
                if not os.access(path, os.R_OK):
                    permission_issues.append(f"No read access to {directory}")
                
                # Check write permission
                if not os.access(path, os.W_OK):
                    permission_issues.append(f"No write access to {directory}")
                
                # Check execute permission (for directories)
                if not os.access(path, os.X_OK):
                    permission_issues.append(f"No execute access to {directory}")
        
        status = len(permission_issues) == 0
        message = "All permissions valid" if status else f"Permission issues: {'; '.join(permission_issues)}"
        
        return {
            "status": status,
            "message": message,
            "details": {"permission_issues": permission_issues}
        }
    
    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity for external services"""
        try:
            import socket
            
            # Test DNS resolution
            socket.gethostbyname("google.com")
            
            # Test HTTP connectivity (basic check)
            import urllib.request
            urllib.request.urlopen("http://google.com", timeout=5)
            
            return {
                "status": True,
                "message": "Network connectivity available",
                "details": {"dns_resolution": True, "http_connectivity": True}
            }
            
        except Exception as e:
            return {
                "status": False,
                "message": f"Network connectivity issues: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def _check_storage_capacity(self) -> Dict[str, Any]:
        """Check storage capacity for test data"""
        # Estimate storage requirements based on test data size
        size_requirements = {
            "small": 100,    # 100MB
            "medium": 500,   # 500MB
            "large": 2048,   # 2GB
            "xl": 5120       # 5GB
        }
        
        required_mb = size_requirements.get(self.config.test_data_size, 1024)
        
        # Check available space
        data_dir = Path(self.config.isolated_data_dir)
        if not data_dir.exists():
            data_dir = data_dir.parent
        
        disk_usage = psutil.disk_usage(str(data_dir))
        available_mb = disk_usage.free / (1024 * 1024)
        
        # Add buffer for temporary files and logs
        required_with_buffer = required_mb * 2
        
        status = available_mb >= required_with_buffer
        message = f"Storage: {available_mb:.0f}MB available, {required_with_buffer:.0f}MB required (with buffer)"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "available_mb": available_mb,
                "required_mb": required_mb,
                "required_with_buffer_mb": required_with_buffer,
                "test_data_size": self.config.test_data_size
            }
        }


class ServiceHealthChecker:
    """Health checker for individual services"""
    
    def __init__(self, environment_manager: TestEnvironmentManager):
        self.environment_manager = environment_manager
        self.health_results = []
        self.logger = logging.getLogger("service_health_checker")
    
    async def check_all_services_health(self) -> Dict[str, Any]:
        """Check health of all services"""
        self.logger.info("Starting service health checks...")
        start_time = time.time()
        
        health_summary = {
            "overall_status": HealthStatus.HEALTHY,
            "check_timestamp": start_time,
            "check_duration_ms": 0,
            "services_checked": 0,
            "services_healthy": 0,
            "services_degraded": 0,
            "services_unhealthy": 0,
            "results": []
        }
        
        # Check each service
        services = self.environment_manager.service_orchestrator.services
        
        for service_name, service in services.items():
            await self._check_service_health(service_name, service)
        
        # Analyze results
        health_summary["results"] = [asdict(result) for result in self.health_results]
        health_summary["services_checked"] = len(self.health_results)
        health_summary["services_healthy"] = sum(1 for r in self.health_results if r.status == HealthStatus.HEALTHY)
        health_summary["services_degraded"] = sum(1 for r in self.health_results if r.status == HealthStatus.DEGRADED)
        health_summary["services_unhealthy"] = sum(1 for r in self.health_results if r.status == HealthStatus.UNHEALTHY)
        
        # Determine overall status
        if health_summary["services_unhealthy"] > 0:
            health_summary["overall_status"] = HealthStatus.UNHEALTHY
        elif health_summary["services_degraded"] > 0:
            health_summary["overall_status"] = HealthStatus.DEGRADED
        
        health_summary["check_duration_ms"] = (time.time() - start_time) * 1000
        
        self.logger.info(f"Health check completed in {health_summary['check_duration_ms']:.1f}ms")
        self.logger.info(f"Overall status: {health_summary['overall_status'].value}")
        
        return health_summary
    
    async def _check_service_health(self, service_name: str, service: Any) -> None:
        """Check health of a specific service"""
        start_time = time.time()
        
        try:
            if service_name == "embedding_service":
                status, message, metrics = await self._check_embedding_service_health(service)
            elif service_name == "memory_manager":
                status, message, metrics = await self._check_memory_manager_health(service)
            elif service_name == "vector_search":
                status, message, metrics = await self._check_vector_search_health(service)
            elif service_name == "orchestrator":
                status, message, metrics = await self._check_orchestrator_health(service)
            else:
                status, message, metrics = await self._check_generic_service_health(service)
            
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Health check failed: {str(e)}"
            metrics = {"error": str(e)}
        
        response_time = (time.time() - start_time) * 1000
        
        health_result = HealthCheckResult(
            service_name=service_name,
            status=status,
            message=message,
            metrics=metrics,
            response_time_ms=response_time
        )
        
        self.health_results.append(health_result)
        self.logger.info(f"{service_name}: {status.value} - {message}")
    
    async def _check_embedding_service_health(self, service: Any) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check embedding service health"""
        if not hasattr(service, 'is_available'):
            return HealthStatus.UNKNOWN, "Service interface not recognized", {}
        
        if not service.is_available:
            return HealthStatus.UNHEALTHY, "Service not available", {"available": False}
        
        # Test basic functionality
        try:
            test_response = service.generate_embedding("test")
            if test_response and test_response.success:
                return HealthStatus.HEALTHY, "Service operational", {
                    "available": True,
                    "test_response_time": test_response.processing_time
                }
            else:
                return HealthStatus.DEGRADED, "Service available but test failed", {
                    "available": True,
                    "test_failed": True
                }
        except Exception as e:
            return HealthStatus.DEGRADED, f"Service test failed: {str(e)}", {
                "available": True,
                "test_error": str(e)
            }
    
    async def _check_memory_manager_health(self, service: Any) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check memory manager health"""
        try:
            # Test basic operations
            test_user_id = f"health_check_user"
            
            # Try creating a conversation
            conversation_id = service.create_conversation(test_user_id, "Health check conversation")
            
            if conversation_id:
                # Try storing a message
                service.store_message(test_user_id, conversation_id, "user", "Health check message")
                
                # Try retrieving conversations
                conversations = service.get_conversations(test_user_id)
                
                return HealthStatus.HEALTHY, "Memory manager operational", {
                    "conversation_created": bool(conversation_id),
                    "message_stored": True,
                    "conversations_retrieved": len(conversations) if conversations else 0
                }
            else:
                return HealthStatus.DEGRADED, "Cannot create conversations", {
                    "conversation_created": False
                }
                
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Memory manager error: {str(e)}", {
                "error": str(e)
            }
    
    async def _check_vector_search_health(self, service: Any) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check vector search health"""
        try:
            # Test basic search functionality
            if hasattr(service, 'search'):
                # Perform a test search
                results = service.search("test query", top_k=1)
                
                return HealthStatus.HEALTHY, "Vector search operational", {
                    "search_completed": True,
                    "results_count": len(results) if results else 0
                }
            else:
                return HealthStatus.UNKNOWN, "Vector search interface not recognized", {}
                
        except Exception as e:
            return HealthStatus.DEGRADED, f"Vector search error: {str(e)}", {
                "error": str(e)
            }
    
    async def _check_orchestrator_health(self, service: Any) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check orchestrator health"""
        try:
            # Check if orchestrator has required methods
            required_methods = ['process_query', 'determine_complexity']
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(service, method):
                    missing_methods.append(method)
            
            if missing_methods:
                return HealthStatus.DEGRADED, f"Missing methods: {', '.join(missing_methods)}", {
                    "missing_methods": missing_methods
                }
            else:
                return HealthStatus.HEALTHY, "Orchestrator interface complete", {
                    "required_methods_present": True
                }
                
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Orchestrator error: {str(e)}", {
                "error": str(e)
            }
    
    async def _check_generic_service_health(self, service: Any) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Generic health check for services"""
        if service is None:
            return HealthStatus.UNHEALTHY, "Service is None", {}
        
        # Basic availability check
        return HealthStatus.HEALTHY, "Service available", {
            "service_type": type(service).__name__
        }


# Convenience functions
async def validate_environment(config: TestEnvironmentConfig) -> Dict[str, Any]:
    """Validate a test environment configuration"""
    validator = EnvironmentValidator(config)
    return await validator.validate_complete_environment()


async def check_service_health(environment_manager: TestEnvironmentManager) -> Dict[str, Any]:
    """Check health of all services in environment"""
    health_checker = ServiceHealthChecker(environment_manager)
    return await health_checker.check_all_services_health()


if __name__ == "__main__":
    async def test_validation():
        """Test the validation system"""
        from test_environment_config import get_standard_config
        
        print("Testing environment validation...")
        
        # Get a standard configuration
        config = get_standard_config(TestEnvironmentType.INTEGRATION, TestComplexity.BASIC)
        
        # Run validation
        validation_results = await validate_environment(config)
        
        print(f"Validation status: {'PASS' if validation_results['overall_status'] else 'FAIL'}")
        print(f"Checks performed: {validation_results['checks_performed']}")
        print(f"Checks passed: {validation_results['checks_passed']}")
        print(f"Critical issues: {validation_results['critical_issues']}")
        print(f"Warnings: {validation_results['warnings']}")
        
        # Show failed checks
        failed_checks = [r for r in validation_results['results'] if not r['status']]
        if failed_checks:
            print("\nFailed checks:")
            for check in failed_checks:
                print(f"  - {check['check_name']}: {check['message']}")
    
    # Run the test
    asyncio.run(test_validation()) 