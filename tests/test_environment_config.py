"""
Test Environment Configuration Management

This module provides comprehensive configuration management for integration testing,
including environment-specific configurations, test scenario configurations,
and validation of test setups.

Features:
- Environment-specific configuration profiles
- Test scenario configuration templates
- Configuration validation and testing
- Resource limit enforcement
- Service mocking configuration
- Performance testing configurations
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum

from test_environment_setup import TestEnvironmentConfig


class TestEnvironmentType(Enum):
    """Types of test environments"""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "end_to_end"
    PERFORMANCE = "performance"
    STRESS = "stress"
    SECURITY = "security"


class TestComplexity(Enum):
    """Test complexity levels"""
    MINIMAL = "minimal"
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    EXHAUSTIVE = "exhaustive"


@dataclass
class ServiceMockConfig:
    """Configuration for service mocking"""
    service_name: str
    mock_enabled: bool = True
    mock_type: str = "mock"  # mock, stub, fake
    mock_data_file: Optional[str] = None
    response_delay_ms: int = 0
    failure_rate: float = 0.0
    custom_responses: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceLimits:
    """Resource limits for test environment"""
    max_memory_mb: int = 2048
    max_cpu_percent: float = 50.0
    max_disk_mb: int = 5120
    max_network_bandwidth_mbps: float = 100.0
    max_file_handles: int = 1000
    max_processes: int = 10
    timeout_seconds: int = 300


@dataclass
class TestDataConfig:
    """Configuration for test data management"""
    use_synthetic_data: bool = True
    data_size: str = "small"  # small, medium, large, xl
    reset_between_tests: bool = True
    preserve_state: bool = False
    data_templates_dir: str = "test_data/templates"
    custom_datasets: List[str] = field(default_factory=list)
    
    # Data generation settings
    num_users: int = 10
    num_conversations_per_user: int = 5
    num_documents: int = 100
    conversation_length_range: tuple = (5, 20)
    document_size_range: tuple = (100, 1000)


@dataclass
class PerformanceTestConfig:
    """Configuration for performance testing"""
    enable_profiling: bool = True
    enable_memory_tracking: bool = True
    enable_cpu_profiling: bool = True
    enable_io_monitoring: bool = True
    
    # Benchmarking settings
    benchmark_iterations: int = 10
    warmup_iterations: int = 3
    measure_latency: bool = True
    measure_throughput: bool = True
    
    # Performance thresholds
    max_response_time_ms: float = 1000.0
    min_throughput_qps: float = 10.0
    max_memory_increase_mb: float = 100.0
    max_cpu_usage_percent: float = 80.0


@dataclass
class SecurityTestConfig:
    """Configuration for security testing"""
    enable_vulnerability_scanning: bool = True
    enable_auth_testing: bool = True
    enable_input_validation_testing: bool = True
    enable_privacy_compliance_testing: bool = True
    
    # Test scenarios
    test_sql_injection: bool = True
    test_xss_protection: bool = True
    test_csrf_protection: bool = True
    test_data_encryption: bool = True
    test_access_controls: bool = True
    
    # Privacy testing
    test_data_anonymization: bool = True
    test_consent_management: bool = True
    test_data_deletion: bool = True


class TestConfigurationManager:
    """Manages test configurations for different scenarios"""
    
    def __init__(self, config_dir: str = "tests/configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.predefined_configs = {}
        self._load_predefined_configs()
    
    def _load_predefined_configs(self) -> None:
        """Load predefined configuration templates"""
        # Unit testing configuration
        self.predefined_configs[TestEnvironmentType.UNIT] = {
            TestComplexity.MINIMAL: self._create_unit_minimal_config(),
            TestComplexity.BASIC: self._create_unit_basic_config(),
            TestComplexity.STANDARD: self._create_unit_standard_config()
        }
        
        # Integration testing configuration
        self.predefined_configs[TestEnvironmentType.INTEGRATION] = {
            TestComplexity.BASIC: self._create_integration_basic_config(),
            TestComplexity.STANDARD: self._create_integration_standard_config(),
            TestComplexity.COMPREHENSIVE: self._create_integration_comprehensive_config()
        }
        
        # End-to-end testing configuration
        self.predefined_configs[TestEnvironmentType.E2E] = {
            TestComplexity.STANDARD: self._create_e2e_standard_config(),
            TestComplexity.COMPREHENSIVE: self._create_e2e_comprehensive_config(),
            TestComplexity.EXHAUSTIVE: self._create_e2e_exhaustive_config()
        }
        
        # Performance testing configuration
        self.predefined_configs[TestEnvironmentType.PERFORMANCE] = {
            TestComplexity.BASIC: self._create_performance_basic_config(),
            TestComplexity.STANDARD: self._create_performance_standard_config(),
            TestComplexity.COMPREHENSIVE: self._create_performance_comprehensive_config()
        }
    
    def get_config(self, 
                   env_type: TestEnvironmentType, 
                   complexity: TestComplexity,
                   custom_overrides: Optional[Dict[str, Any]] = None) -> TestEnvironmentConfig:
        """Get a configuration for specific test environment and complexity"""
        
        if env_type not in self.predefined_configs:
            raise ValueError(f"Unknown environment type: {env_type}")
        
        if complexity not in self.predefined_configs[env_type]:
            # Fall back to closest available complexity
            available = list(self.predefined_configs[env_type].keys())
            complexity = available[0]
        
        base_config = self.predefined_configs[env_type][complexity]
        
        # Apply custom overrides if provided
        if custom_overrides:
            base_config = self._apply_overrides(base_config, custom_overrides)
        
        return base_config
    
    def save_config(self, config: TestEnvironmentConfig, name: str) -> str:
        """Save a configuration to file"""
        config_file = self.config_dir / f"{name}.json"
        
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        return str(config_file)
    
    def load_config(self, name: str) -> TestEnvironmentConfig:
        """Load a configuration from file"""
        config_file = self.config_dir / f"{name}.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        return TestEnvironmentConfig(**config_data)
    
    def validate_config(self, config: TestEnvironmentConfig) -> Dict[str, Any]:
        """Validate a test configuration"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check resource limits
        if config.max_memory_mb < 512:
            validation_results["warnings"].append("Memory limit very low (< 512MB)")
        elif config.max_memory_mb > 8192:
            validation_results["warnings"].append("Memory limit very high (> 8GB)")
        
        if config.max_cpu_percent > 90:
            validation_results["warnings"].append("CPU limit very high (> 90%)")
        
        # Check directory paths
        required_dirs = [
            config.isolated_data_dir,
            config.isolated_cache_dir,
            config.isolated_log_dir,
            config.isolated_temp_dir
        ]
        
        for directory in required_dirs:
            if directory and not Path(directory).exists():
                try:
                    Path(directory).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    validation_results["errors"].append(f"Cannot create directory {directory}: {e}")
        
        # Check configuration consistency
        if config.enable_gpu and config.max_memory_mb < 1024:
            validation_results["warnings"].append("GPU enabled but memory limit may be insufficient")
        
        if config.mock_external_apis and config.enable_external_services:
            validation_results["warnings"].append("External services enabled but APIs mocked - may cause conflicts")
        
        # Performance recommendations
        if config.enable_performance_monitoring and not config.collect_metrics:
            validation_results["recommendations"].append("Consider enabling metric collection for performance monitoring")
        
        # Set overall validity
        validation_results["valid"] = len(validation_results["errors"]) == 0
        
        return validation_results
    
    def _apply_overrides(self, base_config: TestEnvironmentConfig, overrides: Dict[str, Any]) -> TestEnvironmentConfig:
        """Apply custom overrides to base configuration"""
        config_dict = asdict(base_config)
        
        # Deep merge overrides
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_config = deep_merge(config_dict, overrides)
        return TestEnvironmentConfig(**merged_config)
    
    # Configuration creation methods
    def _create_unit_minimal_config(self) -> TestEnvironmentConfig:
        """Create minimal unit testing configuration"""
        return TestEnvironmentConfig(
            environment_id="unit_minimal",
            environment_name="Unit Test - Minimal",
            max_memory_mb=512,
            max_cpu_percent=25.0,
            max_disk_mb=1024,
            use_test_data=False,
            enable_gpu=False,
            enable_external_services=False,
            mock_external_apis=True,
            enable_performance_monitoring=False,
            collect_metrics=False,
            reset_data_between_tests=True
        )
    
    def _create_unit_basic_config(self) -> TestEnvironmentConfig:
        """Create basic unit testing configuration"""
        return TestEnvironmentConfig(
            environment_id="unit_basic",
            environment_name="Unit Test - Basic",
            max_memory_mb=1024,
            max_cpu_percent=50.0,
            max_disk_mb=2048,
            use_test_data=True,
            test_data_size="small",
            enable_gpu=False,
            enable_external_services=False,
            mock_external_apis=True,
            enable_performance_monitoring=True,
            collect_metrics=True,
            reset_data_between_tests=True
        )
    
    def _create_unit_standard_config(self) -> TestEnvironmentConfig:
        """Create standard unit testing configuration"""
        return TestEnvironmentConfig(
            environment_id="unit_standard",
            environment_name="Unit Test - Standard",
            max_memory_mb=2048,
            max_cpu_percent=75.0,
            max_disk_mb=4096,
            use_test_data=True,
            test_data_size="medium",
            enable_gpu=False,
            enable_external_services=False,
            mock_external_apis=True,
            enable_performance_monitoring=True,
            collect_metrics=True,
            enable_memory_profiling=True,
            reset_data_between_tests=True
        )
    
    def _create_integration_basic_config(self) -> TestEnvironmentConfig:
        """Create basic integration testing configuration"""
        return TestEnvironmentConfig(
            environment_id="integration_basic",
            environment_name="Integration Test - Basic",
            max_memory_mb=2048,
            max_cpu_percent=50.0,
            max_disk_mb=4096,
            use_test_data=True,
            test_data_size="medium",
            enable_gpu=False,
            enable_external_services=False,
            mock_external_apis=True,
            enable_performance_monitoring=True,
            collect_metrics=True,
            reset_data_between_tests=True
        )
    
    def _create_integration_standard_config(self) -> TestEnvironmentConfig:
        """Create standard integration testing configuration"""
        return TestEnvironmentConfig(
            environment_id="integration_standard",
            environment_name="Integration Test - Standard",
            max_memory_mb=4096,
            max_cpu_percent=75.0,
            max_disk_mb=8192,
            use_test_data=True,
            test_data_size="large",
            enable_gpu=True,
            enable_external_services=False,
            mock_external_apis=True,
            enable_performance_monitoring=True,
            collect_metrics=True,
            enable_memory_profiling=True,
            reset_data_between_tests=True
        )
    
    def _create_integration_comprehensive_config(self) -> TestEnvironmentConfig:
        """Create comprehensive integration testing configuration"""
        return TestEnvironmentConfig(
            environment_id="integration_comprehensive",
            environment_name="Integration Test - Comprehensive",
            max_memory_mb=8192,
            max_cpu_percent=90.0,
            max_disk_mb=16384,
            use_test_data=True,
            test_data_size="large",
            enable_gpu=True,
            enable_external_services=True,
            mock_external_apis=False,
            enable_performance_monitoring=True,
            collect_metrics=True,
            enable_memory_profiling=True,
            reset_data_between_tests=False,
            preserve_test_artifacts=True
        )
    
    def _create_e2e_standard_config(self) -> TestEnvironmentConfig:
        """Create standard end-to-end testing configuration"""
        return TestEnvironmentConfig(
            environment_id="e2e_standard",
            environment_name="End-to-End Test - Standard",
            max_memory_mb=4096,
            max_cpu_percent=75.0,
            max_disk_mb=8192,
            use_test_data=True,
            test_data_size="large",
            enable_gpu=True,
            enable_external_services=True,
            mock_external_apis=False,
            enable_performance_monitoring=True,
            collect_metrics=True,
            enable_memory_profiling=True,
            reset_data_between_tests=True
        )
    
    def _create_e2e_comprehensive_config(self) -> TestEnvironmentConfig:
        """Create comprehensive end-to-end testing configuration"""
        return TestEnvironmentConfig(
            environment_id="e2e_comprehensive",
            environment_name="End-to-End Test - Comprehensive",
            max_memory_mb=8192,
            max_cpu_percent=90.0,
            max_disk_mb=16384,
            use_test_data=True,
            test_data_size="large",
            enable_gpu=True,
            enable_external_services=True,
            mock_external_apis=False,
            enable_performance_monitoring=True,
            collect_metrics=True,
            enable_memory_profiling=True,
            reset_data_between_tests=False,
            preserve_test_artifacts=True,
            preserve_logs=True
        )
    
    def _create_e2e_exhaustive_config(self) -> TestEnvironmentConfig:
        """Create exhaustive end-to-end testing configuration"""
        return TestEnvironmentConfig(
            environment_id="e2e_exhaustive",
            environment_name="End-to-End Test - Exhaustive",
            max_memory_mb=16384,
            max_cpu_percent=95.0,
            max_disk_mb=32768,
            use_test_data=True,
            test_data_size="xl",
            enable_gpu=True,
            enable_external_services=True,
            mock_external_apis=False,
            enable_performance_monitoring=True,
            collect_metrics=True,
            enable_memory_profiling=True,
            reset_data_between_tests=False,
            preserve_test_artifacts=True,
            preserve_logs=True
        )
    
    def _create_performance_basic_config(self) -> TestEnvironmentConfig:
        """Create basic performance testing configuration"""
        return TestEnvironmentConfig(
            environment_id="performance_basic",
            environment_name="Performance Test - Basic",
            max_memory_mb=4096,
            max_cpu_percent=90.0,
            max_disk_mb=8192,
            use_test_data=True,
            test_data_size="medium",
            enable_gpu=True,
            enable_external_services=False,
            mock_external_apis=True,
            enable_performance_monitoring=True,
            collect_metrics=True,
            enable_memory_profiling=True,
            reset_data_between_tests=True
        )
    
    def _create_performance_standard_config(self) -> TestEnvironmentConfig:
        """Create standard performance testing configuration"""
        return TestEnvironmentConfig(
            environment_id="performance_standard",
            environment_name="Performance Test - Standard",
            max_memory_mb=8192,
            max_cpu_percent=95.0,
            max_disk_mb=16384,
            use_test_data=True,
            test_data_size="large",
            enable_gpu=True,
            enable_external_services=False,
            mock_external_apis=True,
            enable_performance_monitoring=True,
            collect_metrics=True,
            enable_memory_profiling=True,
            reset_data_between_tests=False
        )
    
    def _create_performance_comprehensive_config(self) -> TestEnvironmentConfig:
        """Create comprehensive performance testing configuration"""
        return TestEnvironmentConfig(
            environment_id="performance_comprehensive",
            environment_name="Performance Test - Comprehensive",
            max_memory_mb=16384,
            max_cpu_percent=95.0,
            max_disk_mb=32768,
            use_test_data=True,
            test_data_size="xl",
            enable_gpu=True,
            enable_external_services=True,
            mock_external_apis=False,
            enable_performance_monitoring=True,
            collect_metrics=True,
            enable_memory_profiling=True,
            reset_data_between_tests=False,
            preserve_test_artifacts=True,
            preserve_logs=True
        )


class TestScenarioBuilder:
    """Builder for creating custom test scenarios"""
    
    def __init__(self):
        self.scenario_config = {
            "name": "custom_scenario",
            "description": "",
            "services": [],
            "mock_configs": [],
            "resource_limits": {},
            "test_data": {},
            "performance_config": {},
            "security_config": {}
        }
    
    def with_name(self, name: str) -> 'TestScenarioBuilder':
        """Set scenario name"""
        self.scenario_config["name"] = name
        return self
    
    def with_description(self, description: str) -> 'TestScenarioBuilder':
        """Set scenario description"""
        self.scenario_config["description"] = description
        return self
    
    def with_services(self, services: List[str]) -> 'TestScenarioBuilder':
        """Add services to scenario"""
        self.scenario_config["services"].extend(services)
        return self
    
    def with_service_mock(self, mock_config: ServiceMockConfig) -> 'TestScenarioBuilder':
        """Add service mock configuration"""
        self.scenario_config["mock_configs"].append(asdict(mock_config))
        return self
    
    def with_resource_limits(self, limits: ResourceLimits) -> 'TestScenarioBuilder':
        """Set resource limits"""
        self.scenario_config["resource_limits"] = asdict(limits)
        return self
    
    def with_test_data(self, data_config: TestDataConfig) -> 'TestScenarioBuilder':
        """Set test data configuration"""
        self.scenario_config["test_data"] = asdict(data_config)
        return self
    
    def with_performance_config(self, perf_config: PerformanceTestConfig) -> 'TestScenarioBuilder':
        """Set performance testing configuration"""
        self.scenario_config["performance_config"] = asdict(perf_config)
        return self
    
    def with_security_config(self, sec_config: SecurityTestConfig) -> 'TestScenarioBuilder':
        """Set security testing configuration"""
        self.scenario_config["security_config"] = asdict(sec_config)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the scenario configuration"""
        return self.scenario_config.copy()
    
    def to_environment_config(self) -> TestEnvironmentConfig:
        """Convert scenario to TestEnvironmentConfig"""
        # Extract relevant configuration
        resource_limits = self.scenario_config.get("resource_limits", {})
        test_data = self.scenario_config.get("test_data", {})
        
        return TestEnvironmentConfig(
            environment_id=self.scenario_config["name"],
            environment_name=self.scenario_config.get("description", self.scenario_config["name"]),
            max_memory_mb=resource_limits.get("max_memory_mb", 2048),
            max_cpu_percent=resource_limits.get("max_cpu_percent", 50.0),
            max_disk_mb=resource_limits.get("max_disk_mb", 5120),
            use_test_data=test_data.get("use_synthetic_data", True),
            test_data_size=test_data.get("data_size", "small"),
            reset_data_between_tests=test_data.get("reset_between_tests", True)
        )


# Convenience functions
def get_standard_config(env_type: TestEnvironmentType, complexity: TestComplexity = TestComplexity.STANDARD) -> TestEnvironmentConfig:
    """Get a standard configuration for testing"""
    manager = TestConfigurationManager()
    return manager.get_config(env_type, complexity)


def create_custom_scenario() -> TestScenarioBuilder:
    """Create a new custom test scenario builder"""
    return TestScenarioBuilder()


if __name__ == "__main__":
    # Example usage
    config_manager = TestConfigurationManager()
    
    # Get standard integration test config
    integration_config = config_manager.get_config(
        TestEnvironmentType.INTEGRATION, 
        TestComplexity.STANDARD
    )
    
    print(f"Integration config: {integration_config.environment_name}")
    print(f"Memory limit: {integration_config.max_memory_mb}MB")
    print(f"GPU enabled: {integration_config.enable_gpu}")
    
    # Validate configuration
    validation = config_manager.validate_config(integration_config)
    print(f"Config valid: {validation['valid']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    # Create custom scenario
    scenario = (create_custom_scenario()
                .with_name("custom_integration_test")
                .with_description("Custom integration test with specific requirements")
                .with_services(["orchestrator", "memory_manager", "vector_search"])
                .with_resource_limits(ResourceLimits(max_memory_mb=4096, max_cpu_percent=75.0))
                .build())
    
    print(f"Custom scenario: {scenario['name']}")
    print(f"Services: {scenario['services']}") 