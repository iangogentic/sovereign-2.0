"""
Test Environment Setup and Management

This module provides comprehensive infrastructure for establishing and managing
a dedicated integration testing environment that mirrors production while maintaining
complete isolation from development systems.

Features:
- Isolated test configuration management
- Automated service orchestration
- Test data management and cleanup
- Environment health monitoring
- Resource validation and optimization
- Production-like environment simulation
"""

import os
import sys
import time
import json
import uuid
import shutil
import asyncio
import logging
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sovereign.config import Config, ModelConfig, HardwareConfig, DatabaseConfig, OrchestratorConfig
from sovereign.logger import setup_logger
from sovereign.orchestrator import Orchestrator
from sovereign.memory_manager import MemoryManager
from sovereign.vector_search_engine import VectorSearchEngine
from sovereign.embedding_service import EmbeddingService
from sovereign.performance_monitor import PerformanceMonitor
from sovereign.tool_execution_engine import ToolExecutionEngine
from sovereign.privacy_manager import PrivacyManager


@dataclass
class TestEnvironmentConfig:
    """Configuration for test environment"""
    # Environment identification
    environment_id: str
    environment_name: str = "integration_test"
    
    # Isolation settings
    isolated_data_dir: str = ""
    isolated_cache_dir: str = ""
    isolated_log_dir: str = ""
    isolated_temp_dir: str = ""
    
    # Resource limits
    max_memory_mb: int = 2048
    max_cpu_percent: float = 50.0
    max_disk_mb: int = 5120
    
    # Test data settings
    use_test_data: bool = True
    test_data_size: str = "small"  # small, medium, large
    reset_data_between_tests: bool = True
    
    # Service settings
    enable_gpu: bool = False  # Default to CPU for consistency
    enable_external_services: bool = False
    mock_external_apis: bool = True
    
    # Performance settings
    enable_performance_monitoring: bool = True
    collect_metrics: bool = True
    enable_memory_profiling: bool = True
    
    # Cleanup settings
    cleanup_on_exit: bool = True
    preserve_logs: bool = True
    preserve_test_artifacts: bool = False


class TestEnvironmentError(Exception):
    """Base exception for test environment errors"""
    pass


class ServiceSetupError(TestEnvironmentError):
    """Error during service setup"""
    pass


class EnvironmentValidationError(TestEnvironmentError):
    """Error during environment validation"""
    pass


class TestDataManager:
    """Manages test data lifecycle and isolation"""
    
    def __init__(self, config: TestEnvironmentConfig):
        self.config = config
        self.test_data_dir = Path(config.isolated_data_dir) / "test_data"
        self.backup_dir = Path(config.isolated_data_dir) / "backups"
        self.current_test_id = None
        
    def setup_test_data(self, test_id: str) -> None:
        """Set up isolated test data for a specific test"""
        self.current_test_id = test_id
        test_specific_dir = self.test_data_dir / test_id
        test_specific_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test databases
        self._create_test_database(test_specific_dir)
        self._create_test_vector_store(test_specific_dir)
        self._create_test_embeddings_cache(test_specific_dir)
        
        # Generate test data if needed
        if self.config.use_test_data:
            self._generate_test_data(test_specific_dir)
    
    def cleanup_test_data(self, test_id: str) -> None:
        """Clean up test data for a specific test"""
        if not self.config.reset_data_between_tests:
            return
            
        test_specific_dir = self.test_data_dir / test_id
        if test_specific_dir.exists():
            shutil.rmtree(test_specific_dir)
    
    def backup_current_state(self, backup_name: str) -> str:
        """Create a backup of current test state"""
        if not self.current_test_id:
            raise TestEnvironmentError("No active test to backup")
            
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)
        
        source_dir = self.test_data_dir / self.current_test_id
        if source_dir.exists():
            shutil.copytree(source_dir, backup_path / self.current_test_id, dirs_exist_ok=True)
        
        return str(backup_path)
    
    def restore_from_backup(self, backup_name: str) -> None:
        """Restore test state from backup"""
        backup_path = self.backup_dir / backup_name / self.current_test_id
        if not backup_path.exists():
            raise TestEnvironmentError(f"Backup {backup_name} not found")
            
        target_dir = self.test_data_dir / self.current_test_id
        if target_dir.exists():
            shutil.rmtree(target_dir)
        
        shutil.copytree(backup_path, target_dir)
    
    def _create_test_database(self, test_dir: Path) -> None:
        """Create isolated test database"""
        db_path = test_dir / "test_sovereign.db"
        # Database will be created automatically by MemoryManager
        
    def _create_test_vector_store(self, test_dir: Path) -> None:
        """Create isolated vector store"""
        vector_dir = test_dir / "vectors"
        vector_dir.mkdir(exist_ok=True)
        
    def _create_test_embeddings_cache(self, test_dir: Path) -> None:
        """Create isolated embeddings cache"""
        cache_dir = test_dir / "embeddings_cache"
        cache_dir.mkdir(exist_ok=True)
    
    def _generate_test_data(self, test_dir: Path) -> None:
        """Generate test data based on configuration"""
        # Create sample conversations
        sample_conversations = [
            {
                "user_id": "test_user_1",
                "conversation_id": str(uuid.uuid4()),
                "messages": [
                    {"role": "user", "content": "Hello, test message 1"},
                    {"role": "assistant", "content": "Hello! This is a test response."}
                ]
            },
            {
                "user_id": "test_user_2", 
                "conversation_id": str(uuid.uuid4()),
                "messages": [
                    {"role": "user", "content": "Test query about complex reasoning"},
                    {"role": "assistant", "content": "This is a complex test response that requires reasoning."}
                ]
            }
        ]
        
        # Save test conversations
        conversations_file = test_dir / "test_conversations.json"
        with open(conversations_file, 'w') as f:
            json.dump(sample_conversations, f, indent=2)
        
        # Create test documents for RAG
        test_documents = [
            {"id": "doc1", "content": "Test document 1 content for retrieval testing"},
            {"id": "doc2", "content": "Test document 2 with different content for RAG validation"},
            {"id": "doc3", "content": "Complex test document with technical information for testing"}
        ]
        
        documents_file = test_dir / "test_documents.json"
        with open(documents_file, 'w') as f:
            json.dump(test_documents, f, indent=2)


class ServiceOrchestrator:
    """Orchestrates all Sovereign AI services for testing"""
    
    def __init__(self, config: TestEnvironmentConfig):
        self.config = config
        self.services = {}
        self.service_health = {}
        self.initialization_order = [
            'logger',
            'config',
            'embedding_service',
            'vector_search',
            'memory_manager',
            'performance_monitor',
            'tool_execution',
            'privacy_manager',
            'orchestrator'
        ]
        
    async def initialize_all_services(self) -> Dict[str, Any]:
        """Initialize all services in proper order"""
        results = {}
        
        for service_name in self.initialization_order:
            try:
                service = await self._initialize_service(service_name)
                self.services[service_name] = service
                self.service_health[service_name] = True
                results[service_name] = "initialized"
                
                # Validate service after initialization
                await self._validate_service(service_name)
                
            except Exception as e:
                self.service_health[service_name] = False
                results[service_name] = f"failed: {str(e)}"
                
                # Decide whether to continue or fail fast
                if service_name in ['logger', 'config']:
                    raise ServiceSetupError(f"Critical service {service_name} failed: {e}")
                
        return results
    
    async def _initialize_service(self, service_name: str) -> Any:
        """Initialize a specific service"""
        if service_name == 'logger':
            return self._setup_logger()
        elif service_name == 'config':
            return self._setup_config()
        elif service_name == 'embedding_service':
            return self._setup_embedding_service()
        elif service_name == 'vector_search':
            return self._setup_vector_search()
        elif service_name == 'memory_manager':
            return self._setup_memory_manager()
        elif service_name == 'performance_monitor':
            return self._setup_performance_monitor()
        elif service_name == 'tool_execution':
            return self._setup_tool_execution()
        elif service_name == 'privacy_manager':
            return self._setup_privacy_manager()
        elif service_name == 'orchestrator':
            return self._setup_orchestrator()
        else:
            raise ServiceSetupError(f"Unknown service: {service_name}")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up isolated test logging"""
        log_file = Path(self.config.isolated_log_dir) / f"test_{self.config.environment_id}.log"
        logger = setup_logger(
            name="sovereign.test",
            log_file=str(log_file),
            level="DEBUG" if self.config.collect_metrics else "INFO"
        )
        return logger
    
    def _setup_config(self) -> Config:
        """Set up test-specific configuration"""
        config = Config()
        
        # Override with test-specific settings
        config.database.db_path = str(Path(self.config.isolated_data_dir) / "test_sovereign.db")
        config.database.vector_db_path = str(Path(self.config.isolated_data_dir) / "vectors")
        config.hardware.gpu_enabled = self.config.enable_gpu
        config.hardware.memory_limit_gb = self.config.max_memory_mb // 1024
        
        # Disable external services if configured
        if not self.config.enable_external_services:
            config.models.external_routing_enabled = False
            config.screen_capture.enabled = False
            config.voice.enabled = False
        
        return config
    
    def _setup_embedding_service(self) -> EmbeddingService:
        """Set up embedding service for testing"""
        cache_dir = Path(self.config.isolated_cache_dir) / "embeddings"
        
        service = EmbeddingService(
            cache_dir=str(cache_dir),
            enable_gpu=self.config.enable_gpu,
            max_batch_size=4  # Smaller batches for testing
        )
        return service
    
    def _setup_vector_search(self) -> VectorSearchEngine:
        """Set up vector search engine for testing"""
        vector_dir = Path(self.config.isolated_data_dir) / "vectors"
        embedding_service = self.services.get('embedding_service')
        
        search_engine = VectorSearchEngine(
            vector_store_path=str(vector_dir),
            embedding_service=embedding_service
        )
        return search_engine
    
    def _setup_memory_manager(self) -> MemoryManager:
        """Set up memory manager for testing"""
        config = self.services.get('config')
        vector_search = self.services.get('vector_search')
        
        manager = MemoryManager(
            config=config,
            vector_search_engine=vector_search
        )
        return manager
    
    def _setup_performance_monitor(self) -> PerformanceMonitor:
        """Set up performance monitor for testing"""
        monitor = PerformanceMonitor(
            enable_detailed_metrics=self.config.collect_metrics,
            enable_memory_profiling=self.config.enable_memory_profiling
        )
        return monitor
    
    def _setup_tool_execution(self) -> ToolExecutionEngine:
        """Set up tool execution engine for testing"""
        config = self.services.get('config')
        
        engine = ToolExecutionEngine(config=config)
        return engine
    
    def _setup_privacy_manager(self) -> PrivacyManager:
        """Set up privacy manager for testing"""
        manager = PrivacyManager()
        return manager
    
    def _setup_orchestrator(self) -> Orchestrator:
        """Set up main orchestrator for testing"""
        config = self.services.get('config')
        memory_manager = self.services.get('memory_manager')
        
        orchestrator = Orchestrator(
            config=config,
            memory_manager=memory_manager
        )
        return orchestrator
    
    async def _validate_service(self, service_name: str) -> None:
        """Validate that a service is working correctly"""
        service = self.services.get(service_name)
        if not service:
            raise EnvironmentValidationError(f"Service {service_name} not found")
        
        # Service-specific validation
        if service_name == 'embedding_service':
            if hasattr(service, 'is_available') and not service.is_available:
                raise EnvironmentValidationError("Embedding service not available")
        
        elif service_name == 'memory_manager':
            # Test basic memory operations
            test_user_id = f"test_user_{uuid.uuid4()}"
            conversation_id = service.create_conversation(test_user_id, "Test conversation")
            if not conversation_id:
                raise EnvironmentValidationError("Memory manager failed basic test")
        
        elif service_name == 'orchestrator':
            # Test basic orchestrator functionality
            if not hasattr(service, 'process_query'):
                raise EnvironmentValidationError("Orchestrator missing required methods")
    
    async def shutdown_all_services(self) -> None:
        """Properly shutdown all services"""
        # Shutdown in reverse order
        for service_name in reversed(self.initialization_order):
            if service_name in self.services:
                try:
                    service = self.services[service_name]
                    
                    # Call shutdown method if available
                    if hasattr(service, 'close'):
                        service.close()
                    elif hasattr(service, 'shutdown'):
                        service.shutdown()
                    
                except Exception as e:
                    # Log but don't fail shutdown process
                    print(f"Warning: Error shutting down {service_name}: {e}")
        
        self.services.clear()
        self.service_health.clear()


class EnvironmentHealthMonitor:
    """Monitors the health and performance of the test environment"""
    
    def __init__(self, config: TestEnvironmentConfig):
        self.config = config
        self.metrics = {}
        self.health_checks = []
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self) -> None:
        """Start continuous environment monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop environment monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def add_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Add a custom health check"""
        self.health_checks.append((name, check_func))
    
    def get_environment_status(self) -> Dict[str, Any]:
        """Get current environment status"""
        status = {
            "overall_health": "healthy",
            "timestamp": time.time(),
            "metrics": self.metrics.copy(),
            "health_checks": {}
        }
        
        # Run health checks
        failed_checks = 0
        for name, check_func in self.health_checks:
            try:
                result = check_func()
                status["health_checks"][name] = "pass" if result else "fail"
                if not result:
                    failed_checks += 1
            except Exception as e:
                status["health_checks"][name] = f"error: {str(e)}"
                failed_checks += 1
        
        # Determine overall health
        if failed_checks > 0:
            status["overall_health"] = "degraded" if failed_checks < len(self.health_checks) // 2 else "unhealthy"
        
        return status
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Sleep before next collection
                time.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def _collect_system_metrics(self) -> None:
        """Collect system performance metrics"""
        import psutil
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics["memory_percent"] = memory.percent
        self.metrics["memory_used_mb"] = memory.used / (1024 * 1024)
        
        # CPU usage
        self.metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
        
        # Disk usage for test directories
        if Path(self.config.isolated_data_dir).exists():
            disk = psutil.disk_usage(self.config.isolated_data_dir)
            self.metrics["disk_used_mb"] = disk.used / (1024 * 1024)
            self.metrics["disk_free_mb"] = disk.free / (1024 * 1024)


class TestEnvironmentManager:
    """Main manager for the integration testing environment"""
    
    def __init__(self, config: Optional[TestEnvironmentConfig] = None):
        if config is None:
            config = TestEnvironmentConfig(
                environment_id=str(uuid.uuid4())[:8],
                isolated_data_dir=tempfile.mkdtemp(prefix="sovereign_test_"),
                isolated_cache_dir=tempfile.mkdtemp(prefix="sovereign_cache_"),
                isolated_log_dir=tempfile.mkdtemp(prefix="sovereign_logs_"),
                isolated_temp_dir=tempfile.mkdtemp(prefix="sovereign_temp_")
            )
        
        self.config = config
        self.data_manager = TestDataManager(config)
        self.service_orchestrator = ServiceOrchestrator(config)
        self.health_monitor = EnvironmentHealthMonitor(config)
        
        self.is_setup = False
        self.current_test_context = None
        
        # Setup environment directories
        self._setup_directories()
    
    def _setup_directories(self) -> None:
        """Set up all required directories"""
        directories = [
            self.config.isolated_data_dir,
            self.config.isolated_cache_dir,
            self.config.isolated_log_dir,
            self.config.isolated_temp_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def setup_environment(self) -> Dict[str, Any]:
        """Set up the complete test environment"""
        if self.is_setup:
            return {"status": "already_setup"}
        
        setup_results = {
            "environment_id": self.config.environment_id,
            "timestamp": time.time(),
            "services": {},
            "health_status": {}
        }
        
        try:
            # Initialize all services
            service_results = await self.service_orchestrator.initialize_all_services()
            setup_results["services"] = service_results
            
            # Start health monitoring
            self.health_monitor.start_monitoring()
            
            # Add basic health checks
            self._setup_health_checks()
            
            # Validate environment
            await self._validate_environment()
            
            self.is_setup = True
            setup_results["status"] = "success"
            
        except Exception as e:
            setup_results["status"] = "failed"
            setup_results["error"] = str(e)
            raise TestEnvironmentError(f"Environment setup failed: {e}")
        
        return setup_results
    
    async def teardown_environment(self) -> None:
        """Clean up the test environment"""
        if not self.is_setup:
            return
        
        try:
            # Stop monitoring
            self.health_monitor.stop_monitoring()
            
            # Shutdown services
            await self.service_orchestrator.shutdown_all_services()
            
            # Clean up data if configured
            if self.config.cleanup_on_exit:
                self._cleanup_directories()
            
        except Exception as e:
            print(f"Warning: Error during teardown: {e}")
        finally:
            self.is_setup = False
    
    def _setup_health_checks(self) -> None:
        """Set up standard health checks"""
        def check_memory_usage():
            return self.health_monitor.metrics.get("memory_percent", 0) < 90
        
        def check_disk_space():
            disk_used = self.health_monitor.metrics.get("disk_used_mb", 0)
            return disk_used < self.config.max_disk_mb
        
        def check_services():
            return all(self.service_orchestrator.service_health.values())
        
        self.health_monitor.add_health_check("memory_usage", check_memory_usage)
        self.health_monitor.add_health_check("disk_space", check_disk_space)
        self.health_monitor.add_health_check("services", check_services)
    
    async def _validate_environment(self) -> None:
        """Validate the complete environment setup"""
        # Check that all critical services are running
        critical_services = ['config', 'memory_manager', 'orchestrator']
        for service in critical_services:
            if service not in self.service_orchestrator.services:
                raise EnvironmentValidationError(f"Critical service {service} not initialized")
        
        # Check resource limits
        status = self.health_monitor.get_environment_status()
        if status["overall_health"] == "unhealthy":
            raise EnvironmentValidationError("Environment health check failed")
    
    def _cleanup_directories(self) -> None:
        """Clean up temporary directories"""
        directories_to_clean = [
            self.config.isolated_data_dir,
            self.config.isolated_cache_dir,
            self.config.isolated_temp_dir
        ]
        
        if not self.config.preserve_logs:
            directories_to_clean.append(self.config.isolated_log_dir)
        
        for directory in directories_to_clean:
            if Path(directory).exists():
                try:
                    shutil.rmtree(directory)
                except Exception as e:
                    print(f"Warning: Could not clean up {directory}: {e}")
    
    @asynccontextmanager
    async def test_context(self, test_id: str):
        """Context manager for individual test execution"""
        self.current_test_context = test_id
        
        try:
            # Set up test-specific data
            self.data_manager.setup_test_data(test_id)
            
            # Provide access to services
            yield self.service_orchestrator.services
            
        finally:
            # Clean up test data
            self.data_manager.cleanup_test_data(test_id)
            self.current_test_context = None
    
    def get_service(self, service_name: str) -> Any:
        """Get a specific service instance"""
        return self.service_orchestrator.services.get(service_name)
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get complete environment information"""
        return {
            "config": asdict(self.config),
            "services": list(self.service_orchestrator.services.keys()),
            "health_status": self.health_monitor.get_environment_status(),
            "setup_complete": self.is_setup,
            "current_test": self.current_test_context
        }


# Convenience functions for common test environment operations

async def create_test_environment(config: Optional[TestEnvironmentConfig] = None) -> TestEnvironmentManager:
    """Create and set up a new test environment"""
    manager = TestEnvironmentManager(config)
    await manager.setup_environment()
    return manager


@contextmanager
def temporary_test_environment(config: Optional[TestEnvironmentConfig] = None):
    """Context manager for temporary test environment"""
    manager = None
    
    try:
        # Create and setup environment
        manager = TestEnvironmentManager(config)
        # Note: Sync context manager, so we can't use async setup here
        # This is primarily for synchronous test setup
        yield manager
        
    finally:
        if manager:
            # Cleanup environment
            manager._cleanup_directories()


if __name__ == "__main__":
    async def test_environment_setup():
        """Test the environment setup process"""
        print("Testing integration environment setup...")
        
        # Create test environment
        manager = await create_test_environment()
        
        try:
            # Get environment info
            info = manager.get_environment_info()
            print(f"Environment ID: {info['config']['environment_id']}")
            print(f"Services initialized: {info['services']}")
            print(f"Health status: {info['health_status']['overall_health']}")
            
            # Test service access
            orchestrator = manager.get_service('orchestrator')
            if orchestrator:
                print("✓ Orchestrator service available")
            
            memory_manager = manager.get_service('memory_manager')
            if memory_manager:
                print("✓ Memory manager service available")
            
            # Test context manager
            async with manager.test_context("test_demo") as services:
                print(f"✓ Test context created with {len(services)} services")
            
            print("Environment setup test completed successfully!")
            
        finally:
            await manager.teardown_environment()
    
    # Run the test
    asyncio.run(test_environment_setup()) 