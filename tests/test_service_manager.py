"""
Tests for ServiceManager - Service Orchestration System

Tests the lightweight service manager that handles lazy loading.
"""

import pytest
import logging
from unittest.mock import Mock, patch, AsyncMock

from src.sovereign.core.service_manager import ServiceManager, ServiceStatus


class TestServiceManager:
    """Test the ServiceManager class"""
    
    def test_initialization(self):
        """Test ServiceManager initialization"""
        manager = ServiceManager()
        
        assert manager._services == {}
        assert manager._service_status == {}
        assert manager._initialized is False
    
    def test_initialize_manager(self):
        """Test initializing the service manager"""
        manager = ServiceManager()
        
        manager.initialize()
        
        assert manager._initialized is True
        
        # Check that services are registered
        expected_services = [
            "ModelService",
            "MemoryService", 
            "ToolService",
            "ScreenContextService",
            "VoiceService"
        ]
        
        for service_name in expected_services:
            assert service_name in manager._service_status
            assert manager._service_status[service_name] == ServiceStatus.UNINITIALIZED
    
    def test_initialize_already_initialized(self, caplog):
        """Test initializing already initialized manager"""
        manager = ServiceManager()
        manager.initialize()
        
        with caplog.at_level(logging.WARNING):
            manager.initialize()
        
        assert "already initialized" in caplog.text
    
    def test_get_service_not_initialized(self):
        """Test getting service when manager not initialized"""
        manager = ServiceManager()
        
        with pytest.raises(RuntimeError, match="ServiceManager not initialized"):
            manager.get_service("ModelService")
    
    def test_get_unknown_service(self, caplog):
        """Test getting unknown service"""
        manager = ServiceManager()
        manager.initialize()
        
        with caplog.at_level(logging.ERROR):
            result = manager.get_service("UnknownService")
        
        assert result is None
        assert "Unknown service: UnknownService" in caplog.text
    
    def test_get_service_lazy_loading(self, caplog):
        """Test lazy loading behavior"""
        manager = ServiceManager()
        manager.initialize()
        
        # Test ModelService loading
        model_service = manager.get_service("ModelService")
        assert model_service is not None
        assert manager.get_service_status("ModelService") == ServiceStatus.READY
        
        # Test MemoryService loading
        memory_service = manager.get_service("MemoryService")
        assert memory_service is not None
        assert manager.get_service_status("MemoryService") == ServiceStatus.READY
    
    def test_is_service_ready(self):
        """Test checking if service is ready"""
        manager = ServiceManager()
        manager.initialize()
        
        # Initially not ready
        assert manager.is_service_ready("ModelService") is False
        
        # After "loading" (placeholder), should be ready
        manager.get_service("ModelService")
        assert manager.is_service_ready("ModelService") is True
    
    def test_get_service_status(self):
        """Test getting service status"""
        manager = ServiceManager()
        
        # Unknown service
        assert manager.get_service_status("UnknownService") == ServiceStatus.UNINITIALIZED
        
        manager.initialize()
        
        # Known service, not loaded
        assert manager.get_service_status("ModelService") == ServiceStatus.UNINITIALIZED
        
        # After loading
        manager.get_service("ModelService")
        assert manager.get_service_status("ModelService") == ServiceStatus.READY
    
    def test_list_services(self):
        """Test listing all services"""
        manager = ServiceManager()
        
        # Before initialization
        services = manager.list_services()
        assert services == {}
        
        # After initialization
        manager.initialize()
        services = manager.list_services()
        
        expected_services = [
            "ModelService",
            "MemoryService", 
            "ToolService",
            "ScreenContextService",
            "VoiceService"
        ]
        
        assert len(services) == len(expected_services)
        for service_name in expected_services:
            assert service_name in services
            assert services[service_name] == ServiceStatus.UNINITIALIZED
    
    def test_convenience_methods(self):
        """Test convenience methods for accessing services"""
        manager = ServiceManager()
        manager.initialize()
        
        # Test model() convenience method
        model_service = manager.model()
        assert model_service is not None
        assert manager.is_service_ready("ModelService")
        
        # Should return same instance on subsequent calls
        model_service2 = manager.model()
        assert model_service is model_service2
        
        # Test memory() convenience method
        memory_service = manager.memory()
        assert memory_service is not None
        assert manager.is_service_ready("MemoryService")
        
        # Should return same instance on subsequent calls
        memory_service2 = manager.memory()
        assert memory_service is memory_service2
    
    async def test_shutdown(self):
        """Test shutting down the service manager"""
        manager = ServiceManager()
        manager.initialize()
        
        # Mock a loaded service
        mock_service = AsyncMock()
        manager._services["TestService"] = mock_service
        manager._service_status["TestService"] = ServiceStatus.READY
        
        # Shutdown
        await manager.shutdown()
        
        assert manager._initialized is False
        assert len(manager._services) == 0
        
        # Service close should have been called
        mock_service.close.assert_called_once()
    
    async def test_shutdown_not_initialized(self):
        """Test shutting down when not initialized"""
        manager = ServiceManager()
        
        # Should not raise any errors
        await manager.shutdown()
        
        assert manager._initialized is False


class TestServiceStatus:
    """Test the ServiceStatus enum"""
    
    def test_status_values(self):
        """Test all status enum values"""
        assert ServiceStatus.UNINITIALIZED.value == "uninitialized"
        assert ServiceStatus.INITIALIZING.value == "initializing"
        assert ServiceStatus.READY.value == "ready"
        assert ServiceStatus.ERROR.value == "error"
        assert ServiceStatus.STOPPED.value == "stopped"


@pytest.fixture
def service_manager():
    """Fixture for ServiceManager instance"""
    return ServiceManager()


@pytest.fixture
def initialized_service_manager():
    """Fixture for initialized ServiceManager instance"""
    manager = ServiceManager()
    manager.initialize()
    return manager


class TestServiceManagerIntegration:
    """Integration tests for ServiceManager"""
    
    async def test_complete_service_lifecycle(self, service_manager):
        """Test complete service lifecycle"""
        # Initialize
        service_manager.initialize()
        assert service_manager._initialized is True
        
        # Check initial state
        services = service_manager.list_services()
        for status in services.values():
            assert status == ServiceStatus.UNINITIALIZED
        
        # Load services
        model_service = service_manager.get_service("ModelService")
        assert model_service is not None
        assert service_manager.is_service_ready("ModelService") is True
        
        memory_service = service_manager.get_service("MemoryService")
        assert memory_service is not None
        assert service_manager.is_service_ready("MemoryService") is True
        
        # Shutdown
        await service_manager.shutdown()
        assert service_manager._initialized is False
    
    def test_service_registration_completeness(self, initialized_service_manager):
        """Test that all expected services are registered"""
        services = initialized_service_manager.list_services()
        
        # Must have all the services needed for the refactor
        required_services = {
            "ModelService",      # For Talker/Thinker
            "MemoryService",     # For FAISS + MemoryManager
            "ToolService",       # For tool discovery/execution
            "ScreenContextService",  # For OCR/screen capture
            "VoiceService"       # For microphone/speaker
        }
        
        registered_services = set(services.keys())
        assert required_services == registered_services 