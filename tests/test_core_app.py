"""
Tests for CoreApp - Core Application Shell

Tests the ultra-lightweight core application that orchestrates services.
"""

import pytest
import logging
from unittest.mock import Mock, patch, call, AsyncMock

from src.sovereign.core.app import CoreApp
from src.sovereign.core.service_manager import ServiceManager


class TestCoreApp:
    """Test the CoreApp class"""
    
    def test_initialization(self):
        """Test CoreApp initialization"""
        app = CoreApp()
        
        assert app._service_manager is not None
        assert isinstance(app._service_manager, ServiceManager)
        assert app._running is False
        assert app.service_manager is not None
    
    @patch('src.sovereign.core.app.logging.basicConfig')
    def test_logging_setup(self, mock_basic_config):
        """Test logging configuration"""
        app = CoreApp()
        
        mock_basic_config.assert_called_once_with(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def test_run_application(self):
        """Test running the application"""
        app = CoreApp()
        
        # Mock the service manager
        app._service_manager = Mock()
        
        # Run the application
        app.run()
        
        # Verify service manager was initialized
        app._service_manager.initialize.assert_called_once()
        
        # Verify application is not left running
        assert app._running is False
    
    def test_run_already_running(self, caplog):
        """Test running application when already running"""
        app = CoreApp()
        app._running = True
        
        with caplog.at_level(logging.WARNING):
            app.run()
        
        assert "already running" in caplog.text
    
    def test_run_with_exception(self):
        """Test application run with service manager exception"""
        app = CoreApp()
        app._service_manager = Mock()
        app._service_manager.initialize.side_effect = Exception("Test error")
        
        with pytest.raises(Exception, match="Test error"):
            app.run()
        
        # Verify running state is cleaned up
        assert app._running is False
    
    async def test_stop_application(self):
        """Test stopping the application"""
        app = CoreApp()
        app._running = True
        app._service_manager = AsyncMock()
        
        await app.stop()
        
        app._service_manager.shutdown.assert_called_once()
        assert app._running is False
    
    async def test_stop_not_running(self):
        """Test stopping application when not running"""
        app = CoreApp()
        app._service_manager = AsyncMock()
        
        await app.stop()
        
        # Should not call shutdown if not running
        app._service_manager.shutdown.assert_not_called()
    
    def test_service_manager_property(self):
        """Test service manager property access"""
        app = CoreApp()
        
        service_manager = app.service_manager
        
        assert service_manager is app._service_manager
        assert isinstance(service_manager, ServiceManager)


@pytest.fixture
def core_app():
    """Fixture for CoreApp instance"""
    return CoreApp()


class TestCoreAppIntegration:
    """Integration tests for CoreApp"""
    
    def test_full_lifecycle(self, core_app):
        """Test complete application lifecycle"""
        # Start application
        core_app.run()
        
        # Should complete without errors
        assert core_app._service_manager._initialized is True
        
        # Stop application
        core_app.stop()
        
        assert core_app._running is False
    
    def test_service_manager_integration(self, core_app):
        """Test integration with ServiceManager"""
        # Run application to initialize services
        core_app.run()
        
        # Check service manager state
        service_manager = core_app.service_manager
        assert service_manager._initialized is True
        
        # Check that services are registered
        services = service_manager.list_services()
        expected_services = [
            "ModelService",
            "MemoryService", 
            "ToolService",
            "ScreenContextService",
            "VoiceService"
        ]
        
        for service_name in expected_services:
            assert service_name in services 