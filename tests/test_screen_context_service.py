"""
Tests for ScreenContextService

Tests are organized into two categories:
- not_heavy: Tests service creation and basic functionality without triggering heavy imports
- heavy: Integration tests that load OCR and screen capture components
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from src.sovereign.services.screen_context_service import ScreenContextService, ScreenCaptureResult


class TestScreenContextServiceConstruction:
    """Test service creation and lightweight operations"""
    
    @pytest.mark.not_heavy
    def test_service_creation(self):
        """Test that ScreenContextService can be created without heavy imports"""
        service = ScreenContextService()
        
        # Basic service properties
        assert service is not None
        assert not service.is_capture_system_loaded()
        assert service.get_capture_state() == "uninitialized"
        
        # Statistics are initialized
        stats = service.get_stats()
        assert stats['total_captures'] == 0
        assert stats['successful_captures'] == 0
        assert stats['failed_captures'] == 0
        assert stats['initialization_time'] == 0.0
    
    @pytest.mark.not_heavy
    def test_health_check_uninitialized(self):
        """Test health check before system is loaded"""
        service = ScreenContextService()
        
        health = service.get_health_check()
        
        assert health['service_name'] == 'ScreenContextService'
        assert health['status'] == 'uninitialized'
        assert not health['capture_system_loaded']
        assert health['total_captures'] == 0
        assert health['capture_state'] == 'uninitialized'
        assert not health['components']['screen_manager']
        assert not health['components']['ocr_engine']
        assert not health['components']['privacy_manager']
    
    @pytest.mark.not_heavy
    def test_screen_capture_result_creation(self):
        """Test ScreenCaptureResult creation and properties"""
        # Success case
        result = ScreenCaptureResult(
            success=True,
            capture_id="test123",
            text_content="Hello World",
            bounding_boxes=[(10, 20, 100, 50)],
            processing_time=0.5,
            ocr_confidence=85.0
        )
        
        assert result.success
        assert result.capture_id == "test123"
        assert result.text_content == "Hello World"
        assert len(result.bounding_boxes) == 1
        assert result.bounding_boxes[0] == (10, 20, 100, 50)
        assert result.processing_time == 0.5
        assert result.ocr_confidence == 85.0
        assert result.error is None
    
    @pytest.mark.not_heavy
    def test_screen_capture_result_error_case(self):
        """Test ScreenCaptureResult error case"""
        result = ScreenCaptureResult(
            success=False,
            error="OCR initialization failed",
            processing_time=0.1
        )
        
        assert not result.success
        assert result.error == "OCR initialization failed"
        assert result.capture_id is None
        assert result.text_content == ""
        assert len(result.bounding_boxes) == 0  # Should be empty list from __post_init__
        assert result.processing_time == 0.1
        assert result.ocr_confidence == 0.0


class TestScreenContextServiceIntegration:
    """Test service integration with heavy screen capture and OCR components"""
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_lazy_loading_on_first_capture(self):
        """Test that screen capture components load only on first capture"""
        service = ScreenContextService()
        
        # Mock screen context manager and its dependencies
        with patch('src.sovereign.services.screen_context_service.ScreenContextManager') as mock_manager_class, \
             patch('src.sovereign.services.screen_context_service.ScreenContextConfig') as mock_config_class, \
             patch('src.sovereign.services.screen_context_service.Config') as mock_main_config_class:
            
            # Setup mocks
            mock_main_config = Mock()
            mock_main_config.screen_capture.capture_interval = 5.0
            mock_main_config.screen_capture.max_screenshot_history = 100
            mock_main_config.screen_capture.enabled = True
            mock_main_config.screen_capture.tesseract_cmd_path = None
            mock_main_config_class.return_value = mock_main_config
            
            mock_screen_config = Mock()
            mock_config_class.return_value = mock_screen_config
            
            mock_manager = Mock()
            mock_capture = Mock()
            mock_capture.id = "capture_123"
            mock_capture.text_content = "Test screen text"
            mock_capture.bounding_boxes = [(10, 20, 100, 50)]
            mock_capture.ocr_confidence = 90.0
            mock_manager.capture_screen = AsyncMock(return_value=mock_capture)
            mock_manager_class.return_value = mock_manager
            
            # Verify service is uninitialized
            assert not service.is_capture_system_loaded()
            
            # First capture should trigger loading
            result = await service.capture_and_analyse()
            
            # Verify components were loaded
            assert service.is_capture_system_loaded()
            
            # Verify capture result
            assert result.success
            assert result.capture_id == "capture_123"
            assert result.text_content == "Test screen text"
            assert result.bounding_boxes == [(10, 20, 100, 50)]
            assert result.ocr_confidence == 90.0
            
            # Verify manager was called correctly
            mock_manager.capture_screen.assert_called_once()
            
            # Check statistics were updated
            stats = service.get_stats()
            assert stats['total_captures'] == 1
            assert stats['successful_captures'] == 1
            assert stats['failed_captures'] == 0
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_screen_manager_initialization_failure(self):
        """Test handling of screen manager initialization failure"""
        service = ScreenContextService()
        
        with patch('src.sovereign.services.screen_context_service.ScreenContextManager') as mock_manager_class:
            mock_manager_class.side_effect = Exception("Failed to initialize screen capture")
            
            # Attempt to capture should fail gracefully
            result = await service.capture_and_analyse()
            
            assert not result.success
            assert "Screen capture initialization failed" in result.error
            assert result.capture_id is None
            assert result.processing_time > 0
            
            # Service should not be marked as initialized
            assert not service.is_capture_system_loaded()
            
            # Statistics should reflect the failed attempt
            stats = service.get_stats()
            assert stats['total_captures'] == 1
            assert stats['successful_captures'] == 0
            assert stats['failed_captures'] == 1
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_concurrent_loading_protection(self):
        """Test that concurrent access doesn't cause double loading"""
        service = ScreenContextService()
        
        # Track initialization calls
        init_call_count = 0
        
        def count_init(*args, **kwargs):
            nonlocal init_call_count
            init_call_count += 1
            mock = Mock()
            mock.capture_screen = AsyncMock(return_value=Mock(
                id=f"capture_{init_call_count}",
                text_content="test text",
                bounding_boxes=[],
                ocr_confidence=80.0
            ))
            return mock
        
        with patch('src.sovereign.services.screen_context_service.ScreenContextManager', side_effect=count_init), \
             patch('src.sovereign.services.screen_context_service.ScreenContextConfig'), \
             patch('src.sovereign.services.screen_context_service.Config'):
            
            # Start multiple concurrent captures
            tasks = [
                service.capture_and_analyse(),
                service.capture_and_analyse(),
                service.capture_and_analyse()
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert all(result.success for result in results)
            
            # Screen manager should only be initialized once despite concurrent access
            assert init_call_count == 1
            assert service.is_capture_system_loaded()
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_privacy_blocking_capture(self):
        """Test that privacy settings can block captures"""
        service = ScreenContextService()
        
        with patch('src.sovereign.services.screen_context_service.ScreenContextManager') as mock_manager_class, \
             patch('src.sovereign.services.screen_context_service.ScreenContextConfig'), \
             patch('src.sovereign.services.screen_context_service.Config'):
            
            mock_manager = Mock()
            mock_manager.capture_screen = AsyncMock(return_value=None)  # Privacy blocked
            mock_manager_class.return_value = mock_manager
            
            # Capture should be blocked
            result = await service.capture_and_analyse()
            
            assert not result.success
            assert "Capture blocked by privacy settings" in result.error
            assert result.capture_id is None
            
            # Should be marked as failed capture
            stats = service.get_stats()
            assert stats['total_captures'] == 1
            assert stats['successful_captures'] == 0
            assert stats['failed_captures'] == 1
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_get_recent_captures_with_loading(self):
        """Test getting recent captures triggers loading if needed"""
        service = ScreenContextService()
        
        with patch('src.sovereign.services.screen_context_service.ScreenContextManager') as mock_manager_class, \
             patch('src.sovereign.services.screen_context_service.ScreenContextConfig'), \
             patch('src.sovereign.services.screen_context_service.Config'):
            
            # Create mock captures
            mock_capture1 = Mock()
            mock_capture1.id = "cap1"
            mock_capture1.timestamp = 1000
            mock_capture1.text_content = "First capture"
            mock_capture1.bounding_boxes = []
            mock_capture1.ocr_confidence = 85.0
            mock_capture1.relevance_score = 0.7
            mock_capture1.processing_time = 0.5
            
            mock_capture2 = Mock()
            mock_capture2.id = "cap2"
            mock_capture2.timestamp = 2000
            mock_capture2.text_content = "Second capture"
            mock_capture2.bounding_boxes = [(10, 10, 50, 20)]
            mock_capture2.ocr_confidence = 90.0
            mock_capture2.relevance_score = 0.8
            mock_capture2.processing_time = 0.3
            
            mock_manager = Mock()
            mock_manager.get_recent_captures.return_value = [mock_capture1, mock_capture2]
            mock_manager_class.return_value = mock_manager
            
            # Get recent captures should trigger loading
            captures = await service.get_recent_captures(limit=5)
            
            assert len(captures) == 2
            assert captures[0]['id'] == "cap1"
            assert captures[0]['text_content'] == "First capture"
            assert captures[1]['id'] == "cap2"
            assert captures[1]['text_content'] == "Second capture"
            
            # Should be loaded now
            assert service.is_capture_system_loaded()
            
            # Manager method should have been called with limit
            mock_manager.get_recent_captures.assert_called_once_with(limit=5)
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_search_by_text_functionality(self):
        """Test text search functionality"""
        service = ScreenContextService()
        
        with patch('src.sovereign.services.screen_context_service.ScreenContextManager') as mock_manager_class, \
             patch('src.sovereign.services.screen_context_service.ScreenContextConfig'), \
             patch('src.sovereign.services.screen_context_service.Config'):
            
            # Mock search results
            mock_capture = Mock()
            mock_capture.id = "search_result"
            mock_capture.timestamp = 1500
            mock_capture.text_content = "Found matching text"
            mock_capture.relevance_score = 0.9
            mock_capture.ocr_confidence = 88.0
            
            mock_manager = Mock()
            mock_manager.search_captures_by_text = AsyncMock(return_value=[mock_capture])
            mock_manager_class.return_value = mock_manager
            
            # Search should work
            results = await service.search_by_text("matching", limit=10)
            
            assert len(results) == 1
            assert results[0]['id'] == "search_result"
            assert results[0]['text_content'] == "Found matching text"
            assert results[0]['match_type'] == 'text_search'
            
            # Manager search method should have been called
            mock_manager.search_captures_by_text.assert_called_once_with("matching", 10)
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_continuous_capture_control(self):
        """Test starting and stopping continuous capture"""
        service = ScreenContextService()
        
        with patch('src.sovereign.services.screen_context_service.ScreenContextManager') as mock_manager_class, \
             patch('src.sovereign.services.screen_context_service.ScreenContextConfig'), \
             patch('src.sovereign.services.screen_context_service.Config'):
            
            mock_manager = Mock()
            mock_manager.start_capture = AsyncMock(return_value=None)
            mock_manager.stop_capture = AsyncMock(return_value=None)
            mock_manager_class.return_value = mock_manager
            
            # Start continuous capture
            success = await service.start_continuous_capture()
            assert success
            mock_manager.start_capture.assert_called_once()
            
            # Stop continuous capture
            success = await service.stop_continuous_capture()
            assert success
            mock_manager.stop_capture.assert_called_once()
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_service_cleanup(self):
        """Test proper cleanup of service resources"""
        service = ScreenContextService()
        
        with patch('src.sovereign.services.screen_context_service.ScreenContextManager') as mock_manager_class, \
             patch('src.sovereign.services.screen_context_service.ScreenContextConfig'), \
             patch('src.sovereign.services.screen_context_service.Config'):
            
            mock_manager = Mock()
            mock_manager.stop_capture = AsyncMock()
            mock_manager_class.return_value = mock_manager
            
            # Initialize by starting capture
            await service.start_continuous_capture()
            
            # Verify service is initialized
            assert service.is_capture_system_loaded()
            
            # Cleanup the service
            await service.close()
            
            # Verify cleanup was called on manager
            mock_manager.stop_capture.assert_called()
            
            # Verify service state is reset
            assert not service.is_capture_system_loaded()
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_capture_error_handling(self):
        """Test error handling during capture execution"""
        service = ScreenContextService()
        
        with patch('src.sovereign.services.screen_context_service.ScreenContextManager') as mock_manager_class, \
             patch('src.sovereign.services.screen_context_service.ScreenContextConfig'), \
             patch('src.sovereign.services.screen_context_service.Config'):
            
            mock_manager = Mock()
            # Mock manager to raise an exception during capture
            mock_manager.capture_screen = AsyncMock(side_effect=RuntimeError("Screen capture failed"))
            mock_manager_class.return_value = mock_manager
            
            # Capture should handle the exception gracefully
            result = await service.capture_and_analyse()
            
            assert not result.success
            assert result.error == "Screen capture failed"
            assert result.capture_id is None
            assert result.processing_time > 0
            
            # Statistics should reflect the failure
            stats = service.get_stats()
            assert stats['total_captures'] == 1
            assert stats['successful_captures'] == 0
            assert stats['failed_captures'] == 1 