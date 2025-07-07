"""
Tests for Screen Context Integration with AI Architecture
"""

import pytest
import asyncio
import tempfile
import time
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from src.sovereign.screen_context_integration import (
    ScreenContextIntegration, ContextAccessRequest, ContextAccessLevel, 
    ContextResponse, ScreenElementReference, ContextUpdateType
)
from src.sovereign.screen_context_manager import ScreenContextManager, ScreenContextConfig, ScreenCapture
from src.sovereign.orchestrator import ModelOrchestrator, QueryContext
from src.sovereign.config import Config


class TestScreenContextIntegration:
    """Test screen context integration with AI architecture"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        config = Mock(spec=Config)
        config.enable_screen_context = True
        config.screen_capture_interval = 4.0
        config.max_screen_captures = 100
        config.privacy_mode = False
        return config
    
    @pytest.fixture
    def screen_config(self, temp_dir):
        """Create screen context configuration"""
        return ScreenContextConfig(
            storage_path=temp_dir,
            capture_interval=4.0,
            max_stored_captures=100,
            privacy_mode=False,
            enable_preprocessing=True,
            min_text_confidence=30.0
        )
    
    @pytest.fixture
    async def mock_screen_manager(self, screen_config, mock_config):
        """Create mock screen context manager"""
        manager = Mock(spec=ScreenContextManager)
        manager.config = screen_config
        manager.consent_manager = Mock()
        manager.consent_manager.get_monitoring_state.return_value = Mock()
        manager.consent_manager.add_state_change_callback = Mock()
        manager.consent_manager.add_consent_change_callback = Mock()
        
        # Mock async methods
        manager.initialize = AsyncMock(return_value=True)
        manager.get_recent_captures = Mock(return_value=[])
        manager.get_screen_references = AsyncMock(return_value=[])
        manager.search_captures_by_text = AsyncMock(return_value=[])
        
        return manager
    
    @pytest.fixture
    async def integration(self, mock_screen_manager, mock_config):
        """Create screen context integration instance"""
        integration = ScreenContextIntegration(mock_screen_manager, mock_config)
        await integration.initialize()
        return integration
    
    @pytest.mark.asyncio
    async def test_integration_initialization(self, mock_screen_manager, mock_config):
        """Test integration initialization"""
        integration = ScreenContextIntegration(mock_screen_manager, mock_config)
        
        # Test initialization
        result = await integration.initialize()
        assert result is True
        
        # Verify screen manager was initialized
        mock_screen_manager.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_access_request_basic(self, integration):
        """Test basic context access request"""
        request = ContextAccessRequest(
            requester="talker_model",
            access_level=ContextAccessLevel.BASIC
        )
        
        response = await integration.get_current_context(request)
        
        assert isinstance(response, ContextResponse)
        assert response.success is True
        assert response.access_level == ContextAccessLevel.BASIC
    
    @pytest.mark.asyncio
    async def test_context_access_permission_denied(self, integration):
        """Test context access with insufficient permissions"""
        request = ContextAccessRequest(
            requester="external_model",  # Should have NONE access level
            access_level=ContextAccessLevel.ENHANCED
        )
        
        response = await integration.get_current_context(request)
        
        assert response.success is False
        assert response.privacy_filtered is True
        assert "Access denied" in response.error_message
    
    @pytest.mark.asyncio
    async def test_screen_element_references(self, integration, mock_screen_manager):
        """Test getting screen element references"""
        # Mock screen references
        mock_references = [
            Mock(
                reference_id="ref_1",
                element_type="text",
                text_content="Sample text",
                bounding_box=(10, 20, 100, 30),
                confidence=0.9,
                semantic_tags=["button"]
            )
        ]
        mock_screen_manager.get_screen_references.return_value = mock_references
        
        references = await integration.get_screen_element_references()
        
        assert len(references) == 1
        assert isinstance(references[0], ScreenElementReference)
        assert references[0].ref_id == "ref_1"
        assert references[0].confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_text_search(self, integration, mock_screen_manager):
        """Test text search functionality"""
        # Mock search results
        mock_captures = [
            Mock(
                id="capture_1",
                timestamp=time.time(),
                text_content="Test content for search",
                relevance_score=0.8,
                bounding_boxes=[(0, 0, 100, 50)],
                ocr_confidence=0.9
            )
        ]
        mock_screen_manager.search_captures_by_text.return_value = mock_captures
        
        results = await integration.search_context_by_text("Test content", requester="orchestrator")
        
        assert len(results) == 1
        assert results[0]["capture_id"] == "capture_1"
        assert "Test content" in results[0]["text_content"]
    
    @pytest.mark.asyncio
    async def test_context_summary(self, integration, mock_screen_manager):
        """Test context summary generation"""
        # Mock recent captures
        mock_captures = [
            Mock(
                timestamp=time.time(),
                text_content="Sample screen content",
                bounding_boxes=[(0, 0, 100, 50)]
            )
        ]
        mock_screen_manager.get_recent_captures.return_value = mock_captures
        
        summary = await integration.get_context_summary(format_type="text_summary")
        
        assert summary is not None
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_access_policy_management(self, integration):
        """Test access policy management"""
        # Test setting access policy
        integration.set_access_policy("test_model", ContextAccessLevel.ENHANCED)
        assert integration.access_policies["test_model"] == ContextAccessLevel.ENHANCED
        
        # Test getting integration stats
        stats = integration.get_integration_stats()
        assert "access_stats" in stats
        assert "cache_stats" in stats
        assert "privacy_stats" in stats
    
    @pytest.mark.asyncio
    async def test_caching_mechanism(self, integration):
        """Test context caching mechanism"""
        request = ContextAccessRequest(
            requester="orchestrator",
            access_level=ContextAccessLevel.STANDARD
        )
        
        # First request - should not be cached
        response1 = await integration.get_current_context(request)
        
        # Second identical request - should be cached
        response2 = await integration.get_current_context(request)
        
        # Both should be successful
        assert response1.success is True
        assert response2.success is True
        
        # Check cache hit in stats
        stats = integration.get_integration_stats()
        assert stats["access_stats"]["cache_hits"] > 0
    
    @pytest.mark.asyncio
    async def test_privacy_filtering(self, integration):
        """Test privacy filtering functionality"""
        request = ContextAccessRequest(
            requester="orchestrator",
            access_level=ContextAccessLevel.ENHANCED,
            privacy_aware=True
        )
        
        # Mock context data with sensitive content
        with patch.object(integration, '_get_filtered_captures') as mock_get_captures:
            mock_get_captures.return_value = []
            
            with patch.object(integration, '_apply_privacy_filtering') as mock_filter:
                mock_filter.return_value = ({"filtered": "data"}, True)
                
                response = await integration.get_current_context(request)
                
                assert response.success is True
                assert response.privacy_filtered is True
    
    def test_update_callback_registration(self, integration):
        """Test update callback registration"""
        callback_called = False
        
        def test_callback(update):
            nonlocal callback_called
            callback_called = True
        
        integration.register_update_callback(test_callback)
        assert len(integration.update_callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_cleanup(self, integration):
        """Test integration cleanup"""
        # Add some data to clean up
        integration.context_cache["test"] = (Mock(), time.time())
        integration.update_callbacks.append(Mock())
        
        await integration.cleanup()
        
        # Verify cleanup
        assert len(integration.context_cache) == 0
        assert len(integration.update_callbacks) == 0


class TestOrchestratorScreenContextIntegration:
    """Test orchestrator integration with screen context"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        config = Mock(spec=Config)
        config.enable_screen_context = True
        config.screen_capture_interval = 4.0
        config.max_screen_captures = 100
        config.privacy_mode = False
        return config
    
    @pytest.fixture
    async def mock_orchestrator(self, mock_config):
        """Create mock orchestrator with screen context"""
        orchestrator = ModelOrchestrator(mock_config)
        
        # Mock the models
        orchestrator.talker_model = Mock()
        orchestrator.talker_model.initialize = AsyncMock(return_value=True)
        orchestrator.talker_model.generate_response = AsyncMock(return_value="Mock response")
        orchestrator.talker_model.close = AsyncMock()
        
        orchestrator.thinker_model = Mock()
        orchestrator.thinker_model.initialize = AsyncMock(return_value=True)
        orchestrator.thinker_model.auto_process = AsyncMock(return_value="Mock detailed response")
        orchestrator.thinker_model.close = AsyncMock()
        
        # Mock screen context system
        with patch('src.sovereign.orchestrator.ScreenContextManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.initialize = AsyncMock(return_value=True)
            mock_manager.start_capture = AsyncMock()
            mock_manager.stop_capture = AsyncMock()
            mock_manager.cleanup = AsyncMock()
            mock_manager.consent_manager = Mock()
            mock_manager.consent_manager.get_monitoring_state.return_value = Mock()
            mock_manager.consent_manager.add_state_change_callback = Mock()
            mock_manager.consent_manager.add_consent_change_callback = Mock()
            mock_manager_class.return_value = mock_manager
            
            with patch('src.sovereign.orchestrator.ScreenContextIntegration') as mock_integration_class:
                mock_integration = Mock()
                mock_integration.initialize = AsyncMock(return_value=True)
                mock_integration.get_current_context = AsyncMock()
                mock_integration.get_context_summary = AsyncMock(return_value="Mock summary")
                mock_integration.get_screen_element_references = AsyncMock(return_value=[])
                mock_integration.cleanup = AsyncMock()
                mock_integration_class.return_value = mock_integration
                
                await orchestrator.initialize()
                
                return orchestrator
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization_with_screen_context(self, mock_orchestrator):
        """Test orchestrator initialization includes screen context"""
        assert mock_orchestrator.screen_context_manager is not None
        assert mock_orchestrator.screen_context_integration is not None
        assert mock_orchestrator.enable_screen_context is True
        assert mock_orchestrator.telemetry['screen_context_enabled'] == 1
    
    @pytest.mark.asyncio
    async def test_query_processing_with_screen_context(self, mock_orchestrator):
        """Test query processing with screen context enrichment"""
        # Mock screen context response
        mock_response = Mock()
        mock_response.success = True
        mock_response.timestamp = time.time()
        mock_response.captures_count = 2
        mock_response.references_count = 3
        mock_response.privacy_filtered = False
        mock_response.access_level = Mock()
        mock_response.access_level.value = "enhanced"
        mock_response.data = {"text_content": "Sample screen content"}
        
        mock_orchestrator.screen_context_integration.get_current_context.return_value = mock_response
        
        # Process a query
        result = await mock_orchestrator.process_query("What's on my screen?")
        
        assert result.response is not None
        assert mock_orchestrator.telemetry['screen_context_requests'] > 0
    
    @pytest.mark.asyncio
    async def test_context_enrichment_with_screen_data(self, mock_orchestrator):
        """Test context enrichment with screen data"""
        context = QueryContext(
            user_input="Test query",
            timestamp=time.time(),
            session_id="test",
            previous_queries=[],
            conversation_history=[]
        )
        
        # Mock screen context response
        mock_response = Mock()
        mock_response.success = True
        mock_response.timestamp = time.time()
        mock_response.captures_count = 1
        mock_response.references_count = 2
        mock_response.privacy_filtered = False
        mock_response.access_level = Mock()
        mock_response.access_level.value = "enhanced"
        mock_response.data = {"text_content": "Test screen content"}
        
        mock_orchestrator.screen_context_integration.get_current_context.return_value = mock_response
        
        # Enrich context
        await mock_orchestrator._enrich_context_with_screen_data(context)
        
        # Verify context was enriched
        assert context.screen_context is not None
        assert context.screen_context["captures_count"] == 1
        assert context.screen_context["references_count"] == 2
        assert context.screen_context["has_text_content"] is True
    
    @pytest.mark.asyncio
    async def test_complexity_analysis_with_screen_context(self, mock_orchestrator):
        """Test complexity analysis considers screen context"""
        context = QueryContext(
            user_input="What's currently visible on screen?",
            timestamp=time.time(),
            session_id="test",
            previous_queries=[],
            conversation_history=[]
        )
        
        # Add screen context
        context.screen_context = {
            "has_text_content": True,
            "text_length": 1000,  # Large text content
            "elements": ["elem1", "elem2", "elem3", "elem4", "elem5", "elem6"]  # Many elements
        }
        
        # Analyze complexity
        complexity, confidence = mock_orchestrator.determine_complexity(
            "What's currently visible on screen?", context
        )
        
        # Should be complex due to screen context keywords and content
        assert complexity.value in ["moderate", "complex", "very_complex"]
    
    @pytest.mark.asyncio
    async def test_orchestrator_cleanup_with_screen_context(self, mock_orchestrator):
        """Test orchestrator cleanup includes screen context cleanup"""
        await mock_orchestrator.close()
        
        # Verify screen context was cleaned up
        mock_orchestrator.screen_context_manager.stop_capture.assert_called_once()
        mock_orchestrator.screen_context_manager.cleanup.assert_called_once()
        mock_orchestrator.screen_context_integration.cleanup.assert_called_once()


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_ai_can_reference_screen_elements(self):
        """Test that AI can reference specific screen elements"""
        # This would be a more comprehensive test in a real environment
        # For now, we'll test the API structure
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Mock(spec=Config)
            config.enable_screen_context = True
            
            screen_config = ScreenContextConfig(
                storage_path=temp_dir,
                privacy_mode=False
            )
            
            # Mock screen manager
            screen_manager = Mock(spec=ScreenContextManager)
            screen_manager.initialize = AsyncMock(return_value=True)
            screen_manager.consent_manager = Mock()
            screen_manager.consent_manager.get_monitoring_state.return_value = Mock()
            screen_manager.consent_manager.add_state_change_callback = Mock()
            screen_manager.consent_manager.add_consent_change_callback = Mock()
            screen_manager.get_recent_captures = Mock(return_value=[])
            screen_manager.get_screen_references = AsyncMock(return_value=[])
            
            # Create integration
            integration = ScreenContextIntegration(screen_manager, config)
            await integration.initialize()
            
            # Test AI-friendly element reference
            references = await integration.get_screen_element_references()
            assert isinstance(references, list)
    
    @pytest.mark.asyncio
    async def test_real_time_context_updates(self):
        """Test real-time context update capabilities"""
        # Mock update notifications
        update_received = False
        
        def update_callback(notification):
            nonlocal update_received
            update_received = True
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Mock(spec=Config)
            screen_config = ScreenContextConfig(storage_path=temp_dir)
            
            # Mock screen manager
            screen_manager = Mock(spec=ScreenContextManager)
            screen_manager.initialize = AsyncMock(return_value=True)
            screen_manager.consent_manager = Mock()
            screen_manager.consent_manager.get_monitoring_state.return_value = Mock()
            screen_manager.consent_manager.add_state_change_callback = Mock()
            screen_manager.consent_manager.add_consent_change_callback = Mock()
            
            integration = ScreenContextIntegration(screen_manager, config)
            await integration.initialize()
            
            # Register callback
            integration.register_update_callback(update_callback)
            
            # Verify callback registration
            assert len(integration.update_callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_performance_and_latency(self):
        """Test performance and latency requirements"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Mock(spec=Config)
            screen_config = ScreenContextConfig(storage_path=temp_dir)
            
            # Mock fast screen manager
            screen_manager = Mock(spec=ScreenContextManager)
            screen_manager.initialize = AsyncMock(return_value=True)
            screen_manager.consent_manager = Mock()
            screen_manager.consent_manager.get_monitoring_state.return_value = Mock()
            screen_manager.consent_manager.add_state_change_callback = Mock()
            screen_manager.consent_manager.add_consent_change_callback = Mock()
            screen_manager.get_recent_captures = Mock(return_value=[])
            screen_manager.get_screen_references = AsyncMock(return_value=[])
            
            integration = ScreenContextIntegration(screen_manager, config)
            await integration.initialize()
            
            # Test response time
            start_time = time.time()
            
            request = ContextAccessRequest(
                requester="orchestrator",
                access_level=ContextAccessLevel.STANDARD
            )
            
            response = await integration.get_current_context(request)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Should be under 100ms for minimal latency
            assert response_time < 0.1
            assert response.success is True 