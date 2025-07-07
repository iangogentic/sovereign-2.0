"""
Tests for ModelService - Lazy-loaded AI Model Management

Tests the ModelService that provides on-demand loading of Talker and Thinker models.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, AsyncMock

from src.sovereign.services.model_service import ModelService, ModelType, ModelStatus


class TestModelService:
    """Test the ModelService class"""
    
    @pytest.mark.not_heavy
    def test_initialization(self):
        """Test ModelService initialization (no heavy imports)"""
        service = ModelService()
        
        # Check initial state
        assert service._talker_model is None
        assert service._thinker_model is None
        assert service._model_status[ModelType.TALKER] == ModelStatus.UNLOADED
        assert service._model_status[ModelType.THINKER] == ModelStatus.UNLOADED
        assert service._query_count == 0
        assert service._talker_queries == 0
        assert service._thinker_queries == 0
    
    @pytest.mark.not_heavy
    def test_model_complexity_detection(self):
        """Test query complexity detection logic (no model loading)"""
        service = ModelService()
        
        # Simple queries should use Talker
        simple_queries = [
            "Hello, how are you?",
            "What time is it?",
            "Thanks for your help",
            "Yes, that's correct"
        ]
        
        for query in simple_queries:
            assert not service._should_use_thinker(query), f"Query '{query}' should use Talker"
        
        # Complex queries should use Thinker
        complex_queries = [
            "Analyze the pros and cons of different database architectures",
            "Implement a binary search algorithm in Python",
            "Explain step by step how to debug a memory leak",
            "Write comprehensive code for a REST API with authentication",
            "Plan a strategy for migrating legacy systems"
        ]
        
        for query in complex_queries:
            assert service._should_use_thinker(query), f"Query '{query}' should use Thinker"
    
    @pytest.mark.not_heavy
    def test_status_methods(self):
        """Test status checking methods (no model loading)"""
        service = ModelService()
        
        # Initially no models ready
        assert not service.is_model_ready(ModelType.TALKER)
        assert not service.is_model_ready(ModelType.THINKER)
        
        # Check status dictionary
        status = service.get_status()
        assert status["talker_status"] == "unloaded"
        assert status["thinker_status"] == "unloaded"
        assert status["total_queries"] == 0
        assert status["talker_queries"] == 0
        assert status["thinker_queries"] == 0
        assert status["load_times"] == {}
    
    @pytest.mark.not_heavy
    @patch('src.sovereign.services.model_service.ModelService._load_talker_model')
    async def test_get_talker_success(self, mock_load):
        """Test getting Talker model when loading succeeds"""
        service = ModelService()
        mock_talker = Mock()
        service._talker_model = mock_talker
        mock_load.return_value = True
        
        result = await service.get_talker()
        assert result is mock_talker
        mock_load.assert_called_once()
    
    @pytest.mark.not_heavy
    @patch('src.sovereign.services.model_service.ModelService._load_talker_model')
    async def test_get_talker_failure(self, mock_load):
        """Test getting Talker model when loading fails"""
        service = ModelService()
        mock_load.return_value = False
        
        result = await service.get_talker()
        assert result is None
        mock_load.assert_called_once()
    
    @pytest.mark.not_heavy
    @patch('src.sovereign.services.model_service.ModelService._load_thinker_model')
    async def test_get_thinker_success(self, mock_load):
        """Test getting Thinker model when loading succeeds"""
        service = ModelService()
        mock_thinker = Mock()
        service._thinker_model = mock_thinker
        mock_load.return_value = True
        
        result = await service.get_thinker()
        assert result is mock_thinker
        mock_load.assert_called_once()
    
    @pytest.mark.not_heavy
    @patch('src.sovereign.services.model_service.ModelService._load_thinker_model')
    async def test_get_thinker_failure(self, mock_load):
        """Test getting Thinker model when loading fails"""
        service = ModelService()
        mock_load.return_value = False
        
        result = await service.get_thinker()
        assert result is None
        mock_load.assert_called_once()


class TestModelServiceMocked:
    """Test ModelService with mocked model loading"""
    
    @pytest.mark.not_heavy
    @patch('src.sovereign.services.model_service.ModelService._load_talker_model')
    async def test_query_simple_uses_talker(self, mock_load_talker):
        """Test that simple queries route to Talker model"""
        service = ModelService()
        
        # Mock Talker model
        mock_talker = Mock()  # Use regular Mock for sync methods
        mock_talker.generate_response = AsyncMock(return_value="Hello! I'm the Talker model.")
        mock_talker.detect_complex_query.return_value = False  # No handoff
        service._talker_model = mock_talker
        mock_load_talker.return_value = True
        
        response = await service.query("Hello, how are you?")
        
        assert "Hello! I'm the Talker model." == response
        mock_load_talker.assert_called_once()
        mock_talker.generate_response.assert_called_once()
    
    @pytest.mark.not_heavy
    @patch('src.sovereign.services.model_service.ModelService._load_thinker_model')
    async def test_query_complex_uses_thinker(self, mock_load_thinker):
        """Test that complex queries route to Thinker model"""
        service = ModelService()
        
        # Mock Thinker model
        mock_thinker = AsyncMock()
        mock_thinker.auto_process.return_value = "I'm the Thinker model handling complex analysis."
        service._thinker_model = mock_thinker
        mock_load_thinker.return_value = True
        
        response = await service.query("Analyze the complex architecture patterns")
        
        assert "I'm the Thinker model handling complex analysis." == response
        mock_load_thinker.assert_called_once()
        mock_thinker.auto_process.assert_called_once()
    
    @pytest.mark.not_heavy
    @patch('src.sovereign.services.model_service.ModelService._load_talker_model')
    async def test_talker_fallback_to_thinker(self, mock_load_talker):
        """Test fallback to Thinker when Talker fails to load"""
        service = ModelService()
        mock_load_talker.return_value = False
        
        with patch.object(service, '_query_thinker', return_value="Fallback response") as mock_thinker_query:
            response = await service.query("Simple question")
            
            assert response == "Fallback response"
            mock_load_talker.assert_called_once()
            mock_thinker_query.assert_called_once()
    
    @pytest.mark.not_heavy
    async def test_unload_models(self):
        """Test unloading models to free memory"""
        service = ModelService()
        
        # Mock loaded models
        mock_talker = AsyncMock()
        mock_thinker = AsyncMock()
        service._talker_model = mock_talker
        service._thinker_model = mock_thinker
        service._model_status[ModelType.TALKER] = ModelStatus.READY
        service._model_status[ModelType.THINKER] = ModelStatus.READY
        
        await service.unload_models()
        
        # Check models are unloaded
        assert service._talker_model is None
        assert service._thinker_model is None
        assert service._model_status[ModelType.TALKER] == ModelStatus.UNLOADED
        assert service._model_status[ModelType.THINKER] == ModelStatus.UNLOADED
        
        # Check close methods were called
        mock_talker.close.assert_called_once()
        mock_thinker.close.assert_called_once()
    
    @pytest.mark.not_heavy
    async def test_close_service(self):
        """Test closing the service"""
        service = ModelService()
        
        with patch.object(service, 'unload_models') as mock_unload:
            await service.close()
            mock_unload.assert_called_once()


class TestModelServiceIntegrationHeavy:
    """Heavy integration tests that actually load models"""
    
    async def test_real_model_loading_and_query(self):
        """
        HEAVY TEST: Test actual model loading and querying
        
        This test loads real models and verifies they work correctly.
        It's not marked as @pytest.mark.not_heavy so it runs in full test suite.
        """
        service = ModelService()
        
        try:
            # Test simple query that should load Talker
            response1 = await service.query("Hello, how are you today?")
            
            # Verify Talker was loaded
            status_after_simple = service.get_status()
            assert status_after_simple["talker_queries"] == 1
            assert status_after_simple["total_queries"] == 1
            assert "talker" in status_after_simple["load_times"]
            
            # Test complex query that should load Thinker
            response2 = await service.query("Analyze the trade-offs between different sorting algorithms and recommend the best approach for large datasets")
            
            # Verify both models are now loaded
            status_after_complex = service.get_status()
            assert status_after_complex["total_queries"] == 2
            assert status_after_complex["thinker_queries"] >= 1  # May be 1 or 2 depending on handoff
            
            # Verify no re-loading occurred
            assert len(status_after_complex["load_times"]) == 2  # Both models loaded once
            
            # Verify responses are reasonable
            assert len(response1) > 0
            assert len(response2) > 0
            
            # Test that subsequent queries don't reload models
            load_times_before = status_after_complex["load_times"].copy()
            
            await service.query("Another simple question")
            await service.query("Another complex analysis task")
            
            final_status = service.get_status()
            # Load times should be identical (no reloading)
            assert final_status["load_times"] == load_times_before
            
            print(f"✅ Model loading test passed:")
            print(f"   - Talker loaded in {load_times_before['talker']:.2f}s")
            print(f"   - Thinker loaded in {load_times_before['thinker']:.2f}s")
            print(f"   - Total queries: {final_status['total_queries']}")
            print(f"   - No model reloading occurred")
            
        finally:
            # Clean up
            await service.close()
    
    async def test_concurrent_model_loading(self):
        """
        HEAVY TEST: Test that concurrent queries don't cause double-loading
        """
        service = ModelService()
        
        try:
            # Launch multiple queries simultaneously
            tasks = [
                service.query("Hello there"),
                service.query("Hi, how are you?"),
                service.query("Good morning"),
                service.query("What's up?")
            ]
            
            responses = await asyncio.gather(*tasks)
            
            # All responses should be valid
            for response in responses:
                assert len(response) > 0
            
            # Talker should only be loaded once despite concurrent queries
            status = service.get_status()
            assert len(status["load_times"]) == 1  # Only Talker loaded
            assert "talker" in status["load_times"]
            assert status["talker_queries"] == 4
            assert status["total_queries"] == 4
            
            print(f"✅ Concurrent loading test passed - no double-loading occurred")
            
        finally:
            await service.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 