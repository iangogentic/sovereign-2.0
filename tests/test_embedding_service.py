"""
Tests for EmbeddingService - Local Embedding Generation System

This module tests the embedding service including:
- Model initialization and loading
- Single embedding generation
- Batch embedding processing
- GPU/CPU handling
- Caching mechanisms
- Error handling and fallbacks
- Performance monitoring
"""

import pytest
import tempfile
import numpy as np
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.sovereign.embedding_service import (
    EmbeddingService, EmbeddingModelType, EmbeddingRequest, EmbeddingResponse,
    ModelConfig, ModelPerformanceStats, ProcessingMode
)


class TestEmbeddingModels:
    """Test embedding model types and information"""
    
    def test_model_types(self):
        """Test all model types are properly defined"""
        models = list(EmbeddingModelType)
        assert len(models) >= 3  # At least BGE, E5, and MiniLM
        
        # Test specific models exist
        assert EmbeddingModelType.BGE_BASE_EN_V15 in models
        assert EmbeddingModelType.E5_LARGE_V2 in models
        assert EmbeddingModelType.MINILM_L6_V2 in models
    
    def test_model_config_structure(self):
        """Test model config contains required fields"""
        from src.sovereign.embedding_service import EmbeddingService
        
        service = EmbeddingService()
        configs = service.get_available_models()
        
        for model_name, config in configs.items():
            assert isinstance(config, ModelConfig)
            assert config.model_name
            assert config.dimension > 0
            assert config.max_length > 0
            assert config.gpu_memory_mb > 0


class TestEmbeddingRequest:
    """Test embedding request data structure"""
    
    def test_basic_request(self):
        """Test basic request creation"""
        request = EmbeddingRequest(text="Hello world")
        assert request.text == "Hello world"
        assert request.model_type is None
        assert request.model_name is None
        assert request.metadata == {}
    
    def test_request_with_options(self):
        """Test request with all options"""
        request = EmbeddingRequest(
            text="Test text",
            model_name="custom-model"
        )
        
        assert request.text == "Test text"
        assert request.model_name == "custom-model"
        assert request.request_id is not None


class TestEmbeddingResponse:
    """Test embedding response data structure"""
    
    def test_successful_response(self):
        """Test successful response"""
        embedding = np.random.rand(768).astype(np.float32)
        response = EmbeddingResponse(
            request_id="test-123",
            embedding=embedding,
            model_used="test-model",
            processing_time=0.5,
            tokens_processed=10,
            success=True
        )
        
        assert response.success is True
        assert np.array_equal(response.embedding, embedding)
        assert response.model_used == "test-model"
        assert response.processing_time == 0.5
        assert response.tokens_processed == 10
        assert response.error_message is None
    
    def test_failed_response(self):
        """Test failed response"""
        response = EmbeddingResponse(
            request_id="test-456",
            embedding=None,
            model_used="",
            processing_time=0.0,
            tokens_processed=0,
            success=False,
            error_message="Test error"
        )
        
        assert response.success is False
        assert response.embedding is None
        assert response.error_message == "Test error"


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_transformer_model():
    """Mock transformer model"""
    mock_model = Mock()
    mock_model.encode.return_value = np.random.rand(5, 768).astype(np.float32)
    return mock_model


class TestEmbeddingService:
    """Test the main EmbeddingService class"""
    
    @patch('src.sovereign.embedding_service.TRANSFORMERS_AVAILABLE', True)
    @patch('src.sovereign.embedding_service.AutoModel')
    @patch('src.sovereign.embedding_service.AutoTokenizer')
    def test_service_initialization(self, mock_tokenizer, mock_model, temp_cache_dir):
        """Test service initialization"""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        service = EmbeddingService(
            default_model=EmbeddingModelType.BGE_BASE_EN_V15,
            cache_dir=temp_cache_dir,
            enable_gpu=False,
            max_batch_size=8
        )
        
        assert service._default_model == EmbeddingModelType.BGE_BASE_EN_V15
        assert service._cache_dir == Path(temp_cache_dir)
        assert service._enable_gpu is False
        assert service._max_batch_size == 8
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_service_initialization_failure(self, mock_transformer, temp_cache_dir):
        """Test service initialization with model loading failure"""
        mock_transformer.side_effect = Exception("Model loading failed")
        
        service = EmbeddingService(
            cache_dir=temp_cache_dir,
            enable_gpu=False
        )
        
        assert service.is_available is False
        assert len(service._models) == 0
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_single_embedding_generation(self, mock_transformer, temp_cache_dir):
        """Test generating a single embedding"""
        # Setup mock
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        service = EmbeddingService(cache_dir=temp_cache_dir, enable_gpu=False)
        
        # Generate embedding
        response = service.generate_embedding("Hello world")
        
        assert response.success is True
        assert response.embedding is not None
        assert response.embedding.shape == (768,)
        assert response.model_used is not None
        assert response.processing_time > 0
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_batch_embedding_generation(self, mock_transformer, temp_cache_dir):
        """Test generating multiple embeddings in batch"""
        # Setup mock
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(3, 768).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        service = EmbeddingService(cache_dir=temp_cache_dir, enable_gpu=False)
        
        # Create requests
        requests = [
            EmbeddingRequest(text="Text 1"),
            EmbeddingRequest(text="Text 2"),
            EmbeddingRequest(text="Text 3")
        ]
        
        # Generate embeddings
        responses = service.generate_embeddings(requests)
        
        assert len(responses) == 3
        for response in responses:
            assert response.success is True
            assert response.embedding is not None
            assert response.embedding.shape == (768,)
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_model_switching(self, mock_transformer, temp_cache_dir):
        """Test switching between different models"""
        mock_transformer.return_value = Mock()
        
        service = EmbeddingService(
            default_model=EmbeddingModelType.BGE_BASE_EN_V15,
            cache_dir=temp_cache_dir,
            enable_gpu=False
        )
        
        # Generate with default model
        response1 = service.generate_embedding("Test 1")
        assert response1.success is True
        
        # Generate with different model
        response2 = service.generate_embedding(
            "Test 2", 
            EmbeddingModelType.MINILM_L6_V2
        )
        assert response2.success is True
        
        # Should have loaded both models
        assert len(service._models) >= 1
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_caching_mechanism(self, mock_transformer, temp_cache_dir):
        """Test embedding caching"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        service = EmbeddingService(
            cache_dir=temp_cache_dir,
            enable_gpu=False,
            enable_cache=True
        )
        
        # Generate embedding twice
        text = "Cache test"
        response1 = service.generate_embedding(text)
        response2 = service.generate_embedding(text)
        
        assert response1.success is True
        assert response2.success is True
        
        # Second call should be faster (cached)
        assert response2.processing_time <= response1.processing_time
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_text_truncation(self, mock_transformer, temp_cache_dir):
        """Test text truncation for long inputs"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        service = EmbeddingService(cache_dir=temp_cache_dir, enable_gpu=False)
        
        # Create very long text
        long_text = "word " * 10000
        
        response = service.generate_embedding(long_text)
        
        assert response.success is True
        # Should have called encode with truncated text
        mock_model.encode.assert_called()
        
        call_args = mock_model.encode.call_args[0]
        assert len(call_args[0]) <= 8192  # Default max_seq_length for BGE
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_error_handling(self, mock_transformer, temp_cache_dir):
        """Test error handling during embedding generation"""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Model error")
        mock_transformer.return_value = mock_model
        
        service = EmbeddingService(cache_dir=temp_cache_dir, enable_gpu=False)
        
        response = service.generate_embedding("Test text")
        
        assert response.success is False
        assert response.embedding is None
        assert "Model error" in response.error_message
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_batch_size_limiting(self, mock_transformer, temp_cache_dir):
        """Test batch size limiting for large batches"""
        mock_model = Mock()
        # Return embeddings for each batch
        mock_model.encode.side_effect = [
            np.random.rand(2, 768).astype(np.float32),  # First batch
            np.random.rand(1, 768).astype(np.float32)   # Second batch
        ]
        mock_transformer.return_value = mock_model
        
        service = EmbeddingService(
            cache_dir=temp_cache_dir,
            enable_gpu=False,
            max_batch_size=2  # Small batch size
        )
        
        # Create requests that exceed batch size
        requests = [
            EmbeddingRequest(text=f"Text {i}") 
            for i in range(3)
        ]
        
        responses = service.generate_embeddings(requests)
        
        assert len(responses) == 3
        for response in responses:
            assert response.success is True
        
        # Should have been called twice (2 batches)
        assert mock_model.encode.call_count == 2
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_model_information(self, mock_transformer, temp_cache_dir):
        """Test retrieving model information"""
        mock_transformer.return_value = Mock()
        
        service = EmbeddingService(cache_dir=temp_cache_dir, enable_gpu=False)
        
        info = service.get_model_info(EmbeddingModelType.BGE_BASE_EN_V15)
        
        assert isinstance(info, EmbeddingModelInfo)
        assert info.model_name
        assert info.embedding_dim > 0
        assert info.max_seq_length > 0
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_service_statistics(self, mock_transformer, temp_cache_dir):
        """Test service statistics tracking"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        service = EmbeddingService(cache_dir=temp_cache_dir, enable_gpu=False)
        
        # Generate some embeddings
        service.generate_embedding("Test 1")
        service.generate_embedding("Test 2")
        
        stats = service.get_statistics()
        
        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "failed_requests" in stats
        assert "total_processing_time" in stats
        assert "average_processing_time" in stats
        assert stats["total_requests"] >= 2
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_service_close(self, mock_transformer, temp_cache_dir):
        """Test service cleanup"""
        mock_transformer.return_value = Mock()
        
        service = EmbeddingService(cache_dir=temp_cache_dir, enable_gpu=False)
        
        # Generate embedding to load model
        service.generate_embedding("Test")
        
        # Close service
        service.close()
        
        # Should clear models and executor
        assert len(service._models) == 0
        assert service._executor is None
    
    def test_service_unavailable_fallback(self, temp_cache_dir):
        """Test behavior when service is unavailable"""
        # Create service without mocking SentenceTransformer (will fail to import)
        with patch('src.sovereign.embedding_service.SentenceTransformer', side_effect=ImportError):
            service = EmbeddingService(cache_dir=temp_cache_dir)
            
            assert service.is_available is False
            
            # Should return error response
            response = service.generate_embedding("Test")
            assert response.success is False
            assert "not available" in response.error_message.lower()


class TestEmbeddingIntegration:
    """Integration tests for embedding service"""
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_end_to_end_workflow(self, mock_transformer, temp_cache_dir):
        """Test complete embedding workflow"""
        # Setup mock with realistic behavior
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        service = EmbeddingService(
            default_model=EmbeddingModelType.BGE_BASE_EN_V15,
            cache_dir=temp_cache_dir,
            enable_gpu=False,
            max_batch_size=4
        )
        
        # Test various scenarios
        texts = [
            "Short text",
            "This is a longer text that contains more information and detail.",
            "Another example text for testing purposes.",
            ""  # Empty text edge case
        ]
        
        # Test individual generation
        for text in texts:
            response = service.generate_embedding(text)
            if text:  # Non-empty text
                assert response.success is True
                assert response.embedding is not None
            # Empty text handling depends on implementation
        
        # Test batch generation
        requests = [EmbeddingRequest(text=text) for text in texts if text]
        responses = service.generate_embeddings(requests)
        
        assert len(responses) == len([t for t in texts if t])
        for response in responses:
            assert response.success is True
            assert response.embedding is not None
        
        # Test statistics
        stats = service.get_statistics()
        assert stats["total_requests"] > 0
        
        # Test cleanup
        service.close()
        assert len(service._models) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 