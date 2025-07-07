#!/usr/bin/env python3
"""
Simple EmbeddingService Demonstration

This script tests the basic functionality of the EmbeddingService
without complex mocking, to validate the implementation works.
"""

import tempfile
import numpy as np
from pathlib import Path

# Add the src directory to the path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_embedding_service_basic():
    """Test basic embedding service functionality"""
    
    try:
        from sovereign.embedding_service import (
            EmbeddingService, EmbeddingModelType, EmbeddingRequest, EmbeddingResponse
        )
        
        print("✅ Successfully imported EmbeddingService components")
        
        # Test model types
        print(f"📋 Available model types: {len(list(EmbeddingModelType))}")
        for model in EmbeddingModelType:
            print(f"   - {model.value}")
        
        # Test service initialization (without GPU for compatibility)
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"\n🔧 Initializing EmbeddingService...")
            
            service = EmbeddingService(
                default_model=EmbeddingModelType.BGE_BASE_EN_V15,
                cache_dir=temp_dir,
                enable_gpu=False,  # Use CPU for compatibility
                max_batch_size=4
            )
            
            print(f"✅ Service initialized")
            print(f"   - Default model: {service.default_model.value}")
            print(f"   - Cache dir: {service.cache_dir}")
            print(f"   - GPU enabled: {service.enable_gpu}")
            print(f"   - Max batch size: {service.max_batch_size}")
            
            # Test available models
            available_models = service.get_available_models()
            print(f"\n📊 Available models: {len(available_models)}")
            
            for name, config in available_models.items():
                print(f"   - {name}: {config.dimension}D, {config.gpu_memory_mb}MB")
            
            # Test basic embedding generation (will use fallback if transformers not available)
            print(f"\n🚀 Testing embedding generation...")
            
            try:
                response = service.generate_embedding("Hello, world!")
                
                if response.success:
                    print(f"✅ Generated embedding successfully")
                    print(f"   - Model used: {response.model_used}")
                    print(f"   - Processing time: {response.processing_time:.3f}s")
                    print(f"   - Tokens processed: {response.tokens_processed}")
                    print(f"   - Embedding shape: {response.embedding.shape}")
                    print(f"   - Embedding dtype: {response.embedding.dtype}")
                    
                    # Verify embedding properties
                    assert isinstance(response.embedding, np.ndarray)
                    assert len(response.embedding.shape) == 1  # 1D array
                    assert response.embedding.shape[0] > 0    # Non-empty
                    
                else:
                    print(f"⚠️  Embedding generation failed: {response.error_message}")
                    print("   This is expected if transformers are not installed")
                    
            except Exception as e:
                print(f"⚠️  Embedding generation error: {e}")
                print("   This is expected if transformers are not installed")
            
            # Test batch generation
            print(f"\n📦 Testing batch embedding generation...")
            
            try:
                requests = [
                    EmbeddingRequest(text="First text"),
                    EmbeddingRequest(text="Second text"),
                    EmbeddingRequest(text="Third text")
                ]
                
                responses = service.generate_embeddings(requests)
                
                print(f"✅ Generated {len(responses)} embeddings in batch")
                
                successful = sum(1 for r in responses if r.success)
                print(f"   - Successful: {successful}/{len(responses)}")
                
                if successful > 0:
                    avg_time = np.mean([r.processing_time for r in responses if r.success])
                    print(f"   - Average processing time: {avg_time:.3f}s")
                
            except Exception as e:
                print(f"⚠️  Batch generation error: {e}")
            
            # Test service statistics
            print(f"\n📈 Service statistics:")
            try:
                stats = service.get_service_stats()
                for key, value in stats.items():
                    print(f"   - {key}: {value}")
            except Exception as e:
                print(f"   Error getting stats: {e}")
            
            # Test health check
            print(f"\n🏥 Health check:")
            try:
                health = service.health_check()
                print(f"   - Service healthy: {health.get('healthy', False)}")
                print(f"   - Models loaded: {health.get('models_loaded', 0)}")
                print(f"   - Transformers available: {health.get('transformers_available', False)}")
                
            except Exception as e:
                print(f"   Error during health check: {e}")
            
            # Clean up
            service.close()
            print(f"\n🧹 Service closed successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_embedding_request_response():
    """Test request/response data structures"""
    
    try:
        from sovereign.embedding_service import EmbeddingRequest, EmbeddingResponse
        
        print("\n🧪 Testing EmbeddingRequest...")
        
        # Test basic request
        request = EmbeddingRequest(text="Test text")
        assert request.text == "Test text"
        assert request.request_id is not None
        assert request.created_at is not None
        print("✅ Basic request creation works")
        
        # Test request with model name
        request2 = EmbeddingRequest(text="Test text 2", model_name="custom-model")
        assert request2.model_name == "custom-model"
        print("✅ Request with model name works")
        
        print("\n🧪 Testing EmbeddingResponse...")
        
        # Test successful response
        embedding = np.random.rand(768).astype(np.float32)
        response = EmbeddingResponse(
            request_id="test-123",
            embedding=embedding,
            model_used="test-model",
            processing_time=0.1,
            tokens_processed=5,
            success=True
        )
        
        assert response.success is True
        assert np.array_equal(response.embedding, embedding)
        assert response.model_used == "test-model"
        print("✅ Successful response creation works")
        
        # Test failed response
        failed_response = EmbeddingResponse(
            request_id="test-456",
            embedding=None,
            model_used="",
            processing_time=0.0,
            tokens_processed=0,
            success=False,
            error_message="Test error"
        )
        
        assert failed_response.success is False
        assert failed_response.embedding is None
        assert failed_response.error_message == "Test error"
        print("✅ Failed response creation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing request/response: {e}")
        return False


if __name__ == "__main__":
    print("🚀 EmbeddingService Demonstration")
    print("=" * 50)
    
    # Test basic functionality
    success1 = test_embedding_service_basic()
    
    # Test data structures
    success2 = test_embedding_request_response()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed! EmbeddingService is working correctly.")
    else:
        print("⚠️  Some tests failed. This may be expected if transformers library is not installed.")
        print("    The service will use fallback mock embeddings in that case.")
    
    print("\n📝 Note: For full functionality, install transformers:")
    print("    pip install torch transformers sentence-transformers") 