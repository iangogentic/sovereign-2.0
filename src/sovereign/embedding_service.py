"""
Embedding Service for Sovereign AI Long-Term Memory System

This module provides a comprehensive embedding generation service supporting
multiple state-of-the-art local embedding models. Features include:

- Multiple model support (E5-large-v2, BGE-base-en-v1.5, etc.)
- GPU acceleration with CUDA support  
- CPU fallback for compatibility
- Batch processing for efficiency
- Caching and performance optimization
- Error handling and recovery
- Model lifecycle management
- Memory usage monitoring

Supported Models:
- intfloat/e5-large-v2 (recommended, ~1.2GB VRAM)
- BAAI/bge-base-en-v1.5 (lightweight, ~400MB VRAM)
- sentence-transformers/all-MiniLM-L6-v2 (ultra-light, <300MB VRAM)
- intfloat/multilingual-e5-large-instruct (multilingual support)
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
import gc

# MOVED TORCH IMPORTS TO _load_model() TO FIX STARTUP LAG
# Heavy imports are now deferred until first model loading
try:
    # Just check availability without importing heavy modules
    import importlib.util
    torch_available = importlib.util.find_spec("torch") is not None
    transformers_available = importlib.util.find_spec("transformers") is not None
    TRANSFORMERS_AVAILABLE = torch_available and transformers_available
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Get logger
logger = logging.getLogger(__name__)

class EmbeddingModelType(Enum):
    """Supported embedding model types"""
    E5_LARGE_V2 = "intfloat/e5-large-v2"
    BGE_BASE_EN_V15 = "BAAI/bge-base-en-v1.5"
    MINI_LM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"
    MULTILINGUAL_E5_LARGE = "intfloat/multilingual-e5-large-instruct"
    E5_BASE_V2 = "intfloat/e5-base-v2"
    
class ProcessingMode(Enum):
    """Processing modes for embeddings"""
    SINGLE = "single"
    BATCH = "batch"
    STREAMING = "streaming"

@dataclass
class ModelConfig:
    """Configuration for an embedding model"""
    model_name: str
    model_id: str
    dimension: int
    max_length: int
    gpu_memory_mb: int
    supports_batching: bool = True
    supports_multilingual: bool = False
    description: str = ""

@dataclass
class EmbeddingRequest:
    """Request for embedding generation"""
    text: str
    model_name: str = None
    request_id: str = None
    priority: int = 1
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.request_id is None:
            self.request_id = hashlib.md5(f"{self.text}_{self.created_at}".encode()).hexdigest()[:12]

@dataclass
class EmbeddingResponse:
    """Response from embedding generation"""
    request_id: str
    embedding: np.ndarray
    model_used: str
    processing_time: float
    tokens_processed: int
    success: bool = True
    error_message: str = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ModelPerformanceStats:
    """Performance statistics for a model"""
    total_requests: int = 0
    total_tokens: int = 0
    total_processing_time: float = 0.0
    average_tokens_per_second: float = 0.0
    gpu_memory_used_mb: float = 0.0
    last_used: datetime = None
    error_count: int = 0
    
    def update_stats(self, tokens: int, processing_time: float, gpu_memory_mb: float = 0):
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_processing_time += processing_time
        if self.total_processing_time > 0:
            self.average_tokens_per_second = self.total_tokens / self.total_processing_time
        self.gpu_memory_used_mb = gpu_memory_mb
        self.last_used = datetime.now()

class EmbeddingService:
    """
    High-performance embedding service supporting multiple local models.
    
    Provides GPU-accelerated embedding generation with automatic fallback,
    batching, caching, and comprehensive error handling.
    """
    
    # Default model configurations
    DEFAULT_MODELS = {
        EmbeddingModelType.E5_LARGE_V2: ModelConfig(
            model_name="E5-Large-v2",
            model_id="intfloat/e5-large-v2", 
            dimension=1024,
            max_length=512,
            gpu_memory_mb=1200,
            supports_batching=True,
            supports_multilingual=False,
            description="High-performance general-purpose embedding model"
        ),
        EmbeddingModelType.BGE_BASE_EN_V15: ModelConfig(
            model_name="BGE-Base-EN-v1.5",
            model_id="BAAI/bge-base-en-v1.5",
            dimension=768,
            max_length=512,
            gpu_memory_mb=400,
            supports_batching=True,
            supports_multilingual=False,
            description="Lightweight high-quality embedding model"
        ),
        EmbeddingModelType.MINI_LM_L6_V2: ModelConfig(
            model_name="MiniLM-L6-v2",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            max_length=256,
            gpu_memory_mb=300,
            supports_batching=True,
            supports_multilingual=False,
            description="Ultra-lightweight fast embedding model"
        ),
        EmbeddingModelType.MULTILINGUAL_E5_LARGE: ModelConfig(
            model_name="Multilingual-E5-Large",
            model_id="intfloat/multilingual-e5-large-instruct",
            dimension=1024,
            max_length=512,
            gpu_memory_mb=1400,
            supports_batching=True,
            supports_multilingual=True,
            description="High-quality multilingual embedding model"
        ),
        EmbeddingModelType.E5_BASE_V2: ModelConfig(
            model_name="E5-Base-v2",
            model_id="intfloat/e5-base-v2",
            dimension=768,
            max_length=512,
            gpu_memory_mb=600,
            supports_batching=True,
            supports_multilingual=False,
            description="Balanced performance and efficiency"
        )
    }
    
    def __init__(self, 
                 default_model: EmbeddingModelType = EmbeddingModelType.BGE_BASE_EN_V15,
                 cache_dir: Optional[str] = None,
                 enable_gpu: bool = True,
                 max_batch_size: int = 32,
                 max_workers: int = 2):
        """
        Initialize the Embedding Service.
        
        Args:
            default_model: Primary embedding model to use
            cache_dir: Directory for caching models and embeddings
            enable_gpu: Whether to use GPU acceleration
            max_batch_size: Maximum batch size for processing
            max_workers: Maximum worker threads
        """
        # Check transformers availability
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for embedding service. "
                "Install with: pip install torch transformers sentence-transformers"
            )
        
        self.default_model = default_model
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".sovereign" / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Device configuration (defer torch import)
        self.enable_gpu = enable_gpu
        self.device = None  # Will be set when first model loads
        
        # Processing configuration
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        
        # Model management
        self._loaded_models: Dict[str, Dict[str, Any]] = {}
        self._model_stats: Dict[str, ModelPerformanceStats] = {}
        self._lock = threading.RLock()
        
        # Threading
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Caching
        self._embedding_cache: Dict[str, Tuple[np.ndarray, datetime]] = {}
        self._cache_ttl = timedelta(hours=24)  # 24-hour cache TTL
        
        # Performance tracking
        self._total_requests = 0
        self._total_errors = 0
        self._start_time = time.time()
        
        logger.info(f"EmbeddingService initialized (device will be set on first model load)")
        logger.info(f"Default model: {self.default_model.value}")
        
        # Don't load default model immediately to keep startup fast
        # Model will be loaded on first use
    
    def _load_model(self, model_type: EmbeddingModelType) -> bool:
        """
        Load a specific embedding model.
        
        Args:
            model_type: Type of model to load
            
        Returns:
            bool: True if successfully loaded
        """
        model_config = self.DEFAULT_MODELS[model_type]
        model_id = model_config.model_id
        
        with self._lock:
            # Check if already loaded
            if model_id in self._loaded_models:
                logger.debug(f"Model {model_id} already loaded")
                return True
            
            try:
                # Import torch and transformers when first needed (heavy imports)
                import torch
                import torch.nn.functional as F
                from transformers import AutoTokenizer, AutoModel
                
                # Set up device if not already done
                if self.device is None:
                    self.enable_gpu = self.enable_gpu and torch.cuda.is_available()
                    self.device = torch.device("cuda" if self.enable_gpu else "cpu")
                    logger.info(f"Device configured: {self.device}")
                
                logger.info(f"Loading embedding model: {model_id}")
                start_time = time.time()
                
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True
                )
                
                model = AutoModel.from_pretrained(
                    model_id,
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.enable_gpu else torch.float32
                )
                
                # Move to device
                model = model.to(self.device)
                model.eval()
                
                # Store loaded model
                self._loaded_models[model_id] = {
                    "tokenizer": tokenizer,
                    "model": model,
                    "config": model_config,
                    "loaded_at": datetime.now()
                }
                
                # Initialize stats
                self._model_stats[model_id] = ModelPerformanceStats()
                
                load_time = time.time() - start_time
                logger.info(f"Model {model_id} loaded successfully in {load_time:.2f}s")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                return False
    
    def _unload_model(self, model_type: EmbeddingModelType) -> bool:
        """
        Unload a specific embedding model to free memory.
        
        Args:
            model_type: Type of model to unload
            
        Returns:
            bool: True if successfully unloaded
        """
        model_config = self.DEFAULT_MODELS[model_type]
        model_id = model_config.model_id
        
        with self._lock:
            if model_id not in self._loaded_models:
                return True
            
            try:
                # Clean up model
                model_info = self._loaded_models[model_id]
                model = model_info["model"]
                
                # Move to CPU and delete
                model.cpu()
                del model
                del model_info["model"]
                del model_info["tokenizer"]
                
                # Remove from loaded models
                del self._loaded_models[model_id]
                
                # Force garbage collection
                if self.enable_gpu:
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except ImportError:
                        pass  # torch not available
                gc.collect()
                
                logger.info(f"Model {model_id} unloaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload model {model_id}: {e}")
                return False
    
    def _get_cache_key(self, text: str, model_id: str) -> str:
        """Generate cache key for text and model combination"""
        return hashlib.md5(f"{text}_{model_id}".encode()).hexdigest()
    
    def _check_cache(self, text: str, model_id: str) -> Optional[np.ndarray]:
        """Check if embedding exists in cache"""
        cache_key = self._get_cache_key(text, model_id)
        
        if cache_key in self._embedding_cache:
            embedding, cached_at = self._embedding_cache[cache_key]
            
            # Check if cache is still valid
            if datetime.now() - cached_at < self._cache_ttl:
                return embedding
            else:
                # Remove expired cache entry
                del self._embedding_cache[cache_key]
        
        return None
    
    def _store_cache(self, text: str, model_id: str, embedding: np.ndarray):
        """Store embedding in cache"""
        cache_key = self._get_cache_key(text, model_id)
        self._embedding_cache[cache_key] = (embedding, datetime.now())
        
        # Cleanup old cache entries if cache gets too large
        if len(self._embedding_cache) > 10000:
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = datetime.now()
        expired_keys = [
            key for key, (_, cached_at) in self._embedding_cache.items()
            if current_time - cached_at >= self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._embedding_cache[key]
        
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _average_pool(self, last_hidden_states, attention_mask):
        """Apply average pooling to get sentence embeddings"""
        # Import torch when needed
        import torch
        
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def _encode_texts(self, texts: List[str], model_type: EmbeddingModelType) -> np.ndarray:
        """
        Encode multiple texts into embeddings.
        
        Args:
            texts: List of texts to encode
            model_type: Model to use for encoding
            
        Returns:
            numpy array: Embeddings for all texts
        """
        # Import torch when needed
        import torch
        import torch.nn.functional as F
        
        model_config = self.DEFAULT_MODELS[model_type]
        model_id = model_config.model_id
        
        # Ensure model is loaded
        if model_id not in self._loaded_models:
            if not self._load_model(model_type):
                raise RuntimeError(f"Failed to load model {model_id}")
        
        model_info = self._loaded_models[model_id]
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
        
        try:
            # Tokenize texts
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=model_config.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(**encoded)
                
                # Use pooling to get sentence embeddings
                embeddings = self._average_pool(
                    outputs.last_hidden_state,
                    encoded["attention_mask"]
                )
                
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                # Convert to numpy
                embeddings_np = embeddings.cpu().numpy().astype(np.float32)
                
                return embeddings_np
                
        except Exception as e:
            logger.error(f"Failed to encode texts with model {model_id}: {e}")
            raise
    
    def generate_embedding(self, 
                          text: str, 
                          model_type: Optional[EmbeddingModelType] = None) -> EmbeddingResponse:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            model_type: Model to use (defaults to default_model)
            
        Returns:
            EmbeddingResponse: Response with embedding and metadata
        """
        request = EmbeddingRequest(text=text, model_name=model_type.value if model_type else None)
        return self.generate_embeddings([request])[0]
    
    def generate_embeddings(self, 
                           requests: List[EmbeddingRequest]) -> List[EmbeddingResponse]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            requests: List of embedding requests
            
        Returns:
            List of EmbeddingResponse objects
        """
        if not requests:
            return []
        
        self._total_requests += len(requests)
        start_time = time.time()
        
        try:
            # Group requests by model
            model_groups: Dict[EmbeddingModelType, List[EmbeddingRequest]] = {}
            
            for request in requests:
                # Determine model to use
                if request.model_name:
                    try:
                        model_type = EmbeddingModelType(request.model_name)
                    except ValueError:
                        model_type = self.default_model
                else:
                    model_type = self.default_model
                
                if model_type not in model_groups:
                    model_groups[model_type] = []
                model_groups[model_type].append(request)
            
            # Process each model group
            all_responses = []
            
            for model_type, group_requests in model_groups.items():
                model_config = self.DEFAULT_MODELS[model_type]
                model_id = model_config.model_id
                
                # Check cache first
                cached_responses = []
                uncached_requests = []
                
                for request in group_requests:
                    cached_embedding = self._check_cache(request.text, model_id)
                    if cached_embedding is not None:
                        response = EmbeddingResponse(
                            request_id=request.request_id,
                            embedding=cached_embedding,
                            model_used=model_id,
                            processing_time=0.001,  # Minimal cache lookup time
                            tokens_processed=len(request.text.split()),
                            success=True
                        )
                        cached_responses.append(response)
                    else:
                        uncached_requests.append(request)
                
                all_responses.extend(cached_responses)
                
                # Process uncached requests in batches
                if uncached_requests:
                    batch_responses = self._process_batch(uncached_requests, model_type)
                    all_responses.extend(batch_responses)
                    
                    # Cache new embeddings
                    for request, response in zip(uncached_requests, batch_responses):
                        if response.success:
                            self._store_cache(request.text, model_id, response.embedding)
            
            # Sort responses by original request order
            response_map = {resp.request_id: resp for resp in all_responses}
            ordered_responses = [response_map[req.request_id] for req in requests]
            
            total_time = time.time() - start_time
            logger.debug(f"Processed {len(requests)} embedding requests in {total_time:.3f}s")
            
            return ordered_responses
            
        except Exception as e:
            self._total_errors += len(requests)
            logger.error(f"Failed to generate embeddings: {e}")
            
            # Return error responses
            return [
                EmbeddingResponse(
                    request_id=req.request_id,
                    embedding=np.zeros(768, dtype=np.float32),  # Default dimension
                    model_used="error",
                    processing_time=0.0,
                    tokens_processed=0,
                    success=False,
                    error_message=str(e)
                ) for req in requests
            ]
    
    def _process_batch(self, 
                      requests: List[EmbeddingRequest], 
                      model_type: EmbeddingModelType) -> List[EmbeddingResponse]:
        """
        Process a batch of requests with a specific model.
        
        Args:
            requests: Requests to process
            model_type: Model to use
            
        Returns:
            List of responses
        """
        model_config = self.DEFAULT_MODELS[model_type]
        model_id = model_config.model_id
        
        # Split into smaller batches if needed
        batch_size = min(self.max_batch_size, len(requests))
        responses = []
        
        for i in range(0, len(requests), batch_size):
            batch_requests = requests[i:i + batch_size]
            batch_texts = [req.text for req in batch_requests]
            
            batch_start = time.time()
            
            try:
                # Generate embeddings for batch
                embeddings = self._encode_texts(batch_texts, model_type)
                batch_time = time.time() - batch_start
                
                # Create responses
                for j, (request, embedding) in enumerate(zip(batch_requests, embeddings)):
                    tokens_processed = len(request.text.split())
                    
                    response = EmbeddingResponse(
                        request_id=request.request_id,
                        embedding=embedding,
                        model_used=model_id,
                        processing_time=batch_time / len(batch_requests),
                        tokens_processed=tokens_processed,
                        success=True
                    )
                    responses.append(response)
                
                # Update model stats
                total_tokens = sum(len(req.text.split()) for req in batch_requests)
                gpu_memory = 0
                if self.enable_gpu:
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                
                self._model_stats[model_id].update_stats(total_tokens, batch_time, gpu_memory)
                
                logger.debug(f"Processed batch of {len(batch_requests)} texts in {batch_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Failed to process batch with model {model_id}: {e}")
                
                # Create error responses
                for request in batch_requests:
                    response = EmbeddingResponse(
                        request_id=request.request_id,
                        embedding=np.zeros(model_config.dimension, dtype=np.float32),
                        model_used=model_id,
                        processing_time=0.0,
                        tokens_processed=0,
                        success=False,
                        error_message=str(e)
                    )
                    responses.append(response)
        
        return responses
    
    def get_available_models(self) -> Dict[str, ModelConfig]:
        """Get information about all available models"""
        return {model_type.value: config for model_type, config in self.DEFAULT_MODELS.items()}
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        with self._lock:
            return list(self._loaded_models.keys())
    
    def get_model_stats(self, model_type: Optional[EmbeddingModelType] = None) -> Dict[str, Any]:
        """
        Get performance statistics for models.
        
        Args:
            model_type: Specific model to get stats for (None for all)
            
        Returns:
            Dictionary with model statistics
        """
        if model_type:
            model_id = self.DEFAULT_MODELS[model_type].model_id
            if model_id in self._model_stats:
                return {model_id: asdict(self._model_stats[model_id])}
            else:
                return {}
        else:
            return {model_id: asdict(stats) for model_id, stats in self._model_stats.items()}
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get overall service statistics"""
        uptime = time.time() - self._start_time
        
        return {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "error_rate": self._total_errors / max(1, self._total_requests),
            "uptime_seconds": uptime,
            "cache_size": len(self._embedding_cache),
            "loaded_models": len(self._loaded_models),
            "device": str(self.device),
            "gpu_available": torch.cuda.is_available() if torch else False,
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch and torch.cuda.is_available() else 0,
            "gpu_memory_used": torch.cuda.memory_allocated() / (1024**3) if torch and torch.cuda.is_available() else 0
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the embedding service"""
        health = {
            "status": "healthy",
            "issues": [],
            "device": str(self.device),
            "models_loaded": len(self._loaded_models),
            "cache_size": len(self._embedding_cache)
        }
        
        try:
            # Test default model
            test_response = self.generate_embedding("test embedding health check")
            if not test_response.success:
                health["status"] = "unhealthy"
                health["issues"].append(f"Default model test failed: {test_response.error_message}")
        except Exception as e:
            health["status"] = "unhealthy"
            health["issues"].append(f"Health check failed: {e}")
        
        return health
    
    def clear_cache(self):
        """Clear embedding cache"""
        with self._lock:
            self._embedding_cache.clear()
            logger.info("Embedding cache cleared")
    
    def close(self):
        """Clean up resources"""
        logger.info("Shutting down EmbeddingService...")
        
        # Unload all models
        model_types = list(self.DEFAULT_MODELS.keys())
        for model_type in model_types:
            self._unload_model(model_type)
        
        # Clear cache
        self.clear_cache()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        # Final GPU cleanup
        if self.enable_gpu and torch:
            torch.cuda.empty_cache()
        
        logger.info("EmbeddingService shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Convenience functions for easy integration
def create_embedding_service(model_type: EmbeddingModelType = EmbeddingModelType.BGE_BASE_EN_V15,
                            enable_gpu: bool = True) -> EmbeddingService:
    """Create a new embedding service with default configuration"""
    return EmbeddingService(
        default_model=model_type,
        enable_gpu=enable_gpu,
        max_batch_size=32,
        max_workers=2
    )

def get_embedding(text: str, 
                 model_type: EmbeddingModelType = EmbeddingModelType.BGE_BASE_EN_V15,
                 enable_gpu: bool = True) -> np.ndarray:
    """Quick function to get embedding for a single text"""
    with create_embedding_service(model_type, enable_gpu) as service:
        response = service.generate_embedding(text, model_type)
        if response.success:
            return response.embedding
        else:
            raise RuntimeError(f"Failed to generate embedding: {response.error_message}") 