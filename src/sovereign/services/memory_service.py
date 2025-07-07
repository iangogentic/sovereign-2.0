"""
Memory Service - Lazy-loaded RAG (Retrieval Augmented Generation) Management

Provides on-demand loading of FAISS vector database, SQLite storage, and MemoryManager
to ensure fast application startup. All heavy components are loaded only when first needed.
"""

import asyncio
import logging
import time
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pathlib import Path

# Only import lightweight modules at module level
# Heavy imports (FAISS, SQLite, MemoryManager) are deferred to loading functions

logger = logging.getLogger(__name__)


class MemoryStatus(Enum):
    """Memory service loading status"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class SearchResult:
    """Lightweight search result container"""
    def __init__(self, chunk_id: str, content: str, score: float, 
                 distance: float = 0.0, metadata: Optional[Dict[str, Any]] = None):
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.distance = distance
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'score': self.score,
            'distance': self.distance,
            'metadata': self.metadata
        }


class MemoryService:
    """
    Lazy-loaded memory service that manages RAG functionality
    
    Provides semantic search, memory storage, and retrieval capabilities
    while ensuring components are only loaded when first needed.
    """
    
    def __init__(self, config_path: Optional[str] = None, db_path: Optional[str] = None):
        """Initialize the memory service with no components loaded"""
        self._config_path = config_path
        self._db_path = db_path or "data/memory.db"
        
        # Core components (loaded lazily)
        self._memory_manager: Optional[Any] = None
        self._vector_search_engine: Optional[Any] = None
        self._embedding_service: Optional[Any] = None
        self._config: Optional[Any] = None
        
        # Service status
        self._status = MemoryStatus.UNLOADED
        
        # Thread safety
        self._load_lock = asyncio.Lock()
        
        # Performance tracking
        self._load_time: Optional[float] = None
        self._query_count = 0
        self._total_query_time = 0.0
        
        logger.info("MemoryService initialized - components will be loaded on demand")
    
    async def _load_components(self) -> bool:
        """
        Lazy load all memory components
        
        Returns:
            True if successful, False otherwise
        """
        async with self._load_lock:
            # Check if already loaded
            if self._status == MemoryStatus.READY:
                return True
            
            # Check if currently loading
            if self._status == MemoryStatus.LOADING:
                # Wait for loading to complete
                while self._status == MemoryStatus.LOADING:
                    await asyncio.sleep(0.1)
                return self._status == MemoryStatus.READY
            
            # Start loading
            self._status = MemoryStatus.LOADING
            start_time = time.time()
            
            try:
                logger.info("🔄 Loading memory components...")
                
                # Import heavy modules only when needed
                from ..config import Config
                from ..memory_manager import MemoryManager
                from ..vector_search_engine import VectorSearchEngine
                from ..embedding_service import EmbeddingService, EmbeddingModelType
                
                # Load configuration
                if self._config_path:
                    self._config = Config.from_file(self._config_path)
                else:
                    self._config = Config()
                
                # Ensure data directory exists
                db_path = Path(self._db_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Initialize embedding service
                logger.info("🔄 Initializing embedding service...")
                self._embedding_service = EmbeddingService(
                    default_model=EmbeddingModelType.BGE_BASE_EN_V15,
                    cache_dir=str(db_path.parent / "embeddings"),
                    enable_gpu=True,
                    max_batch_size=8
                )
                
                # Initialize memory manager
                logger.info("🔄 Initializing memory manager...")
                self._memory_manager = MemoryManager(self._config)
                
                # Initialize vector search engine
                logger.info("🔄 Initializing vector search engine...")
                self._vector_search_engine = VectorSearchEngine(
                    db_path=str(db_path),
                    embedding_service=self._embedding_service
                )
                
                # Initialize vector search engine
                await self._vector_search_engine.initialize()
                
                self._status = MemoryStatus.READY
                load_time = time.time() - start_time
                self._load_time = load_time
                
                logger.info(f"✅ Memory components loaded successfully in {load_time:.2f}s")
                return True
                
            except Exception as e:
                self._status = MemoryStatus.ERROR
                logger.error(f"❌ Error loading memory components: {e}")
                return False
    
    async def retrieve(self, query: str, top_k: int = 5, 
                      threshold: float = 0.7) -> List[SearchResult]:
        """
        Retrieve relevant memories using semantic search
        
        Args:
            query: Search query string
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        # Load components if needed
        if not await self._load_components():
            logger.error("Failed to load memory components")
            return []
        
        start_time = time.time()
        self._query_count += 1
        
        try:
            # Import search parameters only when needed
            from ..vector_search_engine import SearchParameters
            
            # Create search parameters
            search_params = SearchParameters(
                top_k=top_k,
                distance_threshold=threshold,
                min_confidence=threshold,
                include_metadata=True
            )
            
            # Perform vector search
            results = await self._vector_search_engine.search(query, search_params)
            
            # Convert to our lightweight format
            search_results = []
            for result in results:
                search_result = SearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=result.score,
                    distance=result.distance,
                    metadata=result.metadata
                )
                search_results.append(search_result)
            
            # Update performance stats
            query_time = time.time() - start_time
            self._total_query_time += query_time
            
            logger.debug(f"Memory retrieval completed in {query_time:.3f}s, found {len(search_results)} results")
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error during memory retrieval: {e}")
            return []
    
    async def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store content in memory with optional metadata
        
        Args:
            content: Content to store
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        # Load components if needed
        if not await self._load_components():
            logger.error("Failed to load memory components")
            return False
        
        try:
            # For now, delegate to memory manager's conversation storage
            # This is a simplified interface - full implementation would depend on use case
            conversation_id = await self._memory_manager.start_conversation(
                title=f"Stored memory: {content[:50]}...",
                description=f"Auto-generated from memory store operation"
            )
            
            if conversation_id:
                # Store as a conversation message
                success = await self._memory_manager.store_conversation_turn(
                    conversation_id=conversation_id,
                    user_message=content,
                    assistant_message="",
                    metadata=metadata
                )
                
                if success:
                    logger.info(f"Successfully stored content in memory (conversation {conversation_id})")
                    return True
            
            logger.error("Failed to store content in memory")
            return False
            
        except Exception as e:
            logger.error(f"Error storing content: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get memory service statistics
        
        Returns:
            Dictionary with performance and status information
        """
        stats = {
            "status": self._status.value,
            "load_time": self._load_time,
            "query_count": self._query_count,
            "total_query_time": self._total_query_time,
            "average_query_time": self._total_query_time / max(1, self._query_count),
            "components_loaded": self._status == MemoryStatus.READY
        }
        
        # Add component-specific stats if loaded
        if self._status == MemoryStatus.READY and self._vector_search_engine:
            try:
                vector_stats = await self._vector_search_engine.get_index_stats()
                stats["vector_index"] = vector_stats
            except Exception as e:
                logger.warning(f"Failed to get vector index stats: {e}")
        
        return stats
    
    def is_ready(self) -> bool:
        """Check if the memory service is ready for use"""
        return self._status == MemoryStatus.READY
    
    async def rebuild_index(self, force: bool = False) -> bool:
        """
        Rebuild the vector search index
        
        Args:
            force: Whether to force rebuild even if index exists
            
        Returns:
            True if successful, False otherwise
        """
        # Load components if needed
        if not await self._load_components():
            logger.error("Failed to load memory components")
            return False
        
        try:
            logger.info("🔄 Rebuilding vector search index...")
            success = await self._vector_search_engine.build_index(force_rebuild=force)
            
            if success:
                logger.info("✅ Vector search index rebuilt successfully")
            else:
                logger.error("❌ Failed to rebuild vector search index")
            
            return success
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            return False
    
    async def clear_cache(self):
        """Clear any internal caches"""
        if self._vector_search_engine:
            # Clear search cache if available
            if hasattr(self._vector_search_engine, 'search_cache'):
                self._vector_search_engine.search_cache.clear()
                logger.info("Cleared vector search cache")
    
    async def close(self):
        """Close the service and clean up resources"""
        try:
            if self._memory_manager:
                self._memory_manager.close()
                self._memory_manager = None
            
            if self._embedding_service:
                self._embedding_service.close()
                self._embedding_service = None
            
            if self._vector_search_engine:
                # Vector search engine doesn't have a close method, but we can clear references
                self._vector_search_engine = None
            
            self._config = None
            self._status = MemoryStatus.UNLOADED
            
            logger.info("MemoryService closed")
            
        except Exception as e:
            logger.error(f"Error closing MemoryService: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the memory service
        
        Returns:
            Dictionary with health status information
        """
        health = {
            "status": self._status.value,
            "components": {},
            "errors": []
        }
        
        if self._status == MemoryStatus.READY:
            # Check memory manager
            if self._memory_manager:
                try:
                    # Simple check - try to get stats
                    stats = self._memory_manager.get_stats()
                    health["components"]["memory_manager"] = "healthy"
                except Exception as e:
                    health["components"]["memory_manager"] = "error"
                    health["errors"].append(f"MemoryManager: {e}")
            
            # Check vector search engine
            if self._vector_search_engine:
                try:
                    await self._vector_search_engine.get_index_stats()
                    health["components"]["vector_search"] = "healthy"
                except Exception as e:
                    health["components"]["vector_search"] = "error"
                    health["errors"].append(f"VectorSearchEngine: {e}")
            
            # Check embedding service
            if self._embedding_service:
                try:
                    if self._embedding_service.is_available:
                        health["components"]["embedding_service"] = "healthy"
                    else:
                        health["components"]["embedding_service"] = "unavailable"
                except Exception as e:
                    health["components"]["embedding_service"] = "error"
                    health["errors"].append(f"EmbeddingService: {e}")
        
        return health 