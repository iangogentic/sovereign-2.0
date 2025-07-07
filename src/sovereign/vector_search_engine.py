"""
Vector Search Engine - FAISS-based semantic similarity search for memory retrieval

This module provides high-performance vector search capabilities using FAISS for
efficient similarity search over conversation history and extracted memories.
"""

import asyncio
import logging
import pickle
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import faiss
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .memory_schema import MemorySchema
from .embedding_service import EmbeddingService


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result with metadata"""
    chunk_id: str
    score: float
    distance: float
    content: str
    timestamp: str
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'chunk_id': self.chunk_id,
            'score': self.score,
            'distance': self.distance,
            'content': self.content,
            'timestamp': self.timestamp,
            'conversation_id': self.conversation_id,
            'metadata': self.metadata
        }


@dataclass
class SearchParameters:
    """Configuration for search operations"""
    top_k: int = 10
    distance_threshold: Optional[float] = None
    include_metadata: bool = True
    filter_conversation_id: Optional[str] = None
    filter_timestamp_range: Optional[Tuple[str, str]] = None
    min_confidence: float = 0.0
    rerank: bool = False
    explain: bool = False


@dataclass 
class IndexMetadata:
    """Metadata for FAISS index"""
    index_type: str
    embedding_dim: int
    total_vectors: int
    created_at: str
    updated_at: str
    embedding_model: str
    distance_metric: str = "L2"
    index_params: Dict[str, Any] = field(default_factory=dict)


class VectorSearchEngine:
    """
    High-performance vector search engine using FAISS for semantic similarity search.
    
    Features:
    - Multiple index types (Flat, IVF, HNSW, PQ)
    - Automatic index selection based on dataset size
    - Efficient batch operations
    - Metadata filtering via SQLite
    - Index persistence and loading
    - Performance monitoring
    - GPU acceleration support
    """
    
    def __init__(self, 
                 db_path: str,
                 embedding_service: EmbeddingService,
                 index_dir: Optional[str] = None):
        """
        Initialize vector search engine
        
        Args:
            db_path: Path to SQLite database
            embedding_service: Embedding service instance
            index_dir: Directory to store FAISS indices (default: same as db)
        """
        self.db_path = Path(db_path)
        self.embedding_service = embedding_service
        self.index_dir = Path(index_dir) if index_dir else self.db_path.parent / "indices"
        self.index_dir.mkdir(exist_ok=True)
        
        # Database connection
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # FAISS index management
        self.index = None
        self.index_metadata = None
        self.chunk_id_map = {}  # Maps FAISS index positions to chunk IDs
        self.reverse_map = {}   # Maps chunk IDs to FAISS index positions
        
        # Performance tracking
        self.stats = {
            'total_searches': 0,
            'total_search_time': 0.0,
            'total_indexing_time': 0.0,
            'cache_hits': 0,
            'last_index_update': None
        }
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Search cache for repeated queries
        self.search_cache = {}
        self.cache_max_size = 1000
        
        logger.info(f"Vector Search Engine initialized with DB: {db_path}")
    
    async def initialize(self) -> None:
        """Initialize the search engine and load existing index if available"""
        try:
            # Ensure database tables exist
            memory_schema = MemorySchema(str(self.db_path))
            if not memory_schema.create_schema():
                logger.warning("Failed to create/verify memory schema")
            memory_schema.close()
            
            # Try to load existing index
            await self.load_index()
            
            if self.index is None:
                logger.info("No existing index found, will build on first search")
            else:
                logger.info(f"Loaded existing index with {self.index.ntotal} vectors")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector search engine: {e}")
            raise
    
    def _get_optimal_index_type(self, num_vectors: int, embedding_dim: int) -> str:
        """
        Determine optimal FAISS index type based on dataset characteristics
        
        Args:
            num_vectors: Number of vectors in dataset
            embedding_dim: Dimension of embeddings
            
        Returns:
            Recommended index type
        """
        if num_vectors < 1000:
            return "Flat"  # Exact search for small datasets
        elif num_vectors < 50000:
            return "IVFFlat"  # Good balance for medium datasets
        elif num_vectors < 1000000:
            return "HNSW"  # Excellent for large datasets
        else:
            return "IVFPQ"  # Compressed index for very large datasets
    
    def _create_faiss_index(self, embeddings: np.ndarray, index_type: str) -> faiss.Index:
        """
        Create FAISS index based on type and embeddings
        
        Args:
            embeddings: Array of embeddings (N, D)
            index_type: Type of index to create
            
        Returns:
            Configured FAISS index
        """
        embedding_dim = embeddings.shape[1]
        num_vectors = embeddings.shape[0]
        
        logger.info(f"Creating {index_type} index for {num_vectors} vectors of dim {embedding_dim}")
        
        if index_type == "Flat":
            # Exact search - best for small datasets
            index = faiss.IndexFlatL2(embedding_dim)
            
        elif index_type == "IVFFlat":
            # Inverted file index - good for medium datasets
            nlist = min(int(np.sqrt(num_vectors)), 16384)  # Number of clusters
            quantizer = faiss.IndexFlatL2(embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
            
            # Train the index
            index.train(embeddings)
            index.nprobe = min(nlist // 4, 128)  # Search in 25% of clusters
            
        elif index_type == "HNSW":
            # Hierarchical Navigable Small World - excellent for large datasets
            M = 16  # Number of bi-directional links for each node
            index = faiss.IndexHNSWFlat(embedding_dim, M)
            index.hnsw.efConstruction = 200  # Size of dynamic candidate list
            index.hnsw.efSearch = 128  # Size of dynamic candidate list during search
            
        elif index_type == "IVFPQ":
            # Inverted file with product quantization - memory efficient for huge datasets
            nlist = min(int(np.sqrt(num_vectors)), 16384)
            m = min(embedding_dim // 4, 64)  # Number of sub-quantizers
            nbits = 8  # Bits per sub-quantizer
            
            quantizer = faiss.IndexFlatL2(embedding_dim)
            index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, nbits)
            
            # Train the index
            index.train(embeddings)
            index.nprobe = min(nlist // 4, 128)
            
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        return index
    
    async def build_index(self, force_rebuild: bool = False) -> bool:
        """
        Build or rebuild the FAISS index from all embeddings in the database
        
        Args:
            force_rebuild: Whether to rebuild even if index exists
            
        Returns:
            True if index was built successfully
        """
        start_time = time.time()
        
        try:
            # Check if rebuild is needed
            if self.index is not None and not force_rebuild:
                logger.info("Index already exists, use force_rebuild=True to rebuild")
                return True
            
            logger.info("Building vector search index...")
            
            # Get all embeddings from database
            embeddings_data = await self._load_all_embeddings()
            
            if not embeddings_data:
                logger.warning("No embeddings found in database")
                return False
            
            embeddings = np.array([item['embedding'] for item in embeddings_data])
            chunk_ids = [item['chunk_id'] for item in embeddings_data]
            
            # Determine optimal index type
            index_type = self._get_optimal_index_type(len(embeddings), embeddings.shape[1])
            
            # Create and populate index
            index = self._create_faiss_index(embeddings, index_type)
            
            # Add vectors to index
            index.add(embeddings)
            
            # Create ID mappings
            chunk_id_map = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
            reverse_map = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
            
            # Update instance variables
            self.index = index
            self.chunk_id_map = chunk_id_map
            self.reverse_map = reverse_map
            
            # Create metadata
            self.index_metadata = IndexMetadata(
                index_type=index_type,
                embedding_dim=embeddings.shape[1],
                total_vectors=len(embeddings),
                created_at=str(time.time()),
                updated_at=str(time.time()),
                embedding_model=self.embedding_service.default_model.value,
                distance_metric="L2"
            )
            
            # Save index to disk
            await self.save_index()
            
            build_time = time.time() - start_time
            self.stats['total_indexing_time'] += build_time
            self.stats['last_index_update'] = time.time()
            
            logger.info(f"Successfully built {index_type} index with {len(embeddings)} vectors in {build_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return False
    
    async def _load_all_embeddings(self) -> List[Dict[str, Any]]:
        """Load all embeddings from the database"""
        def _load_sync():
            with self.engine.connect() as conn:
                query = text("""
                                        SELECT chunk_id, embedding, model_id
                    FROM embeddings
                    WHERE embedding IS NOT NULL
                    ORDER BY chunk_id
                """)
                
                results = []
                for row in conn.execute(query):
                    try:
                        # Deserialize embedding using the schema function
                        from src.sovereign.memory_schema import deserialize_embedding
                        embedding = deserialize_embedding(row.embedding)
                        results.append({
                            'chunk_id': row.chunk_id,
                            'embedding': embedding,
                            'model': row.model_id
                        })
                    except Exception as e:
                        logger.warning(f"Failed to deserialize embedding for chunk {row.chunk_id}: {e}")
                        continue
                
                return results
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _load_sync
        )
    
    async def add_vectors(self, chunk_ids: List[str], embeddings: np.ndarray) -> bool:
        """
        Add new vectors to the existing index
        
        Args:
            chunk_ids: List of chunk IDs
            embeddings: Array of embeddings (N, D)
            
        Returns:
            True if vectors were added successfully
        """
        try:
            if self.index is None:
                logger.info("No index exists, building new index")
                return await self.build_index()
            
            if len(chunk_ids) != len(embeddings):
                raise ValueError("Mismatched chunk_ids and embeddings lengths")
            
            # Check for duplicate chunk IDs
            existing_ids = set(chunk_ids) & set(self.reverse_map.keys())
            if existing_ids:
                logger.warning(f"Skipping {len(existing_ids)} duplicate chunk IDs")
                # Filter out duplicates
                mask = [chunk_id not in existing_ids for chunk_id in chunk_ids]
                chunk_ids = [chunk_ids[i] for i, keep in enumerate(mask) if keep]
                embeddings = embeddings[mask]
            
            if len(chunk_ids) == 0:
                logger.info("No new vectors to add")
                return True
            
            # Add vectors to index
            start_idx = self.index.ntotal
            self.index.add(embeddings)
            
            # Update mappings
            for i, chunk_id in enumerate(chunk_ids):
                idx = start_idx + i
                self.chunk_id_map[idx] = chunk_id
                self.reverse_map[chunk_id] = idx
            
            # Update metadata
            if self.index_metadata:
                self.index_metadata.total_vectors = self.index.ntotal
                self.index_metadata.updated_at = str(time.time())
            
            logger.info(f"Added {len(chunk_ids)} vectors to index (total: {self.index.ntotal})")
            
            # Save updated index
            await self.save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            return False
    
    async def remove_vectors(self, chunk_ids: List[str]) -> bool:
        """
        Remove vectors from the index (requires rebuild)
        
        Args:
            chunk_ids: List of chunk IDs to remove
            
        Returns:
            True if vectors were removed successfully
        """
        try:
            if not chunk_ids:
                return True
            
            # Check which IDs exist
            existing_ids = [cid for cid in chunk_ids if cid in self.reverse_map]
            
            if not existing_ids:
                logger.info("No matching chunk IDs found to remove")
                return True
            
            logger.info(f"Removing {len(existing_ids)} vectors from index")
            
            # FAISS doesn't support efficient removal, so we rebuild
            # First, remove from database (this will be excluded from rebuild)
            with self.engine.connect() as conn:
                for chunk_id in existing_ids:
                    conn.execute(text(
                        "DELETE FROM embeddings WHERE chunk_id = :chunk_id"
                    ), {"chunk_id": chunk_id})
                conn.commit()
            
            # Rebuild index
            return await self.build_index(force_rebuild=True)
            
        except Exception as e:
            logger.error(f"Failed to remove vectors: {e}")
            return False
    
    async def search(self, 
                    query: Union[str, np.ndarray], 
                    params: Optional[SearchParameters] = None) -> List[SearchResult]:
        """
        Perform semantic similarity search
        
        Args:
            query: Query string or embedding vector
            params: Search parameters
            
        Returns:
            List of search results
        """
        if params is None:
            params = SearchParameters()
        
        start_time = time.time()
        
        try:
            # Check if index exists
            if self.index is None:
                logger.info("No index available, building index...")
                if not await self.build_index():
                    return []
            
            # Get query embedding
            if isinstance(query, str):
                # Create consistent cache key that handles None values
                cache_key = f"{query}:{params.top_k}:{params.distance_threshold or 'None'}:{params.min_confidence}:{params.filter_conversation_id or 'None'}"
                if cache_key in self.search_cache:
                    self.stats['cache_hits'] += 1
                    return self.search_cache[cache_key]
                
                query_embedding_result = self.embedding_service.generate_embedding(query)
                query_embedding = query_embedding_result.embedding
                if query_embedding is None:
                    logger.error("Failed to get embedding for query")
                    return []
                # Ensure embedding is flattened and properly shaped
                query_embedding_flat = np.array(query_embedding).flatten()
                query_vector = query_embedding_flat.reshape(1, -1).astype(np.float32)
            else:
                query_vector = query.reshape(1, -1).astype(np.float32)
            
            # Perform FAISS search
            distances, indices = await self._faiss_search(query_vector, params.top_k * 2)  # Get more for filtering
            
            # Convert to results with metadata
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                chunk_id = self.chunk_id_map.get(idx)
                if chunk_id is None:
                    continue
                
                # Apply distance threshold
                if params.distance_threshold is not None and dist > params.distance_threshold:
                    continue
                
                # Convert distance to similarity score
                score = max(0.0, 1.0 / (1.0 + dist))
                
                # Apply confidence threshold
                if score < params.min_confidence:
                    continue
                
                result = SearchResult(
                    chunk_id=chunk_id,
                    score=score,
                    distance=float(dist),
                    content="",  # Will be filled by metadata lookup
                    timestamp=""
                )
                results.append(result)
            
            # Limit to requested number
            results = results[:params.top_k]
            
            # Fetch metadata from database
            if params.include_metadata and results:
                await self._enrich_results_with_metadata(results, params)
            
            # Apply post-search filters
            results = self._apply_filters(results, params)
            
            # Cache results for string queries only
            if isinstance(query, str) and len(self.search_cache) < self.cache_max_size:
                self.search_cache[cache_key] = results
            
            # Update stats
            search_time = time.time() - start_time
            self.stats['total_searches'] += 1
            self.stats['total_search_time'] += search_time
            
            logger.debug(f"Search completed in {search_time:.3f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _faiss_search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform FAISS search in thread pool"""
        def _search_sync():
            # Create a local copy to avoid modifying the original
            local_query_vector = query_vector.copy()
            
            # Ensure query vector is 2D and float32
            if local_query_vector.ndim == 1:
                local_query_vector = local_query_vector.reshape(1, -1)
            
            # Ensure correct dtype for FAISS
            local_query_vector = local_query_vector.astype(np.float32)
            
            # Perform search
            return self.index.search(local_query_vector, k)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _search_sync
        )
    
    async def _enrich_results_with_metadata(self, results: List[SearchResult], params: SearchParameters) -> None:
        """Fetch metadata from database for search results"""
        def _fetch_metadata():
            with self.engine.connect() as conn:
                chunk_ids = [r.chunk_id for r in results]
                
                # Use proper SQLAlchemy parameter binding
                if not chunk_ids:
                    return {}
                
                # Create IN clause with proper parameter binding
                placeholders = ','.join([f':chunk_id_{i}' for i in range(len(chunk_ids))])
                
                # Try with conversations table first, fallback without if it doesn't exist
                try:
                    query = text(f"""
                        SELECT 
                            c.id as chunk_id,
                            c.text,
                            c.created_at as timestamp,
                            c.document_id as conversation_id,
                            c.chunk_type,
                            '' as metadata_json,
                            d.title as conversation_title
                        FROM chunks c
                        LEFT JOIN documents d ON c.document_id = d.id
                        WHERE c.id IN ({placeholders})
                    """)
                    
                    # Create parameter dictionary
                    params = {f'chunk_id_{i}': chunk_id for i, chunk_id in enumerate(chunk_ids)}
                    
                    metadata_dict = {}
                    for row in conn.execute(query, params):
                        metadata_dict[row.chunk_id] = {
                            'content': row.text,
                            'timestamp': row.timestamp,
                            'conversation_id': row.conversation_id,
                            'chunk_type': row.chunk_type,
                            'metadata_json': row.metadata_json,
                            'conversation_title': row.conversation_title
                        }
                    
                    return metadata_dict
                
                except Exception:
                    # Fallback query without conversations table
                    query = text(f"""
                        SELECT 
                            c.id as chunk_id,
                            c.text,
                            c.created_at as timestamp,
                            c.document_id as conversation_id,
                            c.chunk_type,
                            '' as metadata_json,
                            NULL as conversation_title
                        FROM chunks c
                        WHERE c.id IN ({placeholders})
                    """)
                    
                    # Create parameter dictionary
                    params = {f'chunk_id_{i}': chunk_id for i, chunk_id in enumerate(chunk_ids)}
                    
                    metadata_dict = {}
                    for row in conn.execute(query, params):
                        metadata_dict[row.chunk_id] = {
                            'content': row.text,
                            'timestamp': row.timestamp,
                            'conversation_id': row.conversation_id,
                            'chunk_type': row.chunk_type,
                            'metadata_json': row.metadata_json,
                            'conversation_title': row.conversation_title
                        }
                    
                    return metadata_dict
        
        metadata_dict = await asyncio.get_event_loop().run_in_executor(
            self.executor, _fetch_metadata
        )
        
        # Enrich results
        for result in results:
            if result.chunk_id in metadata_dict:
                data = metadata_dict[result.chunk_id]
                result.content = data['content'] or ""
                result.timestamp = data['timestamp'] or ""
                result.conversation_id = data['conversation_id']
                result.metadata = {
                    'chunk_type': data['chunk_type'],
                    'metadata_json': data['metadata_json'],
                    'conversation_title': data['conversation_title']
                }
    
    def _apply_filters(self, results: List[SearchResult], params: SearchParameters) -> List[SearchResult]:
        """Apply post-search filters"""
        filtered = results
        
        # Filter by conversation ID
        if params.filter_conversation_id:
            filtered = [r for r in filtered if r.conversation_id == params.filter_conversation_id]
        
        # Filter by timestamp range
        if params.filter_timestamp_range:
            start_time, end_time = params.filter_timestamp_range
            filtered = [r for r in filtered if start_time <= r.timestamp <= end_time]
        
        return filtered
    
    async def save_index(self) -> bool:
        """Save FAISS index and metadata to disk"""
        try:
            if self.index is None:
                return False
            
            # Save FAISS index
            index_path = self.index_dir / "faiss.index"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata and mappings
            metadata_path = self.index_dir / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.index_metadata,
                    'chunk_id_map': self.chunk_id_map,
                    'reverse_map': self.reverse_map,
                    'stats': self.stats
                }, f)
            
            logger.debug(f"Saved index with {self.index.ntotal} vectors to {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    async def load_index(self) -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            index_path = self.index_dir / "faiss.index"
            metadata_path = self.index_dir / "metadata.pkl"
            
            if not index_path.exists() or not metadata_path.exists():
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata and mappings
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.index_metadata = data['metadata']
                self.chunk_id_map = data['chunk_id_map']
                self.reverse_map = data['reverse_map']
                self.stats.update(data.get('stats', {}))
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors from {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get comprehensive index statistics"""
        stats = {
            'index_exists': self.index is not None,
            'total_vectors': self.index.ntotal if self.index else 0,
            'index_type': self.index_metadata.index_type if self.index_metadata else None,
            'embedding_dim': self.index_metadata.embedding_dim if self.index_metadata else None,
            'embedding_model': self.index_metadata.embedding_model if self.index_metadata else None,
            'performance_stats': self.stats.copy(),
            'cache_size': len(self.search_cache),
            'last_updated': self.index_metadata.updated_at if self.index_metadata else None
        }
        
        # Add database stats
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM chunks")).fetchone()
                stats['total_chunks'] = result[0] if result else 0
                
                result = conn.execute(text("SELECT COUNT(*) FROM embeddings")).fetchone()
                stats['total_embeddings'] = result[0] if result else 0
                
        except Exception as e:
            logger.warning(f"Failed to get database stats: {e}")
        
        return stats
    
    async def optimize_index(self) -> bool:
        """Optimize index performance based on usage patterns"""
        try:
            if self.index is None:
                return False
            
            # Clear search cache when it exceeds max size
            if len(self.search_cache) > self.cache_max_size:
                # Keep only most recent entries
                items = list(self.search_cache.items())
                self.search_cache = dict(items[-self.cache_max_size:])
                logger.info(f"Cleared search cache, reduced to {len(self.search_cache)} entries")
            
            # Check if index rebuild is beneficial
            if (self.stats['total_searches'] > 1000 and 
                self.index_metadata and 
                time.time() - float(self.index_metadata.updated_at) > 86400):  # 24 hours
                
                logger.info("Considering index optimization...")
                
                # Get current vector count
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT COUNT(*) FROM embeddings")).fetchone()
                    current_count = result[0] if result else 0
                
                # Rebuild if significant growth
                if current_count > self.index_metadata.total_vectors * 1.2:
                    logger.info("Rebuilding index due to significant growth")
                    return await self.build_index(force_rebuild=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Index optimization failed: {e}")
            return False
    
    async def batch_search(self, 
                          queries: List[Union[str, np.ndarray]], 
                          params: Optional[SearchParameters] = None) -> List[List[SearchResult]]:
        """
        Perform batch similarity search for multiple queries
        
        Args:
            queries: List of query strings or embedding vectors
            params: Search parameters
            
        Returns:
            List of result lists, one per query
        """
        if not queries:
            return []
        
        if params is None:
            params = SearchParameters()
        
        # Process in parallel with controlled concurrency
        semaphore = asyncio.Semaphore(4)  # Limit concurrent searches
        
        async def search_with_semaphore(query):
            async with semaphore:
                return await self.search(query, params)
        
        tasks = [search_with_semaphore(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch search query {i} failed: {result}")
                final_results.append([])
            else:
                final_results.append(result)
        
        return final_results
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            
            # Clear caches
            self.search_cache.clear()
            
            logger.info("Vector search engine cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except:
            pass


# Public API functions for easy integration

async def create_vector_search_engine(db_path: str, 
                                     embedding_service: EmbeddingService,
                                     index_dir: Optional[str] = None) -> VectorSearchEngine:
    """
    Create and initialize a vector search engine
    
    Args:
        db_path: Path to SQLite database
        embedding_service: Embedding service instance
        index_dir: Optional directory for indices
        
    Returns:
        Initialized VectorSearchEngine
    """
    engine = VectorSearchEngine(db_path, embedding_service, index_dir)
    await engine.initialize()
    return engine


async def search_memories(engine: VectorSearchEngine,
                         query: str,
                         top_k: int = 10,
                         **kwargs) -> List[SearchResult]:
    """
    Convenient function for memory search
    
    Args:
        engine: Vector search engine instance
        query: Search query
        top_k: Number of results to return
        **kwargs: Additional search parameters
        
    Returns:
        List of search results
    """
    params = SearchParameters(top_k=top_k, **kwargs)
    return await engine.search(query, params) 