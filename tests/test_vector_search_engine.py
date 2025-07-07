"""
Vector Search Engine Tests - Comprehensive testing of FAISS-based similarity search

These tests validate the vector search engine functionality including index creation,
search operations, metadata handling, and performance optimization.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import sqlite3
import time
import shutil
import pickle
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

import numpy as np
import faiss

from src.sovereign.vector_search_engine import (
    VectorSearchEngine, SearchResult, SearchParameters, IndexMetadata,
    create_vector_search_engine, search_memories
)
from src.sovereign.embedding_service import EmbeddingService, EmbeddingResponse
from src.sovereign.memory_schema import MemorySchema
from src.sovereign.config import Config


class TestVectorSearchEngine:
    """Test vector search engine functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def test_db_path(self, temp_dir):
        """Create test database path"""
        return Path(temp_dir) / "test_memory.db"
    
    @pytest.fixture
    def embedding_service(self):
        """Create mock embedding service"""
        service = Mock(spec=EmbeddingService)
        service.current_model = "test-model"
        
        # Create a proper async mock that returns different embeddings for different texts
        async def mock_generate_embedding(text):
            # Use hash of text to generate deterministic but different embeddings
            import hashlib
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            embedding = np.random.random(384)
            return EmbeddingResponse(
                request_id="test-request",
                embedding=embedding,
                model_used="test-model",
                processing_time=0.1,
                tokens_processed=10,
                success=True
            )
        
        # Use AsyncMock to track calls properly
        service.generate_embedding = AsyncMock(side_effect=mock_generate_embedding)
        return service
    
    @pytest_asyncio.fixture
    async def vector_engine(self, test_db_path, embedding_service, temp_dir):
        """Create vector search engine instance"""
        engine = VectorSearchEngine(
            db_path=str(test_db_path),
            embedding_service=embedding_service,
            index_dir=str(Path(temp_dir) / "indices")
        )
        await engine.initialize()
        yield engine
        await engine.cleanup()
    
    async def setup_test_data(self, db_path: Path):
        """Set up test data in database"""
        # Create database with the tables expected by vector search engine
        with sqlite3.connect(db_path) as conn:
            # Ensure clean tables
            conn.execute("DROP TABLE IF EXISTS memory_embeddings")
            conn.execute("DROP TABLE IF EXISTS memory_chunks")
            conn.execute("DROP TABLE IF EXISTS conversations")
            
            # Create the tables that vector search engine expects
            conn.execute("""
                CREATE TABLE memory_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    timestamp DATETIME,
                    conversation_id TEXT,
                    chunk_type TEXT DEFAULT 'text',
                    metadata_json TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE memory_embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    embedding_data BLOB,
                    embedding_model TEXT,
                    embedding_timestamp DATETIME,
                    FOREIGN KEY(chunk_id) REFERENCES memory_chunks(chunk_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at DATETIME
                )
            """)
            
            # Insert test data
            test_chunks = [
                ("chunk_1", "This is about machine learning and AI", "2024-01-01T10:00:00", "conv_1"),
                ("chunk_2", "Python programming tutorial for beginners", "2024-01-01T11:00:00", "conv_1"),
                ("chunk_3", "Deep learning neural networks explained", "2024-01-01T12:00:00", "conv_2"),
                ("chunk_4", "Database design and optimization tips", "2024-01-01T13:00:00", "conv_2"),
                ("chunk_5", "Web development with React and Node.js", "2024-01-01T14:00:00", "conv_3"),
            ]
            
            for chunk_id, content, timestamp, conv_id in test_chunks:
                conn.execute("""
                    INSERT INTO memory_chunks (chunk_id, content, timestamp, conversation_id, chunk_type)
                    VALUES (?, ?, ?, ?, 'text')
                """, (chunk_id, content, timestamp, conv_id))
                
                # Create test embeddings (random for testing)
                embedding = np.random.random(384).astype(np.float32)
                embedding_blob = sqlite3.Binary(pickle.dumps(embedding))
                
                conn.execute("""
                    INSERT INTO memory_embeddings (chunk_id, embedding_data, embedding_model, embedding_timestamp)
                    VALUES (?, ?, 'test-model', ?)
                """, (chunk_id, embedding_blob, timestamp))
            
            # Insert conversations
            conversations = [
                ("conv_1", "AI and Programming Discussion"),
                ("conv_2", "Technical Deep Dive"),
                ("conv_3", "Web Development Chat")
            ]
            
            for conv_id, title in conversations:
                conn.execute("""
                    INSERT INTO conversations (conversation_id, title, created_at)
                    VALUES (?, ?, '2024-01-01T09:00:00')
                """, (conv_id, title))
            
            conn.commit()
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, vector_engine, test_db_path):
        """Test vector search engine initialization"""
        assert vector_engine.db_path == test_db_path
        assert vector_engine.embedding_service is not None
        assert vector_engine.index_dir.exists()
        assert vector_engine.stats is not None
        assert vector_engine.search_cache == {}
    
    @pytest.mark.asyncio
    async def test_optimal_index_type_selection(self, vector_engine):
        """Test optimal index type selection logic"""
        # Small dataset
        index_type = vector_engine._get_optimal_index_type(500, 384)
        assert index_type == "Flat"
        
        # Medium dataset
        index_type = vector_engine._get_optimal_index_type(10000, 384)
        assert index_type == "IVFFlat"
        
        # Large dataset
        index_type = vector_engine._get_optimal_index_type(100000, 384)
        assert index_type == "HNSW"
        
        # Very large dataset
        index_type = vector_engine._get_optimal_index_type(2000000, 384)
        assert index_type == "IVFPQ"
    
    @pytest.mark.asyncio
    async def test_faiss_index_creation(self, vector_engine):
        """Test FAISS index creation for different types"""
        embeddings = np.random.random((100, 384)).astype(np.float32)
        
        # Test Flat index
        index = vector_engine._create_faiss_index(embeddings, "Flat")
        assert isinstance(index, faiss.IndexFlatL2)
        assert index.d == 384
        
        # Test IVFFlat index with training
        index = vector_engine._create_faiss_index(embeddings, "IVFFlat")
        assert isinstance(index, faiss.IndexIVFFlat)
        assert index.d == 384
        assert index.is_trained
        
        # Test HNSW index
        index = vector_engine._create_faiss_index(embeddings, "HNSW")
        assert isinstance(index, faiss.IndexHNSWFlat)
        assert index.d == 384
    
    @pytest.mark.asyncio
    async def test_load_all_embeddings(self, vector_engine, test_db_path):
        """Test loading embeddings from database"""
        # Setup test data
        await self.setup_test_data(test_db_path)
        
        # Load embeddings
        embeddings_data = await vector_engine._load_all_embeddings()
        
        assert len(embeddings_data) == 5
        for item in embeddings_data:
            assert 'chunk_id' in item
            assert 'embedding' in item
            assert 'model' in item
            assert isinstance(item['embedding'], np.ndarray)
            assert item['embedding'].shape == (384,)
    
    @pytest.mark.asyncio
    async def test_build_index(self, vector_engine, test_db_path):
        """Test index building from database embeddings"""
        # Setup test data
        await self.setup_test_data(test_db_path)
        
        # Build index
        success = await vector_engine.build_index()
        
        assert success is True
        assert vector_engine.index is not None
        assert vector_engine.index.ntotal == 5
        assert vector_engine.index_metadata is not None
        assert vector_engine.index_metadata.total_vectors == 5
        assert vector_engine.index_metadata.embedding_model == "test-model"
        assert len(vector_engine.chunk_id_map) == 5
        assert len(vector_engine.reverse_map) == 5
    
    @pytest.mark.asyncio
    async def test_add_vectors(self, vector_engine, test_db_path):
        """Test adding new vectors to existing index"""
        # Setup initial data and build index
        await self.setup_test_data(test_db_path)
        await vector_engine.build_index()
        
        initial_count = vector_engine.index.ntotal
        
        # Add new vectors
        new_chunk_ids = ["chunk_6", "chunk_7"]
        new_embeddings = np.random.random((2, 384)).astype(np.float32)
        
        success = await vector_engine.add_vectors(new_chunk_ids, new_embeddings)
        
        assert success is True
        assert vector_engine.index.ntotal == initial_count + 2
        assert "chunk_6" in vector_engine.reverse_map
        assert "chunk_7" in vector_engine.reverse_map
    
    @pytest.mark.asyncio
    async def test_remove_vectors(self, vector_engine, test_db_path):
        """Test removing vectors from index"""
        # Setup test data and build index
        await self.setup_test_data(test_db_path)
        await vector_engine.build_index()
        
        # Remove vectors (this should trigger rebuild)
        success = await vector_engine.remove_vectors(["chunk_1", "chunk_2"])
        
        assert success is True
        # Note: remove_vectors rebuilds index, so we need to check database
        with sqlite3.connect(test_db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memory_embeddings")
            count = cursor.fetchone()[0]
            assert count == 3  # 5 - 2 = 3
    
    @pytest.mark.asyncio
    async def test_search_with_string_query(self, vector_engine, test_db_path, embedding_service):
        """Test search with string query"""
        # Setup test data and build index
        await self.setup_test_data(test_db_path)
        await vector_engine.build_index()
        
        # Mock embedding service to return consistent embedding
        test_embedding = np.random.random(384).tolist()
        embedding_response = EmbeddingResponse(
            request_id="test-request",
            embedding=np.array(test_embedding),
            model_used="test-model",
            processing_time=0.1,
            tokens_processed=10,
            success=True
        )
        embedding_service.generate_embedding.return_value = embedding_response
        
        # Perform search
        results = await vector_engine.search("machine learning")
        
        assert isinstance(results, list)
        assert len(results) <= 10  # Default top_k
        embedding_service.generate_embedding.assert_called_once_with("machine learning")
    
    @pytest.mark.asyncio
    async def test_search_with_vector_query(self, vector_engine, test_db_path):
        """Test search with embedding vector"""
        # Setup test data and build index
        await self.setup_test_data(test_db_path)
        await vector_engine.build_index()
        
        # Create query vector
        query_vector = np.random.random(384).astype(np.float32)
        
        # Perform search
        results = await vector_engine.search(query_vector)
        
        assert isinstance(results, list)
        assert len(results) <= 10
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.chunk_id.startswith("chunk_")
            assert 0 <= result.score <= 1
            assert result.distance >= 0
    
    @pytest.mark.asyncio
    async def test_search_parameters(self, vector_engine, test_db_path):
        """Test search with various parameters"""
        # Setup test data and build index
        await self.setup_test_data(test_db_path)
        await vector_engine.build_index()
        
        query_vector = np.random.random(384).astype(np.float32)
        
        # Test different top_k values
        params = SearchParameters(top_k=3)
        results = await vector_engine.search(query_vector, params)
        assert len(results) <= 3
        
        # Test distance threshold
        params = SearchParameters(distance_threshold=0.5)
        results = await vector_engine.search(query_vector, params)
        for result in results:
            assert result.distance <= 0.5
        
        # Test confidence threshold
        params = SearchParameters(min_confidence=0.8)
        results = await vector_engine.search(query_vector, params)
        for result in results:
            assert result.score >= 0.8
    
    @pytest.mark.asyncio
    async def test_metadata_enrichment(self, vector_engine, test_db_path):
        """Test result enrichment with metadata"""
        # Setup test data and build index
        await self.setup_test_data(test_db_path)
        await vector_engine.build_index()
        
        query_vector = np.random.random(384).astype(np.float32)
        params = SearchParameters(include_metadata=True)
        
        results = await vector_engine.search(query_vector, params)
        
        for result in results:
            assert result.content != ""  # Should be enriched
            assert result.timestamp != ""
            assert result.conversation_id is not None
            assert 'chunk_type' in result.metadata
    
    @pytest.mark.asyncio
    async def test_filtering(self, vector_engine, test_db_path):
        """Test post-search filtering"""
        # Setup test data and build index
        await self.setup_test_data(test_db_path)
        await vector_engine.build_index()
        
        query_vector = np.random.random(384).astype(np.float32)
        
        # Test conversation ID filtering
        params = SearchParameters(
            filter_conversation_id="conv_1",
            include_metadata=True
        )
        results = await vector_engine.search(query_vector, params)
        
        for result in results:
            assert result.conversation_id == "conv_1"
        
        # Test timestamp filtering
        params = SearchParameters(
            filter_timestamp_range=("2024-01-01T10:00:00", "2024-01-01T12:00:00"),
            include_metadata=True
        )
        results = await vector_engine.search(query_vector, params)
        
        for result in results:
            assert "2024-01-01T10:00:00" <= result.timestamp <= "2024-01-01T12:00:00"
    
    @pytest.mark.asyncio
    async def test_search_caching(self, vector_engine, test_db_path, embedding_service):
        """Test search result caching"""
        # Setup test data and build index
        await self.setup_test_data(test_db_path)
        await vector_engine.build_index()
        
        # Configure the AsyncMock to return a consistent embedding
        test_embedding = np.random.random(384).tolist()
        embedding_response = EmbeddingResponse(
            request_id="test-request",
            embedding=np.array(test_embedding),
            model_used="test-model",
            processing_time=0.1,
            tokens_processed=10,
            success=True
        )
        
        # Configure the AsyncMock to return the same response every time
        async def mock_generate_embedding(text):
            return embedding_response
        
        embedding_service.generate_embedding.side_effect = mock_generate_embedding
        
        query = "machine learning"
        
        # First search
        results1 = await vector_engine.search(query)
        assert embedding_service.generate_embedding.call_count == 1
        
        # Second search (should use cache)
        results2 = await vector_engine.search(query)
        assert embedding_service.generate_embedding.call_count == 1  # No additional call
        assert vector_engine.stats['cache_hits'] == 1
        
        # Results should be identical
        assert len(results1) == len(results2)
    
    @pytest.mark.asyncio
    async def test_batch_search(self, vector_engine, test_db_path, embedding_service):
        """Test batch search functionality"""
        # Setup test data and build index
        await self.setup_test_data(test_db_path)
        await vector_engine.build_index()
        
        # Mock embedding service
        test_embedding = np.random.random(384).tolist()
        embedding_response = EmbeddingResponse(
            request_id="test-request",
            embedding=np.array(test_embedding),
            model_used="test-model",
            processing_time=0.1,
            tokens_processed=10,
            success=True
        )
        embedding_service.generate_embedding.return_value = embedding_response
        
        queries = ["machine learning", "python programming", "web development"]
        
        # Perform batch search
        batch_results = await vector_engine.batch_search(queries)
        
        assert len(batch_results) == 3
        for results in batch_results:
            assert isinstance(results, list)
            assert len(results) <= 10
    
    @pytest.mark.asyncio
    async def test_index_persistence(self, vector_engine, test_db_path, temp_dir):
        """Test saving and loading index"""
        # Setup test data and build index
        await self.setup_test_data(test_db_path)
        await vector_engine.build_index()
        
        original_total = vector_engine.index.ntotal
        original_metadata = vector_engine.index_metadata
        
        # Save index
        success = await vector_engine.save_index()
        assert success is True
        
        # Verify files exist
        assert (vector_engine.index_dir / "faiss.index").exists()
        assert (vector_engine.index_dir / "metadata.pkl").exists()
        
        # Create new engine and load index
        new_engine = VectorSearchEngine(
            db_path=str(test_db_path),
            embedding_service=vector_engine.embedding_service,
            index_dir=str(Path(temp_dir) / "indices")
        )
        await new_engine.initialize()
        
        # Verify loaded index
        assert new_engine.index.ntotal == original_total
        assert new_engine.index_metadata.total_vectors == original_metadata.total_vectors
        assert len(new_engine.chunk_id_map) == original_total
        
        await new_engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_index_stats(self, vector_engine, test_db_path):
        """Test index statistics"""
        # Setup test data and build index
        await self.setup_test_data(test_db_path)
        await vector_engine.build_index()
        
        stats = await vector_engine.get_index_stats()
        
        assert stats['index_exists'] is True
        assert stats['total_vectors'] == 5
        assert stats['index_type'] == "Flat"  # Small dataset uses Flat
        assert stats['embedding_dim'] == 384
        assert stats['embedding_model'] == "test-model"
        assert 'performance_stats' in stats
        assert 'total_chunks' in stats
        assert 'total_embeddings' in stats
    
    @pytest.mark.asyncio
    async def test_index_optimization(self, vector_engine, test_db_path):
        """Test index optimization"""
        # Setup test data and build index
        await self.setup_test_data(test_db_path)
        await vector_engine.build_index()
        
        # Fill cache beyond limit
        for i in range(vector_engine.cache_max_size + 100):
            vector_engine.search_cache[f"query_{i}"] = []
        
        assert len(vector_engine.search_cache) > vector_engine.cache_max_size
        
        # Run optimization
        success = await vector_engine.optimize_index()
        assert success is True
        
        # Cache should be reduced
        assert len(vector_engine.search_cache) <= vector_engine.cache_max_size
    
    @pytest.mark.asyncio
    async def test_error_handling(self, vector_engine, test_db_path):
        """Test error handling in various scenarios"""
        # Test search without index
        results = await vector_engine.search("test query")
        assert results == []  # Should handle gracefully
        
        # Test with invalid embeddings
        with patch.object(vector_engine.embedding_service, 'generate_embedding', return_value=None):
            results = await vector_engine.search("test query")
            assert results == []
        
        # Test FAISS search error
        await self.setup_test_data(test_db_path)
        await vector_engine.build_index()
        
        with patch.object(vector_engine.index, 'search', side_effect=Exception("FAISS error")):
            results = await vector_engine.search(np.random.random(384))
            assert results == []
    
    @pytest.mark.asyncio
    async def test_search_result_serialization(self, vector_engine):
        """Test SearchResult serialization"""
        result = SearchResult(
            chunk_id="test_chunk",
            score=0.95,
            distance=0.05,
            content="Test content",
            timestamp="2024-01-01T10:00:00",
            conversation_id="conv_1",
            metadata={"type": "text", "source": "test"}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['chunk_id'] == "test_chunk"
        assert result_dict['score'] == 0.95
        assert result_dict['distance'] == 0.05
        assert result_dict['content'] == "Test content"
        assert result_dict['timestamp'] == "2024-01-01T10:00:00"
        assert result_dict['conversation_id'] == "conv_1"
        assert result_dict['metadata'] == {"type": "text", "source": "test"}
    
    @pytest.mark.asyncio
    async def test_cleanup(self, vector_engine):
        """Test cleanup functionality"""
        # Add some cache entries
        vector_engine.search_cache = {"query1": [], "query2": []}
        
        await vector_engine.cleanup()
        
        assert len(vector_engine.search_cache) == 0


class TestVectorSearchPublicAPI:
    """Test public API functions"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def embedding_service(self):
        """Create mock embedding service"""
        service = Mock(spec=EmbeddingService)
        service.current_model = "test-model"
        
        # Create a proper async mock that returns different embeddings for different texts
        async def mock_generate_embedding(text):
            # Use hash of text to generate deterministic but different embeddings
            import hashlib
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            embedding = np.random.random(384)
            return EmbeddingResponse(
                request_id="test-request",
                embedding=embedding,
                model_used="test-model",
                processing_time=0.1,
                tokens_processed=10,
                success=True
            )
        
        # Use AsyncMock to track calls properly
        service.generate_embedding = AsyncMock(side_effect=mock_generate_embedding)
        return service
    
    @pytest.mark.asyncio
    async def test_create_vector_search_engine(self, temp_dir, embedding_service):
        """Test vector search engine creation via public API"""
        db_path = str(Path(temp_dir) / "test.db")
        
        engine = await create_vector_search_engine(
            db_path=db_path,
            embedding_service=embedding_service,
            index_dir=str(Path(temp_dir) / "indices")
        )
        
        assert isinstance(engine, VectorSearchEngine)
        assert engine.db_path == Path(db_path)
        assert engine.embedding_service == embedding_service
        assert engine.index_dir.exists()
        
        await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_search_memories_convenience_function(self, temp_dir, embedding_service):
        """Test convenience search function"""
        db_path = str(Path(temp_dir) / "test.db")
        
        engine = await create_vector_search_engine(
            db_path=db_path,
            embedding_service=embedding_service
        )
        
        # This should work even with empty index
        results = await search_memories(
            engine=engine,
            query="test query",
            top_k=5,
            include_metadata=True
        )
        
        assert isinstance(results, list)
        assert len(results) <= 5
        
        await engine.cleanup()


class TestVectorSearchPerformance:
    """Performance tests for vector search engine"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def embedding_service(self):
        """Create mock embedding service"""
        service = Mock(spec=EmbeddingService)
        service.current_model = "test-model"
        
        # Create a proper async mock that returns different embeddings for different texts
        async def mock_generate_embedding(text):
            # Use hash of text to generate deterministic but different embeddings
            import hashlib
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            embedding = np.random.random(384)
            return EmbeddingResponse(
                request_id="test-request",
                embedding=embedding,
                model_used="test-model",
                processing_time=0.1,
                tokens_processed=10,
                success=True
            )
        
        # Use AsyncMock to track calls properly
        service.generate_embedding = AsyncMock(side_effect=mock_generate_embedding)
        return service
    
    @pytest.mark.asyncio
    async def test_large_index_performance(self, temp_dir, embedding_service):
        """Test performance with larger dataset"""
        db_path = Path(temp_dir) / "large_test.db"
        
        # Create larger test dataset
        with sqlite3.connect(db_path) as conn:
            # Ensure clean tables
            conn.execute("DROP TABLE IF EXISTS memory_embeddings")
            conn.execute("DROP TABLE IF EXISTS memory_chunks")
            
            # Create the tables that vector search engine expects
            conn.execute("""
                CREATE TABLE memory_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    timestamp DATETIME,
                    conversation_id TEXT,
                    chunk_type TEXT DEFAULT 'text',
                    metadata_json TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE memory_embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    embedding_data BLOB,
                    embedding_model TEXT,
                    embedding_timestamp DATETIME,
                    FOREIGN KEY(chunk_id) REFERENCES memory_chunks(chunk_id)
                )
            """)
            
            # Insert 1000 test chunks with embeddings
            for i in range(1000):
                chunk_id = f"chunk_{i:04d}"
                content = f"Test content {i} about various topics"
                timestamp = f"2024-01-{(i % 30) + 1:02d}T10:00:00"
                conv_id = f"conv_{i % 10}"
                
                conn.execute("""
                    INSERT INTO memory_chunks (chunk_id, content, timestamp, conversation_id, chunk_type)
                    VALUES (?, ?, ?, ?, 'text')
                """, (chunk_id, content, timestamp, conv_id))
                
                # Random embedding
                embedding = np.random.random(384).astype(np.float32)
                embedding_blob = sqlite3.Binary(pickle.dumps(embedding))
                
                conn.execute("""
                    INSERT INTO memory_embeddings (chunk_id, embedding_data, embedding_model, embedding_timestamp)
                    VALUES (?, ?, 'test-model', ?)
                """, (chunk_id, embedding_blob, timestamp))
            
            conn.commit()
        
        # Create engine and build index
        engine = VectorSearchEngine(
            db_path=str(db_path),
            embedding_service=embedding_service,
            index_dir=str(Path(temp_dir) / "indices")
        )
        await engine.initialize()
        
        # Time index building
        start_time = time.time()
        await engine.build_index()
        build_time = time.time() - start_time
        
        assert engine.index.ntotal == 1000
        assert build_time < 10.0  # Should build in reasonable time
        
        # Test search performance
        query_vector = np.random.random(384).astype(np.float32)
        
        start_time = time.time()
        results = await engine.search(query_vector, SearchParameters(top_k=10))
        search_time = time.time() - start_time
        
        assert len(results) == 10
        assert search_time < 1.0  # Should search quickly
        
        await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_batch_search_performance(self, temp_dir, embedding_service):
        """Test batch search performance"""
        db_path = Path(temp_dir) / "batch_test.db"
        
        # Setup test data
        with sqlite3.connect(db_path) as conn:
            # Ensure clean tables
            conn.execute("DROP TABLE IF EXISTS memory_embeddings")
            conn.execute("DROP TABLE IF EXISTS memory_chunks")
            
            # Create the tables that vector search engine expects
            conn.execute("""
                CREATE TABLE memory_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    timestamp DATETIME,
                    conversation_id TEXT,
                    chunk_type TEXT DEFAULT 'text',
                    metadata_json TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE memory_embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    embedding_data BLOB,
                    embedding_model TEXT,
                    embedding_timestamp DATETIME,
                    FOREIGN KEY(chunk_id) REFERENCES memory_chunks(chunk_id)
                )
            """)
            
            for i in range(100):
                chunk_id = f"chunk_{i:03d}"
                content = f"Test content {i}"
                
                conn.execute("""
                    INSERT INTO memory_chunks (chunk_id, content, timestamp, conversation_id, chunk_type)
                    VALUES (?, ?, '2024-01-01T10:00:00', 'conv_1', 'text')
                """, (chunk_id, content))
                
                embedding = np.random.random(384).astype(np.float32)
                embedding_blob = sqlite3.Binary(pickle.dumps(embedding))
                
                conn.execute("""
                    INSERT INTO memory_embeddings (chunk_id, embedding_data, embedding_model, embedding_timestamp)
                    VALUES (?, ?, 'test-model', '2024-01-01T10:00:00')
                """, (chunk_id, embedding_blob))
            
            conn.commit()
        
        engine = VectorSearchEngine(
            db_path=str(db_path),
            embedding_service=embedding_service,
            index_dir=str(Path(temp_dir) / "indices")
        )
        await engine.initialize()
        await engine.build_index()
        
        # Test batch search with 10 queries
        queries = [f"query {i}" for i in range(10)]
        
        start_time = time.time()
        batch_results = await engine.batch_search(queries)
        batch_time = time.time() - start_time
        
        assert len(batch_results) == 10
        assert batch_time < 5.0  # Should complete batch quickly
        
        # Verify each batch result has results
        for i, results in enumerate(batch_results):
            assert len(results) > 0, f"Query {i} returned no results"
            assert all(isinstance(r, SearchResult) for r in results), f"Query {i} returned invalid result types"
        
        # Test that batch search is functional (not performance-dependent comparison)
        sequential_results = []
        for query in queries:
            results = await engine.search(query)
            sequential_results.append(results)
        
        # Verify sequential and batch results are consistent
        assert len(sequential_results) == len(batch_results)
        for i in range(len(queries)):
            assert len(sequential_results[i]) == len(batch_results[i]), f"Result count mismatch for query {i}"
        
        await engine.cleanup()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 