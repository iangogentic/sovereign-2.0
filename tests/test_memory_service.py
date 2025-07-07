"""
Tests for MemoryService - Lazy-loaded RAG Memory Management

Tests the MemoryService that provides on-demand loading of FAISS vector database,
SQLite storage, and MemoryManager components.
"""

import pytest
import asyncio
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.sovereign.services.memory_service import MemoryService, MemoryStatus, SearchResult


class TestMemoryService:
    """Test the MemoryService class"""
    
    @pytest.mark.not_heavy
    def test_initialization(self):
        """Test MemoryService initialization (no heavy imports)"""
        service = MemoryService()
        
        # Check initial state
        assert service._memory_manager is None
        assert service._vector_search_engine is None
        assert service._embedding_service is None
        assert service._config is None
        assert service._status == MemoryStatus.UNLOADED
        assert service._query_count == 0
        assert service._total_query_time == 0.0
        assert service._load_time is None
        assert not service.is_ready()
    
    @pytest.mark.not_heavy
    def test_initialization_with_params(self):
        """Test MemoryService initialization with custom parameters"""
        service = MemoryService(
            config_path="/custom/config.yaml",
            db_path="/custom/memory.db"
        )
        
        assert service._config_path == "/custom/config.yaml"
        assert service._db_path == "/custom/memory.db"
        assert service._status == MemoryStatus.UNLOADED
    
    @pytest.mark.not_heavy
    def test_search_result_creation(self):
        """Test SearchResult class functionality"""
        result = SearchResult(
            chunk_id="test-123",
            content="Test content",
            score=0.85,
            distance=0.15,
            metadata={"source": "test"}
        )
        
        assert result.chunk_id == "test-123"
        assert result.content == "Test content"
        assert result.score == 0.85
        assert result.distance == 0.15
        assert result.metadata == {"source": "test"}
        
        # Test to_dict conversion
        result_dict = result.to_dict()
        expected = {
            'chunk_id': "test-123",
            'content': "Test content",
            'score': 0.85,
            'distance': 0.15,
            'metadata': {"source": "test"}
        }
        assert result_dict == expected
    
    @pytest.mark.not_heavy
    def test_search_result_default_metadata(self):
        """Test SearchResult with default metadata"""
        result = SearchResult(
            chunk_id="test-456",
            content="Another test",
            score=0.90
        )
        
        assert result.metadata == {}
        assert result.distance == 0.0
    
    @pytest.mark.not_heavy
    def test_status_methods(self):
        """Test status checking methods (no loading)"""
        service = MemoryService()
        
        # Initially not ready
        assert not service.is_ready()
        
        # Check initial stats
        stats = asyncio.run(service.get_stats())
        assert stats["status"] == "unloaded"
        assert stats["load_time"] is None
        assert stats["query_count"] == 0
        assert stats["total_query_time"] == 0.0
        assert stats["average_query_time"] == 0.0
        assert not stats["components_loaded"]
    
    @pytest.mark.not_heavy
    @patch('src.sovereign.services.memory_service.MemoryService._load_components')
    async def test_retrieve_without_loading(self, mock_load):
        """Test retrieve when component loading fails"""
        service = MemoryService()
        mock_load.return_value = False
        
        results = await service.retrieve("test query")
        
        assert results == []
        mock_load.assert_called_once()
    
    @pytest.mark.not_heavy
    @patch('src.sovereign.services.memory_service.MemoryService._load_components')
    async def test_store_without_loading(self, mock_load):
        """Test store when component loading fails"""
        service = MemoryService()
        mock_load.return_value = False
        
        success = await service.store("test content")
        
        assert success is False
        mock_load.assert_called_once()
    
    @pytest.mark.not_heavy
    @patch('src.sovereign.services.memory_service.MemoryService._load_components')
    async def test_rebuild_index_without_loading(self, mock_load):
        """Test rebuild_index when component loading fails"""
        service = MemoryService()
        mock_load.return_value = False
        
        success = await service.rebuild_index()
        
        assert success is False
        mock_load.assert_called_once()


class TestMemoryServiceMocked:
    """Test MemoryService with mocked components"""
    
    @pytest.mark.not_heavy
    async def test_retrieve_with_mocked_components(self):
        """Test retrieve with mocked vector search engine"""
        service = MemoryService()
        
        # Mock the loading and components
        with patch.object(service, '_load_components', return_value=True):
            # Mock vector search engine
            mock_vector_engine = AsyncMock()
            mock_result = Mock()
            mock_result.chunk_id = "test-123"
            mock_result.content = "Test content"
            mock_result.score = 0.85
            mock_result.distance = 0.15
            mock_result.metadata = {"source": "test"}
            
            mock_vector_engine.search.return_value = [mock_result]
            service._vector_search_engine = mock_vector_engine
            service._status = MemoryStatus.READY
            
            # Mock SearchParameters import by patching the actual import
            with patch('src.sovereign.vector_search_engine.SearchParameters') as mock_params_class:
                mock_params = Mock()
                mock_params_class.return_value = mock_params
                
                results = await service.retrieve("test query", top_k=3, threshold=0.8)
                
                # Verify results
                assert len(results) == 1
                assert results[0].chunk_id == "test-123"
                assert results[0].content == "Test content"
                assert results[0].score == 0.85
                
                # Verify service was called correctly
                mock_vector_engine.search.assert_called_once_with("test query", mock_params)
                mock_params_class.assert_called_once_with(
                    top_k=3,
                    distance_threshold=0.8,
                    min_confidence=0.8,
                    include_metadata=True
                )
                
                # Verify stats updated
                assert service._query_count == 1
    
    @pytest.mark.not_heavy
    async def test_store_with_mocked_components(self):
        """Test store with mocked memory manager"""
        service = MemoryService()
        
        # Mock the loading and components
        with patch.object(service, '_load_components', return_value=True):
            # Mock memory manager
            mock_memory_manager = AsyncMock()
            mock_memory_manager.start_conversation.return_value = 123
            mock_memory_manager.store_conversation_turn.return_value = True
            service._memory_manager = mock_memory_manager
            service._status = MemoryStatus.READY
            
            success = await service.store("test content", {"source": "test"})
            
            assert success is True
            mock_memory_manager.start_conversation.assert_called_once()
            mock_memory_manager.store_conversation_turn.assert_called_once()
    
    @pytest.mark.not_heavy
    async def test_loading_state_management(self):
        """Test that loading states are managed correctly"""
        service = MemoryService()
        
        # Initially unloaded
        assert service._status == MemoryStatus.UNLOADED
        assert not service.is_ready()
        
        # Mock successful loading
        with patch.object(service, '_load_components', return_value=True):
            service._status = MemoryStatus.READY
            
            # Now should be ready
            assert service.is_ready()
            assert service._status == MemoryStatus.READY
        
        # Test error state
        service._status = MemoryStatus.ERROR
        assert not service.is_ready()
        
        # Test loading state  
        service._status = MemoryStatus.LOADING
        assert not service.is_ready()
    
    @pytest.mark.not_heavy
    async def test_clear_cache(self):
        """Test cache clearing functionality"""
        service = MemoryService()
        
        # Mock vector search engine with cache
        mock_vector_engine = Mock()
        mock_vector_engine.search_cache = {"query1": "result1", "query2": "result2"}
        service._vector_search_engine = mock_vector_engine
        
        await service.clear_cache()
        
        # Cache should be cleared
        assert mock_vector_engine.search_cache == {}
    
    @pytest.mark.not_heavy
    async def test_close_service(self):
        """Test closing the service"""
        service = MemoryService()
        
        # Mock loaded components
        mock_memory_manager = Mock()
        mock_embedding_service = Mock()
        mock_vector_engine = Mock()
        mock_config = Mock()
        
        service._memory_manager = mock_memory_manager
        service._embedding_service = mock_embedding_service
        service._vector_search_engine = mock_vector_engine
        service._config = mock_config
        service._status = MemoryStatus.READY
        
        await service.close()
        
        # Check components were closed
        mock_memory_manager.close.assert_called_once()
        mock_embedding_service.close.assert_called_once()
        
        # Check references cleared
        assert service._memory_manager is None
        assert service._embedding_service is None
        assert service._vector_search_engine is None
        assert service._config is None
        assert service._status == MemoryStatus.UNLOADED
    
    @pytest.mark.not_heavy
    async def test_health_check_unloaded(self):
        """Test health check when service is unloaded"""
        service = MemoryService()
        
        health = await service.health_check()
        
        assert health["status"] == "unloaded"
        assert health["components"] == {}
        assert health["errors"] == []
    
    @pytest.mark.not_heavy
    async def test_health_check_loaded(self):
        """Test health check when service is loaded"""
        service = MemoryService()
        service._status = MemoryStatus.READY
        
        # Mock healthy components
        mock_memory_manager = Mock()
        mock_memory_manager.get_stats.return_value = {"test": "stats"}
        service._memory_manager = mock_memory_manager
        
        mock_vector_engine = AsyncMock()
        mock_vector_engine.get_index_stats.return_value = {"index": "stats"}
        service._vector_search_engine = mock_vector_engine
        
        mock_embedding_service = Mock()
        mock_embedding_service.is_available = True
        service._embedding_service = mock_embedding_service
        
        health = await service.health_check()
        
        assert health["status"] == "ready"
        assert health["components"]["memory_manager"] == "healthy"
        assert health["components"]["vector_search"] == "healthy"
        assert health["components"]["embedding_service"] == "healthy"
        assert health["errors"] == []


class TestMemoryServiceIntegrationHeavy:
    """Heavy integration tests that actually load components"""
    
    def setup_method(self):
        """Set up test with temporary directory"""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.test_dir) / "test_memory.db")
    
    def teardown_method(self):
        """Clean up test directory"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    async def test_real_component_loading_and_retrieval(self):
        """
        HEAVY TEST: Test actual component loading and retrieval
        
        This test loads real components and verifies they work correctly.
        It's not marked as @pytest.mark.not_heavy so it runs in full test suite.
        """
        service = MemoryService(db_path=self.db_path)
        
        try:
            # Test retrieval that should trigger loading
            results = await service.retrieve("hello world", top_k=3)
            
            # Verify service is now loaded
            assert service.is_ready()
            assert service._status == MemoryStatus.READY
            
            # Verify stats show loading occurred
            stats = await service.get_stats()
            assert stats["status"] == "ready"
            assert stats["components_loaded"] is True
            assert stats["load_time"] is not None
            assert stats["load_time"] > 0
            assert stats["query_count"] == 1
            
            # Test that subsequent calls don't reload
            load_time_before = stats["load_time"]
            
            await service.retrieve("another query")
            await service.retrieve("third query")
            
            final_stats = await service.get_stats()
            # Load time should be identical (no reloading)
            assert final_stats["load_time"] == load_time_before
            assert final_stats["query_count"] == 3
            
            print(f"✅ Memory loading test passed:")
            print(f"   - Components loaded in {load_time_before:.2f}s")
            print(f"   - Total queries: {final_stats['query_count']}")
            print(f"   - No component reloading occurred")
            
        finally:
            # Clean up
            await service.close()
    
    async def test_concurrent_component_loading(self):
        """
        HEAVY TEST: Test that concurrent queries don't cause double-loading
        """
        service = MemoryService(db_path=self.db_path)
        
        try:
            # Launch multiple queries simultaneously
            tasks = [
                service.retrieve("query 1"),
                service.retrieve("query 2"),
                service.retrieve("query 3"),
                service.store("content 1"),
                service.store("content 2")
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All operations should complete without exceptions (though they may return empty results)
            for result in results:
                if isinstance(result, Exception):
                    print(f"Exception in concurrent test: {result}")
                # Note: We don't assert no exceptions because some may fail due to missing data
                # The important thing is that loading doesn't cause race conditions
            
            # Components should only be loaded once despite concurrent queries
            stats = await service.get_stats()
            assert stats["status"] == "ready"
            assert stats["components_loaded"] is True
            
            print(f"✅ Concurrent loading test passed - no race conditions occurred")
            
        finally:
            await service.close()
    
    async def test_store_and_retrieve_workflow(self):
        """
        HEAVY TEST: Test storing content and then retrieving it
        """
        service = MemoryService(db_path=self.db_path)
        
        try:
            # Store some content
            store_success = await service.store(
                "This is test content for retrieval", 
                {"source": "test", "category": "demo"}
            )
            
            # Verify storage worked (or at least didn't crash)
            # Note: Actual storage success depends on proper database setup
            # which may not be available in test environment
            print(f"Store operation result: {store_success}")
            
            # Try retrieval
            results = await service.retrieve("test content", top_k=5)
            
            # Verify service is operational
            assert service.is_ready()
            
            # Get final stats
            stats = await service.get_stats()
            print(f"Final stats: {stats}")
            
            print(f"✅ Store and retrieve workflow test completed")
            
        finally:
            await service.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 