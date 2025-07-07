"""
Tests for MemoryManager - Long-Term Memory System Interface

This module tests the high-level MemoryManager interface including:
- User management
- Conversation lifecycle
- Message operations 
- Document management
- Chunk processing
- Search functionality
- Privacy controls
- Performance monitoring
"""

import pytest
import tempfile
import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.sovereign.config import Config
from src.sovereign.memory_manager import (
    MemoryManager, ConversationMeta, MessageData, DocumentData, ChunkData,
    SearchResult, MemoryStats, ConversationStatus, MessageSender, SearchMode
)
from src.sovereign.memory_schema import EntityType, AccessAction, FeedbackType


class TestMemoryManager:
    """Test suite for MemoryManager class"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_memory.db"
            yield str(db_path)
    
    @pytest.fixture
    def config(self, temp_db):
        """Create test configuration"""
        config = Config()
        config.database.db_path = temp_db
        config.database.enable_rag = True
        return config
    
    @pytest.fixture
    def memory_manager(self, config):
        """Create MemoryManager instance"""
        manager = MemoryManager(config)
        yield manager
        manager.close()
    
    def test_memory_manager_initialization(self, memory_manager):
        """Test MemoryManager initialization"""
        assert memory_manager is not None
        assert memory_manager.schema is not None
        assert memory_manager.get_current_user() is not None
        assert memory_manager.get_current_conversation() is None
        
        # Check database was created
        assert Path(memory_manager.db_path).exists()
        
        # Check schema validation
        validation = memory_manager.schema.validate_schema()
        assert validation["valid"] is True
        assert len(validation["issues"]) == 0
    
    def test_user_management(self, memory_manager):
        """Test user creation and management"""
        # Create a new user
        user_id = memory_manager.create_user(
            username="testuser",
            email="test@example.com",
            privacy_level=2,
            settings={"theme": "dark", "notifications": True}
        )
        
        assert user_id > 0
        
        # Test setting current user
        success = memory_manager.set_current_user(user_id)
        assert success is True
        assert memory_manager.get_current_user() == user_id
        
        # Test setting invalid user
        invalid_success = memory_manager.set_current_user(99999)
        assert invalid_success is False
        
        # Test duplicate username
        with pytest.raises(Exception):
            memory_manager.create_user(username="testuser")
    
    def test_conversation_lifecycle(self, memory_manager):
        """Test conversation creation and management"""
        # Start a new conversation
        conversation_id = memory_manager.start_conversation(
            title="Test Conversation",
            privacy_level=1,
            context_window_size=5
        )
        
        assert conversation_id > 0
        assert memory_manager.get_current_conversation() == conversation_id
        
        # Get conversation metadata
        conversation = memory_manager.get_conversation(conversation_id)
        assert conversation is not None
        assert conversation.title == "Test Conversation"
        assert conversation.privacy_level == 1
        assert conversation.context_window_size == 5
        assert conversation.status == ConversationStatus.ACTIVE
        
        # Test conversation setting
        success = memory_manager.set_current_conversation(conversation_id)
        assert success is True
        
        # Test invalid conversation
        invalid_success = memory_manager.set_current_conversation(99999)
        assert invalid_success is False
        
        # Close conversation
        close_success = memory_manager.close_conversation(conversation_id)
        assert close_success is True
        
        # Verify conversation is closed
        updated_conversation = memory_manager.get_conversation(conversation_id)
        assert updated_conversation.status == ConversationStatus.COMPLETED
        assert updated_conversation.ended_at is not None
    
    def test_conversation_listing(self, memory_manager):
        """Test listing conversations"""
        import time
        
        # Create multiple conversations with small delays to ensure different timestamps
        conv1_id = memory_manager.start_conversation("Conversation 1")
        time.sleep(0.1)
        conv2_id = memory_manager.start_conversation("Conversation 2")
        time.sleep(0.1)
        conv3_id = memory_manager.start_conversation("Conversation 3")
        
        # Close one conversation
        memory_manager.close_conversation(conv2_id)
        
        # List all conversations
        all_conversations = memory_manager.list_conversations()
        assert len(all_conversations) == 3
        
        # List active conversations
        active_conversations = memory_manager.list_conversations(status=ConversationStatus.ACTIVE)
        assert len(active_conversations) == 2
        
        # List completed conversations
        completed_conversations = memory_manager.list_conversations(status=ConversationStatus.COMPLETED)
        assert len(completed_conversations) == 1
        
        # Test conversation ordering (most recent first)
        # Since conversations are ordered by started_at DESC, the most recent (conv3) should be first
        assert all_conversations[0].id == conv3_id
        assert all_conversations[1].id == conv2_id
        assert all_conversations[2].id == conv1_id
    
    def test_message_operations(self, memory_manager):
        """Test message creation and retrieval"""
        # Start a conversation
        conversation_id = memory_manager.start_conversation("Message Test")
        
        # Add messages
        msg1_id = memory_manager.add_message(
            message="Hello, this is a test message",
            sender=MessageSender.USER,
            conversation_id=conversation_id
        )
        
        msg2_id = memory_manager.add_message(
            message="This is an assistant response",
            sender=MessageSender.ASSISTANT,
            conversation_id=conversation_id,
            model_used="test-model",
            response_time=1.5,
            token_count=25
        )
        
        assert msg1_id > 0
        assert msg2_id > 0
        
        # Get individual messages
        message1 = memory_manager.get_message(msg1_id)
        assert message1 is not None
        assert message1.message == "Hello, this is a test message"
        assert message1.sender == MessageSender.USER
        assert message1.conversation_id == conversation_id
        
        message2 = memory_manager.get_message(msg2_id)
        assert message2 is not None
        assert message2.message == "This is an assistant response"
        assert message2.sender == MessageSender.ASSISTANT
        assert message2.model_used == "test-model"
        assert message2.response_time == 1.5
        assert message2.token_count == 25
        
        # Get conversation messages
        messages = memory_manager.get_conversation_messages(conversation_id)
        assert len(messages) == 2
        assert messages[0].id == msg1_id
        assert messages[1].id == msg2_id
        
        # Test message with retrieved chunks
        msg3_id = memory_manager.add_message(
            message="Response with context",
            sender=MessageSender.ASSISTANT,
            retrieved_chunk_ids=[1, 2, 3]
        )
        
        message3 = memory_manager.get_message(msg3_id)
        assert message3.retrieved_chunk_ids == [1, 2, 3]
    
    def test_context_window(self, memory_manager):
        """Test context window management"""
        # Start conversation with small context window
        conversation_id = memory_manager.start_conversation(
            "Context Test",
            context_window_size=3
        )
        
        # Add more messages than context window size
        message_ids = []
        for i in range(5):
            msg_id = memory_manager.add_message(
                message=f"Message {i+1}",
                sender=MessageSender.USER if i % 2 == 0 else MessageSender.ASSISTANT
            )
            message_ids.append(msg_id)
        
        # Get context window
        context = memory_manager.get_context_window(conversation_id)
        assert len(context) == 3  # Should be limited by context window size
        
        # Should contain the most recent messages (most recent first)
        assert context[0].message == "Message 5"
        assert context[1].message == "Message 4"
        assert context[2].message == "Message 3"
        
        # Test with custom window size
        larger_context = memory_manager.get_context_window(conversation_id, window_size=4)
        assert len(larger_context) == 4
        assert larger_context[0].message == "Message 5"
    
    def test_document_management(self, memory_manager):
        """Test document creation and retrieval"""
        # Add a document
        doc_id = memory_manager.add_document(
            title="Test Document",
            content="This is a test document with some content for testing purposes.",
            source="test_source",
            document_type="text",
            privacy_level=2
        )
        
        assert doc_id > 0
        
        # Get document
        document = memory_manager.get_document(doc_id)
        assert document is not None
        assert document.title == "Test Document"
        assert document.content == "This is a test document with some content for testing purposes."
        assert document.source == "test_source"
        assert document.document_type == "text"
        assert document.privacy_level == 2
        assert document.file_hash is not None
        
        # Test duplicate document (same content)
        duplicate_id = memory_manager.add_document(
            title="Duplicate Document",
            content="This is a test document with some content for testing purposes.",
            source="different_source"
        )
        
        assert duplicate_id == doc_id  # Should return existing document ID
        
        # List documents
        documents = memory_manager.list_documents()
        assert len(documents) >= 1
        assert any(doc.id == doc_id for doc in documents)
        
        # Delete document
        delete_success = memory_manager.delete_document(doc_id)
        assert delete_success is True
        
        # Verify document is soft deleted
        deleted_document = memory_manager.get_document(doc_id)
        assert deleted_document is None
    
    def test_chunk_operations(self, memory_manager):
        """Test chunk creation and management"""
        # Create a document
        doc_id = memory_manager.add_document(
            title="Chunk Test Document",
            content="This is a longer document that should be split into multiple chunks. " * 50,
            source="test"
        )
        
        # Create chunks
        chunk_ids = memory_manager.create_chunks(
            document_id=doc_id,
            chunk_size=200,
            overlap_size=50
        )
        
        assert len(chunk_ids) > 1
        
        # Get individual chunk
        chunk = memory_manager.get_chunk(chunk_ids[0])
        assert chunk is not None
        assert chunk.document_id == doc_id
        assert chunk.chunk_index == 0
        assert len(chunk.text) <= 200
        
        # Get all chunks for document
        document_chunks = memory_manager.get_document_chunks(doc_id)
        assert len(document_chunks) == len(chunk_ids)
        
        # Verify chunk ordering
        for i, chunk in enumerate(document_chunks):
            assert chunk.chunk_index == i
        
        # Test chunk with overlap
        if len(document_chunks) > 1:
            assert document_chunks[0].overlap_next > 0
            assert document_chunks[1].overlap_prev > 0
    
    def test_embedding_operations(self, memory_manager):
        """Test embedding storage and retrieval"""
        # Create a document and chunks
        doc_id = memory_manager.add_document(
            title="Embedding Test",
            content="This is a longer piece of content for embedding testing that should be sufficient to create chunks with the specified parameters.",
            source="test"
        )
        
        chunk_ids = memory_manager.create_chunks(doc_id, chunk_size=100, overlap_size=20)
        chunk_id = chunk_ids[0]
        
        # Generate and store embedding
        embedding = memory_manager.generate_embedding("test text")
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        
        # Store embedding
        store_success = memory_manager.store_embedding(chunk_id, embedding)
        assert store_success is True
        
        # Retrieve embedding
        retrieved_embedding = memory_manager.get_embedding(chunk_id)
        assert retrieved_embedding is not None
        assert np.array_equal(embedding, retrieved_embedding)
        
        # Test non-existent chunk
        non_existent_embedding = memory_manager.get_embedding(99999)
        assert non_existent_embedding is None
    
    def test_keyword_search(self, memory_manager):
        """Test keyword search functionality"""
        # Create test documents
        doc1_id = memory_manager.add_document(
            title="Python Programming",
            content="Python is a powerful programming language used for web development, data science, and artificial intelligence.",
            source="test1"
        )
        
        doc2_id = memory_manager.add_document(
            title="Machine Learning",
            content="Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data.",
            source="test2"
        )
        
        # Create chunks for searching
        memory_manager.create_chunks(doc1_id, chunk_size=200, overlap_size=50)
        memory_manager.create_chunks(doc2_id, chunk_size=200, overlap_size=50)
        
        # Perform keyword search
        results = memory_manager.search_keyword("artificial intelligence", k=5)
        
        # Should find results containing the keywords
        assert len(results) > 0
        
        # Check result structure
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.chunk_id > 0
            assert result.similarity_score > 0
            assert len(result.chunk_text) > 0
            assert result.document_id > 0
    
    def test_semantic_search_placeholder(self, memory_manager):
        """Test semantic search (placeholder implementation)"""
        # This test ensures the method exists and handles gracefully
        results = memory_manager.search_semantic("test query", k=3)
        assert isinstance(results, list)
        assert len(results) == 0  # Placeholder returns empty list
    
    def test_memory_statistics(self, memory_manager):
        """Test memory statistics collection"""
        # Create some test data
        conversation_id = memory_manager.start_conversation("Stats Test")
        memory_manager.add_message("Test message", MessageSender.USER)
        
        doc_id = memory_manager.add_document(
            title="Stats Document",
            content="Content for statistics testing.",
            source="stats_test"
        )
        
        chunk_ids = memory_manager.create_chunks(doc_id, chunk_size=50)
        
        # Get statistics
        stats = memory_manager.get_memory_stats()
        
        assert isinstance(stats, MemoryStats)
        assert stats.total_conversations >= 1
        assert stats.total_messages >= 1
        assert stats.total_documents >= 1
        assert stats.total_chunks >= len(chunk_ids)
        assert stats.db_size_mb > 0
        assert stats.last_updated is not None
    
    def test_health_check(self, memory_manager):
        """Test system health check"""
        health = memory_manager.health_check()
        
        assert isinstance(health, dict)
        assert "status" in health
        assert "database_connected" in health
        assert "schema_valid" in health
        assert "stats" in health
        assert "issues" in health
        
        # Should be healthy for working system
        assert health["status"] == "healthy"
        assert health["database_connected"] is True
        assert health["schema_valid"] is True
        assert len(health["issues"]) == 0
    
    def test_context_manager(self, temp_db):
        """Test MemoryManager as context manager"""
        config = Config()
        config.database.db_path = temp_db
        
        # Test context manager usage
        with MemoryManager(config) as manager:
            assert manager is not None
            
            # Create some data
            conversation_id = manager.start_conversation("Context Manager Test")
            assert conversation_id > 0
        
        # Manager should be closed after context
        # (Testing this is tricky since we can't easily check internal state)
    
    def test_access_logging(self, memory_manager):
        """Test access logging functionality"""
        # Create a conversation to generate access logs
        conversation_id = memory_manager.start_conversation("Access Log Test")
        
        # Add some messages
        memory_manager.add_message("Test message", MessageSender.USER)
        
        # Check access logs were created
        conn = memory_manager.schema.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM access_logs")
        log_count = cursor.fetchone()[0]
        
        assert log_count > 0
        
        # Check log structure
        cursor.execute("""
            SELECT user_id, entity_type, entity_id, action, success
            FROM access_logs
            ORDER BY created_at DESC
            LIMIT 1
        """)
        
        log_row = cursor.fetchone()
        assert log_row is not None
        assert log_row[0] == memory_manager.get_current_user()
        assert log_row[4] == 1  # success = True
    
    def test_error_handling(self, memory_manager):
        """Test error handling in various scenarios"""
        # Test adding message without conversation
        memory_manager._current_conversation_id = None
        
        with pytest.raises(ValueError):
            memory_manager.add_message("Test", MessageSender.USER)
        
        # Test getting non-existent conversation
        non_existent = memory_manager.get_conversation(99999)
        assert non_existent is None
        
        # Test getting non-existent message
        non_existent_msg = memory_manager.get_message(99999)
        assert non_existent_msg is None
        
        # Test getting non-existent document
        non_existent_doc = memory_manager.get_document(99999)
        assert non_existent_doc is None
        
        # Test creating chunks for non-existent document
        with pytest.raises(ValueError):
            memory_manager.create_chunks(99999)
    
    def test_configuration_integration(self, memory_manager):
        """Test integration with configuration system"""
        # Test that configuration is properly loaded
        assert memory_manager.config is not None
        assert memory_manager.config.database.enable_rag is True
        
        # Test database path from config
        assert str(memory_manager.db_path) == memory_manager.config.database.db_path
    
    def test_threading_safety(self, memory_manager):
        """Test thread safety of operations"""
        import threading
        import concurrent.futures
        
        results = []
        errors = []
        
        def create_conversation(thread_id):
            try:
                conv_id = memory_manager.start_conversation(f"Thread {thread_id}")
                results.append(conv_id)
                return conv_id
            except Exception as e:
                errors.append(e)
                return None
        
        # Create multiple conversations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_conversation, i) for i in range(10)]
            concurrent.futures.wait(futures)
        
        # Should have created conversations without errors
        assert len(errors) == 0
        assert len(results) == 10
        assert len(set(results)) == 10  # All unique IDs
    
    def test_large_content_handling(self, memory_manager):
        """Test handling of large content"""
        # Create a large document
        large_content = "This is a test sentence. " * 1000  # ~25,000 characters
        
        doc_id = memory_manager.add_document(
            title="Large Document",
            content=large_content,
            source="large_test"
        )
        
        assert doc_id > 0
        
        # Create chunks
        chunk_ids = memory_manager.create_chunks(doc_id, chunk_size=500, overlap_size=50)
        
        # Should create multiple chunks
        assert len(chunk_ids) > 10
        
        # Verify chunks are reasonable size
        for chunk_id in chunk_ids[:5]:  # Check first 5 chunks
            chunk = memory_manager.get_chunk(chunk_id)
            assert chunk is not None
            assert len(chunk.text) <= 500
            assert len(chunk.text) > 50  # Not too small
    
    def test_performance_monitoring(self, memory_manager):
        """Test performance monitoring capabilities"""
        # Create some load
        conversation_id = memory_manager.start_conversation("Performance Test")
        
        # Add multiple messages
        for i in range(10):
            memory_manager.add_message(f"Message {i}", MessageSender.USER)
        
        # Create document and chunks
        doc_id = memory_manager.add_document(
            title="Performance Document",
            content="Content for performance testing. " * 100,
            source="perf_test"
        )
        
        chunk_ids = memory_manager.create_chunks(doc_id, chunk_size=200)
        
        # Check that operations completed in reasonable time
        # (This is implicitly tested by the operations completing)
        
        # Verify stats are updated
        stats = memory_manager.get_memory_stats()
        assert stats.total_conversations >= 1
        assert stats.total_messages >= 10
        assert stats.total_documents >= 1
        assert stats.total_chunks >= len(chunk_ids)
    
    def test_cleanup_and_close(self, memory_manager):
        """Test proper cleanup and resource management"""
        # Create some data
        conversation_id = memory_manager.start_conversation("Cleanup Test")
        memory_manager.add_message("Test message", MessageSender.USER)
        
        # Close manager
        memory_manager.close()
        
        # Verify database file still exists
        assert Path(memory_manager.db_path).exists()
        
        # Attempting to use closed manager should handle gracefully
        # (specific behavior depends on implementation) 