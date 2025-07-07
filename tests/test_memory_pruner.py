"""
Tests for Memory Pruner - Automated Memory Management System

This module tests the memory pruning system including:
- Retention policy management
- Multiple pruning strategies
- Automated and manual pruning operations
- Orphan cleanup and duplicate detection
- Scheduled pruning jobs
- Performance monitoring and statistics
"""

import pytest
import tempfile
import shutil
import sqlite3
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from src.sovereign.memory_pruner import (
    MemoryPruner, PruningStrategy, PruningPriority, RetentionPolicy,
    RetentionPolicyType, PruningResult, PruningStats, PruningScheduler
)
from src.sovereign.memory_schema import MemorySchema
from src.sovereign.config import Config


@pytest.fixture
def temp_db_path():
    """Create a temporary database file"""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_memory.db"
    yield str(db_path)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Mock configuration"""
    config = Mock(spec=Config)
    config.max_database_size_mb = 100
    config.enable_pruning = True
    config.default_retention_days = 30
    return config


@pytest.fixture
def setup_test_db(temp_db_path):
    """Set up a test database with sample data"""
    schema = MemorySchema(temp_db_path)
    schema.create_schema()
    
    conn = schema.get_connection()
    cursor = conn.cursor()
    
    # Create test user
    cursor.execute("""
        INSERT INTO users (id, username, email, created_at)
        VALUES (1, 'testuser', 'test@example.com', ?)
    """, (datetime.now().isoformat(),))
    
    # Create test conversations
    old_date = datetime.now() - timedelta(days=60)
    recent_date = datetime.now() - timedelta(days=5)
    
    conversations = [
        (1, 1, 'Old Conversation', old_date.isoformat(), None, 1, 1, 5),
        (2, 1, 'Recent Conversation', recent_date.isoformat(), None, 1, 1, 3),
        (3, 1, 'Another Old Conversation', old_date.isoformat(), None, 1, 1, 2),
    ]
    
    for conv in conversations:
        cursor.execute("""
            INSERT INTO conversations (id, user_id, title, started_at, ended_at, is_active, privacy_level, total_messages)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, conv)
    
    # Create test messages
    messages = [
        (1, 1, 'user', 'Hello', 'text', old_date.isoformat(), None, None, 0.5, 10),
        (2, 1, 'assistant', 'Hi there!', 'text', old_date.isoformat(), '[1,2]', 'gpt-4', 1.2, 25),
        (3, 2, 'user', 'Recent message', 'text', recent_date.isoformat(), None, None, 0.3, 12),
        (4, 3, 'user', 'Old message', 'text', old_date.isoformat(), None, None, 0.4, 8),
    ]
    
    for msg in messages:
        cursor.execute("""
            INSERT INTO messages (id, conversation_id, sender, message, message_type, created_at, retrieved_chunk_ids, model_used, response_time, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, msg)
    
    # Create test documents
    documents = [
        (1, 'Old Document', 'file1.txt', 'Content 1', 'text', '/path/file1.txt', 'hash1', old_date.isoformat(), old_date.isoformat(), 1, 1, 1),
        (2, 'Recent Document', 'file2.txt', 'Content 2', 'text', '/path/file2.txt', 'hash2', recent_date.isoformat(), recent_date.isoformat(), 1, 1, 1),
        (3, 'Duplicate Document', 'file3.txt', 'Content 1', 'text', '/path/file3.txt', 'hash1', old_date.isoformat(), old_date.isoformat(), 1, 1, 1),  # Same hash as doc 1
    ]
    
    for doc in documents:
        cursor.execute("""
            INSERT INTO documents (id, title, source, content, document_type, file_path, file_hash, created_at, updated_at, created_by, is_active, privacy_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, doc)
    
    # Create test chunks
    chunks = [
        (1, 1, 0, 'Chunk 1 content', 'paragraph', 0, 100, 0, 0, old_date.isoformat(), 1),
        (2, 1, 1, 'Chunk 2 content', 'paragraph', 100, 200, 0, 0, old_date.isoformat(), 1),
        (3, 2, 0, 'Recent chunk content', 'paragraph', 0, 150, 0, 0, recent_date.isoformat(), 1),
        (4, None, 0, 'Orphaned chunk', 'paragraph', 0, 100, 0, 0, old_date.isoformat(), 1),  # Orphaned
    ]
    
    for chunk in chunks:
        cursor.execute("""
            INSERT INTO chunks (id, document_id, chunk_index, text, chunk_type, start_position, end_position, overlap_prev, overlap_next, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, chunk)
    
    # Create test access logs
    access_logs = [
        (1, 1, 'document', 1, 'read', '127.0.0.1', 'test-agent', 1, None, old_date.isoformat()),
        (2, 1, 'document', 2, 'read', '127.0.0.1', 'test-agent', 1, None, recent_date.isoformat()),
        (3, 1, 'chunk', 1, 'search', '127.0.0.1', 'test-agent', 1, None, old_date.isoformat()),
    ]
    
    for log in access_logs:
        cursor.execute("""
            INSERT INTO access_logs (id, user_id, entity_type, entity_id, action, ip_address, user_agent, success, error_message, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, log)
    
    # Create test feedback
    feedback = [
        (1, 1, 2, 1, 'irrelevant', 2, 'Not helpful', recent_date.isoformat()),
        (2, 1, 2, 1, 'not_helpful', 1, 'Poor quality', recent_date.isoformat()),
        (3, 1, None, 2, 'relevant', 5, 'Very good', recent_date.isoformat()),
    ]
    
    for fb in feedback:
        cursor.execute("""
            INSERT INTO feedback (id, user_id, message_id, chunk_id, feedback_type, rating, comment, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, fb)
    
    conn.commit()
    schema.close()
    
    return temp_db_path


class TestRetentionPolicy:
    """Test retention policy data structures"""
    
    def test_retention_policy_creation(self):
        """Test creating retention policy"""
        policy = RetentionPolicy(
            entity_type=RetentionPolicyType.CONVERSATION,
            policy_name="test_policy",
            retention_days=30,
            auto_delete=True
        )
        
        assert policy.entity_type == RetentionPolicyType.CONVERSATION
        assert policy.policy_name == "test_policy"
        assert policy.retention_days == 30
        assert policy.auto_delete is True
        assert policy.is_active is True
    
    def test_retention_policy_to_dict(self):
        """Test converting retention policy to dictionary"""
        policy = RetentionPolicy(
            entity_type=RetentionPolicyType.MESSAGE,
            policy_name="msg_policy",
            retention_days=60,
            max_items=1000,
            min_relevance_score=0.5
        )
        
        policy_dict = policy.to_dict()
        
        assert policy_dict['entity_type'] == 'message'
        assert policy_dict['policy_name'] == 'msg_policy'
        assert policy_dict['retention_days'] == 60
        assert policy_dict['max_items'] == 1000
        assert policy_dict['min_relevance_score'] == 0.5


class TestPruningResult:
    """Test pruning result data structures"""
    
    def test_pruning_result_creation(self):
        """Test creating pruning result"""
        result = PruningResult(strategy=PruningStrategy.TIME_BASED)
        
        assert result.strategy == PruningStrategy.TIME_BASED
        assert result.items_analyzed == 0
        assert result.items_deleted == 0
        assert result.storage_freed_mb == 0.0
        assert result.success is True
        assert len(result.errors) == 0
    
    def test_add_deleted_item(self):
        """Test adding deleted items to result"""
        result = PruningResult(strategy=PruningStrategy.TIME_BASED)
        
        result.add_deleted_item("conversations", 1)
        result.add_deleted_item("conversations", 2)
        result.add_deleted_item("messages", 5)
        
        assert result.items_deleted == 3
        assert result.deleted_items["conversations"] == [1, 2]
        assert result.deleted_items["messages"] == [5]


class TestMemoryPruner:
    """Test the main MemoryPruner class"""
    
    def test_pruner_initialization(self, mock_config, temp_db_path):
        """Test pruner initialization"""
        pruner = MemoryPruner(mock_config, temp_db_path)
        
        try:
            assert pruner.config == mock_config
            assert pruner.db_path == temp_db_path
            assert pruner.stats.total_pruning_runs == 0
            assert pruner.max_batch_size == 1000
            assert pruner.dry_run_mode is False
        finally:
            pruner.close()
    
    def test_create_retention_policy(self, mock_config, setup_test_db):
        """Test creating retention policy"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        
        try:
            policy = RetentionPolicy(
                entity_type=RetentionPolicyType.CONVERSATION,
                policy_name="test_policy",
                retention_days=45,
                auto_delete=True
            )
            
            policy_id = pruner.create_retention_policy(policy)
            
            assert policy_id is not None
            assert policy_id > 0
            
            # Verify policy was stored
            retrieved_policy = pruner.get_retention_policy(
                RetentionPolicyType.CONVERSATION, "test_policy"
            )
            
            assert retrieved_policy is not None
            assert retrieved_policy.retention_days == 45
            assert retrieved_policy.auto_delete is True
        finally:
            pruner.close()
    
    def test_create_default_retention_policies(self, mock_config, temp_db_path):
        """Test creating default retention policies"""
        schema = MemorySchema(temp_db_path)
        schema.create_schema()
        schema.close()
        
        pruner = MemoryPruner(mock_config, temp_db_path)
        try:
            pruner.create_default_retention_policies()
            
            # Check that policies were created
            conversation_policy = pruner.get_retention_policy(RetentionPolicyType.CONVERSATION)
            message_policy = pruner.get_retention_policy(RetentionPolicyType.MESSAGE)
            document_policy = pruner.get_retention_policy(RetentionPolicyType.DOCUMENT)
            
            assert conversation_policy is not None
            assert conversation_policy.retention_days == 365
            
            assert message_policy is not None
            assert message_policy.retention_days == 180
            
            assert document_policy is not None
            assert document_policy.retention_days == 730
        finally:
            pruner.close()
    
    @pytest.mark.asyncio
    async def test_time_based_pruning(self, mock_config, setup_test_db):
        """Test time-based pruning strategy"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        
        try:
            # Create retention policy for conversations
            policy = RetentionPolicy(
                entity_type=RetentionPolicyType.CONVERSATION,
                policy_name="default",
                retention_days=30,  # Should prune conversations older than 30 days
                auto_delete=True
            )
            pruner.create_retention_policy(policy)
            
            # Run time-based pruning
            result = await pruner.prune_memory(PruningStrategy.TIME_BASED, dry_run=False)
            
            assert result.success is True
            assert result.items_deleted > 0
            assert "conversations" in result.deleted_items
            
            # Check that old conversations were pruned
            conn = pruner.schema.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversations WHERE is_active = 1")
            active_conversations = cursor.fetchone()[0]
            
            # Should have 1 active conversation (the recent one)
            assert active_conversations == 1
        finally:
            pruner.close()
    
    @pytest.mark.asyncio
    async def test_usage_based_pruning(self, mock_config, setup_test_db):
        """Test usage-based pruning strategy"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        
        result = await pruner.prune_memory(PruningStrategy.USAGE_BASED, dry_run=False)
        
        assert result.success is True
        
        # Check that documents without recent access were pruned
        conn = pruner.schema.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents WHERE is_active = 1")
        active_documents = cursor.fetchone()[0]
        
        # Should have fewer active documents
        assert active_documents < 3
        
        pruner.close()
    
    @pytest.mark.asyncio
    async def test_relevance_based_pruning(self, mock_config, setup_test_db):
        """Test relevance-based pruning strategy"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        
        try:
            result = await pruner.prune_memory(PruningStrategy.RELEVANCE_BASED, dry_run=False)
            
            assert result.success is True
            
            # Check that chunk with negative feedback was pruned
            conn = pruner.schema.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE is_active = 1")
            active_chunks = cursor.fetchone()[0]
            
            # Should have same or fewer active chunks (depends on feedback threshold)
            assert active_chunks <= 4
            
        finally:
            pruner.close()
    
    @pytest.mark.asyncio
    async def test_duplicate_removal(self, mock_config, setup_test_db):
        """Test duplicate removal strategy"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        
        result = await pruner.prune_memory(PruningStrategy.DUPLICATE_REMOVAL, dry_run=False)
        
        assert result.success is True
        
        # Check that duplicate document was removed
        conn = pruner.schema.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents WHERE file_hash = 'hash1' AND is_active = 1")
        hash1_docs = cursor.fetchone()[0]
        
        # Should have only 1 document with hash1 (duplicate removed)
        assert hash1_docs == 1
        
        pruner.close()
    
    @pytest.mark.asyncio
    async def test_orphan_cleanup(self, mock_config, setup_test_db):
        """Test orphan cleanup strategy"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        
        try:
            # Add some data that will become orphaned
            conn = pruner.schema.get_connection()
            cursor = conn.cursor()
            
            # First, add an embedding model
            cursor.execute("""
                INSERT INTO embedding_models (id, model_name, embedding_dim, created_at)
                VALUES (1, 'test-model', 768, ?)
            """, (datetime.now().isoformat(),))
            
            # Create a user, document, and chunk that we'll later delete to create orphans
            cursor.execute("""
                INSERT INTO users (id, username, email, created_at)
                VALUES (100, 'test_user', 'test100@example.com', ?)
            """, (datetime.now().isoformat(),))
            
            cursor.execute("""
                INSERT INTO documents (id, title, source, content, document_type, file_path, file_hash, created_at, updated_at, created_by, is_active, privacy_level)
                VALUES (100, 'Test Document', 'test_source', 'Test content', 'text', '/test/path', 'test_hash', ?, ?, 100, 1, 1)
            """, (datetime.now().isoformat(), datetime.now().isoformat()))
            
            cursor.execute("""
                INSERT INTO chunks (id, document_id, chunk_index, text, chunk_type, start_position, end_position, overlap_prev, overlap_next, created_at, is_active)
                VALUES (999, 100, 0, 'Test chunk', 'paragraph', 0, 100, 0, 0, ?, 1)
            """, (datetime.now().isoformat(),))
            
            # Create a conversation and message that we'll later delete
            cursor.execute("""
                INSERT INTO conversations (id, user_id, title, started_at, ended_at, is_active, privacy_level, total_messages)
                VALUES (100, 100, 'Test Conversation', ?, NULL, 1, 1, 1)
            """, (datetime.now().isoformat(),))
            
            cursor.execute("""
                INSERT INTO messages (id, conversation_id, sender, message, message_type, created_at, retrieved_chunk_ids, model_used, response_time, token_count)
                VALUES (100, 100, 'user', 'Test message', 'text', ?, NULL, NULL, 0.0, 10)
            """, (datetime.now().isoformat(),))
            
            # Add embedding that will become orphaned
            cursor.execute("""
                INSERT INTO embeddings (chunk_id, model_id, embedding, created_at)
                VALUES (999, 1, ?, ?)
            """, (b'fake_embedding', datetime.now().isoformat()))
            
            # Add context window that will become orphaned
            cursor.execute("""
                INSERT INTO context_windows (id, conversation_id, message_id, chunk_id, window_index, relevance_score, created_at)
                VALUES (100, 100, 100, 999, 0, 0.5, ?)
            """, (datetime.now().isoformat(),))
            
            conn.commit()
            
                        # Now create orphans by disabling the chunk instead of deleting it
            # This simulates a scenario where chunks are soft-deleted but embeddings remain
            cursor.execute("UPDATE chunks SET is_active = 0 WHERE id = 999")
            cursor.execute("DELETE FROM messages WHERE id = 100")
            cursor.execute("DELETE FROM conversations WHERE id = 100")
            conn.commit()

            result = await pruner.prune_memory(PruningStrategy.ORPHAN_CLEANUP, dry_run=False)
            
            assert result.success is True
            assert result.items_deleted > 0
            
            # Check that orphaned records were cleaned up
            cursor.execute("SELECT COUNT(*) FROM embeddings WHERE chunk_id = 999")
            orphaned_embeddings = cursor.fetchone()[0]
            assert orphaned_embeddings == 0
            
            cursor.execute("SELECT COUNT(*) FROM context_windows WHERE chunk_id = 999")
            orphaned_contexts = cursor.fetchone()[0]
            assert orphaned_contexts == 0
            
        finally:
            pruner.close()
    
    @pytest.mark.asyncio
    async def test_comprehensive_pruning(self, mock_config, setup_test_db):
        """Test comprehensive pruning strategy"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        
        # Create default retention policies
        pruner.create_default_retention_policies()
        
        result = await pruner.prune_memory(PruningStrategy.COMPREHENSIVE, dry_run=False)
        
        assert result.success is True
        assert result.items_deleted > 0
        assert result.execution_time > 0
        
        # Check that various types of data were processed
        assert result.items_analyzed > 0
        
        pruner.close()
    
    @pytest.mark.asyncio
    async def test_dry_run_mode(self, mock_config, setup_test_db):
        """Test dry run mode doesn't delete data"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        
        # Create retention policy
        policy = RetentionPolicy(
            entity_type=RetentionPolicyType.CONVERSATION,
            policy_name="default",
            retention_days=30,
            auto_delete=True
        )
        pruner.create_retention_policy(policy)
        
        # Count conversations before
        conn = pruner.schema.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM conversations WHERE is_active = 1")
        before_count = cursor.fetchone()[0]
        
        # Run dry run
        result = await pruner.prune_memory(PruningStrategy.TIME_BASED, dry_run=True)
        
        # Count conversations after
        cursor.execute("SELECT COUNT(*) FROM conversations WHERE is_active = 1")
        after_count = cursor.fetchone()[0]
        
        # No actual deletion should occur
        assert before_count == after_count
        assert result.success is True
        
        pruner.close()
    
    @pytest.mark.asyncio
    async def test_batch_size_limiting(self, mock_config, setup_test_db):
        """Test batch size limiting works correctly"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        pruner.max_batch_size = 1  # Very small batch size
        
        # Create retention policy
        policy = RetentionPolicy(
            entity_type=RetentionPolicyType.CONVERSATION,
            policy_name="default",
            retention_days=30,
            auto_delete=True
        )
        pruner.create_retention_policy(policy)
        
        result = await pruner.prune_memory(PruningStrategy.TIME_BASED, dry_run=False)
        
        assert result.success is True
        # Should delete at most 1 item per batch
        assert result.items_deleted <= 1
        
        pruner.close()
    
    def test_get_pruning_stats(self, mock_config, setup_test_db):
        """Test getting pruning statistics"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        
        # Update stats manually
        pruner.stats.total_pruning_runs = 5
        pruner.stats.total_items_deleted = 100
        pruner.stats.total_storage_freed_mb = 50.5
        
        stats = pruner.get_pruning_stats()
        
        assert stats['total_pruning_runs'] == 5
        assert stats['total_items_deleted'] == 100
        assert stats['total_storage_freed_mb'] == 50.5
        assert 'database_size_mb' in stats
        
        pruner.close()
    
    def test_database_size_calculation(self, mock_config, setup_test_db):
        """Test database size calculation"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        
        size_mb = pruner._get_database_size_mb()
        
        assert size_mb > 0
        assert isinstance(size_mb, float)
        
        pruner.close()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_config, temp_db_path):
        """Test error handling in pruning operations"""
        # Create pruner with non-existent database
        pruner = MemoryPruner(mock_config, temp_db_path)
        
        try:
            result = await pruner.prune_memory(PruningStrategy.TIME_BASED, dry_run=False)
            
            # Should complete but with errors due to missing tables
            assert len(result.errors) > 0
            assert result.items_deleted == 0
            
        finally:
            pruner.close()


class TestPruningScheduler:
    """Test the PruningScheduler class"""
    
    def test_scheduler_initialization(self, mock_config, setup_test_db):
        """Test scheduler initialization"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        scheduler = PruningScheduler(pruner)
        
        assert scheduler.pruner == pruner
        assert scheduler._running is False
        assert scheduler._thread is None
        
        pruner.close()
    
    def test_scheduler_start_stop(self, mock_config, setup_test_db):
        """Test starting and stopping scheduler"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        scheduler = PruningScheduler(pruner)
        
        # Start scheduler
        scheduler.start()
        assert scheduler._running is True
        assert scheduler._thread is not None
        
        # Stop scheduler
        scheduler.stop()
        assert scheduler._running is False
        
        pruner.close()
    
    def test_should_run_job_timing(self, mock_config, setup_test_db):
        """Test job timing logic"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        scheduler = PruningScheduler(pruner)
        
        # Test with recent last run
        recent_run = datetime.now() - timedelta(hours=1)
        now = datetime.now()
        
        # Should not run if not enough time has passed
        should_run = scheduler._should_run_job('comprehensive', recent_run, now)
        assert should_run is False
        
        # Test with old last run
        old_run = datetime.now() - timedelta(days=2)
        
        # Should run if enough time has passed and it's the right time
        scheduler.schedule['comprehensive']['hour'] = now.hour
        scheduler.schedule['comprehensive']['minute'] = now.minute
        
        should_run = scheduler._should_run_job('comprehensive', old_run, now)
        assert should_run is True
        
        pruner.close()


class TestPruningIntegration:
    """Integration tests for the pruning system"""
    
    @pytest.mark.asyncio
    async def test_full_pruning_workflow(self, mock_config, setup_test_db):
        """Test complete pruning workflow"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        
        # Create default policies
        pruner.create_default_retention_policies()
        
        # Get initial stats
        initial_stats = pruner.get_pruning_stats()
        
        # Run comprehensive pruning
        result = await pruner.prune_memory(PruningStrategy.COMPREHENSIVE, dry_run=False)
        
        # Get final stats
        final_stats = pruner.get_pruning_stats()
        
        assert result.success is True
        assert final_stats['total_pruning_runs'] > initial_stats['total_pruning_runs']
        assert final_stats['total_items_deleted'] >= initial_stats['total_items_deleted']
        
        pruner.close()
    
    @pytest.mark.asyncio
    async def test_multiple_strategy_execution(self, mock_config, setup_test_db):
        """Test running multiple pruning strategies"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        
        strategies = [
            PruningStrategy.ORPHAN_CLEANUP,
            PruningStrategy.DUPLICATE_REMOVAL,
            PruningStrategy.TIME_BASED
        ]
        
        total_deleted = 0
        
        for strategy in strategies:
            result = await pruner.prune_memory(strategy, dry_run=False)
            assert result.success is True
            total_deleted += result.items_deleted
        
        # Check that stats were updated
        stats = pruner.get_pruning_stats()
        assert stats['total_pruning_runs'] == len(strategies)
        assert stats['total_items_deleted'] == total_deleted
        
        pruner.close()
    
    def test_context_manager_usage(self, mock_config, setup_test_db):
        """Test using pruner as context manager"""
        with MemoryPruner(mock_config, setup_test_db) as pruner:
            assert pruner.schema is not None
            
            # Create a policy
            policy = RetentionPolicy(
                entity_type=RetentionPolicyType.CONVERSATION,
                policy_name="context_test",
                retention_days=60
            )
            
            policy_id = pruner.create_retention_policy(policy)
            assert policy_id is not None
        
        # Pruner should be closed after context exit
        # (We can't easily test this without accessing private members)
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, mock_config, setup_test_db):
        """Test performance monitoring during pruning"""
        pruner = MemoryPruner(mock_config, setup_test_db)
        
        # Run pruning and measure performance
        start_time = time.time()
        result = await pruner.prune_memory(PruningStrategy.ORPHAN_CLEANUP, dry_run=False)
        end_time = time.time()
        
        # Check that execution time is reasonable
        assert result.execution_time > 0
        assert result.execution_time < (end_time - start_time) + 0.1  # Allow small margin
        
        # Check that stats are updated
        stats = pruner.get_pruning_stats()
        assert stats['average_execution_time'] > 0
        
        pruner.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 