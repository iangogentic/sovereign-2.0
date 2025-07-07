"""
Tests for the Memory Schema module of the Sovereign AI Long-Term Memory system.

Tests cover:
- Database schema creation and validation
- Table structure and relationships
- Indexing and FTS functionality
- Utility functions for embeddings
- Privacy and access control features
- Performance monitoring capabilities
"""

import pytest
import sqlite3
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.sovereign.memory_schema import (
    MemorySchema, EntityType, AccessAction, FeedbackType,
    serialize_embedding, deserialize_embedding, calculate_embedding_norm,
    cosine_similarity, validate_entity_type, validate_access_action,
    validate_feedback_type
)

# Module-level fixtures for shared use
@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)

class TestMemorySchema:
    """Test suite for the MemorySchema class"""
    
    @pytest.fixture
    def schema(self, temp_db_path):
        """Create a MemorySchema instance with temporary database"""
        schema = MemorySchema(temp_db_path)
        yield schema
        schema.close()
    
    def test_schema_initialization(self, temp_db_path):
        """Test schema initialization creates proper paths"""
        schema = MemorySchema(temp_db_path)
        assert schema.db_path == Path(temp_db_path)
        assert schema.CURRENT_VERSION == "1.0.0"
        assert schema._connection is None
        schema.close()
    
    def test_database_connection(self, schema):
        """Test database connection configuration"""
        conn = schema.get_connection()
        assert conn is not None
        assert isinstance(conn, sqlite3.Connection)
        
        # Test connection settings
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        assert cursor.fetchone()[0] == 1
        
        cursor.execute("PRAGMA journal_mode")
        assert cursor.fetchone()[0] == 'wal'
        
        cursor.execute("PRAGMA synchronous")
        assert cursor.fetchone()[0] == 1  # NORMAL
    
    def test_schema_creation(self, schema):
        """Test complete schema creation"""
        result = schema.create_schema()
        assert result is True
        
        # Verify all expected tables exist
        conn = schema.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'users', 'documents', 'chunks', 'embedding_models', 'embeddings',
            'conversations', 'messages', 'context_windows', 'metadata',
            'tags', 'entity_tags', 'access_logs', 'retention_policies',
            'privacy_preferences', 'query_performance', 'feedback',
            'system_metrics', 'schema_versions', 'embedding_cache'
        ]
        
        for table in expected_tables:
            assert table in tables, f"Table {table} was not created"
    
    def test_fts_tables_creation(self, schema):
        """Test FTS5 virtual tables creation"""
        schema.create_schema()
        conn = schema.get_connection()
        cursor = conn.cursor()
        
        # Check FTS tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts'")
        fts_tables = [row[0] for row in cursor.fetchall()]
        
        expected_fts_tables = ['chunks_fts', 'messages_fts', 'documents_fts']
        for table in expected_fts_tables:
            assert table in fts_tables, f"FTS table {table} was not created"
    
    def test_indexes_creation(self, schema):
        """Test database indexes creation"""
        schema.create_schema()
        conn = schema.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
        indexes = [row[0] for row in cursor.fetchall()]
        
        # Check for some key indexes
        expected_indexes = [
            'idx_documents_created_by',
            'idx_chunks_document_id',
            'idx_conversations_user_id',
            'idx_messages_conversation_id',
            'idx_access_logs_user_id'
        ]
        
        for index in expected_indexes:
            assert index in indexes, f"Index {index} was not created"
    
    def test_triggers_creation(self, schema):
        """Test FTS triggers creation"""
        schema.create_schema()
        conn = schema.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
        triggers = [row[0] for row in cursor.fetchall()]
        
        expected_triggers = [
            'chunks_fts_insert', 'chunks_fts_update', 'chunks_fts_delete',
            'messages_fts_insert', 'messages_fts_update', 'messages_fts_delete'
        ]
        
        for trigger in expected_triggers:
            assert trigger in triggers, f"Trigger {trigger} was not created"
    
    def test_schema_version_tracking(self, schema):
        """Test schema version tracking"""
        schema.create_schema()
        
        version = schema.get_schema_version()
        assert version == "1.0.0"
        
        # Test migration check
        assert not schema.needs_migration()
    
    def test_schema_validation(self, schema):
        """Test schema validation functionality"""
        schema.create_schema()
        
        validation_result = schema.validate_schema()
        assert validation_result['valid'] is True
        assert validation_result['version'] == "1.0.0"
        assert len(validation_result['tables']) > 0
        assert len(validation_result['indexes']) > 0
        assert len(validation_result['triggers']) > 0
        assert len(validation_result['issues']) == 0
    
    def test_table_info(self, schema):
        """Test table information retrieval"""
        schema.create_schema()
        
        # Test valid table
        info = schema.get_table_info('users')
        assert 'columns' in info
        assert 'row_count' in info
        assert 'indexes' in info
        assert info['row_count'] == 0  # Initially empty
        
        # Test invalid table
        info = schema.get_table_info('nonexistent_table')
        assert 'error' in info
    
    def test_schema_export(self, schema):
        """Test schema export functionality"""
        schema.create_schema()
        
        ddl = schema.export_schema()
        assert isinstance(ddl, str)
        assert len(ddl) > 0
        assert 'CREATE TABLE' in ddl
        assert 'CREATE INDEX' in ddl
        assert 'CREATE TRIGGER' in ddl
    
    def test_default_user_creation(self, schema):
        """Test default user creation"""
        schema.create_schema()
        
        user_id = schema.create_default_user("test_user")
        assert isinstance(user_id, int)
        assert user_id > 0
        
        # Verify user was created
        conn = schema.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "test_user"
    
    def test_default_embedding_model_creation(self, schema):
        """Test default embedding model creation"""
        schema.create_schema()
        
        model_id = schema.insert_default_embedding_model("test-model")
        assert isinstance(model_id, int)
        assert model_id > 0
        
        # Verify model was created
        conn = schema.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT model_name FROM embedding_models WHERE id = ?", (model_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "test-model"
    
    def test_foreign_key_constraints(self, schema):
        """Test foreign key constraints work properly"""
        schema.create_schema()
        conn = schema.get_connection()
        cursor = conn.cursor()
        
        # Insert parent record
        cursor.execute("INSERT INTO users (username) VALUES ('testuser')")
        user_id = cursor.lastrowid
        
        # Insert child record
        cursor.execute("""
            INSERT INTO documents (title, created_by) 
            VALUES ('Test Doc', ?)
        """, (user_id,))
        doc_id = cursor.lastrowid
        
        # Try to delete parent - should fail due to foreign key
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        
        # Delete child first, then parent - should work
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    
    def test_fts_integration(self, schema):
        """Test FTS integration with triggers"""
        schema.create_schema()
        conn = schema.get_connection()
        cursor = conn.cursor()
        
        # Insert a chunk
        cursor.execute("""
            INSERT INTO chunks (text, chunk_index) 
            VALUES ('This is a test chunk for full-text search', 1)
        """)
        chunk_id = cursor.lastrowid
        conn.commit()
        
        # Test FTS search
        cursor.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'test'")
        results = cursor.fetchall()
        assert len(results) == 1
        assert results[0][0] == chunk_id
        
        # Update chunk and test FTS update
        cursor.execute("UPDATE chunks SET text = 'Updated chunk content' WHERE id = ?", (chunk_id,))
        conn.commit()
        
        cursor.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'Updated'")
        results = cursor.fetchall()
        assert len(results) == 1
        assert results[0][0] == chunk_id
    
    def test_embedding_storage(self, schema):
        """Test embedding storage and retrieval"""
        schema.create_schema()
        conn = schema.get_connection()
        cursor = conn.cursor()
        
        # Create test embedding
        test_embedding = np.random.rand(384).astype(np.float32)
        embedding_blob = serialize_embedding(test_embedding)
        
        # Insert embedding model
        cursor.execute("""
            INSERT INTO embedding_models (model_name, embedding_dim) 
            VALUES ('test-model', 384)
        """)
        model_id = cursor.lastrowid
        
        # Insert chunk
        cursor.execute("""
            INSERT INTO chunks (text, chunk_index) 
            VALUES ('Test chunk', 1)
        """)
        chunk_id = cursor.lastrowid
        
        # Insert embedding
        cursor.execute("""
            INSERT INTO embeddings (chunk_id, model_id, embedding, embedding_norm) 
            VALUES (?, ?, ?, ?)
        """, (chunk_id, model_id, embedding_blob, calculate_embedding_norm(test_embedding)))
        
        conn.commit()
        
        # Retrieve and verify embedding
        cursor.execute("SELECT embedding FROM embeddings WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        assert row is not None
        
        retrieved_embedding = deserialize_embedding(row[0])
        assert np.allclose(test_embedding, retrieved_embedding)
    
    def test_conversation_flow(self, schema):
        """Test complete conversation flow with context windows"""
        schema.create_schema()
        conn = schema.get_connection()
        cursor = conn.cursor()
        
        # Create user and conversation
        cursor.execute("INSERT INTO users (username) VALUES ('testuser')")
        user_id = cursor.lastrowid
        
        cursor.execute("""
            INSERT INTO conversations (user_id, title) 
            VALUES (?, 'Test Conversation')
        """, (user_id,))
        conv_id = cursor.lastrowid
        
        # Add messages
        cursor.execute("""
            INSERT INTO messages (conversation_id, sender, message) 
            VALUES (?, 'user', 'Hello, how are you?')
        """, (conv_id,))
        msg1_id = cursor.lastrowid
        
        cursor.execute("""
            INSERT INTO messages (conversation_id, sender, message) 
            VALUES (?, 'assistant', 'I am fine, thank you!')
        """, (conv_id,))
        msg2_id = cursor.lastrowid
        
        # Create chunk and add to context window
        cursor.execute("""
            INSERT INTO chunks (text, chunk_index) 
            VALUES ('Relevant context information', 1)
        """)
        chunk_id = cursor.lastrowid
        
        cursor.execute("""
            INSERT INTO context_windows (conversation_id, message_id, chunk_id, window_index, relevance_score) 
            VALUES (?, ?, ?, 1, 0.95)
        """, (conv_id, msg2_id, chunk_id))
        
        conn.commit()
        
        # Verify conversation structure
        cursor.execute("""
            SELECT m.message, cw.relevance_score 
            FROM messages m 
            LEFT JOIN context_windows cw ON m.id = cw.message_id 
            WHERE m.conversation_id = ? 
            ORDER BY m.created_at
        """, (conv_id,))
        
        results = cursor.fetchall()
        assert len(results) == 2
        assert results[0][0] == 'Hello, how are you?'
        assert results[1][0] == 'I am fine, thank you!'
        assert results[1][1] == 0.95  # Relevance score
    
    def test_privacy_and_audit(self, schema):
        """Test privacy controls and audit logging"""
        schema.create_schema()
        conn = schema.get_connection()
        cursor = conn.cursor()
        
        # Create user
        cursor.execute("INSERT INTO users (username) VALUES ('testuser')")
        user_id = cursor.lastrowid
        
        # Log access
        cursor.execute("""
            INSERT INTO access_logs (user_id, entity_type, entity_id, action, success) 
            VALUES (?, 'document', 1, 'read', 1)
        """, (user_id,))
        
        # Add privacy preference
        cursor.execute("""
            INSERT INTO privacy_preferences (user_id, preference_key, preference_value) 
            VALUES (?, 'data_retention_days', '30')
        """, (user_id,))
        
        # Add retention policy
        cursor.execute("""
            INSERT INTO retention_policies (entity_type, policy_name, retention_days) 
            VALUES ('conversation', 'default', 90)
        """)
        
        conn.commit()
        
        # Verify audit trail
        cursor.execute("SELECT action, success FROM access_logs WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 'read'
        assert row[1] == 1
        
        # Verify privacy preferences
        cursor.execute("SELECT preference_value FROM privacy_preferences WHERE user_id = ? AND preference_key = ?", 
                      (user_id, 'data_retention_days'))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == '30'
    
    def test_performance_tracking(self, schema):
        """Test performance monitoring capabilities"""
        schema.create_schema()
        conn = schema.get_connection()
        cursor = conn.cursor()
        
        # Add query performance record
        cursor.execute("""
            INSERT INTO query_performance (query_type, execution_time, results_count, cache_hit) 
            VALUES ('semantic_search', 0.15, 10, 0)
        """)
        
        # Add feedback
        cursor.execute("""
            INSERT INTO feedback (feedback_type, rating, comment) 
            VALUES ('relevant', 5, 'Very helpful response')
        """)
        
        # Add system metrics
        cursor.execute("""
            INSERT INTO system_metrics (metric_name, metric_value, unit) 
            VALUES ('memory_usage', 256.5, 'MB')
        """)
        
        conn.commit()
        
        # Verify performance data
        cursor.execute("SELECT execution_time, results_count FROM query_performance")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 0.15
        assert row[1] == 10
        
        cursor.execute("SELECT rating FROM feedback")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 5
        
        cursor.execute("SELECT metric_value FROM system_metrics")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 256.5


class TestUtilityFunctions:
    """Test suite for utility functions"""
    
    def test_embedding_serialization(self):
        """Test embedding serialization and deserialization"""
        # Create test embedding
        original_embedding = np.random.rand(384).astype(np.float32)
        
        # Serialize
        serialized = serialize_embedding(original_embedding)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = deserialize_embedding(serialized)
        assert isinstance(deserialized, np.ndarray)
        assert np.allclose(original_embedding, deserialized)
    
    def test_embedding_norm_calculation(self):
        """Test embedding norm calculation"""
        embedding = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        norm = calculate_embedding_norm(embedding)
        assert abs(norm - 5.0) < 1e-6  # 3-4-5 triangle
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        # Test identical vectors
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        similarity = cosine_similarity(a, b)
        assert abs(similarity - 1.0) < 1e-6
        
        # Test orthogonal vectors
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        similarity = cosine_similarity(a, b)
        assert abs(similarity - 0.0) < 1e-6
        
        # Test zero vectors
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        similarity = cosine_similarity(a, b)
        assert similarity == 0.0
    
    def test_entity_type_validation(self):
        """Test entity type validation"""
        assert validate_entity_type('document') is True
        assert validate_entity_type('chunk') is True
        assert validate_entity_type('conversation') is True
        assert validate_entity_type('message') is True
        assert validate_entity_type('user') is True
        assert validate_entity_type('invalid_type') is False
    
    def test_access_action_validation(self):
        """Test access action validation"""
        assert validate_access_action('create') is True
        assert validate_access_action('read') is True
        assert validate_access_action('update') is True
        assert validate_access_action('delete') is True
        assert validate_access_action('search') is True
        assert validate_access_action('export') is True
        assert validate_access_action('import') is True
        assert validate_access_action('invalid_action') is False
    
    def test_feedback_type_validation(self):
        """Test feedback type validation"""
        assert validate_feedback_type('relevant') is True
        assert validate_feedback_type('irrelevant') is True
        assert validate_feedback_type('partially_relevant') is True
        assert validate_feedback_type('helpful') is True
        assert validate_feedback_type('not_helpful') is True
        assert validate_feedback_type('invalid_feedback') is False


class TestSchemaEnum:
    """Test suite for schema enums"""
    
    def test_entity_type_enum(self):
        """Test EntityType enum"""
        assert EntityType.DOCUMENT.value == 'document'
        assert EntityType.CHUNK.value == 'chunk'
        assert EntityType.CONVERSATION.value == 'conversation'
        assert EntityType.MESSAGE.value == 'message'
        assert EntityType.USER.value == 'user'
    
    def test_access_action_enum(self):
        """Test AccessAction enum"""
        assert AccessAction.CREATE.value == 'create'
        assert AccessAction.READ.value == 'read'
        assert AccessAction.UPDATE.value == 'update'
        assert AccessAction.DELETE.value == 'delete'
        assert AccessAction.SEARCH.value == 'search'
        assert AccessAction.EXPORT.value == 'export'
        assert AccessAction.IMPORT.value == 'import'
    
    def test_feedback_type_enum(self):
        """Test FeedbackType enum"""
        assert FeedbackType.RELEVANT.value == 'relevant'
        assert FeedbackType.IRRELEVANT.value == 'irrelevant'
        assert FeedbackType.PARTIALLY_RELEVANT.value == 'partially_relevant'
        assert FeedbackType.HELPFUL.value == 'helpful'
        assert FeedbackType.NOT_HELPFUL.value == 'not_helpful'


class TestSchemaErrorHandling:
    """Test suite for error handling scenarios"""
    
    def test_invalid_database_path(self):
        """Test handling of invalid database path"""
        # Test with a path that can't be created
        with patch('pathlib.Path.mkdir', side_effect=PermissionError()):
            with pytest.raises(PermissionError):
                schema = MemorySchema('/invalid/path/test.db')
                schema.create_schema()
    
    def test_schema_creation_failure(self, temp_db_path):
        """Test schema creation failure handling"""
        schema = MemorySchema(temp_db_path)
        
        # Mock a database error
        with patch.object(schema, 'get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Database error")
            
            result = schema.create_schema()
            assert result is False
    
    def test_connection_failure(self, temp_db_path):
        """Test connection failure handling"""
        schema = MemorySchema(temp_db_path)
        
        # Test with invalid database file
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Connection failed")):
            with pytest.raises(sqlite3.Error):
                schema.get_connection()
    
    def test_schema_version_no_table(self, temp_db_path):
        """Test schema version when table doesn't exist"""
        schema = MemorySchema(temp_db_path)
        
        # Before creating schema, version should be None
        version = schema.get_schema_version()
        assert version is None
        
        # Migration should be needed
        assert schema.needs_migration() is True


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 