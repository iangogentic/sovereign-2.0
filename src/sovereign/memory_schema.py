"""
Database Schema for Sovereign AI Long-Term Memory (RAG) System

This module defines the database schema for storing conversation history,
embeddings, and metadata for the Retrieval-Augmented Generation (RAG) system.
Designed for local-first operation with SQLite, privacy controls, and efficient
semantic search capabilities.

Key Features:
- Normalized schema with proper foreign key relationships
- BLOB storage for embeddings with model versioning
- Full-text search integration (FTS5)
- Privacy controls and access management
- Audit logging and feedback collection
- Context window management
- Schema versioning and migration support
"""

import sqlite3
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

# Get logger
logger = logging.getLogger(__name__)

class EntityType(Enum):
    """Types of entities that can have metadata"""
    DOCUMENT = "document"
    CHUNK = "chunk"
    CONVERSATION = "conversation"
    MESSAGE = "message"
    USER = "user"

class AccessAction(Enum):
    """Types of access actions for audit logging"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"
    EXPORT = "export"
    IMPORT = "import"

class FeedbackType(Enum):
    """Types of feedback for retrieval quality"""
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"

@dataclass
class SchemaVersion:
    """Schema version information"""
    version: str
    applied_at: datetime
    description: str
    migration_script: Optional[str] = None

class MemorySchema:
    """
    Database schema manager for the Long-Term Memory system.
    
    Handles schema creation, versioning, migrations, and provides
    utilities for working with the database structure.
    """
    
    CURRENT_VERSION = "1.0.0"
    
    def __init__(self, db_path: str):
        """
        Initialize the schema manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = None
        
    def get_connection(self) -> sqlite3.Connection:
        """Get or create database connection with proper configuration"""
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            # Enable foreign keys and WAL mode for better performance
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            # Row factory for easier data access
            self._connection.row_factory = sqlite3.Row
        return self._connection
    
    def close(self):
        """Close database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def create_schema(self) -> bool:
        """
        Create the complete database schema.
        
        Returns:
            bool: True if schema was created successfully
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Create all tables
            self._create_core_tables(cursor)
            self._create_embedding_tables(cursor)
            self._create_conversation_tables(cursor)
            self._create_metadata_tables(cursor)
            self._create_privacy_tables(cursor)
            self._create_performance_tables(cursor)
            self._create_schema_version_table(cursor)
            self._create_indexes(cursor)
            self._create_fts_tables(cursor)
            
            # Record schema version
            self._record_schema_version(cursor)
            
            conn.commit()
            logger.info(f"Database schema v{self.CURRENT_VERSION} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            if conn:
                conn.rollback()
            return False
    
    def _create_core_tables(self, cursor: sqlite3.Cursor):
        """Create core document and chunk tables"""
        
        # Users table for multi-user support
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                privacy_level INTEGER DEFAULT 1,
                settings TEXT -- JSON string for user preferences
            )
        """)
        
        # Documents table - source materials
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT,
                content TEXT,
                document_type TEXT DEFAULT 'text',
                file_path TEXT,
                file_hash TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                is_active BOOLEAN DEFAULT 1,
                privacy_level INTEGER DEFAULT 1,
                FOREIGN KEY(created_by) REFERENCES users(id)
            )
        """)
        
        # Chunks table - processed text segments
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                chunk_type TEXT DEFAULT 'paragraph',
                start_position INTEGER,
                end_position INTEGER,
                overlap_prev INTEGER DEFAULT 0,
                overlap_next INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
    
    def _create_embedding_tables(self, cursor: sqlite3.Cursor):
        """Create embedding storage tables"""
        
        # Embedding models table - track different models
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embedding_models (
                id INTEGER PRIMARY KEY,
                model_name TEXT UNIQUE NOT NULL,
                model_version TEXT,
                embedding_dim INTEGER NOT NULL,
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Embeddings table - vector representations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id INTEGER PRIMARY KEY,
                model_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                embedding_norm REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
                FOREIGN KEY(model_id) REFERENCES embedding_models(id)
            )
        """)
        
        # Embedding cache for frequently accessed vectors
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                id INTEGER PRIMARY KEY,
                cache_key TEXT UNIQUE NOT NULL,
                chunk_ids TEXT NOT NULL, -- JSON array of chunk IDs
                similarity_scores TEXT NOT NULL, -- JSON array of scores
                query_embedding BLOB,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1
            )
        """)
    
    def _create_conversation_tables(self, cursor: sqlite3.Cursor):
        """Create conversation and message tables"""
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                title TEXT,
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                ended_at DATETIME,
                is_active BOOLEAN DEFAULT 1,
                privacy_level INTEGER DEFAULT 1,
                context_window_size INTEGER DEFAULT 10,
                total_messages INTEGER DEFAULT 0,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                conversation_id INTEGER NOT NULL,
                sender TEXT NOT NULL, -- 'user', 'assistant', 'system'
                message TEXT NOT NULL,
                message_type TEXT DEFAULT 'text',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                retrieved_chunk_ids TEXT, -- JSON array of chunk IDs used
                model_used TEXT,
                response_time REAL,
                token_count INTEGER,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        
        # Context windows table for efficient context management
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_windows (
                id INTEGER PRIMARY KEY,
                conversation_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                chunk_id INTEGER NOT NULL,
                window_index INTEGER NOT NULL,
                relevance_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            )
        """)
    
    def _create_metadata_tables(self, cursor: sqlite3.Cursor):
        """Create metadata and tagging tables"""
        
        # Metadata table - flexible key-value storage
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                data_type TEXT DEFAULT 'string',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(entity_type, entity_id, key)
            )
        """)
        
        # Tags table for categorical organization
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                color TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                usage_count INTEGER DEFAULT 0
            )
        """)
        
        # Entity tags junction table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_tags (
                id INTEGER PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE,
                UNIQUE(entity_type, entity_id, tag_id)
            )
        """)
    
    def _create_privacy_tables(self, cursor: sqlite3.Cursor):
        """Create privacy and access control tables"""
        
        # Access logs table for audit trail
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                success BOOLEAN DEFAULT 1,
                error_message TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        
        # Data retention policies
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retention_policies (
                id INTEGER PRIMARY KEY,
                entity_type TEXT NOT NULL,
                policy_name TEXT NOT NULL,
                retention_days INTEGER NOT NULL,
                auto_delete BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                UNIQUE(entity_type, policy_name)
            )
        """)
        
        # Privacy preferences
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS privacy_preferences (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                preference_key TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id),
                UNIQUE(user_id, preference_key)
            )
        """)
    
    def _create_performance_tables(self, cursor: sqlite3.Cursor):
        """Create performance monitoring and feedback tables"""
        
        # Query performance metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_performance (
                id INTEGER PRIMARY KEY,
                query_type TEXT NOT NULL,
                query_text TEXT,
                execution_time REAL NOT NULL,
                results_count INTEGER,
                cache_hit BOOLEAN DEFAULT 0,
                model_used TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # User feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                message_id INTEGER,
                chunk_id INTEGER,
                feedback_type TEXT NOT NULL,
                rating INTEGER, -- 1-5 scale
                comment TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(message_id) REFERENCES messages(id),
                FOREIGN KEY(chunk_id) REFERENCES chunks(id)
            )
        """)
        
        # System metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                unit TEXT,
                recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT -- JSON for additional context
            )
        """)
    
    def _create_schema_version_table(self, cursor: sqlite3.Cursor):
        """Create schema version tracking table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_versions (
                version TEXT PRIMARY KEY,
                applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                migration_script TEXT
            )
        """)
    
    def _create_indexes(self, cursor: sqlite3.Cursor):
        """Create database indexes for performance optimization"""
        
        # Core table indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_created_by ON documents(created_by)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(chunk_index)")
        
        # Embedding indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_model_id ON embeddings(model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON embeddings(created_at)")
        
        # Conversation indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_started_at ON conversations(started_at)")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender)")
        
        # Context window indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_windows_conversation_id ON context_windows(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_windows_message_id ON context_windows(message_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_windows_chunk_id ON context_windows(chunk_id)")
        
        # Metadata indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_entity ON metadata(entity_type, entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_key ON metadata(key)")
        
        # Privacy and audit indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_logs_user_id ON access_logs(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_logs_entity ON access_logs(entity_type, entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_logs_created_at ON access_logs(created_at)")
        
        # Performance indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_performance_type ON query_performance(query_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_performance_time ON query_performance(execution_time)")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_message_id ON feedback(message_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_chunk_id ON feedback(chunk_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)")
    
    def _create_fts_tables(self, cursor: sqlite3.Cursor):
        """Create Full-Text Search tables using FTS5"""
        
        # FTS table for chunk text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id'
            )
        """)
        
        # FTS table for message search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                message,
                content='messages',
                content_rowid='id'
            )
        """)
        
        # FTS table for document search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                title,
                content,
                content='documents',
                content_rowid='id'
            )
        """)
        
        # Create triggers to keep FTS tables in sync
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                UPDATE chunks_fts SET text = new.text WHERE rowid = new.id;
            END;
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE rowid = old.id;
            END;
        """)
        
        # Similar triggers for messages and documents
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, message) VALUES (new.id, new.message);
            END;
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
                UPDATE messages_fts SET message = new.message WHERE rowid = new.id;
            END;
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
                DELETE FROM messages_fts WHERE rowid = old.id;
            END;
        """)
    
    def _record_schema_version(self, cursor: sqlite3.Cursor):
        """Record the current schema version"""
        cursor.execute("""
            INSERT OR REPLACE INTO schema_versions (version, description)
            VALUES (?, ?)
        """, (
            self.CURRENT_VERSION,
            f"Initial schema creation for Sovereign AI Long-Term Memory system"
        ))
    
    def get_schema_version(self) -> Optional[str]:
        """Get the current schema version from the database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT version FROM schema_versions ORDER BY applied_at DESC LIMIT 1")
            row = cursor.fetchone()
            return row[0] if row else None
        except sqlite3.OperationalError:
            # Schema version table doesn't exist yet
            return None
    
    def needs_migration(self) -> bool:
        """Check if database needs migration to current version"""
        db_version = self.get_schema_version()
        return db_version != self.CURRENT_VERSION
    
    def validate_schema(self) -> Dict[str, Any]:
        """
        Validate the database schema integrity.
        
        Returns:
            dict: Validation results with table counts and integrity status
        """
        results = {
            "valid": True,
            "version": self.get_schema_version(),
            "tables": {},
            "indexes": {},
            "triggers": {},
            "issues": []
        }
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Check table existence and row counts
            expected_tables = [
                "users", "documents", "chunks", "embedding_models", "embeddings",
                "conversations", "messages", "context_windows", "metadata",
                "tags", "entity_tags", "access_logs", "retention_policies",
                "privacy_preferences", "query_performance", "feedback",
                "system_metrics", "schema_versions", "embedding_cache"
            ]
            
            for table in expected_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                results["tables"][table] = count
            
            # Check foreign key integrity
            cursor.execute("PRAGMA foreign_key_check")
            fk_violations = cursor.fetchall()
            if fk_violations:
                results["valid"] = False
                results["issues"].append(f"Foreign key violations: {fk_violations}")
            
            # Check index existence
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
            indexes = [row[0] for row in cursor.fetchall()]
            results["indexes"] = indexes
            
            # Check trigger existence
            cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
            triggers = [row[0] for row in cursor.fetchall()]
            results["triggers"] = triggers
            
        except Exception as e:
            results["valid"] = False
            results["issues"].append(f"Validation error: {str(e)}")
        
        return results
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific table"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            # Get indexes
            cursor.execute(f"PRAGMA index_list({table_name})")
            indexes = cursor.fetchall()
            
            return {
                "columns": [dict(col) for col in columns],
                "row_count": row_count,
                "indexes": [dict(idx) for idx in indexes]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def export_schema(self) -> str:
        """Export the current schema as SQL DDL"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Get all table creation statements
            cursor.execute("""
                SELECT sql FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            
            ddl_statements = []
            for row in cursor.fetchall():
                if row[0]:  # Some system tables have NULL sql
                    ddl_statements.append(row[0])
            
            # Get indexes
            cursor.execute("""
                SELECT sql FROM sqlite_master 
                WHERE type='index' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            
            for row in cursor.fetchall():
                if row[0]:
                    ddl_statements.append(row[0])
            
            # Get triggers
            cursor.execute("""
                SELECT sql FROM sqlite_master 
                WHERE type='trigger'
                ORDER BY name
            """)
            
            for row in cursor.fetchall():
                if row[0]:
                    ddl_statements.append(row[0])
            
            return ";\n\n".join(ddl_statements) + ";"
            
        except Exception as e:
            logger.error(f"Failed to export schema: {e}")
            return ""
    
    def create_default_user(self, username: str = "default_user") -> int:
        """Create a default user for single-user setups"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR IGNORE INTO users (username, settings) 
                VALUES (?, ?)
            """, (username, json.dumps({"theme": "dark", "context_window": 10})))
            
            conn.commit()
            
            # Get the user ID
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            return row[0] if row else 1
            
        except Exception as e:
            logger.error(f"Failed to create default user: {e}")
            return 1
    
    def insert_default_embedding_model(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> int:
        """Insert a default embedding model"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR IGNORE INTO embedding_models (model_name, embedding_dim, description) 
                VALUES (?, ?, ?)
            """, (
                model_name,
                384,  # Default dimension for MiniLM
                "Default local embedding model for semantic search"
            ))
            
            conn.commit()
            
            # Get the model ID
            cursor.execute("SELECT id FROM embedding_models WHERE model_name = ?", (model_name,))
            row = cursor.fetchone()
            return row[0] if row else 1
            
        except Exception as e:
            logger.error(f"Failed to insert default embedding model: {e}")
            return 1

# Utility functions for embedding serialization
def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize a numpy array embedding to bytes for database storage"""
    return pickle.dumps(embedding.astype(np.float32))

def deserialize_embedding(embedding_bytes: bytes) -> np.ndarray:
    """Deserialize bytes back to numpy array embedding"""
    return pickle.loads(embedding_bytes)

def calculate_embedding_norm(embedding: np.ndarray) -> float:
    """Calculate L2 norm of embedding vector"""
    return float(np.linalg.norm(embedding))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two embedding vectors"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

# Schema validation utilities
def validate_entity_type(entity_type: str) -> bool:
    """Validate that entity type is supported"""
    return entity_type in [e.value for e in EntityType]

def validate_access_action(action: str) -> bool:
    """Validate that access action is supported"""
    return action in [a.value for a in AccessAction]

def validate_feedback_type(feedback_type: str) -> bool:
    """Validate that feedback type is supported"""
    return feedback_type in [f.value for f in FeedbackType]

# Export key classes and functions
__all__ = [
    'MemorySchema',
    'EntityType',
    'AccessAction', 
    'FeedbackType',
    'SchemaVersion',
    'serialize_embedding',
    'deserialize_embedding',
    'calculate_embedding_norm',
    'cosine_similarity',
    'validate_entity_type',
    'validate_access_action',
    'validate_feedback_type'
] 