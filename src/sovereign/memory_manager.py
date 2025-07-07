"""
Memory Manager for Sovereign AI Long-Term Memory (RAG) System

This module provides a high-level interface for managing long-term memory,
including conversation storage, document management, embedding operations,
and semantic search capabilities. Built on top of the MemorySchema database
layer and integrated with the existing configuration system.

Key Features:
- Conversation lifecycle management
- Document and chunk processing
- Embedding generation and storage
- Semantic search and retrieval
- Privacy controls and access management
- Context window optimization
- Performance monitoring
- Import/export capabilities
"""

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
from contextlib import asynccontextmanager

from .config import Config
from .memory_schema import (
    MemorySchema, EntityType, AccessAction, FeedbackType,
    serialize_embedding, deserialize_embedding, cosine_similarity
)
from .embedding_service import (
    EmbeddingService, EmbeddingModelType, EmbeddingRequest, EmbeddingResponse
)
from .context_window_manager import ContextWindowManager, SelectionStrategy, UserPreferences

# Get logger
logger = logging.getLogger(__name__)

class ConversationStatus(Enum):
    """Status of conversations"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class SearchMode(Enum):
    """Search modes for retrieval"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

class MessageSender(Enum):
    """Message sender types"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class ConversationMeta:
    """Metadata for a conversation"""
    id: int
    user_id: int
    title: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    status: ConversationStatus = ConversationStatus.ACTIVE
    total_messages: int = 0
    privacy_level: int = 1
    context_window_size: int = 10

@dataclass
class MessageData:
    """Data structure for a message"""
    id: int
    conversation_id: int
    sender: MessageSender
    message: str
    message_type: str = "text"
    created_at: datetime = None
    retrieved_chunk_ids: List[int] = None
    model_used: str = None
    response_time: float = None
    token_count: int = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.retrieved_chunk_ids is None:
            self.retrieved_chunk_ids = []

@dataclass
class DocumentData:
    """Data structure for a document"""
    id: int
    title: str
    content: str
    source: str = ""
    document_type: str = "text"
    file_path: str = ""
    file_hash: str = ""
    created_at: datetime = None
    privacy_level: int = 1
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ChunkData:
    """Data structure for a text chunk"""
    id: int
    document_id: int
    chunk_index: int
    text: str
    chunk_type: str = "paragraph"
    start_position: int = 0
    end_position: int = 0
    overlap_prev: int = 0
    overlap_next: int = 0

@dataclass
class SearchResult:
    """Result from semantic search"""
    chunk_id: int
    chunk_text: str
    similarity_score: float
    document_id: int
    document_title: str
    document_source: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class MemoryStats:
    """Statistics about memory usage"""
    total_conversations: int = 0
    total_messages: int = 0
    total_documents: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    db_size_mb: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

class MemoryManager:
    """
    High-level interface for the Long-Term Memory system.
    
    Provides comprehensive memory management including conversation storage,
    document processing, embedding operations, and semantic search.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the MemoryManager.
        
        Args:
            config: Main configuration object
        """
        self.config = config
        self.db_path = Path(config.database.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self.schema = MemorySchema(str(self.db_path))
        
        # Threading for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.RLock()
        
        # Current user and conversation context
        self._current_user_id: Optional[int] = None
        self._current_conversation_id: Optional[int] = None
        
        # Caching for performance
        self._embedding_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._conversation_cache: Dict[int, ConversationMeta] = {}
        
        # Performance tracking
        self._stats = MemoryStats()
        self._start_time = time.time()
        
        # Initialize embedding service
        try:
            self._embedding_service = EmbeddingService(
                default_model=EmbeddingModelType.BGE_BASE_EN_V15,  # Balanced performance
                cache_dir=str(self.db_path.parent / "embeddings"),
                enable_gpu=True,
                max_batch_size=16,  # Conservative batch size
                max_workers=2
            )
            logger.info(f"Embedding service initialized with model: {self._embedding_service.default_model.value}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            logger.warning("Falling back to mock embeddings")
            self._embedding_service = None
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database schema and default data"""
        try:
            # Create schema if it doesn't exist
            if not self.schema.create_schema():
                logger.error("Failed to create database schema")
                raise RuntimeError("Database initialization failed")
            
            # Create default user if none exists
            default_user_id = self.schema.create_default_user()
            self._current_user_id = default_user_id
            
            # Create default embedding model
            self.schema.insert_default_embedding_model()
            
            logger.info("Memory database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def close(self):
        """Close the memory manager and cleanup resources"""
        try:
            # Shutdown embedding service
            if self._embedding_service:
                self._embedding_service.close()
                self._embedding_service = None
            
            # Close database and executor
            self.schema.close()
            self._executor.shutdown(wait=True)
            
            # Clear caches
            self._embedding_cache.clear()
            self._conversation_cache.clear()
            
            logger.info("Memory manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing memory manager: {e}")
    
    # User Management
    def create_user(self, username: str, email: str = None, 
                   privacy_level: int = 1, settings: Dict[str, Any] = None) -> int:
        """
        Create a new user.
        
        Args:
            username: Unique username
            email: Optional email address
            privacy_level: Privacy level (1-5)
            settings: User preferences
            
        Returns:
            int: User ID
        """
        try:
            with self._lock:
                conn = self.schema.get_connection()
                cursor = conn.cursor()
                
                settings_json = json.dumps(settings or {})
                
                cursor.execute("""
                    INSERT INTO users (username, email, privacy_level, settings)
                    VALUES (?, ?, ?, ?)
                """, (username, email, privacy_level, settings_json))
                
                user_id = cursor.lastrowid
                conn.commit()
                
                self._log_access(user_id, EntityType.USER, user_id, AccessAction.CREATE)
                logger.info(f"Created user {username} with ID {user_id}")
                
                return user_id
                
        except Exception as e:
            logger.error(f"Failed to create user {username}: {e}")
            raise
    
    def set_current_user(self, user_id: int) -> bool:
        """
        Set the current user context.
        
        Args:
            user_id: User ID to set as current
            
        Returns:
            bool: True if successful
        """
        try:
            # Verify user exists
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM users WHERE id = ? AND is_active = 1", (user_id,))
            if not cursor.fetchone():
                logger.warning(f"User {user_id} not found or inactive")
                return False
            
            self._current_user_id = user_id
            logger.info(f"Set current user to {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set current user {user_id}: {e}")
            return False
    
    def get_current_user(self) -> Optional[int]:
        """Get the current user ID"""
        return self._current_user_id
    
    # Conversation Management
    def start_conversation(self, title: str = "", privacy_level: int = 1,
                          context_window_size: int = 10) -> int:
        """
        Start a new conversation.
        
        Args:
            title: Conversation title
            privacy_level: Privacy level (1-5)
            context_window_size: Number of messages to keep in context
            
        Returns:
            int: Conversation ID
        """
        try:
            with self._lock:
                if not self._current_user_id:
                    raise ValueError("No current user set")
                
                conn = self.schema.get_connection()
                cursor = conn.cursor()
                
                # Generate title if not provided
                if not title:
                    title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                
                cursor.execute("""
                    INSERT INTO conversations (user_id, title, privacy_level, context_window_size)
                    VALUES (?, ?, ?, ?)
                """, (self._current_user_id, title, privacy_level, context_window_size))
                
                conversation_id = cursor.lastrowid
                conn.commit()
                
                # Cache conversation metadata
                self._conversation_cache[conversation_id] = ConversationMeta(
                    id=conversation_id,
                    user_id=self._current_user_id,
                    title=title,
                    started_at=datetime.now(),
                    status=ConversationStatus.ACTIVE,
                    privacy_level=privacy_level,
                    context_window_size=context_window_size
                )
                
                self._current_conversation_id = conversation_id
                self._log_access(self._current_user_id, EntityType.CONVERSATION, 
                               conversation_id, AccessAction.CREATE)
                
                logger.info(f"Started conversation {conversation_id}: {title}")
                return conversation_id
                
        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            raise
    
    def get_conversation(self, conversation_id: int) -> Optional[ConversationMeta]:
        """
        Get conversation metadata.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            ConversationMeta or None if not found
        """
        try:
            # Check cache first
            if conversation_id in self._conversation_cache:
                return self._conversation_cache[conversation_id]
            
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, user_id, title, started_at, ended_at, is_active,
                       privacy_level, context_window_size, total_messages
                FROM conversations 
                WHERE id = ?
            """, (conversation_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Convert to ConversationMeta
            conversation = ConversationMeta(
                id=row[0],
                user_id=row[1],
                title=row[2],
                started_at=datetime.fromisoformat(row[3]),
                ended_at=datetime.fromisoformat(row[4]) if row[4] else None,
                status=ConversationStatus.ACTIVE if row[5] else ConversationStatus.COMPLETED,
                privacy_level=row[6],
                context_window_size=row[7],
                total_messages=row[8]
            )
            
            # Cache for future use
            self._conversation_cache[conversation_id] = conversation
            
            self._log_access(self._current_user_id, EntityType.CONVERSATION, 
                           conversation_id, AccessAction.READ)
            
            return conversation
            
        except Exception as e:
            logger.error(f"Failed to get conversation {conversation_id}: {e}")
            return None
    
    def list_conversations(self, user_id: Optional[int] = None, 
                          status: Optional[ConversationStatus] = None,
                          limit: int = 100) -> List[ConversationMeta]:
        """
        List conversations for a user.
        
        Args:
            user_id: User ID (defaults to current user)
            status: Filter by conversation status
            limit: Maximum number of conversations to return
            
        Returns:
            List of ConversationMeta objects
        """
        try:
            target_user_id = user_id or self._current_user_id
            if not target_user_id:
                return []
            
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            query = """
                SELECT id, user_id, title, started_at, ended_at, is_active,
                       privacy_level, context_window_size, total_messages
                FROM conversations 
                WHERE user_id = ?
            """
            params = [target_user_id]
            
            if status:
                if status == ConversationStatus.ACTIVE:
                    query += " AND is_active = 1"
                elif status == ConversationStatus.COMPLETED:
                    query += " AND is_active = 0"
            
            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            conversations = []
            for row in rows:
                conversation = ConversationMeta(
                    id=row[0],
                    user_id=row[1],
                    title=row[2],
                    started_at=datetime.fromisoformat(row[3]),
                    ended_at=datetime.fromisoformat(row[4]) if row[4] else None,
                    status=ConversationStatus.ACTIVE if row[5] else ConversationStatus.COMPLETED,
                    privacy_level=row[6],
                    context_window_size=row[7],
                    total_messages=row[8]
                )
                conversations.append(conversation)
            
            return conversations
            
        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            return []
    
    def set_current_conversation(self, conversation_id: int) -> bool:
        """
        Set the current conversation context.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            bool: True if successful
        """
        try:
            conversation = self.get_conversation(conversation_id)
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found")
                return False
            
            self._current_conversation_id = conversation_id
            logger.info(f"Set current conversation to {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set current conversation {conversation_id}: {e}")
            return False
    
    def get_current_conversation(self) -> Optional[int]:
        """Get the current conversation ID"""
        return self._current_conversation_id
    
    def close_conversation(self, conversation_id: Optional[int] = None) -> bool:
        """
        Close a conversation.
        
        Args:
            conversation_id: Conversation ID (defaults to current)
            
        Returns:
            bool: True if successful
        """
        try:
            target_id = conversation_id or self._current_conversation_id
            if not target_id:
                return False
            
            with self._lock:
                conn = self.schema.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE conversations 
                    SET ended_at = CURRENT_TIMESTAMP, is_active = 0
                    WHERE id = ?
                """, (target_id,))
                
                conn.commit()
                
                # Update cache
                if target_id in self._conversation_cache:
                    self._conversation_cache[target_id].status = ConversationStatus.COMPLETED
                    self._conversation_cache[target_id].ended_at = datetime.now()
                
                # Clear current conversation if it was closed
                if target_id == self._current_conversation_id:
                    self._current_conversation_id = None
                
                self._log_access(self._current_user_id, EntityType.CONVERSATION, 
                               target_id, AccessAction.UPDATE)
                
                logger.info(f"Closed conversation {target_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to close conversation {target_id}: {e}")
            return False
    
    # Message Management
    def add_message(self, message: str, sender: MessageSender,
                   conversation_id: Optional[int] = None,
                   message_type: str = "text",
                   retrieved_chunk_ids: List[int] = None,
                   model_used: str = None,
                   response_time: float = None,
                   token_count: int = None) -> int:
        """
        Add a message to a conversation.
        
        Args:
            message: Message content
            sender: Message sender (user, assistant, system)
            conversation_id: Conversation ID (defaults to current)
            message_type: Type of message (text, image, etc.)
            retrieved_chunk_ids: List of chunk IDs used for context
            model_used: AI model used to generate response
            response_time: Time taken to generate response
            token_count: Number of tokens in message
            
        Returns:
            int: Message ID
        """
        try:
            target_conversation_id = conversation_id or self._current_conversation_id
            if not target_conversation_id:
                raise ValueError("No conversation context available")
            
            with self._lock:
                conn = self.schema.get_connection()
                cursor = conn.cursor()
                
                # Convert retrieved_chunk_ids to JSON
                chunk_ids_json = json.dumps(retrieved_chunk_ids or [])
                
                cursor.execute("""
                    INSERT INTO messages (
                        conversation_id, sender, message, message_type,
                        retrieved_chunk_ids, model_used, response_time, token_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    target_conversation_id, sender.value, message, message_type,
                    chunk_ids_json, model_used, response_time, token_count
                ))
                
                message_id = cursor.lastrowid
                
                # Update conversation message count
                cursor.execute("""
                    UPDATE conversations 
                    SET total_messages = total_messages + 1
                    WHERE id = ?
                """, (target_conversation_id,))
                
                conn.commit()
                
                # Update cache
                if target_conversation_id in self._conversation_cache:
                    self._conversation_cache[target_conversation_id].total_messages += 1
                
                self._log_access(self._current_user_id, EntityType.MESSAGE, 
                               message_id, AccessAction.CREATE)
                
                logger.info(f"Added message {message_id} to conversation {target_conversation_id}")
                return message_id
                
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            raise
    
    def get_message(self, message_id: int) -> Optional[MessageData]:
        """
        Get a specific message.
        
        Args:
            message_id: Message ID
            
        Returns:
            MessageData or None if not found
        """
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, conversation_id, sender, message, message_type,
                       created_at, retrieved_chunk_ids, model_used, response_time, token_count
                FROM messages 
                WHERE id = ?
            """, (message_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Parse retrieved_chunk_ids
            chunk_ids = json.loads(row[6]) if row[6] else []
            
            message = MessageData(
                id=row[0],
                conversation_id=row[1],
                sender=MessageSender(row[2]),
                message=row[3],
                message_type=row[4],
                created_at=datetime.fromisoformat(row[5]),
                retrieved_chunk_ids=chunk_ids,
                model_used=row[7],
                response_time=row[8],
                token_count=row[9]
            )
            
            self._log_access(self._current_user_id, EntityType.MESSAGE, 
                           message_id, AccessAction.READ)
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to get message {message_id}: {e}")
            return None
    
    def get_conversation_messages(self, conversation_id: int,
                                 limit: int = 100,
                                 offset: int = 0) -> List[MessageData]:
        """
        Get messages from a conversation.
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            
        Returns:
            List of MessageData objects
        """
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, conversation_id, sender, message, message_type,
                       created_at, retrieved_chunk_ids, model_used, response_time, token_count
                FROM messages 
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                LIMIT ? OFFSET ?
            """, (conversation_id, limit, offset))
            
            rows = cursor.fetchall()
            messages = []
            
            for row in rows:
                chunk_ids = json.loads(row[6]) if row[6] else []
                
                message = MessageData(
                    id=row[0],
                    conversation_id=row[1],
                    sender=MessageSender(row[2]),
                    message=row[3],
                    message_type=row[4],
                    created_at=datetime.fromisoformat(row[5]),
                    retrieved_chunk_ids=chunk_ids,
                    model_used=row[7],
                    response_time=row[8],
                    token_count=row[9]
                )
                messages.append(message)
            
            self._log_access(self._current_user_id, EntityType.CONVERSATION, 
                           conversation_id, AccessAction.READ)
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get messages for conversation {conversation_id}: {e}")
            return []
    
    def get_context_window(self, conversation_id: Optional[int] = None,
                          window_size: Optional[int] = None,
                          query: str = "",
                          strategy: SelectionStrategy = SelectionStrategy.HYBRID,
                          model_name: str = "gpt-4") -> List[MessageData]:
        """
        Get the current context window for a conversation with intelligent selection.
        
        This method now provides intelligent context window management using the
        Context Window Manager for optimal memory retrieval.
        
        Args:
            conversation_id: Conversation ID (defaults to current)
            window_size: Maximum number of messages (backwards compatibility)
            query: Query for relevance-based selection
            strategy: Selection strategy to use
            model_name: Target AI model for token limits
            
        Returns:
            List of optimally selected MessageData objects
        """
        try:
            # Use basic method if no query provided (backwards compatibility)
            if not query or strategy == SelectionStrategy.RECENCY_ONLY:
                return self._get_basic_context_window(conversation_id, window_size)
            
            # If context window manager is available and configured, use intelligent selection
            if hasattr(self, '_context_window_manager') and self._context_window_manager:
                try:
                    # Run async context window building in sync context
                    import asyncio
                    
                    # Get or create event loop
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Build intelligent context window
                    context_window = loop.run_until_complete(
                        self._context_window_manager.build_context_window(
                            query=query,
                            conversation_id=conversation_id,
                            model_name=model_name,
                            strategy=strategy,
                            max_messages=window_size or 50
                        )
                    )
                    
                    # Extract selected messages
                    selected_items = context_window.get_selected_items()
                    return [item.message for item in selected_items]
                    
                except Exception as e:
                    logger.warning(f"Intelligent context selection failed, falling back to basic: {e}")
                    return self._get_basic_context_window(conversation_id, window_size)
            
            # Fallback to basic method
            return self._get_basic_context_window(conversation_id, window_size)
            
        except Exception as e:
            logger.error(f"Failed to get context window: {e}")
            return []
    
    def _get_basic_context_window(self, conversation_id: Optional[int] = None,
                                 window_size: Optional[int] = None) -> List[MessageData]:
        """
        Basic context window method (original implementation).
        
        This provides the original recency-based context window selection
        as a fallback when intelligent selection is not available.
        """
        try:
            target_conversation_id = conversation_id or self._current_conversation_id
            if not target_conversation_id:
                return []
            
            # Get conversation metadata for default window size
            conversation = self.get_conversation(target_conversation_id)
            if not conversation:
                return []
            
            effective_window_size = window_size or conversation.context_window_size
            
            # Get recent messages
            messages = self.get_conversation_messages(
                target_conversation_id, 
                limit=100  # Get more than needed to ensure we have enough
            )
            
            # Return most recent messages in reverse order (most recent first)
            recent_messages = messages[-effective_window_size:] if len(messages) > effective_window_size else messages
            return list(reversed(recent_messages))
            
        except Exception as e:
            logger.error(f"Failed to get basic context window: {e}")
            return []
    
    def set_context_window_manager(self, context_manager: 'ContextWindowManager') -> None:
        """
        Set the context window manager for intelligent context selection.
        
        Args:
            context_manager: ContextWindowManager instance
        """
        self._context_window_manager = context_manager
        logger.info("Context Window Manager integrated with Memory Manager")
    
    def get_intelligent_context(self, query: str, 
                               conversation_id: Optional[int] = None,
                               model_name: str = "gpt-4",
                               strategy: SelectionStrategy = SelectionStrategy.HYBRID) -> str:
        """
        Get formatted context text using intelligent selection.
        
        This is a convenience method that provides formatted context text
        ready for use with AI models.
        
        Args:
            query: The query for context relevance
            conversation_id: Target conversation ID
            model_name: Target AI model name
            strategy: Selection strategy
            
        Returns:
            Formatted context text
        """
        try:
            if not hasattr(self, '_context_window_manager') or not self._context_window_manager:
                logger.warning("Context Window Manager not available, using basic context")
                messages = self.get_context_window(conversation_id)
                if not messages:
                    return ""
                
                # Format basic context
                context_parts = []
                for msg in messages:
                    sender = msg.sender.value.upper()
                    timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
                    context_parts.append(f"[{timestamp}] {sender}: {msg.message}")
                
                return "\n".join(context_parts)
            
            # Use intelligent context selection
            import asyncio
            
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            context_text = loop.run_until_complete(
                self._context_window_manager.get_optimized_context(
                    query=query,
                    conversation_id=conversation_id,
                    model_name=model_name,
                    strategy=strategy
                )
            )
            
            return context_text
            
        except Exception as e:
            logger.error(f"Failed to get intelligent context: {e}")
            return ""
    
    # Document Management
    def add_document(self, title: str, content: str, source: str = "",
                    document_type: str = "text", file_path: str = "",
                    privacy_level: int = 1) -> int:
        """
        Add a document to the knowledge base.
        
        Args:
            title: Document title
            content: Document content
            source: Source of the document
            document_type: Type of document (text, pdf, etc.)
            file_path: Original file path
            privacy_level: Privacy level (1-5)
            
        Returns:
            int: Document ID
        """
        try:
            with self._lock:
                conn = self.schema.get_connection()
                cursor = conn.cursor()
                
                # Calculate file hash
                file_hash = hashlib.sha256(content.encode()).hexdigest()
                
                # Check if document already exists
                cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
                existing = cursor.fetchone()
                if existing:
                    logger.info(f"Document with hash {file_hash} already exists")
                    return existing[0]
                
                cursor.execute("""
                    INSERT INTO documents (
                        title, content, source, document_type, file_path, 
                        file_hash, created_by, privacy_level
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    title, content, source, document_type, file_path,
                    file_hash, self._current_user_id, privacy_level
                ))
                
                document_id = cursor.lastrowid
                conn.commit()
                
                self._log_access(self._current_user_id, EntityType.DOCUMENT, 
                               document_id, AccessAction.CREATE)
                
                logger.info(f"Added document {document_id}: {title}")
                return document_id
                
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    def get_document(self, document_id: int) -> Optional[DocumentData]:
        """
        Get a document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            DocumentData or None if not found
        """
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, title, content, source, document_type, file_path,
                       file_hash, created_at, privacy_level
                FROM documents 
                WHERE id = ? AND is_active = 1
            """, (document_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            document = DocumentData(
                id=row[0],
                title=row[1],
                content=row[2],
                source=row[3],
                document_type=row[4],
                file_path=row[5],
                file_hash=row[6],
                created_at=datetime.fromisoformat(row[7]),
                privacy_level=row[8]
            )
            
            self._log_access(self._current_user_id, EntityType.DOCUMENT, 
                           document_id, AccessAction.READ)
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def list_documents(self, limit: int = 100) -> List[DocumentData]:
        """
        List all documents in the knowledge base.
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            List of DocumentData objects
        """
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, title, content, source, document_type, file_path,
                       file_hash, created_at, privacy_level
                FROM documents 
                WHERE is_active = 1
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            documents = []
            
            for row in rows:
                document = DocumentData(
                    id=row[0],
                    title=row[1],
                    content=row[2],
                    source=row[3],
                    document_type=row[4],
                    file_path=row[5],
                    file_hash=row[6],
                    created_at=datetime.fromisoformat(row[7]),
                    privacy_level=row[8]
                )
                documents.append(document)
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document (soft delete).
        
        Args:
            document_id: Document ID
            
        Returns:
            bool: True if successful
        """
        try:
            with self._lock:
                conn = self.schema.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE documents 
                    SET is_active = 0, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (document_id,))
                
                conn.commit()
                
                self._log_access(self._current_user_id, EntityType.DOCUMENT, 
                               document_id, AccessAction.DELETE)
                
                logger.info(f"Deleted document {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    # Chunk Management
    def create_chunks(self, document_id: int, chunk_size: int = 1000,
                     overlap_size: int = 100) -> List[int]:
        """
        Create text chunks from a document.
        
        Args:
            document_id: Document ID
            chunk_size: Size of each chunk in characters
            overlap_size: Overlap between chunks in characters
            
        Returns:
            List of chunk IDs
        """
        try:
            document = self.get_document(document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            content = document.content
            chunks = []
            chunk_ids = []
            
            # Ensure step size is at least 1 to avoid zero step error
            step_size = max(1, chunk_size - overlap_size)
            
            # Split content into chunks
            for i in range(0, len(content), step_size):
                chunk_text = content[i:i + chunk_size]
                if len(chunk_text.strip()) < 50:  # Skip very short chunks
                    continue
                
                overlap_prev = overlap_size if i > 0 else 0
                overlap_next = overlap_size if i + chunk_size < len(content) else 0
                
                chunks.append({
                    'text': chunk_text,
                    'start_position': i,
                    'end_position': i + len(chunk_text),
                    'overlap_prev': overlap_prev,
                    'overlap_next': overlap_next
                })
            
            # Insert chunks into database
            with self._lock:
                conn = self.schema.get_connection()
                cursor = conn.cursor()
                
                for idx, chunk in enumerate(chunks):
                    cursor.execute("""
                        INSERT INTO chunks (
                            document_id, chunk_index, text, start_position, 
                            end_position, overlap_prev, overlap_next
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        document_id, idx, chunk['text'], chunk['start_position'],
                        chunk['end_position'], chunk['overlap_prev'], chunk['overlap_next']
                    ))
                    
                    chunk_ids.append(cursor.lastrowid)
                
                conn.commit()
                
                logger.info(f"Created {len(chunk_ids)} chunks for document {document_id}")
                return chunk_ids
                
        except Exception as e:
            logger.error(f"Failed to create chunks for document {document_id}: {e}")
            raise
    
    def get_chunk(self, chunk_id: int) -> Optional[ChunkData]:
        """
        Get a chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            ChunkData or None if not found
        """
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, document_id, chunk_index, text, chunk_type,
                       start_position, end_position, overlap_prev, overlap_next
                FROM chunks 
                WHERE id = ? AND is_active = 1
            """, (chunk_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            chunk = ChunkData(
                id=row[0],
                document_id=row[1],
                chunk_index=row[2],
                text=row[3],
                chunk_type=row[4],
                start_position=row[5],
                end_position=row[6],
                overlap_prev=row[7],
                overlap_next=row[8]
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None
    
    def get_document_chunks(self, document_id: int) -> List[ChunkData]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of ChunkData objects
        """
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, document_id, chunk_index, text, chunk_type,
                       start_position, end_position, overlap_prev, overlap_next
                FROM chunks 
                WHERE document_id = ? AND is_active = 1
                ORDER BY chunk_index
            """, (document_id,))
            
            rows = cursor.fetchall()
            chunks = []
            
            for row in rows:
                chunk = ChunkData(
                    id=row[0],
                    document_id=row[1],
                    chunk_index=row[2],
                    text=row[3],
                    chunk_type=row[4],
                    start_position=row[5],
                    end_position=row[6],
                    overlap_prev=row[7],
                    overlap_next=row[8]
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {e}")
            return []
    
    def generate_and_store_embeddings(self, document_id: int, 
                                    model_name: str = None) -> List[bool]:
        """
        Generate and store embeddings for all chunks of a document using batch processing.
        
        Args:
            document_id: Document ID to process
            model_name: Embedding model name (optional)
            
        Returns:
            List of success flags for each chunk
        """
        try:
            # Get all chunks for the document
            chunks = self.get_document_chunks(document_id)
            if not chunks:
                logger.warning(f"No chunks found for document {document_id}")
                return []
            
            # Extract texts for batch processing
            chunk_texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings in batch
            embeddings = self.generate_embeddings_batch(chunk_texts, model_name)
            
            # Store embeddings
            results = []
            for chunk, embedding in zip(chunks, embeddings):
                success = self.store_embedding(chunk.id, embedding)
                results.append(success)
            
            successful_count = sum(results)
            logger.info(f"Generated and stored {successful_count}/{len(chunks)} embeddings for document {document_id}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate and store embeddings for document {document_id}: {e}")
            return []
    
    # Embedding Management
    def generate_embedding(self, text: str, model_name: str = None) -> np.ndarray:
        """
        Generate an embedding for text using the embedding service.
        
        Args:
            text: Text to embed
            model_name: Embedding model name (optional)
            
        Returns:
            numpy array: Embedding vector
        """
        try:
            # Use embedding service if available
            if self._embedding_service:
                # Determine model type
                model_type = None
                if model_name:
                    try:
                        model_type = EmbeddingModelType(model_name)
                    except ValueError:
                        logger.warning(f"Unknown model name '{model_name}', using default")
                        model_type = None
                
                # Generate embedding
                response = self._embedding_service.generate_embedding(text, model_type)
                
                if response.success:
                    logger.debug(f"Generated embedding for text (length: {len(text)}) "
                               f"using model {response.model_used} in {response.processing_time:.3f}s")
                    return response.embedding
                else:
                    logger.error(f"Embedding generation failed: {response.error_message}")
                    # Fall through to mock implementation
            
            # Fallback to mock implementation
            logger.warning("Using mock embedding - embedding service unavailable")
            # Return mock embedding with dimension matching BGE-base-en-v1.5
            return np.random.rand(768).astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return mock embedding as fallback
            return np.random.rand(768).astype(np.float32)
    
    def generate_embeddings_batch(self, texts: List[str], 
                                 model_name: str = None) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in a batch for efficiency.
        
        Args:
            texts: List of texts to embed
            model_name: Embedding model name (optional)
            
        Returns:
            List of numpy arrays: Embedding vectors
        """
        try:
            # Use embedding service if available
            if self._embedding_service:
                # Create batch requests
                requests = [
                    EmbeddingRequest(text=text, model_name=model_name) 
                    for text in texts
                ]
                
                # Generate embeddings in batch
                responses = self._embedding_service.generate_embeddings(requests)
                
                # Extract embeddings
                embeddings = []
                for response in responses:
                    if response.success:
                        embeddings.append(response.embedding)
                    else:
                        logger.warning(f"Failed to generate embedding in batch: {response.error_message}")
                        # Use mock embedding as fallback
                        embeddings.append(np.random.rand(768).astype(np.float32))
                
                logger.debug(f"Generated {len(embeddings)} embeddings in batch")
                return embeddings
            
            # Fallback to individual generation
            logger.warning("Using individual embedding generation - service unavailable")
            return [self.generate_embedding(text, model_name) for text in texts]
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings in batch: {e}")
            # Return mock embeddings as fallback
            return [np.random.rand(768).astype(np.float32) for _ in texts]
    
    def store_embedding(self, chunk_id: int, embedding: np.ndarray, 
                       model_id: int = 1) -> bool:
        """
        Store an embedding for a chunk.
        
        Args:
            chunk_id: Chunk ID
            embedding: Embedding vector
            model_id: Embedding model ID
            
        Returns:
            bool: True if successful
        """
        try:
            with self._lock:
                conn = self.schema.get_connection()
                cursor = conn.cursor()
                
                embedding_blob = serialize_embedding(embedding)
                embedding_norm = float(np.linalg.norm(embedding))
                
                cursor.execute("""
                    INSERT OR REPLACE INTO embeddings (chunk_id, model_id, embedding, embedding_norm)
                    VALUES (?, ?, ?, ?)
                """, (chunk_id, model_id, embedding_blob, embedding_norm))
                
                conn.commit()
                
                logger.debug(f"Stored embedding for chunk {chunk_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store embedding for chunk {chunk_id}: {e}")
            return False
    
    def get_embedding(self, chunk_id: int) -> Optional[np.ndarray]:
        """
        Get embedding for a chunk.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            numpy array or None if not found
        """
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT embedding FROM embeddings WHERE chunk_id = ?
            """, (chunk_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return deserialize_embedding(row[0])
            
        except Exception as e:
            logger.error(f"Failed to get embedding for chunk {chunk_id}: {e}")
            return None
    
    # Search and Retrieval (placeholder - will be implemented in next subtask)
    def search_semantic(self, query: str, k: int = 5, 
                       threshold: float = 0.7) -> List[SearchResult]:
        """
        Perform semantic search across all documents using embeddings.
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Generate embedding for query
            query_embedding = self.generate_embedding(query)
            
            # Get all embeddings from database
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT e.chunk_id, e.embedding, c.text, d.id as document_id, 
                       d.title, d.source
                FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE c.is_active = 1 AND d.is_active = 1
            """)
            
            rows = cursor.fetchall()
            if not rows:
                logger.info("No embeddings found for semantic search")
                return []
            
            # Calculate similarities
            similarities = []
            for row in rows:
                chunk_id, embedding_blob, chunk_text, doc_id, doc_title, doc_source = row
                
                try:
                    # Deserialize stored embedding
                    stored_embedding = deserialize_embedding(embedding_blob)
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(query_embedding, stored_embedding)
                    
                    if similarity >= threshold:
                        similarities.append({
                            'chunk_id': chunk_id,
                            'chunk_text': chunk_text,
                            'document_id': doc_id,
                            'document_title': doc_title,
                            'document_source': doc_source,
                            'similarity': float(similarity)
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to process embedding for chunk {chunk_id}: {e}")
                    continue
            
            # Sort by similarity (highest first) and limit results
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            similarities = similarities[:k]
            
            # Convert to SearchResult objects
            results = []
            for sim in similarities:
                result = SearchResult(
                    chunk_id=sim['chunk_id'],
                    chunk_text=sim['chunk_text'],
                    similarity_score=sim['similarity'],
                    document_id=sim['document_id'],
                    document_title=sim['document_title'],
                    document_source=sim['document_source'],
                    metadata={
                        'search_type': 'semantic',
                        'query': query,
                        'threshold': threshold
                    }
                )
                results.append(result)
            
            # Log access
            self._log_access(self._current_user_id, EntityType.CHUNK, 
                           0, AccessAction.SEARCH)
            
            logger.info(f"Semantic search returned {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []
    
    def search_keyword(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        Perform keyword search using FTS.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            # Use FTS for keyword search
            cursor.execute("""
                SELECT c.id, c.text, d.id as document_id, d.title, d.source
                FROM chunks_fts f
                JOIN chunks c ON f.rowid = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, k))
            
            rows = cursor.fetchall()
            results = []
            
            for row in rows:
                result = SearchResult(
                    chunk_id=row[0],
                    chunk_text=row[1],
                    similarity_score=1.0,  # FTS doesn't provide similarity scores
                    document_id=row[2],
                    document_title=row[3],
                    document_source=row[4]
                )
                results.append(result)
            
            self._log_access(self._current_user_id, EntityType.CHUNK, 
                           0, AccessAction.SEARCH)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform keyword search: {e}")
            return []
    
    # Privacy and Access Control
    def _log_access(self, user_id: Optional[int], entity_type: EntityType, 
                   entity_id: int, action: AccessAction, 
                   success: bool = True, error_message: str = None):
        """
        Log access to entities for audit purposes.
        
        Args:
            user_id: User ID
            entity_type: Type of entity accessed
            entity_id: Entity ID
            action: Action performed
            success: Whether the action was successful
            error_message: Error message if unsuccessful
        """
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO access_logs (
                    user_id, entity_type, entity_id, action, success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, entity_type.value, entity_id, action.value, success, error_message))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to log access: {e}")
    
    # Statistics and Monitoring
    def get_memory_stats(self) -> MemoryStats:
        """
        Get memory system statistics.
        
        Returns:
            MemoryStats object
        """
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM conversations WHERE is_active = 1")
            total_conversations = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM messages")
            total_messages = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM documents WHERE is_active = 1")
            total_documents = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE is_active = 1")
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            total_embeddings = cursor.fetchone()[0]
            
            # Get database size
            db_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
            
            stats = MemoryStats(
                total_conversations=total_conversations,
                total_messages=total_messages,
                total_documents=total_documents,
                total_chunks=total_chunks,
                total_embeddings=total_embeddings,
                db_size_mb=db_size_mb,
                last_updated=datetime.now()
            )
            
            self._stats = stats
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return self._stats
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the memory system.
        
        Returns:
            Dict with health check results
        """
        try:
            health = {
                "status": "healthy",
                "issues": [],
                "database_connected": False,
                "schema_valid": False,
                "stats": {}
            }
            
            # Check database connection
            try:
                conn = self.schema.get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                health["database_connected"] = True
            except Exception as e:
                health["status"] = "unhealthy"
                health["issues"].append(f"Database connection failed: {e}")
            
            # Check schema validity
            try:
                schema_validation = self.schema.validate_schema()
                health["schema_valid"] = schema_validation["valid"]
                if not schema_validation["valid"]:
                    health["issues"].extend(schema_validation["issues"])
            except Exception as e:
                health["status"] = "unhealthy"
                health["issues"].append(f"Schema validation failed: {e}")
            
            # Get stats
            try:
                stats = self.get_memory_stats()
                health["stats"] = asdict(stats)
            except Exception as e:
                health["issues"].append(f"Failed to get stats: {e}")
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "issues": [f"Health check failed: {e}"],
                "database_connected": False,
                "schema_valid": False,
                "stats": {}
            }
    
    # Database Utility Methods for Import/Export
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            bool: True if table exists, False otherwise
        """
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,))
            
            return cursor.fetchone() is not None
            
        except Exception as e:
            logger.error(f"Failed to check if table {table_name} exists: {e}")
            return False
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """
        Get column names for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column names
        """
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            return [column[1] for column in columns]  # column[1] is the column name
            
        except Exception as e:
            logger.error(f"Failed to get columns for table {table_name}: {e}")
            return []
    
    def insert_record(self, table_name: str, record: Dict[str, Any]) -> bool:
        """
        Insert a record into a table.
        
        Args:
            table_name: Name of the table
            record: Dictionary of column -> value
            
        Returns:
            bool: True if successful
        """
        try:
            with self._lock:
                conn = self.schema.get_connection()
                cursor = conn.cursor()
                
                columns = list(record.keys())
                values = list(record.values())
                placeholders = ', '.join(['?' for _ in columns])
                columns_str = ', '.join(columns)
                
                sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
                cursor.execute(sql, values)
                conn.commit()
                
                logger.debug(f"Inserted record into {table_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert record into {table_name}: {e}")
            return False
    
    def insert_or_replace_record(self, table_name: str, record: Dict[str, Any]) -> bool:
        """
        Insert or replace a record in a table.
        
        Args:
            table_name: Name of the table
            record: Dictionary of column -> value
            
        Returns:
            bool: True if successful
        """
        try:
            with self._lock:
                conn = self.schema.get_connection()
                cursor = conn.cursor()
                
                columns = list(record.keys())
                values = list(record.values())
                placeholders = ', '.join(['?' for _ in columns])
                columns_str = ', '.join(columns)
                
                sql = f"INSERT OR REPLACE INTO {table_name} ({columns_str}) VALUES ({placeholders})"
                cursor.execute(sql, values)
                conn.commit()
                
                logger.debug(f"Inserted or replaced record in {table_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert or replace record in {table_name}: {e}")
            return False
    
    def insert_or_ignore_record(self, table_name: str, record: Dict[str, Any]) -> bool:
        """
        Insert a record into a table, ignoring if it already exists.
        
        Args:
            table_name: Name of the table
            record: Dictionary of column -> value
            
        Returns:
            bool: True if successful
        """
        try:
            with self._lock:
                conn = self.schema.get_connection()
                cursor = conn.cursor()
                
                columns = list(record.keys())
                values = list(record.values())
                placeholders = ', '.join(['?' for _ in columns])
                columns_str = ', '.join(columns)
                
                sql = f"INSERT OR IGNORE INTO {table_name} ({columns_str}) VALUES ({placeholders})"
                cursor.execute(sql, values)
                conn.commit()
                
                logger.debug(f"Inserted record into {table_name} (ignoring duplicates)")
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert or ignore record in {table_name}: {e}")
            return False

    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

# Export main class
__all__ = ['MemoryManager', 'ConversationMeta', 'MessageData', 'DocumentData', 
           'ChunkData', 'SearchResult', 'MemoryStats', 'ConversationStatus', 
           'SearchMode', 'MessageSender'] 