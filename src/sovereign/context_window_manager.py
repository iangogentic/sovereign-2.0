"""
Context Window Manager for Sovereign AI Long-Term Memory (RAG) System

This module provides intelligent context window management for AI conversations,
ensuring optimal selection of relevant memory chunks while respecting model
context limits and user preferences.

Key Features:
- Token-aware context window management
- Relevance-based chunk selection using semantic similarity
- Recency-based selection for temporal relevance
- Hybrid selection strategies combining multiple factors
- Model-specific context limits
- User preference integration
- Smart truncation preserving message integrity
- Performance optimization with caching
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tiktoken

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .memory_manager import MemoryManager, MessageData, SearchResult, ConversationMeta
    from .vector_search_engine import VectorSearchEngine, SearchParameters
    from .config import Config

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Context window selection strategies"""
    RECENCY_ONLY = "recency_only"      # Most recent messages only
    RELEVANCE_ONLY = "relevance_only"  # Most relevant messages only
    HYBRID = "hybrid"                  # Balanced recency and relevance
    ADAPTIVE = "adaptive"              # Dynamic strategy based on context
    USER_DEFINED = "user_defined"      # Custom user-defined strategy


class TruncationStrategy(Enum):
    """Text truncation strategies"""
    HARD_CUTOFF = "hard_cutoff"        # Cut at exact token limit
    SENTENCE_BOUNDARY = "sentence_boundary"  # Cut at sentence boundaries
    MESSAGE_BOUNDARY = "message_boundary"    # Cut at message boundaries
    SMART_SUMMARIZE = "smart_summarize"      # Summarize older content


@dataclass
class ModelContextLimits:
    """Context limits for different AI models"""
    model_name: str
    total_tokens: int
    max_input_tokens: int
    max_output_tokens: int
    tokens_per_message_overhead: int = 4  # Typical overhead per message
    encoding_name: str = "cl100k_base"    # Default encoding for GPT models
    
    def __post_init__(self):
        """Validate limits"""
        if self.max_input_tokens + self.max_output_tokens > self.total_tokens:
            logger.warning(f"Input + output tokens exceed total for {self.model_name}")


@dataclass
class UserPreferences:
    """User preferences for context window management"""
    relevance_weight: float = 0.6      # Weight for relevance vs recency (0-1)
    recency_weight: float = 0.4        # Weight for recency vs relevance (0-1)
    min_relevance_score: float = 0.3   # Minimum relevance score to include
    max_age_hours: int = 168           # Maximum age in hours (7 days default)
    preserve_conversation_flow: bool = True  # Keep conversation pairs together
    include_system_messages: bool = True     # Include system messages in context
    priority_keywords: List[str] = field(default_factory=list)  # Important keywords
    
    def __post_init__(self):
        """Validate preferences"""
        if abs(self.relevance_weight + self.recency_weight - 1.0) > 0.01:
            logger.warning("Relevance and recency weights should sum to 1.0")
        self.relevance_weight = max(0.0, min(1.0, self.relevance_weight))
        self.recency_weight = max(0.0, min(1.0, self.recency_weight))


@dataclass
class ContextItem:
    """Represents an item in the context window"""
    message: 'MessageData'
    relevance_score: float = 0.0
    recency_score: float = 0.0
    combined_score: float = 0.0
    token_count: int = 0
    selected: bool = False
    truncated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'message_id': self.message.id,
            'conversation_id': self.message.conversation_id,
            'sender': self.message.sender.value,
            'content': self.message.message,
            'created_at': self.message.created_at.isoformat(),
            'relevance_score': self.relevance_score,
            'recency_score': self.recency_score,
            'combined_score': self.combined_score,
            'token_count': self.token_count,
            'selected': self.selected,
            'truncated': self.truncated,
            'metadata': self.metadata
        }


@dataclass
class ContextWindow:
    """Represents a complete context window"""
    items: List[ContextItem]
    total_tokens: int
    available_tokens: int
    strategy_used: SelectionStrategy
    created_at: datetime
    query: str = ""
    conversation_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_selected_items(self) -> List[ContextItem]:
        """Get only selected items"""
        return [item for item in self.items if item.selected]
    
    def get_context_text(self) -> str:
        """Get formatted context text"""
        selected_items = self.get_selected_items()
        if not selected_items:
            return ""
        
        # Sort by message timestamp for chronological order
        selected_items.sort(key=lambda x: x.message.created_at)
        
        context_parts = []
        for item in selected_items:
            sender = item.message.sender.value.upper()
            content = item.message.message
            timestamp = item.message.created_at.strftime("%Y-%m-%d %H:%M:%S")
            
            if item.truncated:
                content += " [TRUNCATED]"
            
            context_parts.append(f"[{timestamp}] {sender}: {content}")
        
        return "\n".join(context_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'items': [item.to_dict() for item in self.items],
            'total_tokens': self.total_tokens,
            'available_tokens': self.available_tokens,
            'strategy_used': self.strategy_used.value,
            'created_at': self.created_at.isoformat(),
            'query': self.query,
            'conversation_id': self.conversation_id,
            'metadata': self.metadata
        }


class ContextWindowManager:
    """
    Intelligent context window management system.
    
    Provides advanced context window management with support for:
    - Multiple selection strategies
    - Token counting and model limits
    - Relevance-based selection
    - User preferences
    - Smart truncation
    - Performance optimization
    """
    
    # Default model limits for common AI models
    DEFAULT_MODEL_LIMITS = {
        "gpt-3.5-turbo": ModelContextLimits(
            model_name="gpt-3.5-turbo",
            total_tokens=4096,
            max_input_tokens=3000,
            max_output_tokens=1000,
            encoding_name="cl100k_base"
        ),
        "gpt-4": ModelContextLimits(
            model_name="gpt-4",
            total_tokens=8192,
            max_input_tokens=6000,
            max_output_tokens=2000,
            encoding_name="cl100k_base"
        ),
        "gpt-4-turbo": ModelContextLimits(
            model_name="gpt-4-turbo",
            total_tokens=128000,
            max_input_tokens=100000,
            max_output_tokens=28000,
            encoding_name="cl100k_base"
        ),
        "claude-3-sonnet": ModelContextLimits(
            model_name="claude-3-sonnet",
            total_tokens=200000,
            max_input_tokens=150000,
            max_output_tokens=50000,
            encoding_name="cl100k_base"  # Approximation
        ),
        "claude-3-opus": ModelContextLimits(
            model_name="claude-3-opus",
            total_tokens=200000,
            max_input_tokens=150000,
            max_output_tokens=50000,
            encoding_name="cl100k_base"  # Approximation
        )
    }
    
    def __init__(self, 
                 memory_manager: 'MemoryManager',
                 vector_search_engine: 'VectorSearchEngine',
                 config: 'Config'):
        """
        Initialize the Context Window Manager.
        
        Args:
            memory_manager: Memory manager instance
            vector_search_engine: Vector search engine instance
            config: Configuration object
        """
        self.memory_manager = memory_manager
        self.vector_search_engine = vector_search_engine
        self.config = config
        
        # Model limits and encoding
        self.model_limits: Dict[str, ModelContextLimits] = self.DEFAULT_MODEL_LIMITS.copy()
        self.encodings: Dict[str, tiktoken.Encoding] = {}
        
        # User preferences
        self.user_preferences: Dict[int, UserPreferences] = {}
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'average_context_size': 0.0,
            'selection_strategy_counts': {strategy.value: 0 for strategy in SelectionStrategy}
        }
        
        # Caching for performance
        self.context_cache: Dict[str, ContextWindow] = {}
        self.cache_ttl = 300  # 5 minutes
        self.max_cache_size = 100
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Context Window Manager initialized")
    
    def add_model_limits(self, model_name: str, limits: ModelContextLimits) -> None:
        """Add or update model context limits"""
        self.model_limits[model_name] = limits
        logger.info(f"Updated limits for model {model_name}: {limits.total_tokens} tokens")
    
    def set_user_preferences(self, user_id: int, preferences: UserPreferences) -> None:
        """Set user preferences for context window management"""
        self.user_preferences[user_id] = preferences
        logger.info(f"Updated preferences for user {user_id}")
    
    def get_user_preferences(self, user_id: int) -> UserPreferences:
        """Get user preferences, returning defaults if not set"""
        return self.user_preferences.get(user_id, UserPreferences())
    
    def _get_encoding(self, encoding_name: str) -> tiktoken.Encoding:
        """Get token encoding, caching for performance"""
        if encoding_name not in self.encodings:
            try:
                self.encodings[encoding_name] = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(f"Failed to get encoding {encoding_name}: {e}")
                # Fallback to basic encoding
                self.encodings[encoding_name] = tiktoken.get_encoding("cl100k_base")
        
        return self.encodings[encoding_name]
    
    def count_tokens(self, text: str, model_name: str = "gpt-4") -> int:
        """Count tokens in text using appropriate encoding"""
        try:
            limits = self.model_limits.get(model_name, self.model_limits["gpt-4"])
            encoding = self._get_encoding(limits.encoding_name)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback to rough estimation
            return len(text) // 4
    
    def _calculate_recency_score(self, message: 'MessageData', max_age_hours: int) -> float:
        """Calculate recency score for a message (0.0 to 1.0)"""
        try:
            now = datetime.now()
            age = now - message.created_at
            age_hours = age.total_seconds() / 3600
            
            if age_hours > max_age_hours:
                return 0.0
            
            # Exponential decay based on age
            return max(0.0, 1.0 - (age_hours / max_age_hours))
        except Exception as e:
            logger.warning(f"Recency score calculation failed: {e}")
            return 0.0
    
    async def _calculate_relevance_score(self, message: 'MessageData', query: str) -> float:
        """Calculate relevance score for a message using semantic similarity"""
        try:
            if not query.strip():
                return 0.5  # Neutral score for empty query
            
            # Search for similar content
            from .vector_search_engine import SearchParameters
            search_params = SearchParameters(
                top_k=1,
                include_metadata=False
            )
            
            # Use vector search engine to find similarity
            results = await self.vector_search_engine.search(query, search_params)
            
            # Find result matching this message's content
            for result in results:
                if result.content and result.content.strip() == message.message.strip():
                    return min(1.0, result.score)
            
            # If no direct match, calculate similarity via embedding
            # This would require generating embeddings for the message
            # For now, return a default score
            return 0.3
            
        except Exception as e:
            logger.warning(f"Relevance score calculation failed: {e}")
            return 0.3
    
    def _combine_scores(self, relevance_score: float, recency_score: float, 
                       preferences: UserPreferences) -> float:
        """Combine relevance and recency scores based on user preferences"""
        combined = (relevance_score * preferences.relevance_weight + 
                   recency_score * preferences.recency_weight)
        return max(0.0, min(1.0, combined))
    
    def _should_preserve_conversation_pair(self, messages: List['MessageData'], 
                                          index: int) -> bool:
        """Check if message should be preserved as part of conversation pair"""
        if index <= 0 or index >= len(messages):
            return False
        
        current_msg = messages[index]
        prev_msg = messages[index - 1]
        
        # Check if this is a user-assistant pair
        if (prev_msg.sender.value == "user" and 
            current_msg.sender.value == "assistant"):
            return True
        
        # Check if messages are close in time (within 2 minutes)
        time_diff = current_msg.created_at - prev_msg.created_at
        return time_diff.total_seconds() < 120
    
    async def _apply_selection_strategy(self, 
                                      context_items: List[ContextItem],
                                      strategy: SelectionStrategy,
                                      available_tokens: int,
                                      preferences: UserPreferences) -> List[ContextItem]:
        """Apply selection strategy to context items"""
        if strategy == SelectionStrategy.RECENCY_ONLY:
            # Sort by recency score (most recent first)
            context_items.sort(key=lambda x: x.recency_score, reverse=True)
        
        elif strategy == SelectionStrategy.RELEVANCE_ONLY:
            # Sort by relevance score (most relevant first)
            context_items.sort(key=lambda x: x.relevance_score, reverse=True)
        
        elif strategy == SelectionStrategy.HYBRID:
            # Sort by combined score
            context_items.sort(key=lambda x: x.combined_score, reverse=True)
        
        elif strategy == SelectionStrategy.ADAPTIVE:
            # Adaptive strategy based on context
            if not any(item.relevance_score > 0.7 for item in context_items):
                # No highly relevant items, favor recency
                context_items.sort(key=lambda x: x.recency_score, reverse=True)
            else:
                # High relevance items exist, use hybrid
                context_items.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Select items within token budget
        selected_items = []
        used_tokens = 0
        
        for item in context_items:
            if used_tokens + item.token_count <= available_tokens:
                item.selected = True
                selected_items.append(item)
                used_tokens += item.token_count
            else:
                item.selected = False
        
        return selected_items
    
    async def build_context_window(self,
                                 query: str,
                                 conversation_id: Optional[int] = None,
                                 model_name: str = "gpt-4",
                                 strategy: SelectionStrategy = SelectionStrategy.HYBRID,
                                 max_messages: int = 50,
                                 user_id: Optional[int] = None) -> ContextWindow:
        """
        Build an intelligent context window for the given query and conversation.
        
        Args:
            query: The query or prompt for context
            conversation_id: Target conversation ID
            model_name: Target AI model name
            strategy: Selection strategy to use
            max_messages: Maximum messages to consider
            user_id: User ID for preferences
            
        Returns:
            ContextWindow object with selected items
        """
        start_time = time.time()
        
        try:
            # Get model limits
            limits = self.model_limits.get(model_name, self.model_limits["gpt-4"])
            
            # Get user preferences
            if user_id is None:
                user_id = self.memory_manager.get_current_user()
            preferences = self.get_user_preferences(user_id or 0)
            
            # Get conversation messages
            target_conversation_id = conversation_id or self.memory_manager.get_current_conversation()
            if not target_conversation_id:
                logger.warning("No conversation specified for context window")
                return ContextWindow(
                    items=[],
                    total_tokens=0,
                    available_tokens=limits.max_input_tokens,
                    strategy_used=strategy,
                    created_at=datetime.now(),
                    query=query
                )
            
            # Get recent messages
            messages = self.memory_manager.get_conversation_messages(
                target_conversation_id, 
                limit=max_messages
            )
            
            if not messages:
                logger.info("No messages found for context window")
                return ContextWindow(
                    items=[],
                    total_tokens=0,
                    available_tokens=limits.max_input_tokens,
                    strategy_used=strategy,
                    created_at=datetime.now(),
                    query=query,
                    conversation_id=target_conversation_id
                )
            
            # Create context items
            context_items = []
            
            for message in messages:
                # Skip messages that are too old
                age_hours = (datetime.now() - message.created_at).total_seconds() / 3600
                if age_hours > preferences.max_age_hours:
                    continue
                
                # Calculate scores
                recency_score = self._calculate_recency_score(message, preferences.max_age_hours)
                relevance_score = await self._calculate_relevance_score(message, query)
                combined_score = self._combine_scores(relevance_score, recency_score, preferences)
                
                # Skip low-relevance messages
                if relevance_score < preferences.min_relevance_score and recency_score < 0.8:
                    continue
                
                # Count tokens
                token_count = self.count_tokens(message.message, model_name)
                
                context_item = ContextItem(
                    message=message,
                    relevance_score=relevance_score,
                    recency_score=recency_score,
                    combined_score=combined_score,
                    token_count=token_count
                )
                
                context_items.append(context_item)
            
            # Apply selection strategy
            selected_items = await self._apply_selection_strategy(
                context_items, strategy, limits.max_input_tokens, preferences
            )
            
            # Calculate total tokens
            total_tokens = sum(item.token_count for item in selected_items)
            
            # Create context window
            context_window = ContextWindow(
                items=context_items,
                total_tokens=total_tokens,
                available_tokens=limits.max_input_tokens - total_tokens,
                strategy_used=strategy,
                created_at=datetime.now(),
                query=query,
                conversation_id=target_conversation_id,
                metadata={
                    'model_name': model_name,
                    'processing_time': time.time() - start_time,
                    'user_preferences': preferences.__dict__,
                    'selected_count': len(selected_items),
                    'total_candidates': len(context_items)
                }
            )
            
            # Update statistics
            self.stats['total_requests'] += 1
            self.stats['total_processing_time'] += time.time() - start_time
            self.stats['selection_strategy_counts'][strategy.value] += 1
            self.stats['average_context_size'] = (
                (self.stats['average_context_size'] * (self.stats['total_requests'] - 1) + 
                 len(selected_items)) / self.stats['total_requests']
            )
            
            logger.info(f"Built context window: {len(selected_items)} items, "
                       f"{total_tokens} tokens, strategy: {strategy.value}")
            
            return context_window
            
        except Exception as e:
            logger.error(f"Failed to build context window: {e}")
            raise
    
    async def get_optimized_context(self,
                                  query: str,
                                  conversation_id: Optional[int] = None,
                                  model_name: str = "gpt-4",
                                  **kwargs) -> str:
        """
        Get optimized context text for a query.
        
        This is a convenience method that builds a context window
        and returns the formatted text.
        
        Args:
            query: The query for context
            conversation_id: Target conversation ID
            model_name: Target AI model name
            **kwargs: Additional arguments for build_context_window
            
        Returns:
            Formatted context text
        """
        context_window = await self.build_context_window(
            query=query,
            conversation_id=conversation_id,
            model_name=model_name,
            **kwargs
        )
        
        return context_window.get_context_text()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.stats,
            'cache_size': len(self.context_cache),
            'models_configured': len(self.model_limits),
            'user_preferences_count': len(self.user_preferences)
        }
    
    def clear_cache(self) -> None:
        """Clear context cache"""
        self.context_cache.clear()
        logger.info("Context cache cleared")
    
    def close(self) -> None:
        """Clean up resources"""
        self.context_cache.clear()
        if self.executor:
            self.executor.shutdown(wait=False)
        logger.info("Context Window Manager closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 