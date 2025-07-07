"""
Tests for Context Window Manager - Intelligent Context Selection System

This module tests the context window management system including:
- Selection strategies (recency, relevance, hybrid, adaptive)
- Token counting and model limits
- User preferences and customization
- Integration with MemoryManager and VectorSearchEngine
- Performance optimization and caching
"""

import pytest
import tempfile
import asyncio
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from src.sovereign.context_window_manager import (
    ContextWindowManager, SelectionStrategy, TruncationStrategy,
    ModelContextLimits, UserPreferences, ContextItem, ContextWindow
)
from src.sovereign.memory_manager import (
    MemoryManager, MessageData, MessageSender, SearchResult
)
from src.sovereign.vector_search_engine import VectorSearchEngine, SearchParameters
from src.sovereign.config import Config


class TestModelContextLimits:
    """Test model context limits configuration"""
    
    def test_model_limits_creation(self):
        """Test creating model context limits"""
        limits = ModelContextLimits(
            model_name="gpt-4",
            total_tokens=8192,
            max_input_tokens=6000,
            max_output_tokens=2000
        )
        
        assert limits.model_name == "gpt-4"
        assert limits.total_tokens == 8192
        assert limits.max_input_tokens == 6000
        assert limits.max_output_tokens == 2000
        assert limits.encoding_name == "cl100k_base"
    
    def test_model_limits_validation(self):
        """Test validation of model limits"""
        # This should trigger a warning but not fail
        limits = ModelContextLimits(
            model_name="test",
            total_tokens=1000,
            max_input_tokens=800,
            max_output_tokens=400  # 800 + 400 > 1000
        )
        
        assert limits.total_tokens == 1000


class TestUserPreferences:
    """Test user preferences configuration"""
    
    def test_default_preferences(self):
        """Test default user preferences"""
        prefs = UserPreferences()
        
        assert prefs.relevance_weight == 0.6
        assert prefs.recency_weight == 0.4
        assert prefs.min_relevance_score == 0.3
        assert prefs.max_age_hours == 168
        assert prefs.preserve_conversation_flow is True
        assert prefs.include_system_messages is True
        assert prefs.priority_keywords == []
    
    def test_custom_preferences(self):
        """Test custom user preferences"""
        prefs = UserPreferences(
            relevance_weight=0.8,
            recency_weight=0.2,
            min_relevance_score=0.5,
            max_age_hours=24,
            preserve_conversation_flow=False
        )
        
        assert prefs.relevance_weight == 0.8
        assert prefs.recency_weight == 0.2
        assert prefs.min_relevance_score == 0.5
        assert prefs.max_age_hours == 24
        assert prefs.preserve_conversation_flow is False
    
    def test_weight_normalization(self):
        """Test weight normalization in preferences"""
        # Weights don't sum to 1.0, should still work
        prefs = UserPreferences(
            relevance_weight=0.7,
            recency_weight=0.5  # Sum is 1.2
        )
        
        # Values should be clamped to valid range
        assert 0.0 <= prefs.relevance_weight <= 1.0
        assert 0.0 <= prefs.recency_weight <= 1.0


class TestContextItem:
    """Test context item data structure"""
    
    def test_context_item_creation(self):
        """Test creating a context item"""
        message = MessageData(
            id=1,
            conversation_id=1,
            sender=MessageSender.USER,
            message="Test message",
            created_at=datetime.now()
        )
        
        item = ContextItem(
            message=message,
            relevance_score=0.8,
            recency_score=0.6,
            combined_score=0.7,
            token_count=10
        )
        
        assert item.message == message
        assert item.relevance_score == 0.8
        assert item.recency_score == 0.6
        assert item.combined_score == 0.7
        assert item.token_count == 10
        assert item.selected is False
        assert item.truncated is False
    
    def test_context_item_to_dict(self):
        """Test context item serialization"""
        message = MessageData(
            id=1,
            conversation_id=1,
            sender=MessageSender.USER,
            message="Test message",
            created_at=datetime.now()
        )
        
        item = ContextItem(
            message=message,
            relevance_score=0.8,
            token_count=10,
            selected=True
        )
        
        data = item.to_dict()
        
        assert data['message_id'] == 1
        assert data['conversation_id'] == 1
        assert data['sender'] == 'user'
        assert data['content'] == "Test message"
        assert data['relevance_score'] == 0.8
        assert data['token_count'] == 10
        assert data['selected'] is True


class TestContextWindow:
    """Test context window data structure"""
    
    def test_context_window_creation(self):
        """Test creating a context window"""
        items = []
        window = ContextWindow(
            items=items,
            total_tokens=100,
            available_tokens=900,
            strategy_used=SelectionStrategy.HYBRID,
            created_at=datetime.now(),
            query="test query"
        )
        
        assert window.items == items
        assert window.total_tokens == 100
        assert window.available_tokens == 900
        assert window.strategy_used == SelectionStrategy.HYBRID
        assert window.query == "test query"
    
    def test_get_selected_items(self):
        """Test getting selected items"""
        message1 = MessageData(
            id=1, conversation_id=1, sender=MessageSender.USER, 
            message="First", created_at=datetime.now()
        )
        message2 = MessageData(
            id=2, conversation_id=1, sender=MessageSender.ASSISTANT, 
            message="Second", created_at=datetime.now()
        )
        
        item1 = ContextItem(message=message1, selected=True)
        item2 = ContextItem(message=message2, selected=False)
        
        window = ContextWindow(
            items=[item1, item2],
            total_tokens=50,
            available_tokens=950,
            strategy_used=SelectionStrategy.HYBRID,
            created_at=datetime.now()
        )
        
        selected = window.get_selected_items()
        assert len(selected) == 1
        assert selected[0] == item1
    
    def test_get_context_text(self):
        """Test generating context text"""
        now = datetime.now()
        message1 = MessageData(
            id=1, conversation_id=1, sender=MessageSender.USER, 
            message="Hello", created_at=now
        )
        message2 = MessageData(
            id=2, conversation_id=1, sender=MessageSender.ASSISTANT, 
            message="Hi there!", created_at=now + timedelta(seconds=1)
        )
        
        item1 = ContextItem(message=message1, selected=True)
        item2 = ContextItem(message=message2, selected=True)
        
        window = ContextWindow(
            items=[item1, item2],
            total_tokens=50,
            available_tokens=950,
            strategy_used=SelectionStrategy.HYBRID,
            created_at=datetime.now()
        )
        
        text = window.get_context_text()
        
        assert "USER: Hello" in text
        assert "ASSISTANT: Hi there!" in text
        assert text.index("USER: Hello") < text.index("ASSISTANT: Hi there!")


@pytest.fixture
def temp_db_path():
    """Create a temporary database file"""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_context.db"
    yield str(db_path)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Create a mock configuration"""
    config = Mock(spec=Config)
    config.database = Mock()
    config.database.db_path = "test.db"
    return config


@pytest.fixture
def mock_memory_manager():
    """Create a mock memory manager"""
    manager = Mock(spec=MemoryManager)
    
    # Mock methods
    manager.get_current_user.return_value = 1
    manager.get_current_conversation.return_value = 1
    
    # Create sample messages
    now = datetime.now()
    messages = [
        MessageData(
            id=1, conversation_id=1, sender=MessageSender.USER,
            message="What is machine learning?", 
            created_at=now - timedelta(minutes=5)
        ),
        MessageData(
            id=2, conversation_id=1, sender=MessageSender.ASSISTANT,
            message="Machine learning is a subset of AI...", 
            created_at=now - timedelta(minutes=4)
        ),
        MessageData(
            id=3, conversation_id=1, sender=MessageSender.USER,
            message="Can you explain neural networks?", 
            created_at=now - timedelta(minutes=2)
        ),
        MessageData(
            id=4, conversation_id=1, sender=MessageSender.ASSISTANT,
            message="Neural networks are computing systems...", 
            created_at=now - timedelta(minutes=1)
        )
    ]
    
    manager.get_conversation_messages.return_value = messages
    
    return manager


@pytest.fixture
def mock_vector_search_engine():
    """Create a mock vector search engine"""
    engine = Mock(spec=VectorSearchEngine)
    
    # Mock search method to return async result
    async def mock_search(query, params=None):
        # Return mock search results based on query
        if "machine learning" in query.lower():
            return [
                Mock(
                    content="Machine learning is a subset of AI...",
                    score=0.9,
                    chunk_id="chunk1"
                )
            ]
        elif "neural" in query.lower():
            return [
                Mock(
                    content="Neural networks are computing systems...",
                    score=0.85,
                    chunk_id="chunk2"
                )
            ]
        else:
            return [
                Mock(
                    content="Generic response...",
                    score=0.3,
                    chunk_id="chunk3"
                )
            ]
    
    engine.search = AsyncMock(side_effect=mock_search)
    
    return engine


class TestContextWindowManager:
    """Test the main Context Window Manager class"""
    
    def test_initialization(self, mock_memory_manager, mock_vector_search_engine, mock_config):
        """Test context window manager initialization"""
        manager = ContextWindowManager(
            memory_manager=mock_memory_manager,
            vector_search_engine=mock_vector_search_engine,
            config=mock_config
        )
        
        assert manager.memory_manager == mock_memory_manager
        assert manager.vector_search_engine == mock_vector_search_engine
        assert manager.config == mock_config
        assert len(manager.model_limits) > 0  # Should have default limits
        assert "gpt-4" in manager.model_limits
    
    def test_add_model_limits(self, mock_memory_manager, mock_vector_search_engine, mock_config):
        """Test adding custom model limits"""
        manager = ContextWindowManager(
            memory_manager=mock_memory_manager,
            vector_search_engine=mock_vector_search_engine,
            config=mock_config
        )
        
        custom_limits = ModelContextLimits(
            model_name="custom-model",
            total_tokens=16000,
            max_input_tokens=12000,
            max_output_tokens=4000
        )
        
        manager.add_model_limits("custom-model", custom_limits)
        
        assert "custom-model" in manager.model_limits
        assert manager.model_limits["custom-model"] == custom_limits
    
    def test_user_preferences(self, mock_memory_manager, mock_vector_search_engine, mock_config):
        """Test user preferences management"""
        manager = ContextWindowManager(
            memory_manager=mock_memory_manager,
            vector_search_engine=mock_vector_search_engine,
            config=mock_config
        )
        
        # Test default preferences
        default_prefs = manager.get_user_preferences(1)
        assert isinstance(default_prefs, UserPreferences)
        assert default_prefs.relevance_weight == 0.6
        
        # Test custom preferences
        custom_prefs = UserPreferences(
            relevance_weight=0.8,
            recency_weight=0.2,
            max_age_hours=24
        )
        
        manager.set_user_preferences(1, custom_prefs)
        retrieved_prefs = manager.get_user_preferences(1)
        
        assert retrieved_prefs.relevance_weight == 0.8
        assert retrieved_prefs.recency_weight == 0.2
        assert retrieved_prefs.max_age_hours == 24
    
    @patch('tiktoken.get_encoding')
    def test_token_counting(self, mock_get_encoding, mock_memory_manager, 
                           mock_vector_search_engine, mock_config):
        """Test token counting functionality"""
        # Mock the tiktoken encoding
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_get_encoding.return_value = mock_encoding
        
        manager = ContextWindowManager(
            memory_manager=mock_memory_manager,
            vector_search_engine=mock_vector_search_engine,
            config=mock_config
        )
        
        token_count = manager.count_tokens("Hello world", "gpt-4")
        assert token_count == 5
        
        # Test caching
        token_count2 = manager.count_tokens("Hello world", "gpt-4")
        assert token_count2 == 5
        
        # Should only call get_encoding once due to caching
        assert mock_get_encoding.call_count == 1
    
    def test_recency_score_calculation(self, mock_memory_manager, 
                                     mock_vector_search_engine, mock_config):
        """Test recency score calculation"""
        manager = ContextWindowManager(
            memory_manager=mock_memory_manager,
            vector_search_engine=mock_vector_search_engine,
            config=mock_config
        )
        
        now = datetime.now()
        
        # Recent message (1 hour ago)
        recent_message = MessageData(
            id=1, conversation_id=1, sender=MessageSender.USER,
            message="Recent", created_at=now - timedelta(hours=1)
        )
        
        # Old message (25 hours ago)
        old_message = MessageData(
            id=2, conversation_id=1, sender=MessageSender.USER,
            message="Old", created_at=now - timedelta(hours=25)
        )
        
        recent_score = manager._calculate_recency_score(recent_message, 24)
        old_score = manager._calculate_recency_score(old_message, 24)
        
        assert recent_score > old_score
        assert 0.0 <= recent_score <= 1.0
        assert old_score == 0.0  # Beyond max age
    
    @pytest.mark.asyncio
    async def test_relevance_score_calculation(self, mock_memory_manager, 
                                             mock_vector_search_engine, mock_config):
        """Test relevance score calculation"""
        manager = ContextWindowManager(
            memory_manager=mock_memory_manager,
            vector_search_engine=mock_vector_search_engine,
            config=mock_config
        )
        
        message = MessageData(
            id=1, conversation_id=1, sender=MessageSender.ASSISTANT,
            message="Machine learning is a subset of AI...",
            created_at=datetime.now()
        )
        
        # Test high relevance query
        relevance_score = await manager._calculate_relevance_score(
            message, "machine learning"
        )
        
        assert 0.0 <= relevance_score <= 1.0
        # Should get some score (mock returns 0.9 for ML queries)
        assert relevance_score > 0.0
    
    def test_score_combination(self, mock_memory_manager, 
                              mock_vector_search_engine, mock_config):
        """Test score combination logic"""
        manager = ContextWindowManager(
            memory_manager=mock_memory_manager,
            vector_search_engine=mock_vector_search_engine,
            config=mock_config
        )
        
        preferences = UserPreferences(relevance_weight=0.7, recency_weight=0.3)
        
        combined_score = manager._combine_scores(0.8, 0.6, preferences)
        
        # Should be weighted average: 0.8*0.7 + 0.6*0.3 = 0.56 + 0.18 = 0.74
        expected = 0.8 * 0.7 + 0.6 * 0.3
        assert abs(combined_score - expected) < 0.01
    
    @pytest.mark.asyncio
    async def test_build_context_window_basic(self, mock_memory_manager, 
                                            mock_vector_search_engine, mock_config):
        """Test basic context window building"""
        manager = ContextWindowManager(
            memory_manager=mock_memory_manager,
            vector_search_engine=mock_vector_search_engine,
            config=mock_config
        )
        
        # Mock token counting to return reasonable values
        manager.count_tokens = Mock(return_value=20)
        
        context_window = await manager.build_context_window(
            query="What is machine learning?",
            model_name="gpt-4",
            strategy=SelectionStrategy.HYBRID
        )
        
        assert isinstance(context_window, ContextWindow)
        assert context_window.query == "What is machine learning?"
        assert context_window.strategy_used == SelectionStrategy.HYBRID
        assert context_window.conversation_id == 1
        assert len(context_window.items) > 0
    
    @pytest.mark.asyncio
    async def test_selection_strategies(self, mock_memory_manager, 
                                      mock_vector_search_engine, mock_config):
        """Test different selection strategies"""
        manager = ContextWindowManager(
            memory_manager=mock_memory_manager,
            vector_search_engine=mock_vector_search_engine,
            config=mock_config
        )
        
        # Mock token counting
        manager.count_tokens = Mock(return_value=50)
        
        # Test recency-only strategy
        recency_window = await manager.build_context_window(
            query="test query",
            strategy=SelectionStrategy.RECENCY_ONLY
        )
        
        # Test relevance-only strategy
        relevance_window = await manager.build_context_window(
            query="test query",
            strategy=SelectionStrategy.RELEVANCE_ONLY
        )
        
        # Test hybrid strategy
        hybrid_window = await manager.build_context_window(
            query="test query",
            strategy=SelectionStrategy.HYBRID
        )
        
        # All should produce valid windows
        assert isinstance(recency_window, ContextWindow)
        assert isinstance(relevance_window, ContextWindow)
        assert isinstance(hybrid_window, ContextWindow)
        
        assert recency_window.strategy_used == SelectionStrategy.RECENCY_ONLY
        assert relevance_window.strategy_used == SelectionStrategy.RELEVANCE_ONLY
        assert hybrid_window.strategy_used == SelectionStrategy.HYBRID
    
    @pytest.mark.asyncio
    async def test_token_limit_enforcement(self, mock_memory_manager, 
                                         mock_vector_search_engine, mock_config):
        """Test that token limits are enforced"""
        manager = ContextWindowManager(
            memory_manager=mock_memory_manager,
            vector_search_engine=mock_vector_search_engine,
            config=mock_config
        )
        
        # Mock token counting to return high values
        manager.count_tokens = Mock(return_value=3000)  # High token count per message
        
        context_window = await manager.build_context_window(
            query="test query",
            model_name="gpt-4"  # Has 6000 max input tokens
        )
        
        # Should not exceed model limits
        assert context_window.total_tokens <= 6000
        
        # Should have selected fewer items due to token limits
        selected_items = context_window.get_selected_items()
        assert len(selected_items) <= 2  # With 3000 tokens each, max 2 items
    
    @pytest.mark.asyncio
    async def test_get_optimized_context(self, mock_memory_manager, 
                                       mock_vector_search_engine, mock_config):
        """Test getting optimized context text"""
        manager = ContextWindowManager(
            memory_manager=mock_memory_manager,
            vector_search_engine=mock_vector_search_engine,
            config=mock_config
        )
        
        # Mock token counting
        manager.count_tokens = Mock(return_value=20)
        
        context_text = await manager.get_optimized_context(
            query="What is machine learning?",
            model_name="gpt-4"
        )
        
        assert isinstance(context_text, str)
        if context_text:  # May be empty if no messages selected
            assert "USER:" in context_text or "ASSISTANT:" in context_text
    
    def test_statistics_tracking(self, mock_memory_manager, 
                                mock_vector_search_engine, mock_config):
        """Test statistics tracking"""
        manager = ContextWindowManager(
            memory_manager=mock_memory_manager,
            vector_search_engine=mock_vector_search_engine,
            config=mock_config
        )
        
        initial_stats = manager.get_statistics()
        
        assert 'total_requests' in initial_stats
        assert 'total_processing_time' in initial_stats
        assert 'selection_strategy_counts' in initial_stats
        assert initial_stats['total_requests'] == 0
    
    def test_cache_management(self, mock_memory_manager, 
                             mock_vector_search_engine, mock_config):
        """Test context cache management"""
        manager = ContextWindowManager(
            memory_manager=mock_memory_manager,
            vector_search_engine=mock_vector_search_engine,
            config=mock_config
        )
        
        # Initially empty
        assert len(manager.context_cache) == 0
        
        # Clear cache (should not error)
        manager.clear_cache()
        assert len(manager.context_cache) == 0
    
    def test_context_manager_protocol(self, mock_memory_manager, 
                                    mock_vector_search_engine, mock_config):
        """Test context manager protocol"""
        with ContextWindowManager(
            memory_manager=mock_memory_manager,
            vector_search_engine=mock_vector_search_engine,
            config=mock_config
        ) as manager:
            assert manager is not None
            assert hasattr(manager, 'memory_manager')
        
        # Should be closed after context


class TestContextWindowIntegration:
    """Integration tests for context window management"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, mock_memory_manager, 
                                     mock_vector_search_engine, mock_config):
        """Test complete context window workflow"""
        manager = ContextWindowManager(
            memory_manager=mock_memory_manager,
            vector_search_engine=mock_vector_search_engine,
            config=mock_config
        )
        
        # Mock token counting
        manager.count_tokens = Mock(return_value=25)
        
        # Set custom user preferences
        custom_prefs = UserPreferences(
            relevance_weight=0.8,
            recency_weight=0.2,
            min_relevance_score=0.1,
            max_age_hours=72
        )
        manager.set_user_preferences(1, custom_prefs)
        
        # Build context window
        context_window = await manager.build_context_window(
            query="Explain neural networks in detail",
            model_name="gpt-4",
            strategy=SelectionStrategy.HYBRID,
            user_id=1
        )
        
        # Verify results
        assert isinstance(context_window, ContextWindow)
        assert context_window.query == "Explain neural networks in detail"
        assert context_window.strategy_used == SelectionStrategy.HYBRID
        
        # Get formatted context
        context_text = context_window.get_context_text()
        
        # Should have some content if messages were selected
        if context_window.get_selected_items():
            assert len(context_text) > 0
            assert any(sender in context_text for sender in ["USER:", "ASSISTANT:"])
        
        # Check statistics were updated
        stats = manager.get_statistics()
        assert stats['total_requests'] >= 1
        assert stats['selection_strategy_counts']['hybrid'] >= 1
        
        # Test serialization
        window_dict = context_window.to_dict()
        assert 'items' in window_dict
        assert 'strategy_used' in window_dict
        assert 'total_tokens' in window_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 