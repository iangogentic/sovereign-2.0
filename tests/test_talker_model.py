"""
Tests for TalkerModel class
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add src to path for importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sovereign.talker_model import TalkerModel
from sovereign.ollama_client import OllamaError


class TestTalkerModel:
    """Test cases for TalkerModel"""
    
    def test_init(self):
        """Test TalkerModel initialization"""
        talker = TalkerModel()
        assert talker.model_name == "gemma2:9b"  # Default from config
        assert not talker.is_initialized
        assert talker.query_count == 0
        assert len(talker.response_times) == 0
    
    def test_custom_model_name(self):
        """Test TalkerModel with custom model name"""
        custom_model = "custom:model"
        talker = TalkerModel(model_name=custom_model)
        assert talker.model_name == custom_model
    
    def test_detect_complex_query_simple(self):
        """Test complexity detection for simple queries"""
        talker = TalkerModel()
        
        simple_queries = [
            "Hello, how are you?",
            "What's the weather like?",
            "Tell me a joke",
            "How old are you?",
        ]
        
        for query in simple_queries:
            assert not talker.detect_complex_query(query), f"Query '{query}' should be simple"
    
    def test_detect_complex_query_complex(self):
        """Test complexity detection for complex queries"""
        talker = TalkerModel()
        
        complex_queries = [
            "Write a Python function to sort a list",
            "Implement a binary search algorithm",
            "Analyze the performance of this code",
            "Search the web for the latest news",
            "Create a detailed plan for project management",
            "Explain the complex architecture of microservices",
            "How do I configure Docker for production deployment?",
        ]
        
        for query in complex_queries:
            assert talker.detect_complex_query(query), f"Query '{query}' should be complex"
    
    def test_detect_complex_query_length(self):
        """Test complexity detection based on query length"""
        talker = TalkerModel()
        
        # Very long query should be considered complex
        long_query = " ".join(["word"] * 60)  # 60 words
        assert talker.detect_complex_query(long_query)
    
    def test_detect_complex_query_multiple_questions(self):
        """Test complexity detection for multiple questions"""
        talker = TalkerModel()
        
        multi_question_query = "What is AI? How does it work? What are the benefits? Where is it used?"
        assert talker.detect_complex_query(multi_question_query)
    
    def test_get_performance_stats_empty(self):
        """Test performance stats when no queries have been made"""
        talker = TalkerModel()
        stats = talker.get_performance_stats()
        
        expected_stats = {
            "total_queries": 0,
            "average_response_time": 0.0,
            "fastest_response": 0.0,
            "slowest_response": 0.0,
            "sub_2s_percentage": 0.0
        }
        
        for key, expected_value in expected_stats.items():
            assert stats[key] == expected_value
    
    def test_get_performance_stats_with_data(self):
        """Test performance stats with mock response times"""
        talker = TalkerModel()
        
        # Mock some response times
        talker.response_times = [1.0, 1.5, 2.5, 0.8, 1.2]
        talker.query_count = 5
        
        stats = talker.get_performance_stats()
        
        assert stats["total_queries"] == 5
        assert stats["average_response_time"] == 1.4  # (1.0+1.5+2.5+0.8+1.2)/5
        assert stats["fastest_response"] == 0.8
        assert stats["slowest_response"] == 2.5
        assert stats["sub_2s_percentage"] == 80.0  # 4 out of 5 under 2s


class TestTalkerModelAsync:
    """Async test cases for TalkerModel"""
    
    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the TalkerModel"""
        talker = TalkerModel()
        
        # Mock the ollama_client close method
        talker.ollama_client.close = AsyncMock()
        
        await talker.close()
        
        assert not talker.is_initialized
        talker.ollama_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('sovereign.talker_model.TalkerModel._generate_complete')
    @patch('sovereign.ollama_client.OllamaClient.check_health')
    @patch('sovereign.ollama_client.OllamaClient.list_models')
    @patch('sovereign.ollama_client.OllamaClient.generate')
    async def test_initialize_success(self, mock_generate, mock_list_models, mock_check_health, mock_gen_complete):
        """Test successful initialization"""
        # Mock successful health check
        mock_check_health.return_value = True
        
        # Mock available models
        mock_list_models.return_value = {
            'models': [{'name': 'gemma2:9b'}, {'name': 'other:model'}]
        }
        
        # Mock successful test generation
        mock_generate.return_value = "Test response"
        
        talker = TalkerModel()
        result = await talker.initialize()
        
        assert result is True
        assert talker.is_initialized is True
        mock_check_health.assert_called_once()
        mock_list_models.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('sovereign.ollama_client.OllamaClient.check_health')
    async def test_initialize_unhealthy_server(self, mock_check_health):
        """Test initialization with unhealthy Ollama server"""
        # Mock unhealthy server
        mock_check_health.return_value = False
        
        talker = TalkerModel()
        result = await talker.initialize()
        
        assert result is False
        assert talker.is_initialized is False
    
    @pytest.mark.asyncio
    @patch('sovereign.talker_model.TalkerModel.initialize')
    @patch('sovereign.talker_model.TalkerModel._generate_complete')
    async def test_generate_response_auto_init(self, mock_gen_complete, mock_init):
        """Test generate_response with automatic initialization"""
        mock_init.return_value = True
        mock_gen_complete.return_value = "Test response"
        
        talker = TalkerModel()
        
        result = await talker.generate_response("Hello")
        
        assert result == "Test response"
        mock_init.assert_called_once()
        assert talker.query_count == 1
    
    @pytest.mark.asyncio
    @patch('sovereign.talker_model.TalkerModel.initialize')
    async def test_generate_response_init_failure(self, mock_init):
        """Test generate_response when initialization fails"""
        mock_init.return_value = False
        
        talker = TalkerModel()
        
        with pytest.raises(RuntimeError, match="Talker model not initialized"):
            await talker.generate_response("Hello")


def test_system_prompt():
    """Test that system prompt is properly set"""
    talker = TalkerModel()
    system_prompt = talker._get_system_prompt()
    
    assert "Sovereign" in system_prompt
    assert "local" in system_prompt.lower()
    assert "privacy" in system_prompt.lower()
    assert "2 seconds" in system_prompt
    assert "Thinker model" in system_prompt


if __name__ == "__main__":
    pytest.main([__file__]) 