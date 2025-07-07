"""
Unit tests for ExternalModelConnector
"""

import pytest
import asyncio
import json
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from aioresponses import aioresponses

from src.sovereign.external_model_connector import (
    ExternalModelConnector,
    ExternalRoutingCriteria,
    RoutingDecision,
    ExternalRequest,
    ExternalResponse
)
from src.sovereign.config import Config


class TestExternalModelConnector:
    """Test suite for ExternalModelConnector"""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        config = Config()
        return config
    
    @pytest.fixture
    def connector(self, config):
        """Create an ExternalModelConnector instance for testing"""
        return ExternalModelConnector(config)
    
    @pytest.fixture
    def mock_openrouter_api_key(self):
        """Mock OpenRouter API key"""
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test_api_key'}):
            yield 'test_api_key'
    
    def test_init(self, connector):
        """Test ExternalModelConnector initialization"""
        assert connector.config is not None
        assert connector.openrouter_base_url == "https://openrouter.ai/api/v1"
        assert connector.default_external_model == "anthropic/claude-3-sonnet"
        assert connector.cache_ttl_hours == 24
        assert len(connector.response_cache) == 0
        assert connector.request_count == 0
        
    def test_routing_patterns_initialization(self, connector):
        """Test that routing patterns are properly initialized"""
        assert len(connector.specialized_knowledge_patterns) > 0
        assert len(connector.complex_tool_patterns) > 0
        assert len(connector.explicit_request_patterns) > 0
        assert len(connector.recent_info_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self, connector):
        """Test initialization fails gracefully without API key"""
        with patch.dict(os.environ, {}, clear=True):
            connector.openrouter_api_key = None
            result = await connector.initialize()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_initialize_with_api_key_success(self, connector, mock_openrouter_api_key):
        """Test successful initialization with API key"""
        with aioresponses() as m:
            # Mock the OpenRouter models endpoint
            m.get("https://openrouter.ai/api/v1/models", status=200, payload={"data": []})
            
            connector.openrouter_api_key = mock_openrouter_api_key
            result = await connector.initialize()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_initialize_with_api_key_failure(self, connector, mock_openrouter_api_key):
        """Test initialization failure with invalid API key"""
        with aioresponses() as m:
            # Mock the OpenRouter models endpoint with error
            m.get("https://openrouter.ai/api/v1/models", status=401, payload={"error": "Unauthorized"})
            
            connector.openrouter_api_key = mock_openrouter_api_key
            result = await connector.initialize()
            assert result is False
    
    def test_determine_external_need_specialized_knowledge(self, connector):
        """Test routing decision for specialized knowledge queries"""
        test_cases = [
            "What are the latest news about AI?",
            "What's the current stock price of Tesla?",
            "What happened in politics today?",
            "What's the weather forecast for tomorrow?",
            "Who won the championship this week?"
        ]
        
        for query in test_cases:
            decision = connector.determine_external_need(query)
            assert decision.should_route is True
            assert ExternalRoutingCriteria.SPECIALIZED_KNOWLEDGE in decision.criteria
            assert decision.confidence >= 0.3
    
    def test_determine_external_need_complex_tool_use(self, connector):
        """Test routing decision for complex tool use queries"""
        test_cases = [
            "Search online for the best Python frameworks",
            "Help me create a REST API endpoint",
            "Deploy this application to AWS",
            "Send an email to the team",
            "Generate a CSV file with this data"
        ]
        
        for query in test_cases:
            decision = connector.determine_external_need(query)
            assert decision.should_route is True
            assert ExternalRoutingCriteria.COMPLEX_TOOL_USE in decision.criteria
            assert decision.confidence >= 0.3
    
    def test_determine_external_need_explicit_request(self, connector):
        """Test routing decision for explicit external requests"""
        test_cases = [
            "Use external model to answer this",
            "Search online for this information",
            "I need the latest up-to-date information",
            "Use OpenRouter to generate this",
            "Use cloud model for this query"
        ]
        
        for query in test_cases:
            decision = connector.determine_external_need(query)
            assert decision.should_route is True
            assert ExternalRoutingCriteria.USER_EXPLICIT_REQUEST in decision.criteria
            assert decision.confidence >= 0.5
    
    def test_determine_external_need_recent_information(self, connector):
        """Test routing decision for recent information queries"""
        test_cases = [
            "What's happening in the world today?",
            "What are the recent developments in AI?",
            "What's the latest update on this project?",
            "What happened recently in technology?",
            "What are the current trends this month?"
        ]
        
        for query in test_cases:
            decision = connector.determine_external_need(query)
            assert decision.should_route is True
            assert ExternalRoutingCriteria.RECENT_INFORMATION in decision.criteria
            assert decision.confidence >= 0.3
    
    def test_determine_external_need_no_routing(self, connector):
        """Test routing decision for queries that don't need external routing"""
        test_cases = [
            "Hello, how are you?",
            "What is Python?",
            "Explain machine learning",
            "Write a simple function",
            "How do I use a for loop?"
        ]
        
        for query in test_cases:
            decision = connector.determine_external_need(query)
            assert decision.should_route is False
            assert len(decision.criteria) == 0
            assert decision.confidence < 0.3
    
    @pytest.mark.asyncio
    async def test_route_request_no_api_key(self, connector):
        """Test routing request without API key"""
        connector.openrouter_api_key = None
        
        response = await connector.route_request("Test query")
        
        assert response.success is False
        assert "no api key configured" in response.response.lower()
        assert response.provider == "openrouter"
    
    @pytest.mark.asyncio
    async def test_route_request_with_cache_hit(self, connector, mock_openrouter_api_key):
        """Test routing request returns cached response"""
        connector.openrouter_api_key = mock_openrouter_api_key
        
        # Pre-populate cache
        cache_key = connector._get_cache_key("test query")
        cached_response = ExternalResponse(
            response="Cached response",
            provider="openrouter",
            model="test-model",
            processing_time=1.0,
            success=True
        )
        connector._cache_response(cache_key, cached_response)
        
        response = await connector.route_request("test query")
        
        assert response.success is True
        assert response.response == "Cached response"
        assert response.cached is True
    
    @pytest.mark.asyncio
    async def test_route_request_successful(self, connector, mock_openrouter_api_key):
        """Test successful external request"""
        connector.openrouter_api_key = mock_openrouter_api_key
        
        mock_response_data = {
            'choices': [
                {
                    'message': {
                        'content': 'Test response from OpenRouter'
                    }
                }
            ]
        }
        
        with aioresponses() as m:
            # Mock the OpenRouter chat completions endpoint
            m.post("https://openrouter.ai/api/v1/chat/completions", status=200, payload=mock_response_data)
            
            response = await connector.route_request("Test query")
            
            assert response.success is True
            assert response.response == "Test response from OpenRouter"
            assert response.provider == "openrouter"
            assert connector.success_count == 1
    
    @pytest.mark.asyncio
    async def test_route_request_failure(self, connector, mock_openrouter_api_key):
        """Test failed external request"""
        connector.openrouter_api_key = mock_openrouter_api_key
        
        with aioresponses() as m:
            # Mock the OpenRouter chat completions endpoint with error
            m.post("https://openrouter.ai/api/v1/chat/completions", status=500, body="Internal server error")
            
            response = await connector.route_request("Test query")
            
            assert response.success is False
            assert "External service error" in response.response
            assert response.provider == "openrouter"
            assert connector.failure_count == 1
    
    def test_cache_key_generation(self, connector):
        """Test cache key generation"""
        key1 = connector._get_cache_key("test query")
        key2 = connector._get_cache_key("test query")
        key3 = connector._get_cache_key("different query")
        
        assert key1 == key2  # Same query should produce same key
        assert key1 != key3  # Different queries should produce different keys
        assert len(key1) == 32  # MD5 hash length
    
    def test_cache_response_and_retrieval(self, connector):
        """Test caching and retrieving responses"""
        response = ExternalResponse(
            response="Test response",
            provider="openrouter",
            model="test-model",
            processing_time=1.0,
            success=True
        )
        
        cache_key = "test_key"
        connector._cache_response(cache_key, response)
        
        # Retrieve cached response
        cached = connector._get_cached_response(cache_key)
        
        assert cached is not None
        assert cached.response == "Test response"
        assert cached.cached is True
    
    def test_cache_expiration(self, connector):
        """Test cache expiration logic"""
        response = ExternalResponse(
            response="Test response",
            provider="openrouter",
            model="test-model",
            processing_time=1.0,
            success=True
        )
        
        cache_key = "test_key"
        connector._cache_response(cache_key, response)
        
        # Manually set expired timestamp
        expired_time = datetime.now() - timedelta(hours=25)
        connector.response_cache[cache_key]['timestamp'] = expired_time.isoformat()
        
        # Try to retrieve - should return None due to expiration
        cached = connector._get_cached_response(cache_key)
        assert cached is None
        assert cache_key not in connector.response_cache  # Should be cleaned up
    
    def test_cache_cleaning(self, connector):
        """Test cache cleaning functionality"""
        # Add multiple cache entries with different timestamps
        current_time = datetime.now()
        
        for i in range(5):
            response = ExternalResponse(
                response=f"Response {i}",
                provider="openrouter", 
                model="test-model",
                processing_time=1.0,
                success=True
            )
            
            if i < 3:
                # First 3 are expired
                timestamp = current_time - timedelta(hours=25)
            else:
                # Last 2 are fresh
                timestamp = current_time - timedelta(hours=1)
            
            connector.response_cache[f"key_{i}"] = {
                'timestamp': timestamp.isoformat(),
                'response': {
                    'response': response.response,
                    'provider': response.provider,
                    'model': response.model,
                    'processing_time': response.processing_time,
                    'success': response.success,
                    'error': response.error
                }
            }
        
        assert len(connector.response_cache) == 5
        
        connector._clean_cache()
        
        # Should only have 2 fresh entries left
        assert len(connector.response_cache) == 2
        assert "key_3" in connector.response_cache
        assert "key_4" in connector.response_cache
    
    def test_consent_callback_management(self, connector):
        """Test user consent callback management"""
        callback1 = AsyncMock(return_value=True)
        callback2 = AsyncMock(return_value=True)
        
        connector.add_consent_callback(callback1)
        connector.add_consent_callback(callback2)
        
        assert len(connector.consent_callbacks) == 2
        assert callback1 in connector.consent_callbacks
        assert callback2 in connector.consent_callbacks
    
    @pytest.mark.asyncio
    async def test_request_user_consent_no_callbacks(self, connector):
        """Test consent request with no callbacks"""
        decision = RoutingDecision(
            should_route=True,
            criteria=[ExternalRoutingCriteria.SPECIALIZED_KNOWLEDGE],
            confidence=0.5,
            reasoning="Test"
        )
        
        result = await connector.request_user_consent("test query", decision)
        assert result is False  # Default to False for privacy
    
    @pytest.mark.asyncio
    async def test_request_user_consent_with_callbacks(self, connector):
        """Test consent request with callbacks"""
        callback1 = AsyncMock(return_value=True)
        callback2 = AsyncMock(return_value=True)
        
        connector.add_consent_callback(callback1)
        connector.add_consent_callback(callback2)
        
        decision = RoutingDecision(
            should_route=True,
            criteria=[ExternalRoutingCriteria.SPECIALIZED_KNOWLEDGE],
            confidence=0.5,
            reasoning="Test"
        )
        
        result = await connector.request_user_consent("test query", decision)
        assert result is True
        
        # Verify callbacks were called
        callback1.assert_called_once_with("test query", decision)
        callback2.assert_called_once_with("test query", decision)
    
    @pytest.mark.asyncio
    async def test_request_user_consent_callback_denial(self, connector):
        """Test consent request when callback denies"""
        callback1 = AsyncMock(return_value=True)
        callback2 = AsyncMock(return_value=False)  # This callback denies
        
        connector.add_consent_callback(callback1)
        connector.add_consent_callback(callback2)
        
        decision = RoutingDecision(
            should_route=True,
            criteria=[ExternalRoutingCriteria.SPECIALIZED_KNOWLEDGE],
            confidence=0.5,
            reasoning="Test"
        )
        
        result = await connector.request_user_consent("test query", decision)
        assert result is False  # Should be False if any callback denies
    
    def test_performance_stats(self, connector):
        """Test performance statistics tracking"""
        # Initial stats
        stats = connector.get_performance_stats()
        assert stats['total_requests'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['api_key_configured'] is False
        
        # Simulate some requests
        connector.request_count = 10
        connector.success_count = 8
        connector.failure_count = 2
        connector.total_processing_time = 25.0
        connector.openrouter_api_key = "test_key"
        
        stats = connector.get_performance_stats()
        assert stats['total_requests'] == 10
        assert stats['successful_requests'] == 8
        assert stats['failed_requests'] == 2
        assert stats['success_rate'] == 0.8
        assert stats['average_processing_time'] == 2.5
        assert stats['api_key_configured'] is True
    
    @pytest.mark.asyncio
    async def test_close(self, connector):
        """Test connector cleanup"""
        await connector.close()
        # Should complete without error
        # In a real implementation, this might close connections, etc.


if __name__ == "__main__":
    pytest.main([__file__]) 