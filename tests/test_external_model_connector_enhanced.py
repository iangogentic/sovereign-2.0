"""
Tests for enhanced ExternalModelConnector with circuit breaker, health monitoring, and improved caching
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from aioresponses import aioresponses

from src.sovereign.external_model_connector import (
    ExternalModelConnector, 
    ExternalResponse, 
    RoutingDecision,
    ExternalRoutingCriteria,
    ServiceHealthStatus,
    CircuitBreakerState,
    CacheStatistics,
    ServiceHealthMetrics
)
from src.sovereign.config import Config


class TestEnhancedExternalModelConnector:
    """Test suite for enhanced external model connector features"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        config = MagicMock()
        config.external_cache_ttl_hours = 1
        config.external_cache_max_size = 100
        return config
    
    @pytest.fixture
    def connector(self, mock_config):
        """Create connector instance for testing"""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            return ExternalModelConnector(mock_config)
    
    @pytest.fixture
    def mock_fallback_callback(self):
        """Mock fallback callback for local processing"""
        async def fallback(query, context):
            return f"Local response for: {query}"
        return fallback
    
    def test_initialization_with_enhanced_features(self, connector):
        """Test that enhanced features are properly initialized"""
        # Check circuit breaker initialization
        assert connector.circuit_breaker.state == "CLOSED"
        assert connector.circuit_breaker.failure_count == 0
        assert connector.circuit_breaker.failure_threshold == 5
        
        # Check cache statistics initialization
        assert connector.cache_stats.hits == 0
        assert connector.cache_stats.misses == 0
        assert connector.cache_stats.evictions == 0
        
        # Check service health initialization
        assert connector.service_health.status == ServiceHealthStatus.UNKNOWN
        assert connector.service_health.consecutive_failures == 0
        
        # Check enhanced performance tracking
        assert connector.retry_count == 0
        assert connector.fallback_count == 0
    
    def test_fallback_callback_registration(self, connector, mock_fallback_callback):
        """Test fallback callback registration"""
        connector.set_fallback_callback(mock_fallback_callback)
        assert connector.fallback_callback == mock_fallback_callback
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opening(self, connector):
        """Test circuit breaker opening after repeated failures"""
        # Simulate repeated failures
        for i in range(5):
            connector._record_circuit_breaker_result(False)
        
        # Circuit breaker should now be open
        assert connector._should_circuit_break() == True
        assert connector.circuit_breaker.state == "OPEN"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_transition(self, connector):
        """Test circuit breaker transition to half-open state"""
        # Open the circuit breaker
        connector.circuit_breaker.state = "OPEN"
        connector.circuit_breaker.last_failure_time = datetime.now() - timedelta(seconds=400)
        
        # Should transition to half-open
        assert connector._should_circuit_break() == False
        assert connector.circuit_breaker.state == "HALF_OPEN"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closing(self, connector):
        """Test circuit breaker closing after successful half-open request"""
        connector.circuit_breaker.state = "HALF_OPEN"
        connector._record_circuit_breaker_result(True)
        
        assert connector.circuit_breaker.state == "CLOSED"
        assert connector.circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, connector):
        """Test successful health check"""
        with aioresponses() as m:
            m.get(
                f"{connector.openrouter_base_url}/models",
                status=200,
                payload={"data": [{"id": "test_model"}]}
            )
            
            result = await connector._perform_health_check()
            assert result == True
            assert connector.service_health.consecutive_successes == 1
            assert connector.service_health.consecutive_failures == 0
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, connector):
        """Test failed health check"""
        with aioresponses() as m:
            m.get(
                f"{connector.openrouter_base_url}/models",
                status=500
            )
            
            result = await connector._perform_health_check()
            assert result == False
            assert connector.service_health.consecutive_failures == 1
            assert connector.service_health.consecutive_successes == 0
    
    @pytest.mark.asyncio
    async def test_cache_statistics_tracking(self, connector):
        """Test cache statistics tracking"""
        query = "test query"
        context = "test context"
        
        # First request should be a cache miss
        cache_key = connector._get_cache_key(query, context)
        cached_response = connector._get_cached_response(cache_key)
        
        assert cached_response is None
        
        # Cache a response
        test_response = ExternalResponse(
            response="test response",
            provider="test",
            model="test_model",
            processing_time=1.0,
            success=True
        )
        
        connector._cache_response(cache_key, test_response)
        
        # Verify cache statistics
        assert connector.cache_stats.cache_size == 1
        assert connector.cache_stats.cache_memory_usage > 0
        
        # Second request should be a cache hit
        cached_response = connector._get_cached_response(cache_key)
        assert cached_response is not None
        assert cached_response.cached == True
    
    @pytest.mark.asyncio
    async def test_cache_eviction_tracking(self, connector):
        """Test cache eviction tracking"""
        # Add expired entries
        expired_response = ExternalResponse(
            response="expired response",
            provider="test",
            model="test_model",
            processing_time=1.0,
            success=True
        )
        
        # Manually add expired cache entry
        expired_key = "expired_key"
        connector.response_cache[expired_key] = {
            'timestamp': (datetime.now() - timedelta(hours=25)).isoformat(),
            'response': {
                'response': expired_response.response,
                'provider': expired_response.provider,
                'model': expired_response.model,
                'processing_time': expired_response.processing_time,
                'success': expired_response.success,
                'error': expired_response.error
            }
        }
        
        # Clean cache
        connector._clean_cache()
        
        # Verify eviction was tracked
        assert connector.cache_stats.evictions == 1
        assert expired_key not in connector.response_cache
    
    @pytest.mark.asyncio
    async def test_fallback_to_local_with_callback(self, connector, mock_fallback_callback):
        """Test fallback to local processing with callback"""
        connector.set_fallback_callback(mock_fallback_callback)
        
        query = "test query"
        context = "test context"
        reason = "test reason"
        
        response = await connector._fallback_to_local(query, context, reason)
        
        assert response.success == True
        assert response.provider == "local"
        assert response.model == "local_fallback"
        assert response.response == "[Local Processing] Local response for: test query"
        assert connector.fallback_count == 1
    
    @pytest.mark.asyncio
    async def test_fallback_to_local_without_callback(self, connector):
        """Test fallback to local processing without callback"""
        query = "test query"
        context = "test context"
        reason = "test reason"
        
        response = await connector._fallback_to_local(query, context, reason)
        
        assert response.success == False
        assert response.provider == "external"
        assert response.model == "none"
        assert "no local fallback configured" in response.response
        assert connector.fallback_count == 1
    
    @pytest.mark.asyncio
    async def test_route_request_with_circuit_breaker_open(self, connector, mock_fallback_callback):
        """Test route request when circuit breaker is open"""
        connector.set_fallback_callback(mock_fallback_callback)
        
        # Open circuit breaker
        connector.circuit_breaker.state = "OPEN"
        connector.circuit_breaker.last_failure_time = datetime.now()
        
        query = "test query"
        response = await connector.route_request(query)
        
        assert response.success == True
        assert response.provider == "local"
        assert "[Local Processing]" in response.response
        assert connector.fallback_count == 1
    
    @pytest.mark.asyncio
    async def test_route_request_with_retry_logic(self, connector, mock_fallback_callback):
        """Test route request with retry logic"""
        connector.set_fallback_callback(mock_fallback_callback)
        
        with aioresponses() as m:
            # Mock health check
            m.get(f"{connector.openrouter_base_url}/models", status=200)
            
            # Mock failing requests (will retry 3 times)
            for i in range(3):
                m.post(
                    f"{connector.openrouter_base_url}/chat/completions",
                    exception=Exception("Connection failed")
                )
            
            # Mock user consent
            connector.consent_callbacks.append(AsyncMock(return_value=True))
            
            query = "test query requiring external routing"
            response = await connector.route_request(query)
            
            # Should fallback to local after retries
            assert response.success == True
            assert response.provider == "local"
            assert connector.retry_count == 3
            assert connector.fallback_count == 1
    
    @pytest.mark.asyncio
    async def test_route_request_cache_hit(self, connector):
        """Test route request with cache hit"""
        query = "test query"
        
        # Pre-populate cache
        cache_key = connector._get_cache_key(query, None)
        test_response = ExternalResponse(
            response="cached response",
            provider="openrouter",
            model="test_model",
            processing_time=1.0,
            success=True
        )
        connector._cache_response(cache_key, test_response)
        
        # Make request
        response = await connector.route_request(query)
        
        assert response.success == True
        assert response.cached == True
        assert response.response == "cached response"
        assert connector.cache_stats.hits == 1
    
    @pytest.mark.asyncio
    async def test_comprehensive_performance_stats(self, connector):
        """Test comprehensive performance statistics"""
        # Simulate some activity
        connector.request_count = 10
        connector.success_count = 7
        connector.failure_count = 3
        connector.retry_count = 2
        connector.fallback_count = 1
        connector.cache_stats.hits = 5
        connector.cache_stats.misses = 3
        connector.cache_stats.evictions = 2
        connector.circuit_breaker.failure_count = 2
        
        stats = connector.get_performance_stats()
        
        # Verify basic metrics
        assert stats['total_requests'] == 10
        assert stats['successful_requests'] == 7
        assert stats['failed_requests'] == 3
        assert stats['retry_count'] == 2
        assert stats['fallback_count'] == 1
        
        # Verify cache statistics
        assert stats['cache_stats']['hits'] == 5
        assert stats['cache_stats']['misses'] == 3
        assert stats['cache_stats']['evictions'] == 2
        assert stats['cache_stats']['hit_rate'] == 5/8  # 5 hits out of 8 total
        
        # Verify circuit breaker status
        assert stats['circuit_breaker']['state'] == 'CLOSED'
        assert stats['circuit_breaker']['failure_count'] == 2
        
        # Verify configuration
        assert stats['configuration']['api_key_configured'] == True
        assert stats['configuration']['fallback_callback_configured'] == False
    
    @pytest.mark.asyncio
    async def test_should_check_health_logic(self, connector):
        """Test health check timing logic"""
        # Should check health initially
        assert await connector._should_check_health() == True
        
        # Set last check to recent
        connector.last_health_check = datetime.now() - timedelta(seconds=30)
        assert await connector._should_check_health() == False
        
        # Set last check to old
        connector.last_health_check = datetime.now() - timedelta(seconds=400)
        assert await connector._should_check_health() == True
    
    @pytest.mark.asyncio
    async def test_cache_memory_usage_estimation(self, connector):
        """Test cache memory usage estimation"""
        # Add some test data
        test_response = ExternalResponse(
            response="test response",
            provider="test",
            model="test_model",
            processing_time=1.0,
            success=True
        )
        
        cache_key = "test_key"
        connector._cache_response(cache_key, test_response)
        
        memory_usage = connector._estimate_cache_memory_usage()
        assert memory_usage > 0
        assert isinstance(memory_usage, int)
    
    @pytest.mark.asyncio
    async def test_route_request_no_api_key(self, mock_config, mock_fallback_callback):
        """Test route request behavior without API key"""
        with patch.dict('os.environ', {}, clear=True):
            connector = ExternalModelConnector(mock_config)
            connector.set_fallback_callback(mock_fallback_callback)
            
            query = "test query"
            response = await connector.route_request(query)
            
            assert response.success == True
            assert response.provider == "local"
            assert "[Local Processing]" in response.response
            assert connector.fallback_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 