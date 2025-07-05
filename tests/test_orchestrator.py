"""
Tests for the ModelOrchestrator class
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.sovereign.orchestrator import (
    ModelOrchestrator, 
    QueryContext, 
    OrchestrationResult,
    QueryComplexity,
    ModelChoice,
    CacheEntry
)
from src.sovereign.config import Config
from src.sovereign.talker_model import TalkerModel
from src.sovereign.thinker_model import ThinkerModel


class TestModelOrchestrator:
    """Test cases for ModelOrchestrator"""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        config = Config()
        # Set test-specific values
        config.orchestrator.cache_max_size = 10
        config.orchestrator.cache_ttl_hours = 1
        config.orchestrator.enable_caching = True
        config.orchestrator.enable_telemetry = True
        return config
    
    @pytest.fixture
    def orchestrator(self, config):
        """Create a test orchestrator instance"""
        return ModelOrchestrator(config)
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample query context"""
        return QueryContext(
            user_input="Test query",
            timestamp=datetime.now(),
            session_id="test_session",
            previous_queries=["Previous query"],
            conversation_history=[{"user": "Hello", "assistant": "Hi there"}]
        )
    
    def test_init(self, orchestrator, config):
        """Test orchestrator initialization"""
        assert orchestrator.config == config
        assert orchestrator.talker_model is None
        assert orchestrator.thinker_model is None
        assert len(orchestrator.response_cache) == 0
        assert orchestrator.telemetry['total_queries'] == 0
        assert len(orchestrator.notification_callbacks) == 0
    
    def test_complexity_patterns_init(self, orchestrator):
        """Test that complexity patterns are initialized correctly"""
        assert len(orchestrator.simple_patterns) > 0
        assert len(orchestrator.complex_patterns) > 0
        assert len(orchestrator.multi_step_patterns) > 0
        assert len(orchestrator.tool_use_patterns) > 0
    
    @pytest.mark.parametrize("query,expected_complexity", [
        ("Hello, how are you?", QueryComplexity.SIMPLE),
        ("What's the weather like?", QueryComplexity.SIMPLE),
        ("Thanks for your help", QueryComplexity.SIMPLE),
        ("Analyze the benefits of machine learning", QueryComplexity.COMPLEX),
        ("Write a Python function to sort data", QueryComplexity.COMPLEX),
        ("Create a step-by-step plan for deployment", QueryComplexity.COMPLEX),
        ("This is a very long query with multiple sentences and complex requirements that should definitely be classified as complex because it involves multiple steps and detailed analysis of various factors.", QueryComplexity.COMPLEX),
    ])
    def test_determine_complexity(self, orchestrator, query, expected_complexity):
        """Test complexity determination for various queries"""
        complexity, confidence = orchestrator.determine_complexity(query)
        assert complexity == expected_complexity
        assert 0.0 <= confidence <= 1.0
    
    def test_determine_complexity_with_context(self, orchestrator):
        """Test complexity determination with context"""
        context = QueryContext(
            user_input="Continue the analysis",
            timestamp=datetime.now(),
            session_id="test",
            previous_queries=["Analyze the market trends"],
            conversation_history=[
                {"user": "Can you analyze this data?", "assistant": "Sure, let me analyze..."}
            ]
        )
        
        complexity, confidence = orchestrator.determine_complexity("Continue the analysis", context)
        # Should be influenced by the complex context
        assert complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]
        assert confidence > 0.0
    
    def test_select_model_simple(self, orchestrator, sample_context):
        """Test model selection for simple queries"""
        choice = orchestrator._select_model(QueryComplexity.SIMPLE, 0.8, sample_context)
        assert choice == ModelChoice.TALKER
    
    def test_select_model_complex(self, orchestrator, sample_context):
        """Test model selection for complex queries"""
        choice = orchestrator._select_model(QueryComplexity.VERY_COMPLEX, 0.9, sample_context)
        assert choice == ModelChoice.THINKER
    
    def test_select_model_moderate(self, orchestrator, sample_context):
        """Test model selection for moderate complexity"""
        choice = orchestrator._select_model(QueryComplexity.MODERATE, 0.6, sample_context)
        assert choice in [ModelChoice.TALKER, ModelChoice.THINKER]
    
    def test_should_handoff_to_thinker(self, orchestrator):
        """Test handoff decision logic"""
        # Test uncertainty phrases
        assert orchestrator._should_handoff_to_thinker(
            "Complex question", 
            "I'm not sure about this", 
            QueryComplexity.MODERATE
        ) == True
        
        # Test short response for complex query
        assert orchestrator._should_handoff_to_thinker(
            "Analyze this complex system", 
            "Yes.", 
            QueryComplexity.COMPLEX
        ) == True
        
        # Test handoff keywords
        assert orchestrator._should_handoff_to_thinker(
            "Analyze in detail the system", 
            "Here's some info", 
            QueryComplexity.SIMPLE
        ) == True
        
        # Test no handoff needed
        assert orchestrator._should_handoff_to_thinker(
            "Simple question", 
            "Here's a comprehensive answer with lots of detail", 
            QueryComplexity.SIMPLE
        ) == False
    
    def test_prepare_handoff_context(self, orchestrator):
        """Test handoff context preparation"""
        context = QueryContext(
            user_input="Test query",
            timestamp=datetime.now(),
            session_id="test",
            previous_queries=["Previous query"],
            conversation_history=[
                {"user": "Hello", "assistant": "Hi there"},
                {"user": "How are you?", "assistant": "I'm doing well"}
            ]
        )
        
        enhanced_context = orchestrator._prepare_handoff_context("Test query", context)
        assert "User Query: Test query" in enhanced_context
        assert "Previous Conversation:" in enhanced_context
        assert "User: Hello" in enhanced_context
        assert "Assistant: Hi there" in enhanced_context
    
    def test_query_hash_generation(self, orchestrator, sample_context):
        """Test query hash generation for caching"""
        hash1 = orchestrator._get_query_hash("test query", sample_context)
        hash2 = orchestrator._get_query_hash("test query", sample_context)
        hash3 = orchestrator._get_query_hash("different query", sample_context)
        
        # Same query should produce same hash
        assert hash1 == hash2
        # Different query should produce different hash
        assert hash1 != hash3
        # Hash should be consistent format
        assert len(hash1) == 32  # MD5 hash length
    
    def test_cache_functionality(self, orchestrator, sample_context):
        """Test response caching functionality"""
        # Create a test result
        result = OrchestrationResult(
            response="Test response",
            model_used=ModelChoice.TALKER,
            complexity_level=QueryComplexity.SIMPLE,
            processing_time=1.0,
            handoff_occurred=False,
            cache_hit=False,
            confidence_score=0.8,
            reasoning="Test reasoning",
            telemetry={}
        )
        
        # Test caching
        orchestrator._cache_response("test query", sample_context, result)
        assert len(orchestrator.response_cache) == 1
        
        # Test cache retrieval
        cached = orchestrator._check_cache("test query", sample_context)
        assert cached is not None
        assert cached.response == "Test response"
        assert cached.model_used == ModelChoice.TALKER
    
    def test_cache_expiration(self, orchestrator, sample_context):
        """Test cache entry expiration"""
        # Create an expired cache entry
        query_hash = orchestrator._get_query_hash("test query", sample_context)
        expired_entry = CacheEntry(
            query_hash=query_hash,
            response="Expired response",
            model_used=ModelChoice.TALKER,
            timestamp=datetime.now() - timedelta(hours=25),  # Expired
            hit_count=0,
            last_accessed=datetime.now() - timedelta(hours=25)
        )
        
        orchestrator.response_cache[query_hash] = expired_entry
        
        # Should return None for expired entry
        cached = orchestrator._check_cache("test query", sample_context)
        assert cached is None
        # Expired entry should be removed
        assert query_hash not in orchestrator.response_cache
    
    def test_should_cache_logic(self, orchestrator):
        """Test caching decision logic"""
        # Simple query with high confidence should be cached
        result1 = OrchestrationResult(
            response="Test", model_used=ModelChoice.TALKER, complexity_level=QueryComplexity.SIMPLE,
            processing_time=1.0, handoff_occurred=False, cache_hit=False, confidence_score=0.8,
            reasoning="Test", telemetry={}
        )
        assert orchestrator._should_cache(result1, QueryComplexity.SIMPLE) == True
        
        # Complex query with long processing time should be cached
        result2 = OrchestrationResult(
            response="Test", model_used=ModelChoice.THINKER, complexity_level=QueryComplexity.COMPLEX,
            processing_time=10.0, handoff_occurred=False, cache_hit=False, confidence_score=0.6,
            reasoning="Test", telemetry={}
        )
        assert orchestrator._should_cache(result2, QueryComplexity.COMPLEX) == True
        
        # Low confidence, quick response should not be cached
        result3 = OrchestrationResult(
            response="Test", model_used=ModelChoice.TALKER, complexity_level=QueryComplexity.MODERATE,
            processing_time=1.0, handoff_occurred=False, cache_hit=False, confidence_score=0.3,
            reasoning="Test", telemetry={}
        )
        assert orchestrator._should_cache(result3, QueryComplexity.MODERATE) == False
    
    def test_cache_cleaning(self, orchestrator):
        """Test cache cleaning when max size is reached"""
        # Fill cache beyond max size
        for i in range(15):  # Config has max_size=10
            context = QueryContext(
                user_input=f"query_{i}",
                timestamp=datetime.now(),
                session_id="test",
                previous_queries=[],
                conversation_history=[]
            )
            result = OrchestrationResult(
                response=f"response_{i}", model_used=ModelChoice.TALKER, 
                complexity_level=QueryComplexity.SIMPLE, processing_time=1.0,
                handoff_occurred=False, cache_hit=False, confidence_score=0.8,
                reasoning="Test", telemetry={}
            )
            orchestrator._cache_response(f"query_{i}", context, result)
        
        # Cache should be cleaned to max size
        assert len(orchestrator.response_cache) == orchestrator.cache_max_size
    
    def test_notification_callbacks(self, orchestrator):
        """Test notification callback system"""
        messages = []
        
        async def test_callback(message: str):
            messages.append(message)
        
        orchestrator.add_notification_callback(test_callback)
        
        # Test notification
        asyncio.run(orchestrator._notify_user("Test message"))
        assert "Test message" in messages
    
    def test_telemetry_tracking(self, orchestrator):
        """Test telemetry data tracking"""
        initial_telemetry = orchestrator._get_telemetry_snapshot()
        
        # Check initial values
        assert initial_telemetry['total_queries'] == 0
        assert initial_telemetry['cache_hit_rate'] == 0
        assert initial_telemetry['handoff_rate'] == 0
        assert 'uptime_seconds' in initial_telemetry
        
        # Simulate some activity
        orchestrator.telemetry['total_queries'] = 10
        orchestrator.telemetry['cache_hits'] = 3
        orchestrator.telemetry['cache_misses'] = 7
        orchestrator.telemetry['handoff_queries'] = 2
        
        updated_telemetry = orchestrator._get_telemetry_snapshot()
        assert updated_telemetry['total_queries'] == 10
        assert updated_telemetry['cache_hit_rate'] == 0.3
        assert updated_telemetry['handoff_rate'] == 0.2
    
    def test_integrate_responses(self, orchestrator):
        """Test response integration from both models"""
        talker_response = "Quick answer"
        thinker_response = "Detailed analysis with comprehensive information"
        
        integrated = asyncio.run(orchestrator.integrate_responses(talker_response, thinker_response))
        
        assert "more comprehensive response" in integrated
        assert thinker_response in integrated
    
    @pytest.mark.asyncio
    async def test_status_reporting(self, orchestrator):
        """Test status reporting functionality"""
        status = await orchestrator.get_status()
        
        assert 'talker_model_ready' in status
        assert 'thinker_model_ready' in status
        assert 'cache_size' in status
        assert 'telemetry' in status
        
        # Initially models should not be ready
        assert status['talker_model_ready'] == False
        assert status['thinker_model_ready'] == False
        assert status['cache_size'] == 0


class TestQueryContext:
    """Test cases for QueryContext dataclass"""
    
    def test_query_context_creation(self):
        """Test QueryContext creation with all fields"""
        context = QueryContext(
            user_input="Test query",
            timestamp=datetime.now(),
            session_id="session_123",
            previous_queries=["Query 1", "Query 2"],
            conversation_history=[{"user": "Hello", "assistant": "Hi"}],
            screen_context={"screen": "data"},
            voice_context={"voice": "data"},
            user_preferences={"pref": "value"}
        )
        
        assert context.user_input == "Test query"
        assert context.session_id == "session_123"
        assert len(context.previous_queries) == 2
        assert len(context.conversation_history) == 1
        assert context.screen_context == {"screen": "data"}
        assert context.voice_context == {"voice": "data"}
        assert context.user_preferences == {"pref": "value"}
    
    def test_query_context_minimal(self):
        """Test QueryContext with minimal required fields"""
        context = QueryContext(
            user_input="Test query",
            timestamp=datetime.now(),
            session_id="session_123",
            previous_queries=[],
            conversation_history=[]
        )
        
        assert context.user_input == "Test query"
        assert context.session_id == "session_123"
        assert context.screen_context is None
        assert context.voice_context is None
        assert context.user_preferences is None


class TestOrchestrationResult:
    """Test cases for OrchestrationResult dataclass"""
    
    def test_orchestration_result_creation(self):
        """Test OrchestrationResult creation"""
        result = OrchestrationResult(
            response="Test response",
            model_used=ModelChoice.TALKER,
            complexity_level=QueryComplexity.SIMPLE,
            processing_time=1.5,
            handoff_occurred=False,
            cache_hit=True,
            confidence_score=0.85,
            reasoning="Simple query handled by Talker",
            telemetry={"queries": 1}
        )
        
        assert result.response == "Test response"
        assert result.model_used == ModelChoice.TALKER
        assert result.complexity_level == QueryComplexity.SIMPLE
        assert result.processing_time == 1.5
        assert result.handoff_occurred == False
        assert result.cache_hit == True
        assert result.confidence_score == 0.85
        assert result.reasoning == "Simple query handled by Talker"
        assert result.telemetry == {"queries": 1}


# Integration tests that require mocked models
class TestOrchestratorIntegration:
    """Integration tests for ModelOrchestrator with mocked models"""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        return Config()
    
    @pytest.fixture
    def orchestrator_with_mocks(self, config):
        """Create orchestrator with mocked models"""
        orchestrator = ModelOrchestrator(config)
        
        # Mock the models
        orchestrator.talker_model = AsyncMock(spec=TalkerModel)
        orchestrator.thinker_model = AsyncMock(spec=ThinkerModel)
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_simple_query_processing(self, orchestrator_with_mocks):
        """Test processing a simple query"""
        # Setup mocks
        orchestrator_with_mocks.talker_model.generate_response.return_value = "Simple answer"
        
        # Process query
        result = await orchestrator_with_mocks.process_query("Hello, how are you?")
        
        # Verify results
        assert result.response == "Simple answer"
        assert result.model_used == ModelChoice.TALKER
        assert result.complexity_level == QueryComplexity.SIMPLE
        assert not result.handoff_occurred
        assert not result.cache_hit
        
        # Verify mock was called
        orchestrator_with_mocks.talker_model.generate_response.assert_called_once_with("Hello, how are you?")
    
    @pytest.mark.asyncio
    async def test_complex_query_processing(self, orchestrator_with_mocks):
        """Test processing a complex query"""
        # Setup mocks
        orchestrator_with_mocks.thinker_model.auto_process.return_value = "Detailed analysis"
        
        # Process complex query
        result = await orchestrator_with_mocks.process_query("Analyze the quantum computing market trends")
        
        # Verify results
        assert result.response == "Detailed analysis"
        assert result.model_used == ModelChoice.THINKER
        assert result.complexity_level in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]
        assert not result.handoff_occurred
        assert not result.cache_hit
        
        # Verify mock was called
        orchestrator_with_mocks.thinker_model.auto_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handoff_processing(self, orchestrator_with_mocks):
        """Test processing with handoff from Talker to Thinker"""
        # Setup mocks for handoff scenario
        orchestrator_with_mocks.talker_model.generate_response.return_value = "I'm not sure about this complex topic"
        orchestrator_with_mocks.thinker_model.auto_process.return_value = "Comprehensive analysis"
        
        # Process query that should trigger handoff
        result = await orchestrator_with_mocks.process_query("Explain quantum mechanics")
        
        # Verify handoff occurred
        assert result.handoff_occurred
        assert result.model_used == ModelChoice.BOTH
        assert "more comprehensive response" in result.response
        assert "Comprehensive analysis" in result.response
    
    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator_with_mocks):
        """Test error handling in query processing"""
        # Setup mock to raise an exception
        orchestrator_with_mocks.talker_model.generate_response.side_effect = Exception("Test error")
        
        # Process query
        result = await orchestrator_with_mocks.process_query("Test query")
        
        # Verify error handling
        assert "error processing your request" in result.response.lower()
        assert result.confidence_score == 0.0
        assert "Error occurred" in result.reasoning


if __name__ == "__main__":
    pytest.main([__file__]) 