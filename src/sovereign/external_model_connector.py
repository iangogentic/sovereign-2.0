"""
External Model Connector - Routes requests to external AI services like OpenRouter
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import os

from .config import Config
from .logger import PerformanceTimer

logger = logging.getLogger(__name__)


class ExternalRoutingCriteria(Enum):
    """Criteria for routing queries to external services"""
    SPECIALIZED_KNOWLEDGE = "specialized_knowledge"
    COMPLEX_TOOL_USE = "complex_tool_use"
    USER_EXPLICIT_REQUEST = "user_explicit_request"
    LOCAL_MODEL_FAILURE = "local_model_failure"
    RECENT_INFORMATION = "recent_information"


class ServiceHealthStatus(Enum):
    """External service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class RoutingDecision:
    """Decision about whether to route to external services"""
    should_route: bool
    criteria: List[ExternalRoutingCriteria]
    confidence: float
    reasoning: str


@dataclass
class ExternalRequest:
    """External service request information"""
    provider: str
    model: str
    query: str
    context: Optional[str] = None
    user_approved: bool = False
    timestamp: datetime = None


@dataclass
class ExternalResponse:
    """External service response information"""
    response: str
    provider: str
    model: str
    processing_time: float
    success: bool
    error: Optional[str] = None
    cached: bool = False


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for external service reliability"""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5
    timeout_seconds: int = 60
    recovery_timeout_seconds: int = 300


@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    average_hit_rate: float = 0.0
    cache_size: int = 0
    cache_memory_usage: int = 0


@dataclass
class ServiceHealthMetrics:
    """Service health monitoring metrics"""
    status: ServiceHealthStatus = ServiceHealthStatus.UNKNOWN
    last_check: Optional[datetime] = None
    response_time: float = 0.0
    uptime_percentage: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class ExternalModelConnector:
    """
    Manages external AI service integration while maintaining local-first philosophy
    Enhanced with circuit breaker, health monitoring, and improved caching
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # API configuration
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        
        # Enhanced caching system
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl_hours = getattr(config, 'external_cache_ttl_hours', 24)
        self.cache_max_size = getattr(config, 'external_cache_max_size', 1000)
        self.cache_stats = CacheStatistics()
        
        # Circuit breaker for reliability
        self.circuit_breaker = CircuitBreakerState()
        
        # Service health monitoring
        self.service_health = ServiceHealthMetrics()
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = None
        self.enable_health_checks = getattr(config, 'external_enable_health_checks', True)
        
        # Backward compatibility: Disable health checks if no API key for testing
        if not self.openrouter_api_key:
            self.enable_health_checks = False
        
        # User consent callbacks
        self.consent_callbacks: List[callable] = []
        
        # Fallback callback for local processing
        self.fallback_callback: Optional[Callable] = None
        
        # Enhanced performance tracking
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_processing_time = 0.0
        self.retry_count = 0
        self.fallback_count = 0
        
        # Routing patterns
        self._init_routing_patterns()
        
        # Default external model
        self.default_external_model = "anthropic/claude-3-sonnet"
        
        self.logger.info("ExternalModelConnector initialized with enhanced features")
    
    def _init_routing_patterns(self):
        """Initialize patterns for external routing decisions"""
        
        # Specialized knowledge patterns
        self.specialized_knowledge_patterns = [
            r'\b(latest|recent|current|up-to-date|new|breaking|today|yesterday|this week|this month|2024|2025)\b',
            r'\b(stock price|market|crypto|bitcoin|ethereum|financial|trading)\b',
            r'\b(news|events|current events|politics|elections|government)\b',
            r'\b(weather|forecast|temperature|climate|hurricane|earthquake)\b',
            r'\b(sports|game|match|score|championship|tournament)\b',
            r'\b(celebrity|famous person|actor|musician|politician)\b',
            r'\b(company|startup|business|IPO|merger|acquisition)\b',
        ]
        
        # Complex tool use patterns
        self.complex_tool_patterns = [
            r'\b(search|google|bing|find online|browse web|lookup online)\b',
            r'\b(API|REST|GraphQL|endpoint|webhook|integration)\b',
            r'\b(database|SQL|query|schema|migration)\b',
            r'\b(deploy|deployment|CI/CD|docker|kubernetes|aws|azure|gcp)\b',
            r'\b(email|send message|notification|alert|communication)\b',
            r'\b(file|document|spreadsheet|pdf|csv|json|xml)\b',
        ]
        
        # User explicit request patterns
        self.explicit_request_patterns = [
            r'\b(use external|use cloud|use online|search online|look up online)\b',
            r'\b(openrouter|claude|gpt|gemini|external model|cloud model)\b',
            r'\bneed\b.*(fresh|current|latest|up-to-date)',
        ]
        
        # Recent information patterns
        self.recent_info_patterns = [
            r'\b(what\'s happening|what happened|recent developments|latest news)\b',
            r'\b(current status|latest update|recent changes|new features)\b',
            r'\b(today|yesterday|this week|this month|recently|lately)\b',
        ]
    
    def set_fallback_callback(self, callback: Callable):
        """Set callback for local model fallback"""
        self.fallback_callback = callback
        self.logger.info("Fallback callback registered for local model integration")
    
    async def initialize(self) -> bool:
        """
        Initialize the external model connector with health checks
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing ExternalModelConnector...")
            
            # Check if API key is available
            if not self.openrouter_api_key:
                self.logger.warning("No OpenRouter API key found. External routing will be disabled.")
                self.service_health.status = ServiceHealthStatus.UNHEALTHY
                return False
            
            # Perform initial health check
            health_check_result = await self._perform_health_check()
            if health_check_result:
                self.service_health.status = ServiceHealthStatus.HEALTHY
                self.logger.info("External model connector initialized successfully")
                return True
            else:
                self.service_health.status = ServiceHealthStatus.UNHEALTHY
                self.logger.warning("External model connector initialized but service is unhealthy")
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing ExternalModelConnector: {e}")
            self.service_health.status = ServiceHealthStatus.UNHEALTHY
            return False
    
    async def _perform_health_check(self) -> bool:
        """Perform health check on external service"""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.openrouter_api_key}',
                    'Content-Type': 'application/json'
                }
                
                async with session.get(
                    f"{self.openrouter_base_url}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    response_time = time.time() - start_time
                    self.service_health.response_time = response_time
                    self.service_health.last_check = datetime.now()
                    
                    if response.status == 200:
                        self.service_health.consecutive_successes += 1
                        self.service_health.consecutive_failures = 0
                        self.logger.debug(f"Health check passed in {response_time:.2f}s")
                        return True
                    else:
                        self.service_health.consecutive_failures += 1
                        self.service_health.consecutive_successes = 0
                        self.logger.warning(f"Health check failed with status {response.status}")
                        return False
                        
        except Exception as e:
            self.service_health.consecutive_failures += 1
            self.service_health.consecutive_successes = 0
            self.service_health.last_check = datetime.now()
            self.logger.error(f"Health check error: {e}")
            return False
    
    def _should_circuit_break(self) -> bool:
        """Check if circuit breaker should prevent requests"""
        now = datetime.now()
        
        # Check if we're in OPEN state and should transition to HALF_OPEN
        if self.circuit_breaker.state == "OPEN":
            if (self.circuit_breaker.last_failure_time and 
                now - self.circuit_breaker.last_failure_time > timedelta(seconds=self.circuit_breaker.recovery_timeout_seconds)):
                self.circuit_breaker.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN state")
                return False
            return True
        
        # Check if we should open the circuit
        if (self.circuit_breaker.state == "CLOSED" and 
            self.circuit_breaker.failure_count >= self.circuit_breaker.failure_threshold):
            self.circuit_breaker.state = "OPEN"
            self.circuit_breaker.last_failure_time = now
            self.logger.warning("Circuit breaker opened due to repeated failures")
            return True
        
        return False
    
    def _record_circuit_breaker_result(self, success: bool):
        """Record circuit breaker result and update state"""
        if success:
            self.circuit_breaker.success_count += 1
            if self.circuit_breaker.state == "HALF_OPEN":
                self.circuit_breaker.state = "CLOSED"
                self.circuit_breaker.failure_count = 0
                self.logger.info("Circuit breaker closed after successful request")
        else:
            self.circuit_breaker.failure_count += 1
            self.circuit_breaker.last_failure_time = datetime.now()
            if self.circuit_breaker.state == "HALF_OPEN":
                self.circuit_breaker.state = "OPEN"
                self.logger.warning("Circuit breaker opened after failed half-open request")
    
    async def _should_check_health(self) -> bool:
        """Determine if a health check should be performed"""
        if not self.last_health_check:
            return True
        
        time_since_last_check = (datetime.now() - self.last_health_check).total_seconds()
        return time_since_last_check > self.health_check_interval
    
    async def _fallback_to_local(self, query: str, context: Optional[str], reason: str) -> ExternalResponse:
        """Fallback to local processing when external service is unavailable"""
        self.fallback_count += 1
        
        if self.fallback_callback:
            try:
                self.logger.info(f"Attempting local fallback due to: {reason}")
                
                # Call the local model through the fallback callback
                local_response = await self.fallback_callback(query, context)
                
                return ExternalResponse(
                    response=f"[Local Processing] {local_response}",
                    provider="local",
                    model="local_fallback",
                    processing_time=0.0,
                    success=True,
                    cached=False
                )
                
            except Exception as e:
                self.logger.error(f"Local fallback failed: {e}")
                return ExternalResponse(
                    response=f"External service unavailable ({reason}) and local fallback failed. Please try again later.",
                    provider="local",
                    model="local_fallback",
                    processing_time=0.0,
                    success=False,
                    error=f"Fallback failed: {str(e)}"
                )
        else:
            # No fallback callback available - return error consistent with original behavior
            if "no api key configured" in reason.lower():
                return ExternalResponse(
                    response="External services are not available (no api key configured)",
                    provider="openrouter",
                    model="none",
                    processing_time=0.0,
                    success=False,
                    error="No API key configured"
                )
            else:
                return ExternalResponse(
                    response=f"External service unavailable ({reason}) and no local fallback configured. Please try again later.",
                    provider="external",
                    model="none",
                    processing_time=0.0,
                    success=False,
                    error=f"No fallback available: {reason}"
                )

    def determine_external_need(self, query: str, context: Optional[str] = None) -> RoutingDecision:
        """
        Determine if a query should be routed to external services
        
        Args:
            query: User's query
            context: Additional context
            
        Returns:
            RoutingDecision object with routing recommendation
        """
        criteria = []
        confidence = 0.0
        reasoning_parts = []
        
        query_lower = query.lower()
        
        # Check for explicit user request first (highest priority)
        for pattern in self.explicit_request_patterns:
            if len([m for m in __import__('re').finditer(pattern, query_lower)]) > 0:
                criteria.append(ExternalRoutingCriteria.USER_EXPLICIT_REQUEST)
                confidence += 0.5
                reasoning_parts.append("User explicitly requested external service")
                break
        
        # Check for complex tool use patterns
        for pattern in self.complex_tool_patterns:
            if len([m for m in __import__('re').finditer(pattern, query_lower)]) > 0:
                criteria.append(ExternalRoutingCriteria.COMPLEX_TOOL_USE)
                confidence += 0.4
                reasoning_parts.append("Query requires complex tool usage")
                break
        
        # Check for recent information need
        for pattern in self.recent_info_patterns:
            if len([m for m in __import__('re').finditer(pattern, query_lower)]) > 0:
                criteria.append(ExternalRoutingCriteria.RECENT_INFORMATION)
                confidence += 0.4
                reasoning_parts.append("Query requires recent/current information")
                break
        
        # Check for specialized knowledge patterns (lowest priority)
        for pattern in self.specialized_knowledge_patterns:
            if len([m for m in __import__('re').finditer(pattern, query_lower)]) > 0:
                criteria.append(ExternalRoutingCriteria.SPECIALIZED_KNOWLEDGE)
                confidence += 0.3
                reasoning_parts.append("Query requires specialized/current knowledge")
                break
        
        # Determine if should route
        should_route = confidence >= 0.3 and len(criteria) > 0
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No external routing criteria met"
        
        return RoutingDecision(
            should_route=should_route,
            criteria=criteria,
            confidence=confidence,
            reasoning=reasoning
        )
    
    async def route_request(self, query: str, context: Optional[str] = None) -> ExternalResponse:
        """
        Route a request to an external service with enhanced fallback logic
        
        Args:
            query: User's query
            context: Additional context
            
        Returns:
            ExternalResponse object with the result
        """
        self.cache_stats.total_requests += 1
        
        # Check if API key is available
        if not self.openrouter_api_key:
            self.logger.warning("No OpenRouter API key configured, attempting local fallback")
            return await self._fallback_to_local(query, context, "No API key configured")
        
        # Check cache first
        cache_key = self._get_cache_key(query, context)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            self.cache_stats.hits += 1
            self.logger.info("Returning cached external response")
            return cached_response
        
        self.cache_stats.misses += 1
        
        # Check circuit breaker
        if self._should_circuit_break():
            self.logger.warning("Circuit breaker is open, falling back to local processing")
            return await self._fallback_to_local(query, context, "Circuit breaker open")
        
        # Check service health (if enabled)
        if self.enable_health_checks and await self._should_check_health():
            health_ok = await self._perform_health_check()
            self.last_health_check = datetime.now()
            if not health_ok:
                self.logger.warning("Service health check failed, falling back to local processing")
                return await self._fallback_to_local(query, context, "Service health check failed")
        
        # Check routing decision and request user consent (only if specific criteria are met)
        routing_decision = self.determine_external_need(query, context)
        if routing_decision.should_route and self.consent_callbacks:
            # Only request consent if we have consent callbacks AND routing decision recommends external
            consent_granted = await self.request_user_consent(query, routing_decision)
            if not consent_granted:
                self.logger.info("User denied consent for external routing")
                return ExternalResponse(
                    response="External service request was denied by user. Please try rephrasing your query for local processing.",
                    provider="openrouter",
                    model="none",
                    processing_time=0.0,
                    success=False,
                    error="User denied consent"
                )
        # For backward compatibility: if we have an API key and passed health checks,
        # attempt external routing regardless of routing decision (unless consent was explicitly denied)
        # (This preserves original behavior where any query with API key goes external)
        
        # Make request with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            with PerformanceTimer() as timer:
                try:
                    response = await self._make_openrouter_request(query, context)
                    
                    # Record circuit breaker result
                    self._record_circuit_breaker_result(response.success)
                    
                    # Cache successful response
                    if response.success:
                        self._cache_response(cache_key, response)
                        self.success_count += 1
                    else:
                        self.failure_count += 1
                        
                        # If this was our last attempt, try local fallback
                        if attempt == max_retries - 1:
                            self.logger.warning(f"External request failed after {max_retries} attempts, falling back to local")
                            return await self._fallback_to_local(query, context, f"External service failed: {response.error}")
                    
                    self.request_count += 1
                    self.total_processing_time += timer.elapsed_time
                    
                    return response
                    
                except Exception as e:
                    self.logger.error(f"Error making external request (attempt {attempt + 1}): {e}")
                    self.failure_count += 1
                    self.retry_count += 1
                    
                    # Record circuit breaker failure
                    self._record_circuit_breaker_result(False)
                    
                    # If this was our last attempt, try local fallback
                    if attempt == max_retries - 1:
                        self.logger.warning(f"External request failed after {max_retries} attempts, falling back to local")
                        return await self._fallback_to_local(query, context, f"Connection error: {str(e)}")
                    
                    # Wait before retry (exponential backoff)
                    await asyncio.sleep(2 ** attempt)
        
        # Fallback if all retries failed
        return await self._fallback_to_local(query, context, "All retry attempts failed")
    
    async def _make_openrouter_request(self, query: str, context: Optional[str] = None) -> ExternalResponse:
        """Make a request to OpenRouter API"""
        
        # Prepare the full prompt
        full_prompt = query
        if context:
            full_prompt = f"Context: {context}\n\nUser: {query}"
        
        headers = {
            'Authorization': f'Bearer {self.openrouter_api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://sovereign-ai.local',
            'X-Title': 'Sovereign AI Agent'
        }
        
        payload = {
            'model': self.default_external_model,
            'messages': [
                {
                    'role': 'user',
                    'content': full_prompt
                }
            ],
            'temperature': 0.7,
            'max_tokens': 2048
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    response_text = data['choices'][0]['message']['content']
                    
                    return ExternalResponse(
                        response=response_text,
                        provider="openrouter",
                        model=self.default_external_model,
                        processing_time=0.0,  # Will be filled by caller
                        success=True
                    )
                else:
                    error_text = await response.text()
                    return ExternalResponse(
                        response=f"External service error: {error_text}",
                        provider="openrouter",
                        model=self.default_external_model,
                        processing_time=0.0,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
    
    def _get_cache_key(self, query: str, context: Optional[str] = None) -> str:
        """Generate cache key for query"""
        content = f"{query}:{context or ''}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[ExternalResponse]:
        """Get cached response if available and not expired"""
        if cache_key not in self.response_cache:
            return None
        
        cached_data = self.response_cache[cache_key]
        cached_time = datetime.fromisoformat(cached_data['timestamp'])
        
        if datetime.now() - cached_time > timedelta(hours=self.cache_ttl_hours):
            # Cache expired
            del self.response_cache[cache_key]
            return None
        
        # Return cached response
        cached_response = ExternalResponse(**cached_data['response'])
        cached_response.cached = True
        return cached_response
    
    def _cache_response(self, cache_key: str, response: ExternalResponse):
        """Cache a response with enhanced statistics"""
        self.response_cache[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'response': {
                'response': response.response,
                'provider': response.provider,
                'model': response.model,
                'processing_time': response.processing_time,
                'success': response.success,
                'error': response.error
            }
        }
        
        # Update cache statistics
        self.cache_stats.cache_size = len(self.response_cache)
        self.cache_stats.cache_memory_usage = self._estimate_cache_memory_usage()
        
        # Clean old cache entries if needed
        if len(self.response_cache) > self.cache_max_size:
            self._clean_cache()
    
    def _clean_cache(self):
        """Clean expired cache entries and track evictions"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, data in self.response_cache.items():
            cached_time = datetime.fromisoformat(data['timestamp'])
            if current_time - cached_time > timedelta(hours=self.cache_ttl_hours):
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            del self.response_cache[key]
        
        # Update statistics
        self.cache_stats.evictions += len(expired_keys)
        self.cache_stats.cache_size = len(self.response_cache)
        self.cache_stats.cache_memory_usage = self._estimate_cache_memory_usage()
        
        # Update hit rate
        total_requests = self.cache_stats.hits + self.cache_stats.misses
        if total_requests > 0:
            self.cache_stats.average_hit_rate = self.cache_stats.hits / total_requests
        
        self.logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
    
    def _estimate_cache_memory_usage(self) -> int:
        """Estimate memory usage of the cache in bytes"""
        total_size = 0
        for key, data in self.response_cache.items():
            # Rough estimate: key size + data size
            total_size += len(key) * 2  # Unicode characters
            total_size += len(str(data)) * 2  # JSON string representation
        return total_size
    
    def add_consent_callback(self, callback: callable):
        """Add a callback for user consent requests"""
        self.consent_callbacks.append(callback)
    
    async def request_user_consent(self, query: str, routing_decision: RoutingDecision) -> bool:
        """
        Request user consent for external routing
        
        Args:
            query: The user's query
            routing_decision: The routing decision
            
        Returns:
            True if user consents, False otherwise
        """
        if not self.consent_callbacks:
            # No consent mechanism available, default to False for privacy
            self.logger.warning("No consent callback available for external routing")
            return False
        
        # Call all consent callbacks
        for callback in self.consent_callbacks:
            try:
                result = await callback(query, routing_decision)
                if not result:
                    return False
            except Exception as e:
                self.logger.error(f"Error in consent callback: {e}")
                return False
        
        return True
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics with backward compatibility"""
        avg_processing_time = (
            self.total_processing_time / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        success_rate = (
            self.success_count / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        # Calculate cache hit rate
        total_cache_requests = self.cache_stats.hits + self.cache_stats.misses
        cache_hit_rate = (
            self.cache_stats.hits / total_cache_requests 
            if total_cache_requests > 0 else 0.0
        )
        
        return {
            # Backward compatible fields (original format)
            'total_requests': self.request_count,
            'successful_requests': self.success_count,
            'failed_requests': self.failure_count,
            'success_rate': success_rate,
            'average_processing_time': avg_processing_time,
            'cached_responses': len(self.response_cache),  # Backward compatibility
            'api_key_configured': bool(self.openrouter_api_key),  # Backward compatibility
            
            # Enhanced metrics
            'retry_count': self.retry_count,
            'fallback_count': self.fallback_count,
            
            # Cache statistics
            'cache_stats': {
                'hits': self.cache_stats.hits,
                'misses': self.cache_stats.misses,
                'hit_rate': cache_hit_rate,
                'evictions': self.cache_stats.evictions,
                'current_size': self.cache_stats.cache_size,
                'max_size': self.cache_max_size,
                'memory_usage_bytes': self.cache_stats.cache_memory_usage,
                'ttl_hours': self.cache_ttl_hours
            },
            
            # Circuit breaker status
            'circuit_breaker': {
                'state': self.circuit_breaker.state,
                'failure_count': self.circuit_breaker.failure_count,
                'success_count': self.circuit_breaker.success_count,
                'failure_threshold': self.circuit_breaker.failure_threshold,
                'last_failure_time': self.circuit_breaker.last_failure_time.isoformat() if self.circuit_breaker.last_failure_time else None
            },
            
            # Service health
            'service_health': {
                'status': self.service_health.status.value,
                'last_check': self.service_health.last_check.isoformat() if self.service_health.last_check else None,
                'response_time': self.service_health.response_time,
                'consecutive_failures': self.service_health.consecutive_failures,
                'consecutive_successes': self.service_health.consecutive_successes
            },
            
            # Configuration
            'configuration': {
                'api_key_configured': bool(self.openrouter_api_key),
                'default_model': self.default_external_model,
                'fallback_callback_configured': bool(self.fallback_callback),
                'consent_callbacks_count': len(self.consent_callbacks),
                'health_checks_enabled': self.enable_health_checks
            }
        }
    
    async def close(self):
        """Clean up resources"""
        self.logger.info("ExternalModelConnector closing...")
        # Clean up any resources if needed
        self.logger.info("ExternalModelConnector closed") 