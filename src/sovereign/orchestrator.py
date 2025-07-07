import asyncio
import hashlib
import json
import re
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict

from .config import Config
from .logger import setup_logger, get_performance_logger
from .talker_model import TalkerModel
from .thinker_model import ThinkerModel
from .screen_context_manager import ScreenContextManager, ScreenContextConfig
from .screen_context_integration import ScreenContextIntegration, ContextAccessRequest, ContextAccessLevel
import logging


class QueryComplexity(Enum):
    """Enum for query complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ModelChoice(Enum):
    """Enum for model selection"""
    TALKER = "talker"
    THINKER = "thinker"
    BOTH = "both"  # For queries that benefit from both models


@dataclass
class QueryContext:
    """Context information for queries"""
    user_input: str
    timestamp: datetime
    session_id: str
    previous_queries: List[str]
    conversation_history: List[Dict[str, Any]]
    screen_context: Optional[Dict[str, Any]] = None
    voice_context: Optional[Dict[str, Any]] = None
    user_preferences: Optional[Dict[str, Any]] = None


@dataclass
class OrchestrationResult:
    """Result of orchestration processing"""
    response: str
    model_used: ModelChoice
    complexity_level: QueryComplexity
    processing_time: float
    handoff_occurred: bool
    cache_hit: bool
    confidence_score: float
    reasoning: str
    telemetry: Dict[str, Any]


@dataclass
class CacheEntry:
    """Cache entry for responses"""
    query_hash: str
    response: str
    model_used: ModelChoice
    timestamp: datetime
    hit_count: int
    last_accessed: datetime


class ModelOrchestrator:
    """
    Intelligent orchestration system that manages the handoff between
    Talker and Thinker models based on query complexity and context.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.perf_logger = get_performance_logger()
        
        # Initialize models
        self.talker_model: Optional[TalkerModel] = None
        self.thinker_model: Optional[ThinkerModel] = None
        
        # Screen context system
        self.screen_context_manager: Optional[ScreenContextManager] = None
        self.screen_context_integration: Optional[ScreenContextIntegration] = None
        self.enable_screen_context = getattr(config, 'enable_screen_context', True)
        
        # Response cache
        self.response_cache: Dict[str, CacheEntry] = {}
        self.cache_max_size = 1000
        self.cache_ttl_hours = 24
        
        # Telemetry data
        self.telemetry = {
            'total_queries': 0,
            'talker_queries': 0,
            'thinker_queries': 0,
            'handoff_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0.0,
            'complexity_distribution': defaultdict(int),
            'error_count': 0,
            'uptime_start': datetime.now(),
            'screen_context_requests': 0,
            'screen_context_enabled': 0
        }
        
        # Complexity detection patterns
        self._init_complexity_patterns()
        
        # User notification callbacks
        self.notification_callbacks = []
        
        self.logger.info("ModelOrchestrator initialized")
    
    def _init_complexity_patterns(self):
        """Initialize patterns for complexity detection"""
        # Simple patterns - typically handled by Talker
        self.simple_patterns = [
            r'\b(hi|hello|hey|what\'s up|good morning|good evening)\b',
            r'\b(how are you|how\'s it going|what\'s new)\b',
            r'\b(thank you|thanks|thx)\b',
            r'\b(yes|no|ok|okay|sure|definitely|absolutely)\b',
            r'\b(what time|what date|current time|current date)\b',
            r'\b(weather|temperature)\b',
            r'\b(define|meaning of|what is|what does .* mean)\b',
        ]
        
        # Complex patterns - require Thinker
        self.complex_patterns = [
            r'\b(analyze|analysis|examine|evaluate|assess|compare|contrast)\b',
            r'\b(explain .* steps|how do I|step by step|walkthrough)\b',
            r'\b(code|programming|debug|algorithm|function|class|method)\b',
            r'\b(research|investigate|find information|look up)\b',
            r'\b(plan|strategy|approach|methodology|framework)\b',
            r'\b(calculate|compute|math|equation|formula|solve)\b',
            r'\b(create|generate|build|design|develop|implement)\b',
            r'\b(optimize|improve|enhance|refactor|refine)\b',
            r'\b(troubleshoot|fix|repair|resolve|solve problem)\b',
            r'\b(pros and cons|advantages|disadvantages|trade-offs)\b',
            r'\b(recommend|suggest|advise|best practices|alternatives)\b',
        ]
        
        # Multi-step reasoning indicators
        self.multi_step_patterns = [
            r'\b(first.*then|step 1.*step 2|initially.*after)\b',
            r'\b(because.*therefore|if.*then|assuming.*would)\b',
            r'\b(considering.*we should|given that.*the best)\b',
            r'\b(let\'s start.*then move|begin with.*proceed to)\b',
        ]
        
        # Tool use indicators
        self.tool_use_patterns = [
            r'\b(search|google|look up|find online|browse)\b',
            r'\b(send email|create file|save|download|upload)\b',
            r'\b(open|launch|start|run|execute)\b',
            r'\b(schedule|calendar|appointment|meeting)\b',
            r'\b(translate|conversion|convert|transform)\b',
        ]
    
    async def initialize(self):
        """Initialize the orchestrator and both models"""
        try:
            print("DEBUG: Starting ModelOrchestrator initialization...")
            self.logger.info("Initializing ModelOrchestrator...")
            
            print("DEBUG: Creating TalkerModel instance...")
            # Initialize Talker model - FIXED: Pass model name string, not entire Config object
            self.talker_model = TalkerModel(self.config.models.talker_model)
            print("DEBUG: TalkerModel instance created, about to initialize...")
            await self.talker_model.initialize()
            print("DEBUG: TalkerModel initialized successfully.")
            
            print("DEBUG: Creating ThinkerModel instance...")
            # Initialize Thinker model - FIXED: Pass model name string, not entire Config object
            self.thinker_model = ThinkerModel(self.config.models.thinker_model)
            print("DEBUG: ThinkerModel instance created, about to initialize...")
            await self.thinker_model.initialize()
            print("DEBUG: ThinkerModel initialized successfully.")
            
            print("DEBUG: Checking if screen context should be enabled...")
            # Initialize screen context system if enabled
            if self.enable_screen_context:
                print("DEBUG: Screen context enabled, initializing...")
                await self._initialize_screen_context()
                print("DEBUG: Screen context initialization completed.")
            else:
                print("DEBUG: Screen context disabled, skipping initialization.")
            
            print("DEBUG: ModelOrchestrator initialization completed successfully.")
            self.logger.info("ModelOrchestrator initialization complete")
            
        except Exception as e:
            print(f"DEBUG: ModelOrchestrator initialization failed with error: {e}")
            self.logger.error(f"Failed to initialize ModelOrchestrator: {e}")
            raise
    
    async def _initialize_screen_context(self):
        """Initialize screen context management system"""
        try:
            print("DEBUG: Starting screen context initialization...")
            self.logger.info("Initializing screen context system...")
            
            print("DEBUG: Creating screen context configuration...")
            # Create screen context configuration
            screen_config = ScreenContextConfig(
                capture_interval=getattr(self.config, 'screen_capture_interval', 4.0),
                max_stored_captures=getattr(self.config, 'max_screen_captures', 100),
                privacy_mode=getattr(self.config, 'privacy_mode', False),
                enable_preprocessing=True,
                min_text_confidence=30.0
            )
            print("DEBUG: Screen context configuration created.")
            
            print("DEBUG: Creating ScreenContextManager...")
            # Initialize screen context manager
            self.screen_context_manager = ScreenContextManager(screen_config, self.config)
            print("DEBUG: ScreenContextManager created, about to initialize...")
            if await self.screen_context_manager.initialize(self.config):
                print("DEBUG: ScreenContextManager initialized successfully.")
                self.logger.info("✅ Screen context manager initialized successfully")
                
                print("DEBUG: Creating ScreenContextIntegration...")
                # Initialize integration layer
                self.screen_context_integration = ScreenContextIntegration(
                    self.screen_context_manager, self.config
                )
                print("DEBUG: ScreenContextIntegration created, about to initialize...")
                
                if await self.screen_context_integration.initialize():
                    print("DEBUG: ScreenContextIntegration initialized successfully.")
                    self.logger.info("✅ Screen context integration initialized successfully")
                    self.telemetry['screen_context_enabled'] = 1
                    
                    print("DEBUG: Checking if screen capture should be started...")
                    # Start screen capture if not in privacy mode
                    if not screen_config.privacy_mode:
                        print("DEBUG: Privacy mode off, starting screen capture...")
                        await self.screen_context_manager.start_capture()
                        print("DEBUG: Screen capture started.")
                        self.logger.info("🔄 Screen capture started")
                    else:
                        print("DEBUG: Privacy mode on, skipping screen capture start.")
                else:
                    print("DEBUG: Failed to initialize ScreenContextIntegration.")
                    self.logger.error("❌ Failed to initialize screen context integration")
                    self.screen_context_integration = None
            else:
                print("DEBUG: Failed to initialize ScreenContextManager.")
                self.logger.error("❌ Failed to initialize screen context manager")
                self.screen_context_manager = None
                
        except Exception as e:
            print(f"DEBUG: Screen context initialization failed with error: {e}")
            self.logger.error(f"Screen context initialization failed: {e}")
            self.screen_context_manager = None
            self.screen_context_integration = None
    
    async def process_query(self, user_input: str, context: Optional[QueryContext] = None) -> OrchestrationResult:
        """
        Main entry point for processing user queries.
        Determines complexity and routes to appropriate model(s).
        """
        start_time = time.time()
        
        try:
            # Update telemetry
            self.telemetry['total_queries'] += 1
            
            # Create context if not provided
            if context is None:
                context = QueryContext(
                    user_input=user_input,
                    timestamp=datetime.now(),
                    session_id="default",
                    previous_queries=[],
                    conversation_history=[]
                )
            
            # Enrich context with screen context if available
            await self._enrich_context_with_screen_data(context)
            
            # Check cache first
            cache_result = self._check_cache(user_input, context)
            if cache_result:
                self.telemetry['cache_hits'] += 1
                processing_time = time.time() - start_time
                return OrchestrationResult(
                    response=cache_result.response,
                    model_used=cache_result.model_used,
                    complexity_level=QueryComplexity.SIMPLE,  # Cached responses are typically simple
                    processing_time=processing_time,
                    handoff_occurred=False,
                    cache_hit=True,
                    confidence_score=0.9,  # High confidence for cached responses
                    reasoning="Response retrieved from cache",
                    telemetry=self._get_telemetry_snapshot()
                )
            
            self.telemetry['cache_misses'] += 1
            
            # Determine query complexity
            complexity, confidence = self.determine_complexity(user_input, context)
            self.telemetry['complexity_distribution'][complexity.value] += 1
            
            # Select appropriate model
            model_choice = self._select_model(complexity, confidence, context)
            
            # Process query with selected model(s)
            result = await self._process_with_model(user_input, context, model_choice, complexity)
            
            # Cache the result if appropriate
            if self._should_cache(result, complexity):
                self._cache_response(user_input, context, result)
            
            # Update telemetry
            processing_time = time.time() - start_time
            self.telemetry['avg_response_time'] = (
                (self.telemetry['avg_response_time'] * (self.telemetry['total_queries'] - 1) + processing_time) /
                self.telemetry['total_queries']
            )
            
            result.processing_time = processing_time
            result.confidence_score = confidence
            result.telemetry = self._get_telemetry_snapshot()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            self.telemetry['error_count'] += 1
            
            # Return fallback response
            return OrchestrationResult(
                response="I apologize, but I encountered an error processing your request. Please try again.",
                model_used=ModelChoice.TALKER,
                complexity_level=QueryComplexity.SIMPLE,
                processing_time=time.time() - start_time,
                handoff_occurred=False,
                cache_hit=False,
                confidence_score=0.0,
                reasoning=f"Error occurred: {str(e)}",
                telemetry=self._get_telemetry_snapshot()
            )
    
    def determine_complexity(self, query: str, context: Optional[QueryContext] = None) -> Tuple[QueryComplexity, float]:
        """
        Analyze query complexity using multiple heuristics.
        Returns complexity level and confidence score.
        """
        query_lower = query.lower()
        confidence_factors = []
        
        # 1. Pattern matching
        simple_matches = sum(1 for pattern in self.simple_patterns if re.search(pattern, query_lower))
        complex_matches = sum(1 for pattern in self.complex_patterns if re.search(pattern, query_lower))
        multi_step_matches = sum(1 for pattern in self.multi_step_patterns if re.search(pattern, query_lower))
        tool_use_matches = sum(1 for pattern in self.tool_use_patterns if re.search(pattern, query_lower))
        
        pattern_score = 0
        if simple_matches > 0:
            pattern_score -= simple_matches * 0.3
        if complex_matches > 0:
            pattern_score += complex_matches * 0.5
        if multi_step_matches > 0:
            pattern_score += multi_step_matches * 0.7
        if tool_use_matches > 0:
            pattern_score += tool_use_matches * 0.6
        
        confidence_factors.append(('pattern_matching', min(abs(pattern_score), 1.0)))
        
        # 2. Query length and structure
        word_count = len(query.split())
        sentence_count = len([s for s in query.split('.') if s.strip()])
        question_marks = query.count('?')
        
        length_score = 0
        if word_count > 20:
            length_score += 0.3
        if word_count > 50:
            length_score += 0.4
        if sentence_count > 2:
            length_score += 0.3
        if question_marks > 1:
            length_score += 0.2
        
        confidence_factors.append(('length_structure', min(length_score, 1.0)))
        
        # 3. Technical terminology
        technical_terms = [
            'algorithm', 'function', 'variable', 'database', 'api', 'framework',
            'optimize', 'refactor', 'deploy', 'architecture', 'scalability',
            'performance', 'security', 'encryption', 'authentication'
        ]
        tech_score = sum(0.1 for term in technical_terms if term in query_lower)
        confidence_factors.append(('technical_terms', min(tech_score, 1.0)))
        
        # 4. Context analysis
        context_score = 0
        if context:
            # Consider conversation history
            if len(context.conversation_history) > 0:
                recent_complex = any(
                    'complex' in str(item).lower() or 'analyze' in str(item).lower()
                    for item in context.conversation_history[-3:]
                )
                if recent_complex:
                    context_score += 0.3
            
            # Consider previous queries
            if len(context.previous_queries) > 0:
                recent_query = context.previous_queries[-1].lower()
                if any(pattern in recent_query for pattern in ['explain', 'analyze', 'compare']):
                    context_score += 0.2
            
            # Consider screen context
            if hasattr(context, 'screen_context') and context.screen_context:
                screen_ctx = context.screen_context
                
                # Queries about visible screen content may be more complex
                if any(keyword in query_lower for keyword in ['on screen', 'visible', 'current', 'this page', 'what i see']):
                    if screen_ctx.get('has_text_content', False):
                        context_score += 0.4  # High boost for screen-aware queries
                
                # Large amount of screen text suggests complex analysis
                text_length = screen_ctx.get('text_length', 0)
                if text_length > 500:
                    context_score += 0.2
                elif text_length > 1000:
                    context_score += 0.3
                
                # Multiple screen elements suggest complexity
                elements_count = len(screen_ctx.get('elements', []))
                if elements_count > 5:
                    context_score += 0.2
        
        confidence_factors.append(('context_analysis', min(context_score, 1.0)))
        
        # Calculate final scores
        total_score = pattern_score + length_score + tech_score + context_score
        avg_confidence = sum(factor[1] for factor in confidence_factors) / len(confidence_factors)
        
        # Determine complexity level
        if total_score < -0.5:
            complexity = QueryComplexity.SIMPLE
        elif total_score < 0.5:
            complexity = QueryComplexity.MODERATE
        elif total_score < 1.5:
            complexity = QueryComplexity.COMPLEX
        else:
            complexity = QueryComplexity.VERY_COMPLEX
        
        self.logger.debug(f"Complexity analysis: {complexity.value} (score: {total_score:.2f}, confidence: {avg_confidence:.2f})")
        
        return complexity, avg_confidence
    
    def _select_model(self, complexity: QueryComplexity, confidence: float, context: QueryContext) -> ModelChoice:
        """Select appropriate model based on complexity and confidence"""
        if complexity == QueryComplexity.SIMPLE and confidence > 0.7:
            return ModelChoice.TALKER
        elif complexity == QueryComplexity.VERY_COMPLEX or confidence > 0.8:
            return ModelChoice.THINKER
        elif complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]:
            # For moderate complexity, consider context
            if len(context.conversation_history) > 0:
                # If we're in a complex conversation, use Thinker
                return ModelChoice.THINKER
            else:
                # Try Talker first, with potential handoff
                return ModelChoice.TALKER
        else:
            # Default to Talker with handoff capability
            return ModelChoice.TALKER
    
    async def _process_with_model(self, user_input: str, context: QueryContext, model_choice: ModelChoice, complexity: QueryComplexity) -> OrchestrationResult:
        """Process query with the selected model"""
        handoff_occurred = False
        
        if model_choice == ModelChoice.TALKER:
            # Try Talker first
            self.telemetry['talker_queries'] += 1
            await self._notify_user("Processing with Talker model...")
            
            response = await self.talker_model.generate_response(user_input)
            
            # Check if handoff is needed
            if self._should_handoff_to_thinker(user_input, response, complexity):
                handoff_occurred = True
                self.telemetry['handoff_queries'] += 1
                await self._notify_user("Consulting Thinker model for deeper analysis...")
                
                # Hand off to Thinker
                thinker_response = await self.handle_model_handoff(user_input, context)
                response = await self.integrate_responses(response, thinker_response)
                model_choice = ModelChoice.THINKER
            
            return OrchestrationResult(
                response=response,
                model_used=model_choice,
                complexity_level=complexity,
                processing_time=0.0,  # Will be set by caller
                handoff_occurred=handoff_occurred,
                cache_hit=False,
                confidence_score=0.0,  # Will be set by caller
                reasoning="Processed with Talker model" + (" with Thinker handoff" if handoff_occurred else ""),
                telemetry={}  # Will be set by caller
            )
        
        elif model_choice == ModelChoice.THINKER:
            # Use Thinker directly
            self.telemetry['thinker_queries'] += 1
            await self._notify_user("Processing with Thinker model...")
            
            response = await self.thinker_model.auto_process(user_input)
            
            return OrchestrationResult(
                response=response,
                model_used=ModelChoice.THINKER,
                complexity_level=complexity,
                processing_time=0.0,  # Will be set by caller
                handoff_occurred=False,
                cache_hit=False,
                confidence_score=0.0,  # Will be set by caller
                reasoning="Processed with Thinker model",
                telemetry={}  # Will be set by caller
            )
        
        else:
            # This shouldn't happen with current logic
            raise ValueError(f"Unsupported model choice: {model_choice}")
    
    def _should_handoff_to_thinker(self, user_input: str, talker_response: str, complexity: QueryComplexity) -> bool:
        """Determine if a handoff to Thinker is needed"""
        # Check if Talker response indicates uncertainty
        uncertainty_phrases = [
            "i'm not sure", "i don't know", "that's complex", "that's complicated",
            "i can't", "i'm unable", "that's beyond", "that requires", "you might want to"
        ]
        
        response_lower = talker_response.lower()
        if any(phrase in response_lower for phrase in uncertainty_phrases):
            return True
        
        # Check if response is too short for complex query
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            if len(talker_response.split()) < 20:
                return True
        
        # Check if user input has specific keywords that typically require deep analysis
        handoff_keywords = [
            'analyze', 'explain in detail', 'step by step', 'comprehensive',
            'thorough', 'detailed', 'research', 'investigate', 'compare thoroughly'
        ]
        
        input_lower = user_input.lower()
        if any(keyword in input_lower for keyword in handoff_keywords):
            return True
        
        return False
    
    async def handle_model_handoff(self, query: str, context: QueryContext) -> str:
        """Handle handoff from Talker to Thinker model"""
        try:
            # Prepare enhanced context for Thinker
            enhanced_context = self._prepare_handoff_context(query, context)
            
            # Use Thinker model with enhanced context
            response = await self.thinker_model.auto_process(enhanced_context)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error during model handoff: {e}")
            return "I apologize, but I encountered an issue while processing your complex request. Please try rephrasing your question."
    
    def _prepare_handoff_context(self, query: str, context: QueryContext) -> str:
        """Prepare enhanced context for Thinker model during handoff"""
        context_parts = [
            f"User Query: {query}",
            ""
        ]
        
        # Add conversation history if available
        if context.conversation_history:
            context_parts.append("Previous Conversation:")
            for item in context.conversation_history[-3:]:  # Last 3 exchanges
                if isinstance(item, dict):
                    if 'user' in item:
                        context_parts.append(f"User: {item['user']}")
                    if 'assistant' in item:
                        context_parts.append(f"Assistant: {item['assistant']}")
            context_parts.append("")
        
        # Add screen context if available
        if context.screen_context:
            context_parts.append("Screen Context:")
            context_parts.append(str(context.screen_context))
            context_parts.append("")
        
        context_parts.append("Please provide a comprehensive and detailed response to the user's query.")
        
        return "\n".join(context_parts)
    
    async def _enrich_context_with_screen_data(self, context: QueryContext):
        """Enrich QueryContext with current screen context data"""
        if not self.screen_context_integration:
            return
        
        try:
            self.telemetry['screen_context_requests'] += 1
            
            # Create context access request
            request = ContextAccessRequest(
                requester="orchestrator",
                access_level=ContextAccessLevel.ENHANCED,
                max_age_seconds=300,  # Last 5 minutes
                include_references=True,
                privacy_aware=True
            )
            
            # Get current screen context
            response = await self.screen_context_integration.get_current_context(request)
            
            if response.success:
                # Add screen context to QueryContext
                context.screen_context = {
                    "timestamp": response.timestamp,
                    "captures_count": response.captures_count,
                    "references_count": response.references_count,
                    "privacy_filtered": response.privacy_filtered,
                    "summary": await self.screen_context_integration.get_context_summary(
                        format_type="text_summary",
                        max_age_seconds=300
                    ),
                    "elements": (await self.screen_context_integration.get_screen_element_references(
                        max_age_seconds=300
                    ))[:10],  # Limit to 10 most recent elements
                    "access_level": response.access_level.value
                }
                
                # Add screen context indicators for complexity analysis
                if response.data.get("text_content"):
                    context.screen_context["has_text_content"] = True
                    context.screen_context["text_length"] = len(response.data["text_content"])
                else:
                    context.screen_context["has_text_content"] = False
                    context.screen_context["text_length"] = 0
                
                self.logger.debug(f"Enriched context with screen data: {response.captures_count} captures, {response.references_count} references")
            else:
                self.logger.warning(f"Failed to get screen context: {response.error_message}")
                context.screen_context = {
                    "error": response.error_message,
                    "privacy_filtered": response.privacy_filtered
                }
                
        except Exception as e:
            self.logger.error(f"Error enriching context with screen data: {e}")
            context.screen_context = {"error": str(e)}
    
    async def integrate_responses(self, talker_response: str, thinker_response: str) -> str:
        """Integrate responses from both models"""
        # Simple integration strategy - use Thinker response but acknowledge the handoff
        integrated = f"After further analysis, here's a more comprehensive response:\n\n{thinker_response}"
        
        # Could be enhanced with more sophisticated integration logic
        return integrated
    
    def _check_cache(self, query: str, context: QueryContext) -> Optional[CacheEntry]:
        """Check if query response is cached"""
        query_hash = self._get_query_hash(query, context)
        
        if query_hash in self.response_cache:
            entry = self.response_cache[query_hash]
            
            # Check if cache entry is still valid
            if datetime.now() - entry.timestamp < timedelta(hours=self.cache_ttl_hours):
                entry.hit_count += 1
                entry.last_accessed = datetime.now()
                return entry
            else:
                # Remove expired entry
                del self.response_cache[query_hash]
        
        return None
    
    def _cache_response(self, query: str, context: QueryContext, result: OrchestrationResult):
        """Cache a response"""
        query_hash = self._get_query_hash(query, context)
        
        # Clean cache if it's getting too large
        if len(self.response_cache) >= self.cache_max_size:
            self._clean_cache()
        
        self.response_cache[query_hash] = CacheEntry(
            query_hash=query_hash,
            response=result.response,
            model_used=result.model_used,
            timestamp=datetime.now(),
            hit_count=0,
            last_accessed=datetime.now()
        )
    
    def _get_query_hash(self, query: str, context: QueryContext) -> str:
        """Generate a hash for query caching"""
        # Create a simplified context for hashing
        cache_context = {
            'query': query.lower().strip(),
            'session_id': context.session_id,
            # Don't include timestamp or conversation history for broader cache hits
        }
        
        context_str = json.dumps(cache_context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _should_cache(self, result: OrchestrationResult, complexity: QueryComplexity) -> bool:
        """Determine if a result should be cached"""
        # Cache simple and moderate queries (basic factual queries should be cached)
        if complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]:
            return True
        
        # Cache complex queries that took significant time
        if result.processing_time > 5.0 and result.confidence_score > 0.5:
            return True
        
        # Cache any successful response with reasonable confidence
        if result.confidence_score > 0.3:
            return True
        
        return False
    
    def _clean_cache(self):
        """Clean old or rarely used cache entries"""
        # Remove oldest entries first
        sorted_entries = sorted(
            self.response_cache.items(),
            key=lambda x: (x[1].hit_count, x[1].last_accessed)
        )
        
        # Remove 20% of entries
        remove_count = len(sorted_entries) // 5
        for i in range(remove_count):
            del self.response_cache[sorted_entries[i][0]]
    
    async def _notify_user(self, message: str):
        """Notify user of orchestration actions"""
        for callback in self.notification_callbacks:
            try:
                await callback(message)
            except Exception as e:
                self.logger.error(f"Error in notification callback: {e}")
    
    def add_notification_callback(self, callback):
        """Add a notification callback"""
        self.notification_callbacks.append(callback)
    
    def _get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Get current telemetry data"""
        return {
            **self.telemetry,
            'uptime_seconds': (datetime.now() - self.telemetry['uptime_start']).total_seconds(),
            'cache_hit_rate': (
                self.telemetry['cache_hits'] / 
                (self.telemetry['cache_hits'] + self.telemetry['cache_misses'])
                if (self.telemetry['cache_hits'] + self.telemetry['cache_misses']) > 0 else 0
            ),
            'handoff_rate': (
                self.telemetry['handoff_queries'] / self.telemetry['total_queries']
                if self.telemetry['total_queries'] > 0 else 0
            )
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            'talker_model_ready': self.talker_model is not None,
            'thinker_model_ready': self.thinker_model is not None,
            'cache_size': len(self.response_cache),
            'telemetry': self._get_telemetry_snapshot()
        }
    
    async def close(self):
        """Clean up resources"""
        if self.talker_model:
            await self.talker_model.close()
        if self.thinker_model:
            await self.thinker_model.close()
        
        # Close screen context system
        if self.screen_context_manager:
            await self.screen_context_manager.stop_capture()
            await self.screen_context_manager.cleanup()
        
        if self.screen_context_integration:
            await self.screen_context_integration.cleanup()
        
        self.logger.info("ModelOrchestrator closed") 