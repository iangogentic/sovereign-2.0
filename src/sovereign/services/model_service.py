"""
Model Service - Lazy-loaded AI Model Management

Provides on-demand loading of Talker and Thinker models to ensure fast application startup.
Manages model lifecycle, memory usage, and intelligent routing between models.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Union
from enum import Enum

# Only import lightweight modules at module level
# Heavy imports (TalkerModel, ThinkerModel) are deferred to loading functions

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available model types"""
    TALKER = "talker"
    THINKER = "thinker"


class ModelStatus(Enum):
    """Model loading status"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class ModelService:
    """
    Lazy-loaded model service that manages Talker and Thinker models
    
    Provides intelligent routing between models based on query complexity
    while ensuring models are only loaded when first needed.
    """
    
    def __init__(self):
        """Initialize the model service with no models loaded"""
        self._talker_model: Optional[Any] = None
        self._thinker_model: Optional[Any] = None
        
        # Track model status
        self._model_status: Dict[ModelType, ModelStatus] = {
            ModelType.TALKER: ModelStatus.UNLOADED,
            ModelType.THINKER: ModelStatus.UNLOADED
        }
        
        # Thread safety locks
        self._talker_lock = asyncio.Lock()
        self._thinker_lock = asyncio.Lock()
        
        # Performance tracking
        self._load_times: Dict[ModelType, float] = {}
        self._query_count = 0
        self._talker_queries = 0
        self._thinker_queries = 0
        
        logger.info("ModelService initialized - models will be loaded on demand")
    
    async def _load_talker_model(self) -> bool:
        """
        Lazy load the Talker model
        
        Returns:
            True if successful, False otherwise
        """
        async with self._talker_lock:
            # Check if already loaded
            if self._model_status[ModelType.TALKER] == ModelStatus.READY:
                return True
            
            # Check if currently loading
            if self._model_status[ModelType.TALKER] == ModelStatus.LOADING:
                # Wait for loading to complete
                while self._model_status[ModelType.TALKER] == ModelStatus.LOADING:
                    await asyncio.sleep(0.1)
                return self._model_status[ModelType.TALKER] == ModelStatus.READY
            
            # Start loading
            self._model_status[ModelType.TALKER] = ModelStatus.LOADING
            start_time = time.time()
            
            try:
                logger.info("🔄 Loading Talker model...")
                
                # Import TalkerModel only when needed (heavy import deferred)
                from ..talker_model import TalkerModel
                
                # Create and initialize the model
                self._talker_model = TalkerModel()
                success = await self._talker_model.initialize()
                
                if success:
                    self._model_status[ModelType.TALKER] = ModelStatus.READY
                    load_time = time.time() - start_time
                    self._load_times[ModelType.TALKER] = load_time
                    logger.info(f"✅ Talker model loaded successfully in {load_time:.2f}s")
                    return True
                else:
                    self._model_status[ModelType.TALKER] = ModelStatus.ERROR
                    logger.error("❌ Failed to initialize Talker model")
                    return False
                    
            except Exception as e:
                self._model_status[ModelType.TALKER] = ModelStatus.ERROR
                logger.error(f"❌ Error loading Talker model: {e}")
                return False
    
    async def _load_thinker_model(self) -> bool:
        """
        Lazy load the Thinker model
        
        Returns:
            True if successful, False otherwise
        """
        async with self._thinker_lock:
            # Check if already loaded
            if self._model_status[ModelType.THINKER] == ModelStatus.READY:
                return True
            
            # Check if currently loading
            if self._model_status[ModelType.THINKER] == ModelStatus.LOADING:
                # Wait for loading to complete
                while self._model_status[ModelType.THINKER] == ModelStatus.LOADING:
                    await asyncio.sleep(0.1)
                return self._model_status[ModelType.THINKER] == ModelStatus.READY
            
            # Start loading
            self._model_status[ModelType.THINKER] = ModelStatus.LOADING
            start_time = time.time()
            
            try:
                logger.info("🔄 Loading Thinker model...")
                
                # Import ThinkerModel only when needed (heavy import deferred)
                from ..thinker_model import ThinkerModel
                
                # Create and initialize the model
                self._thinker_model = ThinkerModel()
                success = await self._thinker_model.initialize()
                
                if success:
                    self._model_status[ModelType.THINKER] = ModelStatus.READY
                    load_time = time.time() - start_time
                    self._load_times[ModelType.THINKER] = load_time
                    logger.info(f"✅ Thinker model loaded successfully in {load_time:.2f}s")
                    return True
                else:
                    self._model_status[ModelType.THINKER] = ModelStatus.ERROR
                    logger.error("❌ Failed to initialize Thinker model")
                    return False
                    
            except Exception as e:
                self._model_status[ModelType.THINKER] = ModelStatus.ERROR
                logger.error(f"❌ Error loading Thinker model: {e}")
                return False
    
    def _should_use_thinker(self, query: str) -> bool:
        """
        Determine if query should use Thinker model based on complexity
        
        Args:
            query: User query to analyze
            
        Returns:
            True if Thinker model should be used
        """
        # Simple heuristics for model selection
        query_lower = query.lower()
        
        # Complex reasoning indicators
        complex_keywords = [
            'analyze', 'explain', 'compare', 'evaluate', 'assess',
            'code', 'implement', 'debug', 'algorithm', 'function',
            'step by step', 'detailed', 'comprehensive', 'research',
            'plan', 'strategy', 'solve problem', 'calculate'
        ]
        
        # Check for complex keywords
        if any(keyword in query_lower for keyword in complex_keywords):
            return True
        
        # Long queries likely need deeper reasoning
        if len(query.split()) > 30:
            return True
        
        # Multiple questions
        if query.count('?') > 2:
            return True
        
        # Default to Talker for simple queries
        return False
    
    async def query(self, user_input: str, context: Optional[str] = None) -> str:
        """
        Process a user query using the appropriate model
        
        Args:
            user_input: User's query
            context: Optional additional context
            
        Returns:
            AI response
        """
        self._query_count += 1
        
        # Determine which model to use
        use_thinker = self._should_use_thinker(user_input)
        
        if use_thinker:
            return await self._query_thinker(user_input, context)
        else:
            return await self._query_talker(user_input, context)
    
    async def _query_talker(self, user_input: str, context: Optional[str] = None) -> str:
        """Query using Talker model with lazy loading"""
        # Load Talker model if needed
        if not await self._load_talker_model():
            # Fallback to Thinker if Talker fails
            logger.warning("Talker model unavailable, falling back to Thinker")
            return await self._query_thinker(user_input, context)
        
        self._talker_queries += 1
        
        try:
            response = await self._talker_model.generate_response(user_input, context)
            
            # Check if Talker suggests handoff to Thinker
            if self._talker_model.detect_complex_query(user_input, context):
                logger.info("Talker detected complex query, handing off to Thinker")
                thinker_response = await self._query_thinker(user_input, context)
                # Combine responses or use better one
                return f"{response}\n\n[Deeper Analysis]: {thinker_response}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in Talker query: {e}")
            return f"Sorry, I encountered an error: {e}"
    
    async def _query_thinker(self, user_input: str, context: Optional[str] = None) -> str:
        """Query using Thinker model with lazy loading"""
        # Load Thinker model if needed
        if not await self._load_thinker_model():
            return "I'm sorry, but I'm unable to process complex queries right now. Please try a simpler question or check if the AI models are properly configured."
        
        self._thinker_queries += 1
        
        try:
            return await self._thinker_model.auto_process(user_input, context)
        except Exception as e:
            logger.error(f"Error in Thinker query: {e}")
            return f"Sorry, I encountered an error with complex reasoning: {e}"
    
    async def get_talker(self):
        """
        Get Talker model instance (loads if needed)
        
        Returns:
            TalkerModel instance or None if failed to load
        """
        if await self._load_talker_model():
            return self._talker_model
        return None
    
    async def get_thinker(self):
        """
        Get Thinker model instance (loads if needed)
        
        Returns:
            ThinkerModel instance or None if failed to load
        """
        if await self._load_thinker_model():
            return self._thinker_model
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of all models
        
        Returns:
            Dictionary with model status information
        """
        return {
            "talker_status": self._model_status[ModelType.TALKER].value,
            "thinker_status": self._model_status[ModelType.THINKER].value,
            "total_queries": self._query_count,
            "talker_queries": self._talker_queries,
            "thinker_queries": self._thinker_queries,
            "load_times": {k.value: v for k, v in self._load_times.items()}
        }
    
    def is_model_ready(self, model_type: ModelType) -> bool:
        """Check if a specific model is ready"""
        return self._model_status[model_type] == ModelStatus.READY
    
    async def unload_models(self):
        """
        Unload all models to free memory
        
        Useful for memory management when models aren't needed
        """
        async with self._talker_lock:
            if self._talker_model:
                try:
                    await self._talker_model.close()
                except Exception as e:
                    logger.error(f"Error closing Talker model: {e}")
                finally:
                    self._talker_model = None
                    self._model_status[ModelType.TALKER] = ModelStatus.UNLOADED
        
        async with self._thinker_lock:
            if self._thinker_model:
                try:
                    await self._thinker_model.close()
                except Exception as e:
                    logger.error(f"Error closing Thinker model: {e}")
                finally:
                    self._thinker_model = None
                    self._model_status[ModelType.THINKER] = ModelStatus.UNLOADED
        
        logger.info("All models unloaded")
    
    async def close(self):
        """Close the service and clean up resources"""
        await self.unload_models()
        logger.info("ModelService closed") 