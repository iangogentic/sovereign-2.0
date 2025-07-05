"""
Talker Model - Fast conversational AI interface using Ollama
"""

import time
import asyncio
import logging
from typing import Optional, Dict, Any, AsyncGenerator
import re

from .config import config
from .ollama_client import OllamaClient, OllamaError
from .logger import PerformanceTimer

logger = logging.getLogger(__name__)


class TalkerModel:
    """
    Fast conversational model using Ollama
    
    This is the primary interface for user interactions, designed to provide
    responses in under 2 seconds while detecting when complex queries need
    to be handed off to the Thinker model.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize TalkerModel
        
        Args:
            model_name: Ollama model name (defaults to config value)
        """
        self.model_name = model_name or config.models.talker_model
        self.ollama_client = OllamaClient(
            base_url=config.models.ollama_endpoint,
            timeout=config.models.ollama_timeout
        )
        self.is_initialized = False
        self.system_prompt = self._get_system_prompt()
        
        # Performance tracking
        self.response_times = []
        self.query_count = 0
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Talker model"""
        return """You are Sovereign, a fast and helpful AI assistant running locally on the user's machine. 

Key characteristics:
- You provide quick, helpful responses in under 2 seconds
- You are direct and concise but friendly
- You can handle most conversational queries, basic questions, and simple tasks
- When you encounter complex queries that require deep reasoning, code generation, or tool use, you should indicate that you'll consult your "deeper reasoning capabilities" and hand off to the Thinker model
- You maintain context about the user's privacy and local setup

Complex queries that should be handed off include:
- Multi-step reasoning problems
- Code generation and debugging
- Complex analysis or research
- Tool usage (web search, file operations, etc.)
- Long-form content creation
- Technical troubleshooting

Respond naturally and helpfully. If a query is complex, say something like "Let me engage my deeper reasoning capabilities for this..." and indicate the handoff."""
    
    async def initialize(self) -> bool:
        """
        Initialize the Talker model
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.is_initialized:
            return True
            
        try:
            logger.info(f"Initializing Talker model: {self.model_name}")
            
            # Check if Ollama server is running
            if not await self.ollama_client.check_health():
                logger.error("Ollama server is not accessible")
                return False
            
            # Check if model is available
            models = await self.ollama_client.list_models()
            available_models = [model['name'] for model in models.get('models', [])]
            
            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found locally. Attempting to pull...")
                if not await self.ollama_client.pull_model(self.model_name):
                    logger.error(f"Failed to pull model {self.model_name}")
                    return False
                logger.info(f"Successfully pulled model {self.model_name}")
            
            # Test generation to ensure everything works
            test_response = await self.ollama_client.generate(
                model=self.model_name,
                prompt="Hello, test if you're working.",
                system=self.system_prompt,
                temperature=config.models.ollama_temperature,
                top_p=config.models.ollama_top_p,
                max_tokens=50
            )
            
            if test_response:
                logger.info("Talker model initialized successfully")
                self.is_initialized = True
                return True
            else:
                logger.error("Model test generation failed")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing Talker model: {e}")
            return False
    
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        stream: bool = True
    ) -> str:
        """
        Generate a response to the user's prompt
        
        Args:
            prompt: User's input prompt
            context: Optional additional context
            stream: Whether to use streaming response
            
        Returns:
            Generated response text
        """
        if not self.is_initialized:
            if not await self.initialize():
                raise RuntimeError("Talker model not initialized")
        
        self.query_count += 1
        
        # Prepare full prompt with context
        full_prompt = prompt
        if context:
            full_prompt = f"Context: {context}\n\nUser: {prompt}"
        
        try:
            with PerformanceTimer() as timer:
                if stream and config.models.ollama_stream:
                    response = await self._generate_streaming(full_prompt)
                else:
                    response = await self._generate_complete(full_prompt)
            
            # Track response time
            response_time = timer.elapsed_time
            self.response_times.append(response_time)
            
            # Log performance
            logger.info(f"Response generated in {response_time:.2f}s (Query #{self.query_count})")
            
            # Check if we're meeting the sub-2-second goal
            if response_time > config.models.talker_timeout:
                logger.warning(f"Response time ({response_time:.2f}s) exceeded target ({config.models.talker_timeout}s)")
            
            return response.strip()
            
        except OllamaError as e:
            logger.error(f"Ollama error during generation: {e}")
            return "I'm sorry, I'm having trouble connecting to my language model. Please check if Ollama is running."
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            return "I encountered an unexpected error. Please try again."
    
    async def _generate_streaming(self, prompt: str) -> str:
        """Generate response using streaming"""
        response_parts = []
        
        async for chunk in self.ollama_client.generate_stream(
            model=self.model_name,
            prompt=prompt,
            system=self.system_prompt,
            temperature=config.models.ollama_temperature,
            top_p=config.models.ollama_top_p
        ):
            response_parts.append(chunk)
        
        return ''.join(response_parts)
    
    async def _generate_complete(self, prompt: str) -> str:
        """Generate complete response"""
        return await self.ollama_client.generate(
            model=self.model_name,
            prompt=prompt,
            system=self.system_prompt,
            temperature=config.models.ollama_temperature,
            top_p=config.models.ollama_top_p
        )
    
    def detect_complex_query(self, prompt: str, context: Optional[str] = None) -> bool:
        """
        Detect if a query requires the Thinker model
        
        Args:
            prompt: User's input prompt
            context: Optional additional context
            
        Returns:
            True if query should be handed off to Thinker model
        """
        # Convert to lowercase for pattern matching
        text = prompt.lower()
        if context:
            text += " " + context.lower()
        
        # Patterns that indicate complex queries
        complex_patterns = [
            # Code-related
            r'\b(write|create|generate|debug|fix).*(code|function|script|program|algorithm)\b',
            r'\b(implement|develop|build).*(feature|application|system|algorithm|function)\b',
            r'\b(refactor|optimize|review).*(code|implementation)\b',
            
            # Analysis and reasoning
            r'\b(analyze|compare|evaluate|assess|review).*(data|performance|options)\b',
            r'\b(explain|describe).*(complex|detailed|technical|architecture)\b',
            r'\b(solve|calculate|compute).*(problem|equation|optimization)\b',
            
            # Research and tools
            r'\b(search|find|look up|research)\b.*\b(web|internet|online)\b',
            r'\b(file|directory|folder).*(operation|management|search)\b',
            r'\b(install|configure|setup|deploy)\b',
            
            # Multi-step tasks
            r'\b(step by step|detailed|comprehensive|thorough)\b',
            r'\b(plan|strategy|approach|methodology)\b',
            r'\b(multiple|several|various).*(steps|tasks|components)\b',
            
            # Content creation
            r'\b(write|create|generate).*(article|essay|report|documentation|tutorial)\b',
            r'\b(draft|compose|author)\b',
        ]
        
        # Check for complex patterns
        for pattern in complex_patterns:
            if re.search(pattern, text):
                return True
        
        # Check for length-based complexity (very long prompts)
        if len(prompt.split()) > 50:
            return True
        
        # Check for question chains (multiple questions)
        question_count = text.count('?')
        if question_count > 2:
            return True
        
        return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.response_times:
            return {
                "total_queries": 0,
                "average_response_time": 0.0,
                "fastest_response": 0.0,
                "slowest_response": 0.0,
                "sub_2s_percentage": 0.0
            }
        
        avg_time = sum(self.response_times) / len(self.response_times)
        sub_2s_count = sum(1 for t in self.response_times if t <= config.models.talker_timeout)
        sub_2s_percentage = (sub_2s_count / len(self.response_times)) * 100
        
        return {
            "total_queries": self.query_count,
            "average_response_time": avg_time,
            "fastest_response": min(self.response_times),
            "slowest_response": max(self.response_times),
            "sub_2s_percentage": sub_2s_percentage,
            "target_time": config.models.talker_timeout
        }
    
    async def close(self):
        """Close the Ollama client connection"""
        await self.ollama_client.close()
        self.is_initialized = False
        logger.info("Talker model closed") 