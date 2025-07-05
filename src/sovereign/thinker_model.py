"""
Thinker Model - Advanced reasoning and complex task AI interface using Ollama

This module implements the "Thinker" half of the dual-model architecture,
designed for complex reasoning, code generation, and tool use planning.
"""

import time
import asyncio
import logging
from typing import Optional, Dict, Any, List, AsyncGenerator
from enum import Enum
import json

from .config import config
from .ollama_client import OllamaClient, OllamaError
from .logger import PerformanceTimer, get_performance_logger

logger = logging.getLogger(__name__)
performance_logger = get_performance_logger()


class TaskType(Enum):
    """Types of tasks the Thinker model can handle"""
    DEEP_REASONING = "deep_reasoning"
    CODE_GENERATION = "code_generation"
    TOOL_USE_PLANNING = "tool_use_planning"
    ANALYSIS = "analysis"
    PROBLEM_SOLVING = "problem_solving"


class ThinkerModel:
    """
    Advanced reasoning model using Ollama
    
    This is the sophisticated reasoning engine designed for complex tasks
    that require deep analysis, multi-step problem-solving, code generation,
    and tool use planning. Complements the fast TalkerModel.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize ThinkerModel
        
        Args:
            model_name: Override default model name from config
        """
        self.model_name = model_name or config.models.thinker_model
        self.client = OllamaClient(
            base_url=config.models.ollama_endpoint,
            timeout=config.models.thinker_timeout
        )
        
        self.is_initialized = False
        self.task_count = 0
        self.processing_times = []
        self.last_task_type = None
        
        # Performance tracking
        self.gpu_memory_usage = []
        self.context_lengths = []
        
        logger.info(f"ThinkerModel initialized with model: {self.model_name}")
    
    async def initialize(self) -> bool:
        """
        Initialize the Thinker model
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            with PerformanceTimer("ThinkerModel initialization", performance_logger):
                # Check Ollama health
                if not await self.client.check_health():
                    logger.error("Ollama server is not healthy")
                    return False
                
                # Check if model is available
                available_models = await self.client.list_models()
                if self.model_name not in available_models:
                    logger.warning(f"Model {self.model_name} not found. Attempting to pull...")
                    
                    # Try to pull the model
                    success = await self.client.pull_model(self.model_name)
                    if not success:
                        logger.error(f"Failed to pull model {self.model_name}")
                        return False
                
                # Test the model with a simple query
                test_response = await self.client.generate(
                    model=self.model_name,
                    prompt="Respond with 'READY' if you can process this request."
                )
                
                if test_response and "READY" in test_response.upper():
                    self.is_initialized = True
                    logger.info(f"✅ ThinkerModel successfully initialized with {self.model_name}")
                    return True
                else:
                    logger.error("Model test failed - unexpected response")
                    return False
                    
        except Exception as e:
            logger.error(f"ThinkerModel initialization failed: {e}")
            return False
    
    def _get_system_prompt(self, task_type: TaskType) -> str:
        """
        Get specialized system prompt based on task type
        
        Args:
            task_type: Type of task being performed
            
        Returns:
            Optimized system prompt for the task type
        """
        base_prompt = """You are Sovereign, an advanced AI reasoning engine. You are the "Thinker" component of a dual-model system, specialized in deep analysis, complex problem-solving, and sophisticated reasoning.

Your capabilities include:
- Multi-step logical reasoning and analysis
- Advanced code generation and debugging
- Tool use planning and orchestration
- Complex problem decomposition
- Technical documentation and explanation

Always think step-by-step and show your reasoning process."""
        
        if task_type == TaskType.DEEP_REASONING:
            return base_prompt + """

For reasoning tasks:
1. Break down complex problems into manageable steps
2. Consider multiple perspectives and approaches
3. Validate your reasoning at each step
4. Provide clear explanations of your thought process
5. Identify potential edge cases or limitations"""
            
        elif task_type == TaskType.CODE_GENERATION:
            return base_prompt + """

For code generation tasks:
1. Understand the requirements thoroughly
2. Choose appropriate algorithms and data structures
3. Write clean, well-documented, and efficient code
4. Consider error handling and edge cases
5. Provide explanations of your implementation choices
6. Include relevant tests when appropriate"""
            
        elif task_type == TaskType.TOOL_USE_PLANNING:
            return base_prompt + """

For tool use planning:
1. Analyze what tools or external resources are needed
2. Plan the sequence of tool calls and data flow
3. Consider error handling and fallback strategies
4. Optimize for efficiency and reliability
5. Provide clear justification for tool choices"""
            
        elif task_type == TaskType.ANALYSIS:
            return base_prompt + """

For analysis tasks:
1. Examine data or information systematically
2. Identify patterns, trends, and relationships
3. Draw evidence-based conclusions
4. Highlight key insights and implications
5. Consider limitations and potential biases"""
            
        else:  # PROBLEM_SOLVING
            return base_prompt + """

For problem-solving tasks:
1. Define the problem clearly and precisely
2. Identify constraints and requirements
3. Generate and evaluate multiple solution approaches
4. Select the optimal solution with justification
5. Consider implementation details and potential challenges"""
    
    def _detect_task_type(self, prompt: str) -> TaskType:
        """
        Automatically detect the type of task based on the prompt
        
        Args:
            prompt: User's input prompt
            
        Returns:
            Detected task type
        """
        prompt_lower = prompt.lower()
        
        # Code generation patterns
        if any(keyword in prompt_lower for keyword in [
            'write code', 'implement', 'function', 'class', 'algorithm',
            'debug', 'fix code', 'optimize code', 'program', 'script'
        ]):
            return TaskType.CODE_GENERATION
        
        # Tool use patterns
        elif any(keyword in prompt_lower for keyword in [
            'search', 'api', 'tool', 'external', 'web search', 'database',
            'file system', 'execute', 'run command'
        ]):
            return TaskType.TOOL_USE_PLANNING
        
        # Analysis patterns
        elif any(keyword in prompt_lower for keyword in [
            'analyze', 'examine', 'compare', 'evaluate', 'assess',
            'review', 'study', 'investigate'
        ]):
            return TaskType.ANALYSIS
        
        # Problem solving patterns
        elif any(keyword in prompt_lower for keyword in [
            'solve', 'problem', 'challenge', 'issue', 'difficulty',
            'how to', 'strategy', 'approach'
        ]):
            return TaskType.PROBLEM_SOLVING
        
        # Default to deep reasoning
        return TaskType.DEEP_REASONING
    
    async def deep_reasoning(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Perform deep reasoning and analysis
        
        Args:
            prompt: The reasoning task or question
            context: Additional context information
            
        Returns:
            Detailed reasoning response
        """
        return await self._process_task(prompt, context, TaskType.DEEP_REASONING)
    
    async def code_generation(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate code based on requirements
        
        Args:
            prompt: Code generation requirements
            context: Additional context (existing code, specifications)
            
        Returns:
            Generated code with explanations
        """
        return await self._process_task(prompt, context, TaskType.CODE_GENERATION)
    
    async def tool_use_planning(self, prompt: str, context: Optional[str] = None, 
                               available_tools: Optional[List[str]] = None) -> str:
        """
        Plan tool usage and external integrations
        
        Args:
            prompt: Task requiring tool use
            context: Additional context
            available_tools: List of available tools/APIs
            
        Returns:
            Tool usage plan and implementation strategy
        """
        enhanced_context = context or ""
        if available_tools:
            enhanced_context += f"\n\nAvailable tools: {', '.join(available_tools)}"
        
        return await self._process_task(prompt, enhanced_context, TaskType.TOOL_USE_PLANNING)
    
    async def _process_task(self, prompt: str, context: Optional[str], 
                           task_type: TaskType) -> str:
        """
        Internal method to process tasks with appropriate prompting
        
        Args:
            prompt: User prompt
            context: Additional context
            task_type: Type of task being processed
            
        Returns:
            AI response
        """
        if not self.is_initialized:
            logger.info("ThinkerModel not initialized, attempting auto-initialization...")
            if not await self.initialize():
                return "❌ ThinkerModel is not available. Please check Ollama server and model availability."
        
        start_time = time.time()
        self.task_count += 1
        self.last_task_type = task_type
        
        try:
            with PerformanceTimer(f"ThinkerModel {task_type.value}", performance_logger):
                # Build the full prompt with system prompt and context
                system_prompt = self._get_system_prompt(task_type)
                
                full_prompt = system_prompt + "\n\n"
                if context:
                    full_prompt += f"Context: {context}\n\n"
                full_prompt += f"Task: {prompt}\n\nResponse:"
                
                # Track context length
                self.context_lengths.append(len(full_prompt))
                
                # Generate response
                response = await self.client.generate(
                    model=self.model_name,
                    prompt=full_prompt,
                    temperature=config.models.thinker_temperature,
                    top_p=config.models.ollama_top_p
                )
                
                if not response:
                    return "❌ Failed to generate response from ThinkerModel."
                
                # Track performance
                elapsed_time = time.time() - start_time
                self.processing_times.append(elapsed_time)
                
                logger.info(f"ThinkerModel completed {task_type.value} task in {elapsed_time:.2f}s")
                
                return response.strip()
                
        except OllamaError as e:
            logger.error(f"Ollama error in ThinkerModel: {e}")
            return f"❌ ThinkerModel error: {e}"
        except Exception as e:
            logger.error(f"Unexpected error in ThinkerModel: {e}")
            return f"❌ ThinkerModel encountered an unexpected error: {e}"
    
    async def auto_process(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Automatically detect task type and process accordingly
        
        Args:
            prompt: User input
            context: Additional context
            
        Returns:
            AI response optimized for the detected task type
        """
        task_type = self._detect_task_type(prompt)
        logger.info(f"Auto-detected task type: {task_type.value}")
        return await self._process_task(prompt, context, task_type)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the ThinkerModel
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.processing_times:
            return {
                "task_count": 0,
                "average_processing_time": 0,
                "last_task_type": None,
                "total_processing_time": 0
            }
        
        return {
            "task_count": self.task_count,
            "average_processing_time": sum(self.processing_times) / len(self.processing_times),
            "min_processing_time": min(self.processing_times),
            "max_processing_time": max(self.processing_times),
            "last_task_type": self.last_task_type.value if self.last_task_type else None,
            "total_processing_time": sum(self.processing_times),
            "average_context_length": sum(self.context_lengths) / len(self.context_lengths) if self.context_lengths else 0,
            "model_name": self.model_name
        }
    
    async def close(self):
        """Clean up resources"""
        if self.client:
            await self.client.close()
        logger.info("ThinkerModel closed") 