"""
Tests for ThinkerModel class
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add src to path for importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sovereign.thinker_model import ThinkerModel, TaskType
from sovereign.ollama_client import OllamaError


class TestThinkerModel:
    """Test cases for ThinkerModel"""
    
    def test_init(self):
        """Test ThinkerModel initialization"""
        thinker = ThinkerModel()
        assert thinker.model_name == "deepseek-r1:14b"  # Default from config
        assert not thinker.is_initialized
        assert thinker.task_count == 0
        assert len(thinker.processing_times) == 0
    
    def test_custom_model_name(self):
        """Test ThinkerModel with custom model name"""
        custom_model = "custom:model"
        thinker = ThinkerModel(model_name=custom_model)
        assert thinker.model_name == custom_model
    
    def test_task_type_detection(self):
        """Test automatic task type detection"""
        thinker = ThinkerModel()
        
        # Test code generation detection
        assert thinker._detect_task_type("Write a Python function") == TaskType.CODE_GENERATION
        assert thinker._detect_task_type("Implement a sorting algorithm") == TaskType.CODE_GENERATION
        assert thinker._detect_task_type("Debug this code") == TaskType.CODE_GENERATION
        
        # Test tool use detection
        assert thinker._detect_task_type("Search the web for information") == TaskType.TOOL_USE_PLANNING
        assert thinker._detect_task_type("Call the API to get data") == TaskType.TOOL_USE_PLANNING
        assert thinker._detect_task_type("Execute this command") == TaskType.TOOL_USE_PLANNING
        
        # Test analysis detection
        assert thinker._detect_task_type("Analyze this data") == TaskType.ANALYSIS
        assert thinker._detect_task_type("Compare these options") == TaskType.ANALYSIS
        assert thinker._detect_task_type("Evaluate the performance") == TaskType.ANALYSIS
        
        # Test problem solving detection
        assert thinker._detect_task_type("How to solve this problem") == TaskType.PROBLEM_SOLVING
        assert thinker._detect_task_type("Find a strategy for this challenge") == TaskType.PROBLEM_SOLVING
        
        # Test default (deep reasoning)
        assert thinker._detect_task_type("What is the meaning of life?") == TaskType.DEEP_REASONING
    
    def test_system_prompt_generation(self):
        """Test system prompt generation for different task types"""
        thinker = ThinkerModel()
        
        # Test that different task types get different prompts
        reasoning_prompt = thinker._get_system_prompt(TaskType.DEEP_REASONING)
        code_prompt = thinker._get_system_prompt(TaskType.CODE_GENERATION)
        tool_prompt = thinker._get_system_prompt(TaskType.TOOL_USE_PLANNING)
        analysis_prompt = thinker._get_system_prompt(TaskType.ANALYSIS)
        problem_prompt = thinker._get_system_prompt(TaskType.PROBLEM_SOLVING)
        
        # All should contain the base prompt
        base_content = "You are Sovereign"
        assert base_content in reasoning_prompt
        assert base_content in code_prompt
        assert base_content in tool_prompt
        assert base_content in analysis_prompt
        assert base_content in problem_prompt
        
        # Each should have specific instructions
        assert "reasoning tasks" in reasoning_prompt
        assert "code generation" in code_prompt
        assert "tool use planning" in tool_prompt
        assert "analysis tasks" in analysis_prompt
        assert "problem-solving" in problem_prompt
    
    def test_performance_stats_empty(self):
        """Test performance stats with no processing history"""
        thinker = ThinkerModel()
        stats = thinker.get_performance_stats()
        
        assert stats["task_count"] == 0
        assert stats["average_processing_time"] == 0
        assert stats["last_task_type"] is None
        assert stats["total_processing_time"] == 0
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """Test initialization failure scenarios"""
        thinker = ThinkerModel()
        
        # Mock client to fail health check
        thinker.client.health_check = AsyncMock(return_value=False)
        
        result = await thinker.initialize()
        assert result is False
        assert not thinker.is_initialized
    
    @pytest.mark.asyncio
    async def test_initialization_success(self):
        """Test successful initialization"""
        thinker = ThinkerModel()
        
        # Mock successful initialization
        thinker.client.health_check = AsyncMock(return_value=True)
        thinker.client.list_models = AsyncMock(return_value=["deepseek-r1:14b"])
        thinker.client.generate = AsyncMock(return_value="READY")
        
        result = await thinker.initialize()
        assert result is True
        assert thinker.is_initialized
    
    @pytest.mark.asyncio
    async def test_model_pulling(self):
        """Test automatic model pulling when model not available"""
        thinker = ThinkerModel()
        
        # Mock scenario where model needs to be pulled
        thinker.client.health_check = AsyncMock(return_value=True)
        thinker.client.list_models = AsyncMock(return_value=[])  # Model not available
        thinker.client.pull_model = AsyncMock(return_value=True)
        thinker.client.generate = AsyncMock(return_value="READY")
        
        result = await thinker.initialize()
        assert result is True
        
        # Verify pull_model was called
        thinker.client.pull_model.assert_called_once_with("deepseek-r1:14b")
    
    @pytest.mark.asyncio
    async def test_auto_process_without_initialization(self):
        """Test auto_process when model is not initialized"""
        thinker = ThinkerModel()
        
        # Mock failed initialization
        thinker.client.health_check = AsyncMock(return_value=False)
        
        response = await thinker.auto_process("Test prompt")
        assert "ThinkerModel is not available" in response
    
    @pytest.mark.asyncio
    async def test_successful_processing(self):
        """Test successful task processing"""
        thinker = ThinkerModel()
        thinker.is_initialized = True
        
        # Mock successful generation
        mock_response = "This is a detailed analysis of the problem..."
        thinker.client.generate = AsyncMock(return_value=mock_response)
        
        response = await thinker.deep_reasoning("Analyze this complex problem")
        
        assert response == mock_response
        assert thinker.task_count == 1
        assert len(thinker.processing_times) == 1
        assert thinker.last_task_type == TaskType.DEEP_REASONING
    
    @pytest.mark.asyncio
    async def test_tool_use_planning_with_available_tools(self):
        """Test tool use planning with available tools list"""
        thinker = ThinkerModel()
        thinker.is_initialized = True
        
        mock_response = "Here's the tool usage plan..."
        thinker.client.generate = AsyncMock(return_value=mock_response)
        
        available_tools = ["web_search", "database_query", "file_reader"]
        response = await thinker.tool_use_planning(
            "Plan how to gather information", 
            context="Need to research a topic",
            available_tools=available_tools
        )
        
        assert response == mock_response
        # Verify the tools were included in the context
        call_args = thinker.client.generate.call_args[1]  # kwargs
        assert "web_search, database_query, file_reader" in call_args["prompt"]
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling during processing"""
        thinker = ThinkerModel()
        thinker.is_initialized = True
        
        # Mock OllamaError
        thinker.client.generate = AsyncMock(side_effect=OllamaError("Connection failed"))
        
        response = await thinker.auto_process("Test prompt")
        assert "ThinkerModel error:" in response
        assert "Connection failed" in response
    
    @pytest.mark.asyncio
    async def test_context_length_tracking(self):
        """Test that context lengths are tracked"""
        thinker = ThinkerModel()
        thinker.is_initialized = True
        
        thinker.client.generate = AsyncMock(return_value="Response")
        
        await thinker.deep_reasoning("Short prompt")
        await thinker.code_generation("Write a very long and detailed function with many parameters and complex logic")
        
        assert len(thinker.context_lengths) == 2
        assert thinker.context_lengths[1] > thinker.context_lengths[0]  # Second prompt should be longer
    
    @pytest.mark.asyncio
    async def test_performance_stats_with_data(self):
        """Test performance stats with actual processing data"""
        thinker = ThinkerModel()
        thinker.is_initialized = True
        
        thinker.client.generate = AsyncMock(return_value="Response")
        
        # Simulate some processing
        await thinker.deep_reasoning("Prompt 1")
        await thinker.code_generation("Prompt 2")
        
        stats = thinker.get_performance_stats()
        
        assert stats["task_count"] == 2
        assert stats["average_processing_time"] > 0
        assert stats["min_processing_time"] > 0
        assert stats["max_processing_time"] > 0
        assert stats["total_processing_time"] > 0
        assert stats["last_task_type"] == TaskType.CODE_GENERATION.value
        assert stats["average_context_length"] > 0
        assert stats["model_name"] == "deepseek-r1:14b"
    
    @pytest.mark.asyncio
    async def test_close(self):
        """Test model cleanup"""
        thinker = ThinkerModel()
        thinker.client.close = AsyncMock()
        
        await thinker.close()
        thinker.client.close.assert_called_once()


class TestTaskType:
    """Test TaskType enum"""
    
    def test_task_type_values(self):
        """Test that all task types have correct string values"""
        assert TaskType.DEEP_REASONING.value == "deep_reasoning"
        assert TaskType.CODE_GENERATION.value == "code_generation"
        assert TaskType.TOOL_USE_PLANNING.value == "tool_use_planning"
        assert TaskType.ANALYSIS.value == "analysis"
        assert TaskType.PROBLEM_SOLVING.value == "problem_solving"


if __name__ == "__main__":
    pytest.main([__file__]) 