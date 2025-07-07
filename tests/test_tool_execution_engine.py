"""
Test suite for Enhanced Tool Execution Engine
"""

import pytest
import asyncio
import time
import tempfile
import json
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional

from src.sovereign.tool_execution_engine import (
    EnhancedExecutionEngine,
    ParameterValidator,
    ExecutionSandbox,
    ResultFormatter,
    ExecutionContext,
    ExecutionStats,
    ValidationRule,
    ValidationError,
    ExecutionError,
    SecurityError,
    TimeoutError,
    ParameterType,
    ExecutionState
)
from src.sovereign.tool_integration_framework import (
    BaseTool,
    ToolMetadata,
    ToolParameter,
    ToolExecutionResult,
    ToolSecurityLevel,
    ToolExecutionMode,
    ToolType,
    PermissionManager
)
from src.sovereign.config import Config


class MockTool(BaseTool):
    """Mock tool for testing"""
    
    def __init__(self, name: str = "mock_tool", should_fail: bool = False, delay: float = 0.0):
        # Create dummy metadata for BaseTool
        dummy_metadata = ToolMetadata(
            name=name,
            description="Mock tool for testing",
            version="1.0.0",
            tool_type=ToolType.PYTHON,
            security_level=ToolSecurityLevel.SAFE,
            execution_mode=ToolExecutionMode.SYNCHRONOUS,
            parameters=[],
            return_type="dict"
        )
        super().__init__(dummy_metadata)
        
        self.name = name
        self.should_fail = should_fail
        self.delay = delay
        self.execution_count = 0
    
    async def execute(self, parameters: Dict[str, Any]) -> Any:
        self.execution_count += 1
        
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        if self.should_fail:
            raise Exception(f"Mock tool {self.name} failed")
        
        return {"result": f"Mock execution with params: {parameters}", "execution_count": self.execution_count}
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameters - always return True for testing"""
        return True, None


class TestParameterValidator:
    """Test parameter validation functionality"""
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        validator = ParameterValidator()
        assert validator is not None
        assert len(validator.custom_validators) > 0
        assert 'email' in validator.custom_validators
        assert 'url' in validator.custom_validators
    
    def test_builtin_validators(self):
        """Test built-in validation functions"""
        validator = ParameterValidator()
        
        # Test email validator
        assert validator.custom_validators['email']("test@example.com") == True
        assert validator.custom_validators['email']("invalid-email") == False
        
        # Test URL validator
        assert validator.custom_validators['url']("https://example.com") == True
        assert validator.custom_validators['url']("invalid-url") == False
        
        # Test JSON validator
        assert validator.custom_validators['json']('{"key": "value"}') == True
        assert validator.custom_validators['json']("invalid-json") == False
        
        # Test safe string validator
        assert validator.custom_validators['safe_string']("safe text") == True
        assert validator.custom_validators['safe_string']("<script>alert('xss')</script>") == False
    
    def test_parameter_type_validation(self):
        """Test parameter type validation"""
        validator = ParameterValidator()
        
        # String parameter
        string_param = ToolParameter(
            name="test_string",
            type="string",
            required=True,
            description="Test string parameter"
        )
        
        valid, error = validator.validate_parameter(string_param, "test value")
        assert valid == True
        assert error == ""
        
        valid, error = validator.validate_parameter(string_param, 123)
        assert valid == False
        assert "must be of type" in error
        
        # Integer parameter
        int_param = ToolParameter(
            name="test_int",
            type="integer",
            required=True,
            description="Test integer parameter"
        )
        
        valid, error = validator.validate_parameter(int_param, 42)
        assert valid == True
        
        valid, error = validator.validate_parameter(int_param, "not a number")
        assert valid == False
    
    def test_required_parameter_validation(self):
        """Test required parameter validation"""
        validator = ParameterValidator()
        
        required_param = ToolParameter(
            name="required_param",
            type="string",
            required=True,
            description="Required parameter"
        )
        
        # Missing required parameter
        valid, error = validator.validate_parameter(required_param, None)
        assert valid == False
        assert "is required" in error
        
        # Empty string for required parameter
        valid, error = validator.validate_parameter(required_param, "")
        assert valid == False
        assert "is required" in error
        
        # Valid required parameter
        valid, error = validator.validate_parameter(required_param, "value")
        assert valid == True
    
    def test_optional_parameter_validation(self):
        """Test optional parameter validation"""
        validator = ParameterValidator()
        
        optional_param = ToolParameter(
            name="optional_param",
            type="string",
            required=False,
            description="Optional parameter"
        )
        
        # None value for optional parameter should be valid
        valid, error = validator.validate_parameter(optional_param, None)
        assert valid == True
        assert error == ""
        
        # Valid value for optional parameter
        valid, error = validator.validate_parameter(optional_param, "value")
        assert valid == True
    
    def test_custom_validation_rules(self):
        """Test custom validation rules"""
        validator = ParameterValidator()
        
        # Add custom rule
        rule = ValidationRule(
            parameter_name="test_param",
            validator=lambda x: x > 10,
            error_message="Value must be greater than 10"
        )
        
        validator.add_validation_rule("test_tool", rule)
        
        # Test rule application
        assert "test_tool" in validator.validation_rules
        assert len(validator.validation_rules["test_tool"]) == 1
    
    def test_validate_all_parameters(self):
        """Test validation of all parameters"""
        validator = ParameterValidator()
        
        # Create tool metadata
        tool_metadata = ToolMetadata(
            name="test_tool",
            description="Test tool",
            version="1.0.0",
            tool_type=ToolType.PYTHON,
            security_level=ToolSecurityLevel.SAFE,
            execution_mode=ToolExecutionMode.SYNCHRONOUS,
            parameters=[
                ToolParameter(
                    name="param1",
                    type="string",
                    required=True,
                    description="First parameter"
                ),
                ToolParameter(
                    name="param2",
                    type="integer",
                    required=False,
                    description="Second parameter"
                )
            ],
            return_type="dict"
        )
        
        # Valid parameters
        valid_params = {"param1": "value1", "param2": 42}
        is_valid, errors = validator.validate_all_parameters(tool_metadata, valid_params)
        assert is_valid == True
        assert len(errors) == 0
        
        # Invalid parameters
        invalid_params = {"param1": "", "param2": "not_an_int", "unexpected": "value"}
        is_valid, errors = validator.validate_all_parameters(tool_metadata, invalid_params)
        assert is_valid == False
        assert len(errors) > 0


class TestExecutionSandbox:
    """Test execution sandbox functionality"""
    
    def test_sandbox_initialization(self):
        """Test sandbox initialization"""
        config = Config()
        sandbox = ExecutionSandbox(config)
        assert sandbox is not None
        assert sandbox.config == config
    
    def test_execution_environment_creation(self):
        """Test execution environment creation"""
        config = Config()
        sandbox = ExecutionSandbox(config)
        
        context = ExecutionContext(
            execution_id="test_exec_001",
            tool_name="test_tool",
            parameters={"param": "value"}
        )
        
        with sandbox.create_execution_environment(context) as env:
            assert 'temp_dir' in env
            assert 'env_id' in env
            assert 'context' in env
            assert env['env_id'] == "test_exec_001"
            assert os.path.exists(env['temp_dir'])
    
    def test_resource_monitoring(self):
        """Test resource monitoring"""
        config = Config()
        sandbox = ExecutionSandbox(config)
        
        context = ExecutionContext(
            execution_id="test_exec_002",
            tool_name="test_tool",
            parameters={"param": "value"},
            timeout=1.0,
            max_memory_mb=50,
            max_cpu_percent=80.0
        )
        
        # Test with short timeout
        with sandbox.create_execution_environment(context) as env:
            time.sleep(0.1)  # Brief execution
            # Should not timeout immediately
            assert context.state != ExecutionState.TIMEOUT
    
    def test_cleanup_environment(self):
        """Test environment cleanup"""
        config = Config()
        sandbox = ExecutionSandbox(config)
        
        context = ExecutionContext(
            execution_id="test_exec_003",
            tool_name="test_tool",
            parameters={"param": "value"}
        )
        
        temp_dir = None
        with sandbox.create_execution_environment(context) as env:
            temp_dir = env['temp_dir']
            assert os.path.exists(temp_dir)
        
        # After context exit, temp directory should be cleaned up
        # Note: cleanup is async, so we check that it was attempted
        assert context.execution_id not in sandbox.temp_dirs


class TestResultFormatter:
    """Test result formatting functionality"""
    
    def test_formatter_initialization(self):
        """Test formatter initialization"""
        formatter = ResultFormatter()
        assert formatter is not None
    
    def test_basic_result_formatting(self):
        """Test basic result formatting"""
        formatter = ResultFormatter()
        
        context = ExecutionContext(
            execution_id="test_exec_004",
            tool_name="test_tool",
            parameters={"param": "value"},
            security_level=ToolSecurityLevel.SAFE,
            execution_mode=ToolExecutionMode.SYNCHRONOUS
        )
        
        original_result = ToolExecutionResult(
            execution_id="test_exec_004",
            tool_name="test_tool",
            success=True,
            result={"message": "success"},
            execution_time=0.5,
            metadata={"original": "metadata"}
        )
        
        formatted_result = formatter.format_result(original_result, context)
        
        assert formatted_result.success == True
        assert formatted_result.result == {"message": "success"}
        assert 'execution_id' in formatted_result.metadata
        assert 'tool_name' in formatted_result.metadata
        assert formatted_result.metadata['execution_id'] == "test_exec_004"
    
    def test_binary_data_formatting(self):
        """Test formatting of binary data"""
        formatter = ResultFormatter()
        
        binary_data = b"test binary data"
        formatted_data = formatter._format_data(binary_data)
        
        assert formatted_data['type'] == 'binary'
        assert formatted_data['encoding'] == 'base64'
        assert 'data' in formatted_data
    
    def test_datetime_formatting(self):
        """Test formatting of datetime objects"""
        formatter = ResultFormatter()
        
        test_datetime = datetime.now()
        formatted_data = formatter._format_data(test_datetime)
        
        assert formatted_data['type'] == 'datetime'
        assert 'iso' in formatted_data
        assert 'timestamp' in formatted_data
    
    def test_nested_data_formatting(self):
        """Test formatting of nested data structures"""
        formatter = ResultFormatter()
        
        nested_data = {
            "string": "value",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {
                "binary": b"binary_data",
                "datetime": datetime.now()
            }
        }
        
        formatted_data = formatter._format_data(nested_data)
        
        assert formatted_data['string'] == "value"
        assert formatted_data['number'] == 42
        assert formatted_data['list'] == [1, 2, 3]
        assert formatted_data['nested']['binary']['type'] == 'binary'
        assert formatted_data['nested']['datetime']['type'] == 'datetime'


class TestExecutionStats:
    """Test execution statistics functionality"""
    
    def test_stats_initialization(self):
        """Test stats initialization"""
        stats = ExecutionStats()
        assert stats.total_executions == 0
        assert stats.successful_executions == 0
        assert stats.failed_executions == 0
        assert stats.min_execution_time == float('inf')
    
    def test_stats_update_successful(self):
        """Test stats update for successful execution"""
        stats = ExecutionStats()
        
        context = ExecutionContext(
            execution_id="test_exec_005",
            tool_name="test_tool",
            parameters={"param": "value"},
            state=ExecutionState.COMPLETED,
            started_at=datetime.now() - timedelta(seconds=1),
            completed_at=datetime.now()
        )
        
        stats.update(context)
        
        assert stats.total_executions == 1
        assert stats.successful_executions == 1
        assert stats.failed_executions == 0
        assert stats.executions_by_tool["test_tool"] == 1
        assert stats.average_execution_time > 0
    
    def test_stats_update_failed(self):
        """Test stats update for failed execution"""
        stats = ExecutionStats()
        
        context = ExecutionContext(
            execution_id="test_exec_006",
            tool_name="test_tool",
            parameters={"param": "value"},
            state=ExecutionState.FAILED,
            error="Test error",
            started_at=datetime.now() - timedelta(seconds=1),
            completed_at=datetime.now()
        )
        
        stats.update(context)
        
        assert stats.total_executions == 1
        assert stats.successful_executions == 0
        assert stats.failed_executions == 1
        assert stats.executions_by_tool["test_tool"] == 1


class TestEnhancedExecutionEngine:
    """Test enhanced execution engine"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return Config()
    
    @pytest.fixture
    def permission_manager(self):
        """Create mock permission manager"""
        mock_pm = Mock(spec=PermissionManager)
        mock_pm.check_permission = Mock(return_value=(True, None))
        return mock_pm
    
    @pytest.fixture
    def execution_engine(self, config, permission_manager):
        """Create execution engine instance"""
        return EnhancedExecutionEngine(config, permission_manager)
    
    @pytest.fixture
    def mock_tool(self):
        """Create mock tool"""
        return MockTool("test_tool")
    
    @pytest.fixture
    def tool_metadata(self):
        """Create tool metadata"""
        return ToolMetadata(
            name="test_tool",
            description="Test tool for execution",
            version="1.0.0",
            tool_type=ToolType.PYTHON,
            security_level=ToolSecurityLevel.SAFE,
            execution_mode=ToolExecutionMode.SYNCHRONOUS,
            parameters=[
                ToolParameter(
                    name="message",
                    type="string",
                    required=True,
                    description="Test message"
                )
            ],
            return_type="dict"
        )
    
    def test_engine_initialization(self, execution_engine):
        """Test engine initialization"""
        assert execution_engine is not None
        assert execution_engine.parameter_validator is not None
        assert execution_engine.execution_sandbox is not None
        assert execution_engine.result_formatter is not None
    
    def test_execution_id_generation(self, execution_engine):
        """Test execution ID generation"""
        id1 = execution_engine.generate_execution_id()
        id2 = execution_engine.generate_execution_id()
        
        assert id1 != id2
        assert id1.startswith("exec_")
        assert len(id1) > 10
    
    def test_execution_context_creation(self, execution_engine):
        """Test execution context creation"""
        context = execution_engine.create_execution_context(
            tool_name="test_tool",
            parameters={"param": "value"},
            user_id="user123",
            security_level=ToolSecurityLevel.MODERATE
        )
        
        assert context.tool_name == "test_tool"
        assert context.parameters == {"param": "value"}
        assert context.user_id == "user123"
        assert context.security_level == ToolSecurityLevel.MODERATE
        assert context.execution_id in execution_engine.execution_contexts
    
    def test_cache_functionality(self, execution_engine):
        """Test result caching"""
        cache_key = execution_engine.generate_cache_key("test_tool", {"param": "value"})
        
        # Should not find cache initially
        result = execution_engine.get_cached_result(cache_key)
        assert result is None
        
        # Cache a result
        test_result = ToolExecutionResult(
            execution_id="cache_test",
            tool_name="test_tool",
            success=True,
            result={"cached": "data"},
            execution_time=0.1
        )
        execution_engine.cache_result(cache_key, test_result)
        
        # Should find cached result
        cached_result = execution_engine.get_cached_result(cache_key)
        assert cached_result is not None
        assert cached_result.result == {"cached": "data"}
    
    def test_cache_key_generation(self, execution_engine):
        """Test cache key generation"""
        key1 = execution_engine.generate_cache_key("tool1", {"param": "value"})
        key2 = execution_engine.generate_cache_key("tool1", {"param": "value"})
        key3 = execution_engine.generate_cache_key("tool2", {"param": "value"})
        
        assert key1 == key2  # Same tool and params should generate same key
        assert key1 != key3  # Different tools should generate different keys
    
    @pytest.mark.asyncio
    async def test_successful_tool_execution(self, execution_engine, mock_tool, tool_metadata):
        """Test successful tool execution"""
        parameters = {"message": "test message"}
        
        result = await execution_engine.execute_tool_call(
            tool_name="test_tool",
            parameters=parameters,
            tool_instance=mock_tool,
            tool_metadata=tool_metadata
        )
        
        assert result.success == True
        assert "Mock execution" in str(result.result)
        assert result.execution_time >= 0
        assert mock_tool.execution_count == 1
    
    @pytest.mark.asyncio
    async def test_failed_tool_execution(self, execution_engine, tool_metadata):
        """Test failed tool execution"""
        failing_tool = MockTool("failing_tool", should_fail=True)
        parameters = {"message": "test message"}
        
        result = await execution_engine.execute_tool_call(
            tool_name="failing_tool",
            parameters=parameters,
            tool_instance=failing_tool,
            tool_metadata=tool_metadata
        )
        
        assert result.success == False
        assert "failed" in result.error
        assert failing_tool.execution_count == 1
    
    @pytest.mark.asyncio
    async def test_parameter_validation_in_execution(self, execution_engine, mock_tool, tool_metadata):
        """Test parameter validation during execution"""
        # Missing required parameter
        invalid_parameters = {}
        
        result = await execution_engine.execute_tool_call(
            tool_name="test_tool",
            parameters=invalid_parameters,
            tool_instance=mock_tool,
            tool_metadata=tool_metadata
        )
        
        assert result.success == False
        assert "Parameter validation failed" in result.error
        assert mock_tool.execution_count == 0  # Should not have executed
    
    @pytest.mark.asyncio
    async def test_permission_validation_in_execution(self, execution_engine, mock_tool, tool_metadata):
        """Test permission validation during execution"""
        # Mock permission manager to deny access
        execution_engine.permission_manager.check_permission = Mock(return_value=(False, "Access denied for testing"))
        
        parameters = {"message": "test message"}
        
        result = await execution_engine.execute_tool_call(
            tool_name="test_tool",
            parameters=parameters,
            tool_instance=mock_tool,
            tool_metadata=tool_metadata
        )
        
        assert result.success == False
        assert "Permission denied" in result.error
        assert mock_tool.execution_count == 0  # Should not have executed
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, execution_engine, tool_metadata):
        """Test timeout handling"""
        # Create tool with long delay
        slow_tool = MockTool("slow_tool", delay=2.0)
        parameters = {"message": "test message"}
        
        result = await execution_engine.execute_tool_call(
            tool_name="slow_tool",
            parameters=parameters,
            tool_instance=slow_tool,
            tool_metadata=tool_metadata,
            timeout=0.5  # Short timeout
        )
        
        assert result.success == False
        assert ("timeout" in result.error.lower() or "timed out" in result.error.lower())
    
    def test_execution_stats_tracking(self, execution_engine):
        """Test execution statistics tracking"""
        initial_stats = execution_engine.get_execution_stats()
        assert initial_stats.total_executions == 0
        
        # Stats should be updated after executions
        assert isinstance(initial_stats, ExecutionStats)
    
    def test_execution_history_tracking(self, execution_engine):
        """Test execution history tracking"""
        history = execution_engine.get_execution_history()
        assert isinstance(history, list)
        assert len(history) == 0  # Initially empty
    
    def test_active_executions_tracking(self, execution_engine):
        """Test active executions tracking"""
        active = execution_engine.get_active_executions()
        assert isinstance(active, list)
        assert len(active) == 0  # Initially empty
    
    def test_execution_cancellation(self, execution_engine):
        """Test execution cancellation"""
        # Create a context
        context = execution_engine.create_execution_context(
            tool_name="test_tool",
            parameters={"param": "value"}
        )
        
        # Cancel execution
        result = execution_engine.cancel_execution(context.execution_id)
        assert result == True
        assert context.state == ExecutionState.CANCELLED
        
        # Try to cancel non-existent execution
        result = execution_engine.cancel_execution("non_existent_id")
        assert result == False
    
    def test_engine_cleanup(self, execution_engine):
        """Test engine cleanup"""
        # Create some state
        context = execution_engine.create_execution_context(
            tool_name="test_tool",
            parameters={"param": "value"}
        )
        
        # Cleanup should not raise exceptions
        execution_engine.cleanup()
        
        # Verify cleanup effects
        assert len(execution_engine.result_cache) == 0
        assert len(execution_engine.execution_contexts) == 0


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios"""
    
    @pytest.fixture
    def full_setup(self):
        """Full setup for integration tests"""
        config = Config()
        permission_manager = Mock(spec=PermissionManager)
        permission_manager.check_permission = Mock(return_value=(True, None))
        
        engine = EnhancedExecutionEngine(config, permission_manager)
        
        tool = MockTool("integration_tool")
        tool_metadata = ToolMetadata(
            name="integration_tool",
            description="Integration test tool",
            version="1.0.0",
            tool_type=ToolType.PYTHON,
            security_level=ToolSecurityLevel.SAFE,
            execution_mode=ToolExecutionMode.SYNCHRONOUS,
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    required=True,
                    description="Action to perform"
                ),
                ToolParameter(
                    name="value",
                    type="integer",
                    required=False,
                    description="Optional value"
                )
            ],
            return_type="dict"
        )
        
        # Add cacheable attribute for caching test
        tool_metadata.cacheable = True
        
        return engine, tool, tool_metadata
    
    @pytest.mark.asyncio
    async def test_complete_execution_flow(self, full_setup):
        """Test complete execution flow"""
        engine, tool, tool_metadata = full_setup
        
        parameters = {"action": "test_action", "value": 42}
        
        # Execute tool
        result = await engine.execute_tool_call(
            tool_name="integration_tool",
            parameters=parameters,
            tool_instance=tool,
            tool_metadata=tool_metadata,
            user_id="test_user",
            session_id="test_session"
        )
        
        # Verify success
        assert result.success == True
        assert "Mock execution" in str(result.result)
        
        # Verify stats were updated
        stats = engine.get_execution_stats()
        assert stats.total_executions == 1
        assert stats.successful_executions == 1
        assert "integration_tool" in stats.executions_by_tool
        
        # Verify history was updated
        history = engine.get_execution_history()
        assert len(history) == 1
        assert history[0].tool_name == "integration_tool"
        assert history[0].state == ExecutionState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_caching_integration(self, full_setup):
        """Test caching integration"""
        engine, tool, tool_metadata = full_setup
        
        parameters = {"action": "cache_test", "value": 100}
        
        # First execution
        result1 = await engine.execute_tool_call(
            tool_name="integration_tool",
            parameters=parameters,
            tool_instance=tool,
            tool_metadata=tool_metadata
        )
        
        # Second execution should use cache
        result2 = await engine.execute_tool_call(
            tool_name="integration_tool",
            parameters=parameters,
            tool_instance=tool,
            tool_metadata=tool_metadata
        )
        
        # Both should be successful
        assert result1.success == True
        assert result2.success == True
        
        # Tool should have been executed only once (second was cached)
        assert tool.execution_count == 1
        
        # Results should be similar
        assert result1.result == result2.result
    
    @pytest.mark.asyncio
    async def test_security_integration(self, full_setup):
        """Test security integration"""
        engine, tool, tool_metadata = full_setup
        
        # Test with elevated security level
        elevated_metadata = ToolMetadata(
            name="elevated_tool",
            description="Elevated security tool",
            version="1.0.0",
            tool_type=ToolType.PYTHON,
            security_level=ToolSecurityLevel.ELEVATED,
            execution_mode=ToolExecutionMode.SYNCHRONOUS,
            parameters=[
                ToolParameter(
                    name="sensitive_data",
                    type="string",
                    required=True,
                    description="Sensitive data parameter"
                )
            ],
            return_type="dict"
        )
        
        # Should work with permission
        result = await engine.execute_tool_call(
            tool_name="elevated_tool",
            parameters={"sensitive_data": "secret"},
            tool_instance=tool,
            tool_metadata=elevated_metadata,
            security_level=ToolSecurityLevel.ELEVATED
        )
        
        assert result.success == True
        
        # Test permission denial
        engine.permission_manager.check_permission = Mock(return_value=(False, "Access denied for testing"))
        
        result = await engine.execute_tool_call(
            tool_name="elevated_tool",
            parameters={"sensitive_data": "secret"},
            tool_instance=tool,
            tool_metadata=elevated_metadata,
            security_level=ToolSecurityLevel.ELEVATED
        )
        
        assert result.success == False
        assert "Permission denied" in result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 