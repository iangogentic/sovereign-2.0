"""
Tests for Tool Integration Framework

This module contains comprehensive tests for the tool integration framework,
including unit tests for all core components and integration tests for
end-to-end functionality.
"""

import pytest
import asyncio
import tempfile
import time
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from src.sovereign.tool_integration_framework import (
    ToolIntegrationFramework, BaseTool, ToolRegistry, PermissionManager,
    ExecutionEngine, ToolMetadata, ToolParameter, ToolExecutionRequest,
    ToolExecutionResult, ToolType, ToolSecurityLevel, ToolExecutionMode,
    ToolNotFoundError, ToolPermissionError, ToolExecutionError
)
from src.sovereign.tools.example_tools import (
    EchoTool, CalculatorTool, FileInfoTool, SystemInfoTool, MockApiTool,
    create_example_tools, register_example_tools
)
from src.sovereign.config import Config


class TestBaseTool:
    """Test the BaseTool abstract class and its implementations"""
    
    def test_echo_tool_creation(self):
        """Test creating an echo tool"""
        tool = EchoTool()
        assert tool.metadata.name == "echo"
        assert tool.metadata.tool_type == ToolType.PYTHON
        assert tool.metadata.security_level == ToolSecurityLevel.SAFE
        assert len(tool.metadata.parameters) == 2
        assert tool.execution_count == 0
    
    @pytest.mark.asyncio
    async def test_echo_tool_execution(self):
        """Test executing the echo tool"""
        tool = EchoTool()
        
        # Test basic echo
        result = await tool.execute({"message": "Hello World"})
        assert result == "Hello World"
        
        # Test uppercase echo
        result = await tool.execute({"message": "Hello World", "uppercase": True})
        assert result == "HELLO WORLD"
        
        # Check statistics
        assert tool.execution_count == 2
        assert tool.last_execution is not None
    
    @pytest.mark.asyncio
    async def test_echo_tool_validation(self):
        """Test parameter validation for echo tool"""
        tool = EchoTool()
        
        # Valid parameters
        valid, error = await tool.validate_parameters({"message": "test"})
        assert valid is True
        assert error is None
        
        # Missing required parameter
        valid, error = await tool.validate_parameters({})
        assert valid is False
        assert "missing" in error.lower()
        
        # Invalid parameter type
        valid, error = await tool.validate_parameters({"message": 123})
        assert valid is False
        assert "string" in error.lower()
    
    @pytest.mark.asyncio
    async def test_calculator_tool_operations(self):
        """Test calculator tool operations"""
        tool = CalculatorTool()
        
        # Test addition
        result = await tool.execute({"operation": "add", "a": 5, "b": 3})
        assert result == 8
        
        # Test multiplication
        result = await tool.execute({"operation": "multiply", "a": 4, "b": 7})
        assert result == 28
        
        # Test division
        result = await tool.execute({"operation": "divide", "a": 10, "b": 2})
        assert result == 5.0
        
        # Test division by zero
        with pytest.raises(ToolExecutionError):
            await tool.execute({"operation": "divide", "a": 10, "b": 0})
    
    @pytest.mark.asyncio
    async def test_file_info_tool(self):
        """Test file info tool"""
        tool = FileInfoTool()
        
        # Test with existing file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            result = await tool.execute({"file_path": temp_path})
            assert result["exists"] is True
            assert "size" in result
            assert "modified" in result
            assert result["is_directory"] is False
        finally:
            Path(temp_path).unlink(missing_ok=True)
        
        # Test with non-existent file
        result = await tool.execute({"file_path": "/nonexistent/file.txt"})
        assert result["exists"] is False
    
    @pytest.mark.asyncio
    async def test_system_info_tool(self):
        """Test system info tool"""
        tool = SystemInfoTool()
        
        # Test basic info
        result = await tool.execute({"info_type": "basic"})
        assert "platform" in result
        assert "python_version" in result
        assert "timestamp" in result
        
        # Test detailed info
        result = await tool.execute({"info_type": "detailed"})
        assert "platform_release" in result
        assert "processor" in result
    
    @pytest.mark.asyncio
    async def test_mock_api_tool(self):
        """Test mock API tool"""
        tool = MockApiTool()
        
        # Test GET request
        result = await tool.execute({"endpoint": "/users", "method": "GET"})
        assert result["status"] == 200
        assert "users" in result["data"]
        
        # Test health endpoint
        result = await tool.execute({"endpoint": "/health"})
        assert result["status"] == 200
        assert result["data"]["status"] == "healthy"
        
        # Test non-existent endpoint
        result = await tool.execute({"endpoint": "/nonexistent"})
        assert result["status"] == 404


class TestToolRegistry:
    """Test the ToolRegistry class"""
    
    def test_registry_creation(self):
        """Test creating a tool registry"""
        registry = ToolRegistry()
        assert len(registry.tools) == 0
        assert len(registry.tool_groups) == 0
    
    def test_tool_registration(self):
        """Test registering tools"""
        registry = ToolRegistry()
        tool = EchoTool()
        
        # Register tool
        success = registry.register_tool(tool)
        assert success is True
        assert len(registry.tools) == 1
        assert "echo" in registry.tools
        
        # Check tool groups
        assert "utility" in registry.tool_groups
        assert "echo" in registry.tool_groups["utility"]
    
    def test_tool_retrieval(self):
        """Test retrieving tools"""
        registry = ToolRegistry()
        echo_tool = EchoTool()
        calc_tool = CalculatorTool()
        
        registry.register_tool(echo_tool)
        registry.register_tool(calc_tool)
        
        # Get specific tool
        retrieved = registry.get_tool("echo")
        assert retrieved is echo_tool
        
        # Get non-existent tool
        retrieved = registry.get_tool("nonexistent")
        assert retrieved is None
    
    def test_tool_listing_and_filtering(self):
        """Test listing and filtering tools"""
        registry = ToolRegistry()
        
        # Register multiple tools
        tools = create_example_tools()
        for tool in tools:
            registry.register_tool(tool)
        
        # List all tools
        all_tools = registry.list_tools()
        assert len(all_tools) == len(tools)
        
        # Filter by type
        python_tools = registry.list_tools(tool_type=ToolType.PYTHON)
        assert len(python_tools) >= 2  # Echo and Calculator
        
        # Filter by security level
        safe_tools = registry.list_tools(security_level=ToolSecurityLevel.SAFE)
        assert len(safe_tools) >= 2
        
        # Filter by tags
        utility_tools = registry.list_tools(tags=["utility"])
        assert len(utility_tools) >= 1
    
    def test_tool_search(self):
        """Test searching tools"""
        registry = ToolRegistry()
        
        # Register tools
        tools = create_example_tools()
        for tool in tools:
            registry.register_tool(tool)
        
        # Search by name
        results = registry.search_tools("echo")
        assert len(results) == 1
        assert results[0].metadata.name == "echo"
        
        # Search by description
        results = registry.search_tools("mathematical")
        assert len(results) >= 1
        assert any(tool.metadata.name == "calculator" for tool in results)
    
    def test_tool_unregistration(self):
        """Test unregistering tools"""
        registry = ToolRegistry()
        tool = EchoTool()
        
        # Register and then unregister
        registry.register_tool(tool)
        assert len(registry.tools) == 1
        
        success = registry.unregister_tool("echo")
        assert success is True
        assert len(registry.tools) == 0
        
        # Try to unregister non-existent tool
        success = registry.unregister_tool("nonexistent")
        assert success is False
    
    def test_registry_stats(self):
        """Test registry statistics"""
        registry = ToolRegistry()
        
        # Register tools
        tools = create_example_tools()
        for tool in tools:
            registry.register_tool(tool)
        
        stats = registry.get_registry_stats()
        assert stats["total_tools"] == len(tools)
        assert "tool_types" in stats
        assert "security_levels" in stats
        assert stats["tool_types"]["python"] >= 2


class TestPermissionManager:
    """Test the PermissionManager class"""
    
    def test_permission_manager_creation(self):
        """Test creating a permission manager"""
        config = Config()
        pm = PermissionManager(config)
        assert pm.config is config
        assert isinstance(pm.global_policies, dict)
    
    def test_permission_checking(self):
        """Test permission checking"""
        config = Config()
        pm = PermissionManager(config)
        
        # Create tool metadata
        safe_tool = ToolMetadata(
            name="safe_tool",
            description="A safe tool",
            version="1.0.0",
            tool_type=ToolType.PYTHON,
            security_level=ToolSecurityLevel.SAFE,
            execution_mode=ToolExecutionMode.SYNCHRONOUS,
            parameters=[],
            return_type="string"
        )
        
        # Check permission for safe tool
        has_permission, error = pm.check_permission("safe_tool", "test_user", safe_tool)
        assert has_permission is True
        assert error is None
        
        # Create elevated tool
        elevated_tool = ToolMetadata(
            name="elevated_tool",
            description="An elevated tool",
            version="1.0.0",
            tool_type=ToolType.SYSTEM,
            security_level=ToolSecurityLevel.ELEVATED,
            execution_mode=ToolExecutionMode.SYNCHRONOUS,
            parameters=[],
            return_type="string"
        )
        
        # Check permission for elevated tool (should be denied)
        has_permission, error = pm.check_permission("elevated_tool", "test_user", elevated_tool)
        assert has_permission is False
        assert "consent" in error.lower()
    
    def test_permission_granting_and_revoking(self):
        """Test granting and revoking permissions"""
        config = Config()
        pm = PermissionManager(config)
        
        # Grant permission
        success = pm.grant_permission("test_tool", "test_user")
        assert success is True
        
        # Revoke permission
        success = pm.revoke_permission("test_tool", "test_user")
        assert success is True
    
    def test_global_policies(self):
        """Test global policy management"""
        config = Config()
        pm = PermissionManager(config)
        
        # Set global policy
        success = pm.set_global_policy("allow_system_tools", True)
        assert success is True
        assert pm.global_policies["allow_system_tools"] is True
        
        # Try to set invalid policy
        success = pm.set_global_policy("invalid_policy", True)
        assert success is False
    
    def test_permissions_summary(self):
        """Test getting permissions summary"""
        config = Config()
        pm = PermissionManager(config)
        
        summary = pm.get_permissions_summary()
        assert "global_policies" in summary
        assert "tool_permissions" in summary
        assert "user_permissions" in summary


class TestExecutionEngine:
    """Test the ExecutionEngine class"""
    
    def test_execution_engine_creation(self):
        """Test creating an execution engine"""
        config = Config()
        engine = ExecutionEngine(config)
        assert engine.config is config
        assert len(engine.active_executions) == 0
    
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test executing tools through the engine"""
        config = Config()
        engine = ExecutionEngine(config)
        tool = EchoTool()
        
        # Create execution request
        request = ToolExecutionRequest(
            tool_name="echo",
            parameters={"message": "test"},
            requester="test_user"
        )
        
        # Execute tool
        result = await engine.execute_tool(tool, request)
        
        assert result.success is True
        assert result.result == "test"
        assert result.tool_name == "echo"
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_tool_execution_timeout(self):
        """Test tool execution timeout"""
        config = Config()
        engine = ExecutionEngine(config)
        
        # Create a mock tool that takes too long
        class SlowTool(BaseTool):
            def __init__(self):
                metadata = ToolMetadata(
                    name="slow_tool",
                    description="A slow tool",
                    version="1.0.0",
                    tool_type=ToolType.PYTHON,
                    security_level=ToolSecurityLevel.SAFE,
                    execution_mode=ToolExecutionMode.SYNCHRONOUS,
                    parameters=[],
                    return_type="string",
                    max_execution_time=1  # 1 second timeout
                )
                super().__init__(metadata)
            
            async def execute(self, parameters: Dict[str, Any]) -> Any:
                await asyncio.sleep(2)  # Sleep for 2 seconds
                return "slow result"
            
            async def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
                return True, None
        
        slow_tool = SlowTool()
        request = ToolExecutionRequest(
            tool_name="slow_tool",
            parameters={},
            requester="test_user"
        )
        
        # Execute tool (should timeout)
        result = await engine.execute_tool(slow_tool, request)
        
        assert result.success is False
        assert "timeout" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_tool_validation_failure(self):
        """Test tool execution with validation failure"""
        config = Config()
        engine = ExecutionEngine(config)
        tool = EchoTool()
        
        # Create request with invalid parameters
        request = ToolExecutionRequest(
            tool_name="echo",
            parameters={},  # Missing required 'message' parameter
            requester="test_user"
        )
        
        # Execute tool (should fail validation)
        result = await engine.execute_tool(tool, request)
        
        assert result.success is False
        assert "validation" in result.error.lower()
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        config = Config()
        engine = ExecutionEngine(config)
        
        # Check rate limit for new tool
        allowed = engine._check_rate_limit("test_tool")
        assert allowed is True
        
        # Simulate many rapid requests
        for _ in range(65):  # Exceed the 60 request limit
            engine._check_rate_limit("test_tool")
        
        # Should now be rate limited
        allowed = engine._check_rate_limit("test_tool")
        assert allowed is False
    
    def test_execution_stats(self):
        """Test execution statistics"""
        config = Config()
        engine = ExecutionEngine(config)
        
        # Initially no executions
        stats = engine.get_execution_stats()
        assert stats["total_executions"] == 0
        assert stats["successful_executions"] == 0
        
        # Add some mock execution results
        engine.execution_history.append(ToolExecutionResult(
            execution_id="test1",
            tool_name="test_tool",
            success=True,
            execution_time=1.0
        ))
        
        engine.execution_history.append(ToolExecutionResult(
            execution_id="test2",
            tool_name="test_tool",
            success=False,
            execution_time=0.5
        ))
        
        stats = engine.get_execution_stats()
        assert stats["total_executions"] == 2
        assert stats["successful_executions"] == 1
        assert stats["avg_execution_time"] == 0.75


class TestToolIntegrationFramework:
    """Test the main ToolIntegrationFramework class"""
    
    def test_framework_creation(self):
        """Test creating the framework"""
        config = Config()
        framework = ToolIntegrationFramework(config)
        
        assert framework.config is config
        assert framework.registry is not None
        assert framework.permission_manager is not None
        assert framework.execution_engine is not None
        assert framework.initialized is False
    
    @pytest.mark.asyncio
    async def test_framework_initialization(self):
        """Test framework initialization"""
        config = Config()
        framework = ToolIntegrationFramework(config)
        
        # Initialize framework
        success = await framework.initialize()
        assert success is True
        assert framework.initialized is True
    
    @pytest.mark.asyncio
    async def test_tool_execution_through_framework(self):
        """Test executing tools through the framework"""
        config = Config()
        framework = ToolIntegrationFramework(config)
        
        # Initialize framework
        await framework.initialize()
        
        # Register tools
        registered_count = await register_example_tools(framework)
        assert registered_count > 0
        
        # Execute echo tool
        result = await framework.execute_tool(
            "echo",
            {"message": "Hello Framework"},
            "test_user"
        )
        
        assert result.success is True
        assert result.result == "Hello Framework"
        
        # Execute calculator tool
        result = await framework.execute_tool(
            "calculator",
            {"operation": "add", "a": 10, "b": 5},
            "test_user"
        )
        
        assert result.success is True
        assert result.result == 15
    
    @pytest.mark.asyncio
    async def test_tool_not_found_error(self):
        """Test error when tool is not found"""
        config = Config()
        framework = ToolIntegrationFramework(config)
        
        await framework.initialize()
        
        # Try to execute non-existent tool
        with pytest.raises(ToolNotFoundError):
            await framework.execute_tool(
                "nonexistent_tool",
                {},
                "test_user"
            )
    
    @pytest.mark.asyncio
    async def test_permission_denied_error(self):
        """Test permission denied error"""
        config = Config()
        framework = ToolIntegrationFramework(config)
        
        await framework.initialize()
        
        # Create and register a restricted tool
        class RestrictedTool(BaseTool):
            def __init__(self):
                metadata = ToolMetadata(
                    name="restricted_tool",
                    description="A restricted tool",
                    version="1.0.0",
                    tool_type=ToolType.SYSTEM,
                    security_level=ToolSecurityLevel.RESTRICTED,
                    execution_mode=ToolExecutionMode.SYNCHRONOUS,
                    parameters=[],
                    return_type="string"
                )
                super().__init__(metadata)
            
            async def execute(self, parameters: Dict[str, Any]) -> Any:
                return "restricted result"
            
            async def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
                return True, None
        
        restricted_tool = RestrictedTool()
        framework.registry.register_tool(restricted_tool)
        
        # Try to execute restricted tool
        with pytest.raises(ToolPermissionError):
            await framework.execute_tool(
                "restricted_tool",
                {},
                "test_user"
            )
    
    def test_get_available_tools(self):
        """Test getting available tools"""
        config = Config()
        framework = ToolIntegrationFramework(config)
        
        # Register tools
        tools = create_example_tools()
        for tool in tools:
            framework.registry.register_tool(tool)
        
        # Get available tools
        available = framework.get_available_tools("test_user")
        assert len(available) > 0
        
        # Check structure
        for tool_info in available:
            assert "name" in tool_info
            assert "description" in tool_info
            assert "type" in tool_info
            assert "parameters" in tool_info
    
    def test_framework_stats(self):
        """Test getting framework statistics"""
        config = Config()
        framework = ToolIntegrationFramework(config)
        
        # Register tools
        tools = create_example_tools()
        for tool in tools:
            framework.registry.register_tool(tool)
        
        stats = framework.get_framework_stats()
        assert "framework" in stats
        assert "registry" in stats
        assert "execution" in stats
        assert "permissions" in stats
    
    @pytest.mark.asyncio
    async def test_framework_cleanup(self):
        """Test framework cleanup"""
        config = Config()
        framework = ToolIntegrationFramework(config)
        
        await framework.initialize()
        assert framework.initialized is True
        
        await framework.cleanup()
        assert framework.initialized is False


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete tool integration workflow"""
        config = Config()
        framework = ToolIntegrationFramework(config)
        
        # Initialize framework
        await framework.initialize()
        
        # Register example tools
        registered_count = await register_example_tools(framework)
        assert registered_count == 5
        
        # Test multiple tool executions
        results = []
        
        # Echo tool
        result = await framework.execute_tool("echo", {"message": "Hello"}, "user1")
        results.append(result)
        
        # Calculator tool
        result = await framework.execute_tool("calculator", {"operation": "multiply", "a": 6, "b": 7}, "user1")
        results.append(result)
        
        # Mock API tool
        result = await framework.execute_tool("mock_api", {"endpoint": "/health"}, "user1")
        results.append(result)
        
        # Verify all executions succeeded
        assert all(result.success for result in results)
        assert len(results) == 3
        
        # Check framework statistics
        stats = framework.get_framework_stats()
        assert stats["framework"]["total_requests"] == 3
        assert stats["framework"]["successful_requests"] == 3
    
    @pytest.mark.asyncio
    async def test_mixed_success_failure_scenario(self):
        """Test scenario with both successful and failed executions"""
        config = Config()
        framework = ToolIntegrationFramework(config)
        
        await framework.initialize()
        await register_example_tools(framework)
        
        # Successful execution
        result1 = await framework.execute_tool("echo", {"message": "success"}, "user1")
        assert result1.success is True
        
        # Failed execution (division by zero)
        result2 = await framework.execute_tool("calculator", {"operation": "divide", "a": 10, "b": 0}, "user1")
        assert result2.success is False
        assert "zero" in result2.error.lower()
        
        # Another successful execution
        result3 = await framework.execute_tool("mock_api", {"endpoint": "/users"}, "user1")
        assert result3.success is True
        
        # Check statistics
        stats = framework.get_framework_stats()
        assert stats["framework"]["total_requests"] == 3
        assert stats["framework"]["successful_requests"] == 2
        assert stats["framework"]["failed_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_executions(self):
        """Test concurrent tool executions"""
        config = Config()
        framework = ToolIntegrationFramework(config)
        
        await framework.initialize()
        await register_example_tools(framework)
        
        # Execute multiple tools concurrently
        tasks = [
            framework.execute_tool("echo", {"message": f"Message {i}"}, f"user{i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        assert len(results) == 5
        assert all(isinstance(result, ToolExecutionResult) for result in results)
        assert all(result.success for result in results)
    
    @pytest.mark.asyncio
    async def test_tool_usage_statistics(self):
        """Test tool usage statistics tracking"""
        config = Config()
        framework = ToolIntegrationFramework(config)
        
        await framework.initialize()
        await register_example_tools(framework)
        
        # Use echo tool multiple times
        for i in range(3):
            await framework.execute_tool("echo", {"message": f"Test {i}"}, "user1")
        
        # Check tool statistics
        echo_tool = framework.registry.get_tool("echo")
        stats = echo_tool.get_usage_stats()
        
        assert stats["execution_count"] == 3
        assert stats["total_execution_time"] > 0
        assert stats["avg_execution_time"] > 0
        assert stats["last_execution"] is not None 