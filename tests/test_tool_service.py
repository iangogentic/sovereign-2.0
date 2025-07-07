"""
Test Tool Service - Lazy-loaded tool discovery and execution system

Tests both lightweight construction and full integration scenarios.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import asdict

# Import the service directly for testing
from src.sovereign.services.tool_service import ToolService, ToolResult


class TestToolServiceConstruction:
    """Test lightweight service construction and configuration"""
    
    @pytest.mark.not_heavy
    def test_service_creation(self):
        """Test that ToolService can be created without heavy imports"""
        service = ToolService()
        
        # Should be created but not initialized
        assert not service.is_registry_loaded()
        assert service._registry is None
        assert service._execution_engine is None
        assert service._permission_manager is None
        assert service._framework is None
        assert not service._initialized
        
        # Should have statistics tracking ready
        stats = service.get_stats()
        assert stats['total_executions'] == 0
        assert stats['successful_executions'] == 0
        assert stats['failed_executions'] == 0
        assert stats['loaded_tools'] == 0

    @pytest.mark.not_heavy
    def test_health_check_uninitialized(self):
        """Test health check on uninitialized service"""
        service = ToolService()
        health = service.get_health_check()
        
        assert health['service_name'] == 'ToolService'
        assert health['status'] == 'uninitialized'
        assert not health['registry_loaded']
        assert health['total_executions'] == 0
        assert not health['components']['registry']
        assert not health['components']['execution_engine']
        assert not health['components']['permission_manager']
        assert not health['components']['framework']

    @pytest.mark.not_heavy
    def test_tool_result_creation(self):
        """Test ToolResult data structure"""
        result = ToolResult(
            success=True,
            result={"data": "test"},
            execution_time=0.123,
            tool_name="test_tool",
            execution_id="exec_123"
        )
        
        assert result.success
        assert result.result == {"data": "test"}
        assert result.error is None
        assert result.execution_time == 0.123
        assert result.tool_name == "test_tool"
        assert result.execution_id == "exec_123"
        
        # Test dictionary conversion
        result_dict = result.to_dict()
        expected = {
            'success': True,
            'result': {"data": "test"},
            'error': None,
            'execution_time': 0.123,
            'tool_name': "test_tool",
            'execution_id': "exec_123"
        }
        assert result_dict == expected

    @pytest.mark.not_heavy
    def test_tool_result_error_case(self):
        """Test ToolResult for error scenarios"""
        result = ToolResult(
            success=False,
            error="Tool not found",
            execution_time=0.05,
            tool_name="nonexistent_tool"
        )
        
        assert not result.success
        assert result.result is None
        assert result.error == "Tool not found"
        assert result.execution_time == 0.05
        assert result.tool_name == "nonexistent_tool"
        assert result.execution_id == ""


class TestToolServiceIntegration:
    """Test full tool service integration with framework loading"""
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_lazy_loading_on_first_execute(self):
        """Test that components are lazy-loaded on first tool execution"""
        service = ToolService()
        
        # Initially not loaded
        assert not service.is_registry_loaded()
        
        # Mock the framework components to avoid actual tool loading
        with patch('src.sovereign.tool_integration_framework.ToolIntegrationFramework') as mock_framework_class:
            # Setup mock framework
            mock_framework = Mock()
            mock_framework.initialize = AsyncMock(return_value=True)
            mock_framework.registry = Mock()
            mock_framework.registry.tools = {"test_tool": Mock()}
            mock_framework.execution_engine = Mock()
            mock_framework.permission_manager = Mock()
            mock_framework.get_framework_stats.return_value = {"framework_loaded": True}
            mock_framework.execute_tool = AsyncMock()
            
            # Setup mock execution result
            mock_result = Mock()
            mock_result.success = True
            mock_result.result = "tool executed"
            mock_result.execution_id = "exec_123"
            mock_framework.execute_tool.return_value = mock_result
            
            mock_framework_class.return_value = mock_framework
            
            # Execute a tool - this should trigger lazy loading
            result = await service.execute_tool("test_tool", {"param": "value"})
            
            # Verify lazy loading occurred
            assert service.is_registry_loaded()
            assert service._framework is not None
            assert service._registry is not None
            assert service._execution_engine is not None
            
            # Verify tool execution
            assert result.success
            assert result.result == "tool executed"
            assert result.tool_name == "test_tool"
            assert result.execution_id == "exec_123"
            
            # Verify framework was called correctly
            mock_framework.initialize.assert_called_once()
            mock_framework.execute_tool.assert_called_once_with(
                "test_tool", {"param": "value"}, "system"
            )
            
            # Check statistics were updated
            stats = service.get_stats()
            assert stats['total_executions'] == 1
            assert stats['successful_executions'] == 1
            assert stats['failed_executions'] == 0

    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_framework_initialization_failure(self):
        """Test handling of framework initialization failure"""
        service = ToolService()
        
        with patch('src.sovereign.tool_integration_framework.ToolIntegrationFramework') as mock_framework_class:
            mock_framework = Mock()
            mock_framework.initialize = AsyncMock(return_value=False)  # Initialization fails
            mock_framework.get_framework_stats.return_value = {"framework_loaded": False}
            mock_framework_class.return_value = mock_framework
            
            # Attempt to execute tool should fail gracefully
            result = await service.execute_tool("test_tool", {})
            
            assert not result.success
            assert "Tool framework initialization failed" in result.error
            assert result.tool_name == "test_tool"
            
            # Service should not be marked as initialized
            assert not service.is_registry_loaded()
            
            # Statistics should reflect the failed execution attempt
            stats = service.get_stats()
            assert stats['total_executions'] == 1
            assert stats['successful_executions'] == 0
            assert stats['failed_executions'] == 1

    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_concurrent_loading_protection(self):
        """Test that concurrent access doesn't cause double loading"""
        service = ToolService()
        
        # Track initialization calls
        init_call_count = 0
        
        def count_init():
            nonlocal init_call_count
            init_call_count += 1
            return True
        
        with patch('src.sovereign.tool_integration_framework.ToolIntegrationFramework') as mock_framework_class:
            mock_framework = Mock()
            mock_framework.initialize = AsyncMock(side_effect=lambda: count_init())
            mock_framework.registry = Mock()
            mock_framework.registry.tools = {}
            mock_framework.execution_engine = Mock()
            mock_framework.permission_manager = Mock()
            mock_framework.get_framework_stats.return_value = {"framework_loaded": True}
            mock_framework.execute_tool = AsyncMock()
            
            mock_result = Mock()
            mock_result.success = True
            mock_result.result = "executed"
            mock_result.execution_id = "exec_123"
            mock_framework.execute_tool.return_value = mock_result
            
            mock_framework_class.return_value = mock_framework
            
            # Start multiple concurrent tool executions
            tasks = [
                service.execute_tool("tool1", {}),
                service.execute_tool("tool2", {}),
                service.execute_tool("tool3", {})
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert all(result.success for result in results)
            
            # Framework should only be initialized once despite concurrent access
            assert init_call_count == 1
            assert service.is_registry_loaded()

    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_get_available_tools_with_loading(self):
        """Test getting available tools triggers loading if needed"""
        service = ToolService()
        
        with patch('src.sovereign.tool_integration_framework.ToolIntegrationFramework') as mock_framework_class:
            mock_framework = Mock()
            mock_framework.initialize = AsyncMock(return_value=True)
            mock_framework.registry = Mock()
            mock_framework.registry.tools = {"tool1": Mock(), "tool2": Mock()}
            mock_framework.execution_engine = Mock()
            mock_framework.permission_manager = Mock()
            mock_framework.get_framework_stats.return_value = {"framework_loaded": True}
            mock_framework.get_available_tools.return_value = [
                {"name": "tool1", "description": "Test tool 1"},
                {"name": "tool2", "description": "Test tool 2"}
            ]
            
            mock_framework_class.return_value = mock_framework
            
            # Get available tools should trigger loading
            tools = await service.get_available_tools("test_requester")
            
            assert len(tools) == 2
            assert tools[0]["name"] == "tool1"
            assert tools[1]["name"] == "tool2"
            
            # Should be loaded now
            assert service.is_registry_loaded()
            
            # Framework method should have been called with requester
            mock_framework.get_available_tools.assert_called_once_with("test_requester")

    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_service_cleanup(self):
        """Test proper cleanup of service resources"""
        service = ToolService()
        
        with patch('src.sovereign.tool_integration_framework.ToolIntegrationFramework') as mock_framework_class:
            mock_framework = Mock()
            mock_framework.initialize = AsyncMock(return_value=True)
            mock_framework.registry = Mock()
            mock_framework.registry.tools = {"test_tool": Mock()}
            mock_framework.execution_engine = Mock()
            mock_framework.permission_manager = Mock()
            mock_framework.get_framework_stats.return_value = {"framework_loaded": True}
            mock_framework.cleanup = AsyncMock()
            mock_framework.execute_tool = AsyncMock()
            
            mock_framework_class.return_value = mock_framework
            
            # Initialize by executing a tool
            mock_result = Mock()
            mock_result.success = True
            mock_result.result = "executed"
            mock_result.execution_id = "exec_123"
            mock_framework.execute_tool.return_value = mock_result
            
            await service.execute_tool("test_tool", {})
            
            # Verify service is initialized
            assert service.is_registry_loaded()
            assert service._framework is not None
            
            # Cleanup the service
            await service.close()
            
            # Verify cleanup was called on framework
            mock_framework.cleanup.assert_called_once()
            
            # Verify service state is reset
            assert not service.is_registry_loaded()
            assert service._framework is None
            assert service._registry is None
            assert service._execution_engine is None

    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_tool_execution_error_handling(self):
        """Test error handling during tool execution"""
        service = ToolService()
        
        with patch('src.sovereign.tool_integration_framework.ToolIntegrationFramework') as mock_framework_class:
            mock_framework = Mock()
            mock_framework.initialize = AsyncMock(return_value=True)
            mock_framework.registry = Mock()
            mock_framework.registry.tools = {"failing_tool": Mock()}
            mock_framework.execution_engine = Mock()
            mock_framework.permission_manager = Mock()
            mock_framework.get_framework_stats.return_value = {"framework_loaded": True}
            
            # Mock framework to raise an exception during execution
            mock_framework.execute_tool = AsyncMock(side_effect=RuntimeError("Tool execution failed"))
            
            mock_framework_class.return_value = mock_framework
            
            # Execute tool should handle the exception gracefully
            result = await service.execute_tool("failing_tool", {})
            
            assert not result.success
            assert result.error == "Tool execution failed"
            assert result.tool_name == "failing_tool"
            assert result.execution_time > 0
            
            # Statistics should reflect the failure
            stats = service.get_stats()
            assert stats['total_executions'] == 1
            assert stats['successful_executions'] == 0
            assert stats['failed_executions'] == 1 