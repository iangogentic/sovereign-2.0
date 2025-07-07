"""
Core Tools Test Suite - Comprehensive testing for the four core tools

This test suite validates the functionality of all core tools:
- InternetSearchTool
- EnhancedSystemInfoTool  
- FileAccessTool
- AdvancedCalculationTool
"""

import pytest
import asyncio
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.sovereign.tools.core_tools import (
    InternetSearchTool, EnhancedSystemInfoTool, 
    FileAccessTool, AdvancedCalculationTool,
    create_core_tools, register_core_tools
)
from src.sovereign.tool_integration_framework import (
    ToolIntegrationFramework, ToolExecutionError
)
from src.sovereign.config import Config


class TestInternetSearchTool:
    """Test suite for InternetSearchTool"""
    
    @pytest.fixture
    def search_tool(self):
        return InternetSearchTool()
    
    @pytest.mark.asyncio
    async def test_tool_initialization(self, search_tool):
        """Test that search tool initializes correctly"""
        assert search_tool.metadata.name == "internet_search"
        assert search_tool.metadata.security_level.value == "moderate"
        assert search_tool.metadata.requires_consent == True
        assert "search" in search_tool.metadata.tags
        assert "privacy" in search_tool.metadata.tags
    
    @pytest.mark.asyncio
    async def test_parameter_validation_valid(self, search_tool):
        """Test parameter validation with valid inputs"""
        valid_params = {
            "query": "Python programming",
            "max_results": 5,
            "region": "us-en"
        }
        is_valid, error = await search_tool.validate_parameters(valid_params)
        assert is_valid == True
        assert error is None
    
    @pytest.mark.asyncio
    async def test_parameter_validation_invalid(self, search_tool):
        """Test parameter validation with invalid inputs"""
        # Missing query
        invalid_params = {"max_results": 5}
        is_valid, error = await search_tool.validate_parameters(invalid_params)
        assert is_valid == False
        assert "query" in error
        
        # Empty query
        invalid_params = {"query": ""}
        is_valid, error = await search_tool.validate_parameters(invalid_params)
        assert is_valid == False
        
        # Query too long
        invalid_params = {"query": "x" * 501}
        is_valid, error = await search_tool.validate_parameters(invalid_params)
        assert is_valid == False
    
    @pytest.mark.asyncio
    async def test_search_execution_mocked(self, search_tool):
        """Test search execution with mocked API response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Abstract": "Python is a programming language",
            "AbstractURL": "https://python.org",
            "RelatedTopics": [
                {
                    "Text": "Python tutorial for beginners",
                    "FirstURL": "https://example.com/tutorial"
                }
            ]
        }
        
        with patch('requests.get', return_value=mock_response):
            params = {"query": "Python programming", "max_results": 3}
            result = await search_tool.execute(params)
            
            assert result["query"] == "Python programming"
            assert len(result["results"]) > 0
            assert result["total_results"] > 0
            assert "search_time" in result
            assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_fallback_search(self, search_tool):
        """Test fallback search when API returns no results"""
        fallback_results = await search_tool._fallback_search("test query", 5)
        
        assert len(fallback_results) == 1
        assert "fallback" in fallback_results[0]["type"]
        assert "test query" in fallback_results[0]["title"]


class TestEnhancedSystemInfoTool:
    """Test suite for EnhancedSystemInfoTool"""
    
    @pytest.fixture
    def system_tool(self):
        return EnhancedSystemInfoTool()
    
    @pytest.mark.asyncio
    async def test_tool_initialization(self, system_tool):
        """Test that system info tool initializes correctly"""
        assert system_tool.metadata.name == "enhanced_system_info"
        assert system_tool.metadata.security_level.value == "moderate"
        assert system_tool.metadata.requires_consent == True
        assert "system" in system_tool.metadata.tags
    
    @pytest.mark.asyncio
    async def test_overview_info(self, system_tool):
        """Test overview information collection"""
        result = await system_tool._get_overview_info()
        
        assert "platform" in result
        assert "architecture" in result
        assert "python_version" in result
        assert "cpu_count" in result
        assert "memory_total_gb" in result
        assert isinstance(result["cpu_count"], int)
        assert isinstance(result["memory_total_gb"], float)
    
    @pytest.mark.asyncio
    async def test_hardware_info(self, system_tool):
        """Test hardware information collection"""
        result = await system_tool._get_hardware_info()
        
        assert "processor" in result
        assert "memory" in result
        assert "swap" in result
        assert "total_gb" in result["memory"]
        assert "available_gb" in result["memory"]
        assert isinstance(result["memory"]["total_gb"], float)
    
    @pytest.mark.asyncio
    async def test_performance_info(self, system_tool):
        """Test performance metrics collection"""
        result = await system_tool._get_performance_info()
        
        assert "cpu_usage_percent" in result
        assert "memory_usage_percent" in result
        assert "process_count" in result
        assert isinstance(result["cpu_usage_percent"], (int, float))
        assert isinstance(result["process_count"], int)
    
    @pytest.mark.asyncio
    async def test_network_info_safe(self, system_tool):
        """Test network info without sensitive data"""
        result = await system_tool._get_network_info(include_sensitive=False)
        
        assert "bytes_sent" in result
        assert "bytes_recv" in result
        assert "connections_count" in result
        assert "hostname" not in result  # Should not include sensitive data
        assert "local_ip" not in result
    
    @pytest.mark.asyncio
    async def test_network_info_sensitive(self, system_tool):
        """Test network info with sensitive data"""
        result = await system_tool._get_network_info(include_sensitive=True)
        
        assert "bytes_sent" in result
        assert "hostname" in result  # Should include sensitive data
        assert "local_ip" in result
    
    @pytest.mark.asyncio
    async def test_execute_with_different_categories(self, system_tool):
        """Test execution with different info categories"""
        categories = ["overview", "hardware", "performance", "network"]
        
        for category in categories:
            params = {"info_category": category}
            result = await system_tool.execute(params)
            
            assert result["category"] == category
            assert "timestamp" in result
            assert len(result) > 2  # Should have actual data


class TestFileAccessTool:
    """Test suite for FileAccessTool"""
    
    @pytest.fixture
    def file_tool(self):
        return FileAccessTool()
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_tool_initialization(self, file_tool):
        """Test that file access tool initializes correctly"""
        assert file_tool.metadata.name == "file_access"
        assert file_tool.metadata.security_level.value == "elevated"
        assert file_tool.metadata.requires_consent == True
        assert "file" in file_tool.metadata.tags
        assert "security" in file_tool.metadata.tags
    
    @pytest.mark.asyncio
    async def test_parameter_validation(self, file_tool):
        """Test parameter validation"""
        # Valid parameters
        valid_params = {"operation": "read", "path": "/tmp/test.txt"}
        is_valid, error = await file_tool.validate_parameters(valid_params)
        assert is_valid == True
        
        # Missing operation
        invalid_params = {"path": "/tmp/test.txt"}
        is_valid, error = await file_tool.validate_parameters(invalid_params)
        assert is_valid == False
        
        # Invalid operation
        invalid_params = {"operation": "invalid", "path": "/tmp/test.txt"}
        is_valid, error = await file_tool.validate_parameters(invalid_params)
        assert is_valid == False
        
        # Write operation without content
        invalid_params = {"operation": "write", "path": "/tmp/test.txt"}
        is_valid, error = await file_tool.validate_parameters(invalid_params)
        assert is_valid == False
    
    @pytest.mark.asyncio
    async def test_file_write_and_read(self, file_tool, temp_dir):
        """Test writing and reading files"""
        test_file = Path(temp_dir) / "test.txt"
        test_content = "Hello, World!"
        
        # Override safe directory check for testing
        file_tool.safe_directories.append(temp_dir)
        
        # Write file
        write_params = {
            "operation": "write",
            "path": str(test_file),
            "content": test_content
        }
        write_result = await file_tool.execute(write_params)
        
        assert write_result["success"] == True
        assert "bytes_written" in write_result
        
        # Read file
        read_params = {
            "operation": "read",
            "path": str(test_file)
        }
        read_result = await file_tool.execute(read_params)
        
        assert read_result["success"] == True
        assert read_result["content"] == test_content
        assert read_result["size"] == len(test_content)
    
    @pytest.mark.asyncio
    async def test_directory_listing(self, file_tool, temp_dir):
        """Test directory listing functionality"""
        # Create test files
        test_file1 = Path(temp_dir) / "file1.txt"
        test_file2 = Path(temp_dir) / "file2.txt"
        test_dir = Path(temp_dir) / "subdir"
        
        test_file1.write_text("content1")
        test_file2.write_text("content2")
        test_dir.mkdir()
        
        # Override safe directory check for testing
        file_tool.safe_directories.append(temp_dir)
        
        list_params = {
            "operation": "list",
            "path": temp_dir
        }
        result = await file_tool.execute(list_params)
        
        assert result["success"] == True
        assert result["total_files"] == 2
        assert result["total_directories"] == 1
        assert len(result["files"]) == 2
        assert len(result["directories"]) == 1
    
    @pytest.mark.asyncio
    async def test_file_info(self, file_tool, temp_dir):
        """Test file information retrieval"""
        test_file = Path(temp_dir) / "info_test.txt"
        test_file.write_text("test content")
        
        # Override safe directory check for testing
        file_tool.safe_directories.append(temp_dir)
        
        info_params = {
            "operation": "info",
            "path": str(test_file)
        }
        result = await file_tool.execute(info_params)
        
        assert result["success"] == True
        assert result["name"] == "info_test.txt"
        assert result["is_file"] == True
        assert result["is_directory"] == False
        assert "created" in result
        assert "modified" in result
    
    @pytest.mark.asyncio
    async def test_safety_checks(self, file_tool):
        """Test security and safety mechanisms"""
        # Test forbidden path
        dangerous_params = {
            "operation": "read",
            "path": "/etc/passwd",
            "safe_mode": True
        }
        
        with pytest.raises(ToolExecutionError):
            await file_tool.execute(dangerous_params)


class TestAdvancedCalculationTool:
    """Test suite for AdvancedCalculationTool"""
    
    @pytest.fixture
    def calc_tool(self):
        return AdvancedCalculationTool()
    
    @pytest.mark.asyncio
    async def test_tool_initialization(self, calc_tool):
        """Test that calculation tool initializes correctly"""
        assert calc_tool.metadata.name == "advanced_calculator"
        assert calc_tool.metadata.security_level.value == "safe"
        assert "math" in calc_tool.metadata.tags
        assert "calculation" in calc_tool.metadata.tags
    
    @pytest.mark.asyncio
    async def test_basic_operations(self, calc_tool):
        """Test basic mathematical operations"""
        # Square root
        sqrt_params = {
            "operation": "basic",
            "function": "sqrt",
            "numbers": [16]
        }
        result = await calc_tool.execute(sqrt_params)
        assert result["result"] == 4.0
        
        # Power
        pow_params = {
            "operation": "basic",
            "function": "pow",
            "numbers": [2, 3]
        }
        result = await calc_tool.execute(pow_params)
        assert result["result"] == 8.0
        
        # Sum
        sum_params = {
            "operation": "basic",
            "function": "sum",
            "numbers": [1, 2, 3, 4, 5]
        }
        result = await calc_tool.execute(sum_params)
        assert result["result"] == 15
    
    @pytest.mark.asyncio
    async def test_trigonometry_operations(self, calc_tool):
        """Test trigonometric operations"""
        import math
        
        # Sine in radians
        sin_params = {
            "operation": "trigonometry",
            "function": "sin",
            "numbers": [math.pi/2],
            "angle_unit": "radians"
        }
        result = await calc_tool.execute(sin_params)
        assert abs(result["result"] - 1.0) < 0.001  # sin(π/2) = 1
        
        # Cosine in degrees
        cos_params = {
            "operation": "trigonometry",
            "function": "cos",
            "numbers": [60],
            "angle_unit": "degrees"
        }
        result = await calc_tool.execute(cos_params)
        assert abs(result["result"] - 0.5) < 0.001  # cos(60°) = 0.5
    
    @pytest.mark.asyncio
    async def test_statistics_operations(self, calc_tool):
        """Test statistical operations"""
        numbers = [1, 2, 3, 4, 5]
        
        # Mean
        mean_params = {
            "operation": "statistics",
            "function": "mean",
            "numbers": numbers
        }
        result = await calc_tool.execute(mean_params)
        assert result["result"] == 3.0
        
        # Median
        median_params = {
            "operation": "statistics",
            "function": "median",
            "numbers": numbers
        }
        result = await calc_tool.execute(median_params)
        assert result["result"] == 3.0
        
        # Standard deviation
        std_params = {
            "operation": "statistics",
            "function": "std",
            "numbers": numbers
        }
        result = await calc_tool.execute(std_params)
        assert result["result"] > 0  # Should be positive
    
    @pytest.mark.asyncio
    async def test_expression_evaluation(self, calc_tool):
        """Test mathematical expression evaluation"""
        import math
        
        expr_params = {
            "operation": "expression",
            "expression": "2 * pi * 5"
        }
        result = await calc_tool.execute(expr_params)
        expected = 2 * math.pi * 5
        assert abs(result["result"] - expected) < 0.001
    
    @pytest.mark.asyncio
    async def test_matrix_operations(self, calc_tool):
        """Test matrix operations"""
        # 2x2 matrix determinant
        matrix_params = {
            "operation": "matrix",
            "function": "determinant",
            "numbers": [1, 2, 3, 4]  # [[1,2], [3,4]]
        }
        result = await calc_tool.execute(matrix_params)
        expected = 1*4 - 2*3  # -2
        assert result["result"] == expected
    
    @pytest.mark.asyncio
    async def test_sequence_operations(self, calc_tool):
        """Test sequence operations"""
        # Fibonacci sequence
        fib_params = {
            "operation": "sequence",
            "function": "fibonacci",
            "numbers": [10]
        }
        result = await calc_tool.execute(fib_params)
        
        # First 10 Fibonacci numbers: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        assert result["result"] == expected
    
    @pytest.mark.asyncio
    async def test_precision_handling(self, calc_tool):
        """Test precision formatting"""
        # Test with custom precision
        sqrt_params = {
            "operation": "basic",
            "function": "sqrt",
            "numbers": [2],
            "precision": 3
        }
        result = await calc_tool.execute(sqrt_params)
        
        # sqrt(2) ≈ 1.414, with precision 3 should be 1.414
        assert result["precision"] == 3
        assert len(str(result["result"]).split(".")[-1]) <= 3


class TestCoreToolsIntegration:
    """Test suite for core tools integration"""
    
    @pytest.mark.asyncio
    async def test_create_core_tools(self):
        """Test core tools factory function"""
        tools = create_core_tools()
        
        assert len(tools) == 4
        tool_names = [tool.metadata.name for tool in tools]
        assert "internet_search" in tool_names
        assert "enhanced_system_info" in tool_names
        assert "file_access" in tool_names
        assert "advanced_calculator" in tool_names
    
    @pytest.mark.asyncio
    async def test_register_core_tools(self):
        """Test core tools registration with framework"""
        config = Config()
        framework = ToolIntegrationFramework(config)
        await framework.initialize()
        
        registered_count = await register_core_tools(framework)
        
        assert registered_count == 4
        
        # Verify tools are registered
        available_tools = framework.get_available_tools()
        tool_names = [tool["name"] for tool in available_tools]
        assert "internet_search" in tool_names
        assert "enhanced_system_info" in tool_names
        assert "file_access" in tool_names
        assert "advanced_calculator" in tool_names
    
    @pytest.mark.asyncio
    async def test_end_to_end_tool_execution(self):
        """Test end-to-end tool execution through framework"""
        config = Config()
        framework = ToolIntegrationFramework(config)
        await framework.initialize()
        
        # Register core tools
        await register_core_tools(framework)
        
        # Test calculator execution through framework
        result = await framework.execute_tool(
            "advanced_calculator",
            {
                "operation": "basic",
                "function": "sum",
                "numbers": [1, 2, 3, 4, 5]
            }
        )
        
        assert result.success == True
        assert result.result["result"] == 15
        assert result.tool_name == "advanced_calculator"


# Performance and stress tests
class TestCoreToolsPerformance:
    """Performance tests for core tools"""
    
    @pytest.mark.asyncio
    async def test_calculation_performance(self):
        """Test calculation tool performance with large datasets"""
        calc_tool = AdvancedCalculationTool()
        
        # Large dataset
        large_numbers = list(range(1, 10001))  # 1 to 10,000
        
        params = {
            "operation": "statistics",
            "function": "mean",
            "numbers": large_numbers
        }
        
        import time
        start_time = time.time()
        result = await calc_tool.execute(params)
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 1.0  # Should complete within 1 second
        assert result["result"] == 5000.5  # Mean of 1 to 10,000
    
    @pytest.mark.asyncio
    async def test_file_tool_large_file_handling(self):
        """Test file tool with large file operations"""
        file_tool = FileAccessTool()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_tool.safe_directories.append(temp_dir)
            
            # Test max file size limit
            large_content = "x" * (15 * 1024 * 1024)  # 15MB (larger than 10MB limit)
            test_file = Path(temp_dir) / "large_test.txt"
            test_file.write_text(large_content)
            
            read_params = {
                "operation": "read",
                "path": str(test_file),
                "max_size_mb": 10
            }
            
            result = await file_tool.execute(read_params)
            assert result["success"] == False
            assert "too large" in result["error"]


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"]) 