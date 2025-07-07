"""
Example Tool Implementations - Demonstrating the Tool Integration Framework

This module contains example implementations of various tool types to showcase
the framework's capabilities and provide templates for future tool development.
"""

import asyncio
import json
import os
import subprocess
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..tool_integration_framework import (
    BaseTool, ToolMetadata, ToolParameter, ToolType, 
    ToolSecurityLevel, ToolExecutionMode, ToolExecutionError
)


class EchoTool(BaseTool):
    """Simple echo tool for testing and demonstration"""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="echo",
            description="Simple echo tool that returns the input message",
            version="1.0.0",
            tool_type=ToolType.PYTHON,
            security_level=ToolSecurityLevel.SAFE,
            execution_mode=ToolExecutionMode.SYNCHRONOUS,
            parameters=[
                ToolParameter(
                    name="message",
                    type="string",
                    description="The message to echo back",
                    required=True
                ),
                ToolParameter(
                    name="uppercase",
                    type="boolean",
                    description="Whether to convert message to uppercase",
                    required=False,
                    default=False
                )
            ],
            return_type="string",
            examples=[
                {
                    "input": {"message": "Hello World", "uppercase": False},
                    "output": "Hello World"
                },
                {
                    "input": {"message": "Hello World", "uppercase": True},
                    "output": "HELLO WORLD"
                }
            ],
            tags=["utility", "test", "demo"],
            max_execution_time=5
        )
        super().__init__(metadata)
    
    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute the echo tool"""
        message = parameters["message"]
        uppercase = parameters.get("uppercase", False)
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        result = message.upper() if uppercase else message
        self.logger.info(f"Echo tool executed: {message} -> {result}")
        
        # Update statistics
        self.execution_count += 1
        self.last_execution = datetime.now()
        
        return result
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameters for the echo tool"""
        if "message" not in parameters:
            return False, "Missing required parameter: message"
        
        if not isinstance(parameters["message"], str):
            return False, "Parameter 'message' must be a string"
        
        if "uppercase" in parameters and not isinstance(parameters["uppercase"], bool):
            return False, "Parameter 'uppercase' must be a boolean"
        
        return True, None


class FileInfoTool(BaseTool):
    """Tool for getting file information"""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="file_info",
            description="Get information about a file (size, modified time, etc.)",
            version="1.0.0",
            tool_type=ToolType.FILE,
            security_level=ToolSecurityLevel.MODERATE,
            execution_mode=ToolExecutionMode.SYNCHRONOUS,
            parameters=[
                ToolParameter(
                    name="file_path",
                    type="string",
                    description="Path to the file to analyze",
                    required=True
                )
            ],
            return_type="object",
            examples=[
                {
                    "input": {"file_path": "/tmp/test.txt"},
                    "output": {
                        "exists": True,
                        "size": 1024,
                        "modified": "2024-01-01T12:00:00Z",
                        "is_directory": False
                    }
                }
            ],
            tags=["file", "utility", "info"],
            max_execution_time=10,
            requires_consent=True,
            data_handling="Reads file metadata only, does not access file contents"
        )
        super().__init__(metadata)
    
    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute the file info tool"""
        file_path = parameters["file_path"]
        
        try:
            if not os.path.exists(file_path):
                return {
                    "exists": False,
                    "error": "File not found"
                }
            
            stat = os.stat(file_path)
            
            result = {
                "exists": True,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_directory": os.path.isdir(file_path),
                "permissions": oct(stat.st_mode)[-3:],
                "owner": stat.st_uid if hasattr(stat, 'st_uid') else None
            }
            
            self.logger.info(f"File info retrieved for: {file_path}")
            return result
            
        except Exception as e:
            raise ToolExecutionError(f"Failed to get file info: {e}")
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameters for the file info tool"""
        if "file_path" not in parameters:
            return False, "Missing required parameter: file_path"
        
        if not isinstance(parameters["file_path"], str):
            return False, "Parameter 'file_path' must be a string"
        
        if not parameters["file_path"].strip():
            return False, "Parameter 'file_path' cannot be empty"
        
        return True, None


class CalculatorTool(BaseTool):
    """Simple calculator tool for mathematical operations"""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="calculator",
            description="Perform basic mathematical calculations",
            version="1.0.0",
            tool_type=ToolType.PYTHON,
            security_level=ToolSecurityLevel.SAFE,
            execution_mode=ToolExecutionMode.SYNCHRONOUS,
            parameters=[
                ToolParameter(
                    name="operation",
                    type="string",
                    description="The mathematical operation to perform",
                    required=True,
                    allowed_values=["add", "subtract", "multiply", "divide", "power"]
                ),
                ToolParameter(
                    name="a",
                    type="number",
                    description="First number",
                    required=True
                ),
                ToolParameter(
                    name="b",
                    type="number",
                    description="Second number",
                    required=True
                )
            ],
            return_type="number",
            examples=[
                {
                    "input": {"operation": "add", "a": 5, "b": 3},
                    "output": 8
                },
                {
                    "input": {"operation": "multiply", "a": 4, "b": 7},
                    "output": 28
                }
            ],
            tags=["math", "utility", "calculator"],
            max_execution_time=5
        )
        super().__init__(metadata)
    
    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute the calculator tool"""
        operation = parameters["operation"]
        a = parameters["a"]
        b = parameters["b"]
        
        try:
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    raise ToolExecutionError("Division by zero")
                result = a / b
            elif operation == "power":
                result = a ** b
            else:
                raise ToolExecutionError(f"Unknown operation: {operation}")
            
            self.logger.info(f"Calculator executed: {a} {operation} {b} = {result}")
            return result
            
        except Exception as e:
            raise ToolExecutionError(f"Calculation failed: {e}")
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameters for the calculator tool"""
        if "operation" not in parameters:
            return False, "Missing required parameter: operation"
        
        if "a" not in parameters or "b" not in parameters:
            return False, "Missing required parameters: a and b"
        
        operation = parameters["operation"]
        if operation not in ["add", "subtract", "multiply", "divide", "power"]:
            return False, f"Invalid operation: {operation}"
        
        try:
            float(parameters["a"])
            float(parameters["b"])
        except (ValueError, TypeError):
            return False, "Parameters 'a' and 'b' must be numbers"
        
        return True, None


class SystemInfoTool(BaseTool):
    """Tool for getting system information"""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="system_info",
            description="Get basic system information",
            version="1.0.0",
            tool_type=ToolType.SYSTEM,
            security_level=ToolSecurityLevel.MODERATE,
            execution_mode=ToolExecutionMode.SYNCHRONOUS,
            parameters=[
                ToolParameter(
                    name="info_type",
                    type="string",
                    description="Type of system information to retrieve",
                    required=False,
                    default="basic",
                    allowed_values=["basic", "detailed", "performance"]
                )
            ],
            return_type="object",
            examples=[
                {
                    "input": {"info_type": "basic"},
                    "output": {
                        "platform": "linux",
                        "python_version": "3.9.0",
                        "timestamp": "2024-01-01T12:00:00Z"
                    }
                }
            ],
            tags=["system", "info", "diagnostics"],
            max_execution_time=15,
            requires_consent=True,
            data_handling="Collects basic system information for diagnostics"
        )
        super().__init__(metadata)
    
    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute the system info tool"""
        import platform
        import sys
        
        info_type = parameters.get("info_type", "basic")
        
        try:
            basic_info = {
                "platform": platform.system().lower(),
                "architecture": platform.machine(),
                "python_version": sys.version.split()[0],
                "timestamp": datetime.now().isoformat()
            }
            
            if info_type == "basic":
                result = basic_info
            elif info_type == "detailed":
                result = {
                    **basic_info,
                    "platform_release": platform.release(),
                    "platform_version": platform.version(),
                    "processor": platform.processor(),
                    "python_implementation": platform.python_implementation()
                }
            elif info_type == "performance":
                # Basic performance metrics
                result = {
                    **basic_info,
                    "cpu_count": os.cpu_count(),
                    "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
                }
            else:
                raise ToolExecutionError(f"Unknown info_type: {info_type}")
            
            self.logger.info(f"System info retrieved: {info_type}")
            return result
            
        except Exception as e:
            raise ToolExecutionError(f"Failed to get system info: {e}")
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameters for the system info tool"""
        if "info_type" in parameters:
            info_type = parameters["info_type"]
            if info_type not in ["basic", "detailed", "performance"]:
                return False, f"Invalid info_type: {info_type}"
        
        return True, None


class MockApiTool(BaseTool):
    """Mock API tool for testing network-based tools"""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="mock_api",
            description="Mock API tool for testing HTTP requests",
            version="1.0.0",
            tool_type=ToolType.API,
            security_level=ToolSecurityLevel.MODERATE,
            execution_mode=ToolExecutionMode.ASYNCHRONOUS,
            parameters=[
                ToolParameter(
                    name="endpoint",
                    type="string",
                    description="API endpoint to call",
                    required=True
                ),
                ToolParameter(
                    name="method",
                    type="string",
                    description="HTTP method",
                    required=False,
                    default="GET",
                    allowed_values=["GET", "POST", "PUT", "DELETE"]
                ),
                ToolParameter(
                    name="data",
                    type="object",
                    description="Request data",
                    required=False
                )
            ],
            return_type="object",
            examples=[
                {
                    "input": {"endpoint": "/users", "method": "GET"},
                    "output": {"status": 200, "data": {"users": []}}
                }
            ],
            tags=["api", "network", "test"],
            max_execution_time=30,
            rate_limit=100,  # 100 requests per minute
            requires_consent=False,
            data_handling="Makes HTTP requests to specified endpoints"
        )
        super().__init__(metadata)
    
    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute the mock API tool"""
        endpoint = parameters["endpoint"]
        method = parameters.get("method", "GET")
        data = parameters.get("data", None)
        
        # Simulate API call delay
        await asyncio.sleep(0.5)
        
        # Mock response based on endpoint
        if endpoint == "/users":
            mock_response = {
                "status": 200,
                "data": {
                    "users": [
                        {"id": 1, "name": "John Doe", "email": "john@example.com"},
                        {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
                    ]
                }
            }
        elif endpoint == "/health":
            mock_response = {
                "status": 200,
                "data": {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat()
                }
            }
        else:
            mock_response = {
                "status": 404,
                "error": "Endpoint not found"
            }
        
        # Add method and data to response for testing
        mock_response["request"] = {
            "method": method,
            "endpoint": endpoint,
            "data": data
        }
        
        self.logger.info(f"Mock API call: {method} {endpoint} -> {mock_response['status']}")
        return mock_response
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameters for the mock API tool"""
        if "endpoint" not in parameters:
            return False, "Missing required parameter: endpoint"
        
        if not isinstance(parameters["endpoint"], str):
            return False, "Parameter 'endpoint' must be a string"
        
        if "method" in parameters:
            method = parameters["method"]
            if method not in ["GET", "POST", "PUT", "DELETE"]:
                return False, f"Invalid HTTP method: {method}"
        
        return True, None


# Tool factory function
def create_example_tools() -> List[BaseTool]:
    """Create and return a list of example tools"""
    return [
        EchoTool(),
        FileInfoTool(),
        CalculatorTool(),
        SystemInfoTool(),
        MockApiTool()
    ]


# Tool registration helper
async def register_example_tools(framework) -> int:
    """Register example tools with the framework"""
    tools = create_example_tools()
    registered_count = 0
    
    for tool in tools:
        if framework.registry.register_tool(tool):
            registered_count += 1
    
    return registered_count 