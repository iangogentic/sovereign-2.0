"""
Tool Integration Framework - Core architecture for external tool integration

This module provides the foundational architecture for integrating external tools
with the Sovereign AI system, enabling the AI to interact with various APIs,
CLI tools, and services in a secure and controlled manner.
"""

import asyncio
import logging
import json
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import subprocess
import tempfile
import os
import sys
from pathlib import Path
import uuid

from .config import Config
from .consent_manager import ConsentLevel, MonitoringState
from .logger import setup_logger, get_performance_logger


class ToolType(Enum):
    """Types of tools that can be integrated"""
    API = "api"                 # REST/HTTP API calls
    CLI = "cli"                 # Command-line interface tools
    PYTHON = "python"           # Python modules/functions
    SYSTEM = "system"           # System operations
    FILE = "file"               # File operations
    NETWORK = "network"         # Network operations
    DATABASE = "database"       # Database operations
    CUSTOM = "custom"           # Custom tool implementations


class ToolSecurityLevel(Enum):
    """Security levels for tool execution"""
    SAFE = "safe"               # Read-only, no system access
    MODERATE = "moderate"       # Limited system access
    ELEVATED = "elevated"       # Full system access (requires explicit consent)
    RESTRICTED = "restricted"   # Disabled/blocked tool


class ToolExecutionMode(Enum):
    """Execution modes for tools"""
    SYNCHRONOUS = "sync"        # Blocking execution
    ASYNCHRONOUS = "async"      # Non-blocking execution
    BACKGROUND = "background"   # Long-running background tasks


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""
    name: str
    type: str                   # 'string', 'integer', 'boolean', 'array', 'object'
    description: str
    required: bool = True
    default: Any = None
    validation_regex: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    sensitive: bool = False     # Whether parameter contains sensitive data


@dataclass
class ToolMetadata:
    """Metadata describing a tool's capabilities and requirements"""
    name: str
    description: str
    version: str
    tool_type: ToolType
    security_level: ToolSecurityLevel
    execution_mode: ToolExecutionMode
    parameters: List[ToolParameter]
    return_type: str
    examples: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    author: str = "Unknown"
    documentation_url: Optional[str] = None
    max_execution_time: int = 30  # seconds
    rate_limit: Optional[int] = None  # requests per minute
    requires_consent: bool = False
    data_handling: str = "unknown"  # Description of how data is handled
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ToolExecutionRequest:
    """Request for tool execution"""
    tool_name: str
    parameters: Dict[str, Any]
    requester: str              # ID of the component requesting execution
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timeout: Optional[int] = None
    priority: int = 1           # 1 = highest, 10 = lowest
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ToolExecutionResult:
    """Result of tool execution"""
    execution_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization"""
        return {
            'execution_id': self.execution_id,
            'tool_name': self.tool_name,
            'success': self.success,
            'result': self.result,
            'error': self.error,
            'execution_time': self.execution_time,
            'metadata': self.metadata,
            'logs': self.logs,
            'created_at': self.created_at.isoformat()
        }


class ToolException(Exception):
    """Base exception for tool-related errors"""
    pass


class ToolNotFoundError(ToolException):
    """Tool not found in registry"""
    pass


class ToolSecurityError(ToolException):
    """Security violation in tool execution"""
    pass


class ToolExecutionError(ToolException):
    """Error during tool execution"""
    pass


class ToolPermissionError(ToolException):
    """Permission denied for tool execution"""
    pass


class BaseTool(ABC):
    """Abstract base class for all tools"""
    
    def __init__(self, metadata: ToolMetadata):
        self.metadata = metadata
        self.logger = logging.getLogger(f"tool.{metadata.name}")
        self.execution_count = 0
        self.last_execution = None
        self.total_execution_time = 0.0
        
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute the tool with given parameters"""
        pass
    
    @abstractmethod
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate tool parameters"""
        pass
    
    async def pre_execute(self, parameters: Dict[str, Any]) -> None:
        """Pre-execution hook (can be overridden)"""
        pass
    
    async def post_execute(self, result: Any, parameters: Dict[str, Any]) -> None:
        """Post-execution hook (can be overridden)"""
        pass
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        return {
            'execution_count': self.execution_count,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'total_execution_time': self.total_execution_time,
            'avg_execution_time': self.total_execution_time / max(1, self.execution_count)
        }
    
    def __str__(self) -> str:
        return f"Tool({self.metadata.name})"
    
    def __repr__(self) -> str:
        return f"Tool(name='{self.metadata.name}', type='{self.metadata.tool_type.value}')"


class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_groups: Dict[str, List[str]] = {}
        self.logger = logging.getLogger(__name__)
        
    def register_tool(self, tool: BaseTool) -> bool:
        """Register a new tool"""
        try:
            if tool.metadata.name in self.tools:
                self.logger.warning(f"Tool '{tool.metadata.name}' already registered, updating...")
            
            self.tools[tool.metadata.name] = tool
            
            # Add to appropriate groups
            for tag in tool.metadata.tags:
                if tag not in self.tool_groups:
                    self.tool_groups[tag] = []
                if tool.metadata.name not in self.tool_groups[tag]:
                    self.tool_groups[tag].append(tool.metadata.name)
            
            self.logger.info(f"Successfully registered tool: {tool.metadata.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool {tool.metadata.name}: {e}")
            return False
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool"""
        try:
            if tool_name not in self.tools:
                raise ToolNotFoundError(f"Tool '{tool_name}' not found")
            
            tool = self.tools[tool_name]
            
            # Remove from groups
            for tag in tool.metadata.tags:
                if tag in self.tool_groups and tool_name in self.tool_groups[tag]:
                    self.tool_groups[tag].remove(tool_name)
            
            del self.tools[tool_name]
            self.logger.info(f"Unregistered tool: {tool_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister tool {tool_name}: {e}")
            return False
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a registered tool by name"""
        return self.tools.get(tool_name)
    
    def list_tools(self, 
                   tool_type: Optional[ToolType] = None,
                   security_level: Optional[ToolSecurityLevel] = None,
                   tags: Optional[List[str]] = None) -> List[BaseTool]:
        """List available tools with optional filtering"""
        tools = list(self.tools.values())
        
        if tool_type:
            tools = [tool for tool in tools if tool.metadata.tool_type == tool_type]
        
        if security_level:
            tools = [tool for tool in tools if tool.metadata.security_level == security_level]
        
        if tags:
            tools = [tool for tool in tools if any(tag in tool.metadata.tags for tag in tags)]
        
        return tools
    
    def search_tools(self, query: str) -> List[BaseTool]:
        """Search tools by name or description"""
        query_lower = query.lower()
        matching_tools = []
        
        for tool in self.tools.values():
            if (query_lower in tool.metadata.name.lower() or 
                query_lower in tool.metadata.description.lower()):
                matching_tools.append(tool)
        
        return matching_tools
    
    def get_tool_groups(self) -> Dict[str, List[str]]:
        """Get all tool groups"""
        return self.tool_groups.copy()
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        type_counts = {}
        security_counts = {}
        
        for tool in self.tools.values():
            tool_type = tool.metadata.tool_type.value
            security_level = tool.metadata.security_level.value
            
            type_counts[tool_type] = type_counts.get(tool_type, 0) + 1
            security_counts[security_level] = security_counts.get(security_level, 0) + 1
        
        return {
            'total_tools': len(self.tools),
            'tool_types': type_counts,
            'security_levels': security_counts,
            'tool_groups': len(self.tool_groups)
        }


class PermissionManager:
    """Manages permissions and access control for tool execution"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Permission policies
        self.tool_permissions: Dict[str, Dict[str, bool]] = {}
        self.user_permissions: Dict[str, List[str]] = {}
        self.global_policies: Dict[str, bool] = {
            'allow_system_tools': False,
            'allow_network_tools': True,
            'allow_file_tools': False,
            'require_consent_for_elevated': True
        }
        
    def check_permission(self, tool_name: str, requester: str, tool_metadata: ToolMetadata) -> Tuple[bool, Optional[str]]:
        """Check if requester has permission to execute tool"""
        try:
            # Check global policies first
            if tool_metadata.security_level == ToolSecurityLevel.RESTRICTED:
                return False, "Tool is restricted and cannot be executed"
            
            if tool_metadata.security_level == ToolSecurityLevel.ELEVATED:
                if self.global_policies.get('require_consent_for_elevated', True):
                    if not self._has_elevated_consent(requester):
                        return False, "Elevated permissions required - user consent needed"
            
            # Check tool-specific permissions
            if tool_name in self.tool_permissions:
                if requester in self.tool_permissions[tool_name]:
                    return self.tool_permissions[tool_name][requester], None
            
            # Check user permissions
            if requester in self.user_permissions:
                if tool_name in self.user_permissions[requester]:
                    return True, None
            
            # Default permission based on security level
            if tool_metadata.security_level == ToolSecurityLevel.SAFE:
                return True, None
            elif tool_metadata.security_level == ToolSecurityLevel.MODERATE:
                return self.global_policies.get('allow_moderate_tools', True), None
            else:
                return False, "Permission denied for this tool"
                
        except Exception as e:
            self.logger.error(f"Permission check failed for {tool_name}: {e}")
            return False, f"Permission check error: {e}"
    
    def grant_permission(self, tool_name: str, requester: str) -> bool:
        """Grant permission for tool execution"""
        try:
            if tool_name not in self.tool_permissions:
                self.tool_permissions[tool_name] = {}
            
            self.tool_permissions[tool_name][requester] = True
            self.logger.info(f"Granted permission for {requester} to use {tool_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to grant permission: {e}")
            return False
    
    def revoke_permission(self, tool_name: str, requester: str) -> bool:
        """Revoke permission for tool execution"""
        try:
            if tool_name in self.tool_permissions:
                if requester in self.tool_permissions[tool_name]:
                    del self.tool_permissions[tool_name][requester]
            
            self.logger.info(f"Revoked permission for {requester} to use {tool_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to revoke permission: {e}")
            return False
    
    def _has_elevated_consent(self, requester: str) -> bool:
        """Check if requester has elevated consent"""
        # This would integrate with the consent manager
        # For now, return False to be safe
        return False
    
    def set_global_policy(self, policy_name: str, value: bool) -> bool:
        """Set a global permission policy"""
        if policy_name in self.global_policies:
            self.global_policies[policy_name] = value
            self.logger.info(f"Set global policy {policy_name} = {value}")
            return True
        return False
    
    def get_permissions_summary(self) -> Dict[str, Any]:
        """Get summary of current permissions"""
        return {
            'global_policies': self.global_policies,
            'tool_permissions': len(self.tool_permissions),
            'user_permissions': len(self.user_permissions)
        }


class ExecutionEngine:
    """Manages secure execution of tools"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.perf_logger = get_performance_logger()
        
        # Execution state
        self.active_executions: Dict[str, ToolExecutionRequest] = {}
        self.execution_history: List[ToolExecutionResult] = []
        self.max_history_size = 1000
        
        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = {}
        
        # Resource monitoring
        self.resource_limits = {
            'max_memory_mb': 512,
            'max_cpu_percent': 80,
            'max_execution_time': 300
        }
    
    async def execute_tool(self, tool: BaseTool, request: ToolExecutionRequest) -> ToolExecutionResult:
        """Execute a tool with the given request"""
        start_time = time.time()
        result = ToolExecutionResult(
            execution_id=request.execution_id,
            tool_name=request.tool_name,
            success=False  # Default to False, will be set to True on success
        )
        
        try:
            # Check rate limits
            if not self._check_rate_limit(tool.metadata.name):
                raise ToolExecutionError(f"Rate limit exceeded for tool {tool.metadata.name}")
            
            # Validate parameters
            valid, error = await tool.validate_parameters(request.parameters)
            if not valid:
                raise ToolExecutionError(f"Parameter validation failed: {error}")
            
            # Record execution start
            self.active_executions[request.execution_id] = request
            self.logger.info(f"Starting execution of {tool.metadata.name} (ID: {request.execution_id})")
            
            # Pre-execution hook
            await tool.pre_execute(request.parameters)
            
            # Execute with timeout
            timeout = request.timeout or tool.metadata.max_execution_time
            
            try:
                execution_result = await asyncio.wait_for(
                    tool.execute(request.parameters),
                    timeout=timeout
                )
                
                result.success = True
                result.result = execution_result
                
            except asyncio.TimeoutError:
                raise ToolExecutionError(f"Tool execution timed out after {timeout} seconds")
            
            # Post-execution hook
            await tool.post_execute(result.result, request.parameters)
            
            # Update tool statistics
            tool.execution_count += 1
            tool.last_execution = datetime.now()
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            self.logger.error(f"Tool execution failed: {e}")
            
        finally:
            # Clean up
            if request.execution_id in self.active_executions:
                del self.active_executions[request.execution_id]
            
            # Record execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            if result.success:
                tool.total_execution_time += execution_time
            
            # Add to history
            self.execution_history.append(result)
            if len(self.execution_history) > self.max_history_size:
                self.execution_history.pop(0)
            
            self.logger.info(f"Completed execution of {tool.metadata.name} in {execution_time:.2f}s")
        
        return result
    
    def _check_rate_limit(self, tool_name: str) -> bool:
        """Check if tool is within rate limits"""
        # Simple rate limiting implementation
        current_time = time.time()
        
        if tool_name not in self.rate_limits:
            self.rate_limits[tool_name] = []
        
        # Clean old entries (older than 1 minute)
        self.rate_limits[tool_name] = [
            timestamp for timestamp in self.rate_limits[tool_name]
            if current_time - timestamp < 60
        ]
        
        # Check limit (default 60 requests per minute)
        max_requests = 60
        if len(self.rate_limits[tool_name]) >= max_requests:
            return False
        
        # Add current request
        self.rate_limits[tool_name].append(current_time)
        return True
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for result in self.execution_history if result.success)
        
        avg_execution_time = 0.0
        if total_executions > 0:
            avg_execution_time = sum(result.execution_time for result in self.execution_history) / total_executions
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'failure_rate': (total_executions - successful_executions) / max(1, total_executions),
            'avg_execution_time': avg_execution_time,
            'active_executions': len(self.active_executions)
        }
    
    def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get list of currently active executions"""
        return [
            {
                'execution_id': req.execution_id,
                'tool_name': req.tool_name,
                'requester': req.requester,
                'started_at': req.created_at.isoformat(),
                'runtime': (datetime.now() - req.created_at).total_seconds()
            }
            for req in self.active_executions.values()
        ]


class ToolIntegrationFramework:
    """Main framework orchestrating tool integration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.perf_logger = get_performance_logger()
        
        # Core components
        self.registry = ToolRegistry()
        self.permission_manager = PermissionManager(config)
        self.execution_engine = ExecutionEngine(config)
        
        # Framework state
        self.initialized = False
        self.framework_stats = {
            'start_time': datetime.now(),
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0
        }
        
    async def initialize(self) -> bool:
        """Initialize the framework"""
        try:
            self.logger.info("Initializing Tool Integration Framework...")
            
            # Load default tools
            await self._load_default_tools()
            
            # Set up monitoring
            await self._setup_monitoring()
            
            self.initialized = True
            self.logger.info("Tool Integration Framework initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize framework: {e}")
            return False
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], 
                          requester: str = "system") -> ToolExecutionResult:
        """Execute a tool with given parameters"""
        if not self.initialized:
            raise ToolException("Framework not initialized")
        
        self.framework_stats['total_requests'] += 1
        
        try:
            # Get tool
            tool = self.registry.get_tool(tool_name)
            if not tool:
                raise ToolNotFoundError(f"Tool '{tool_name}' not found")
            
            # Check permissions
            has_permission, error = self.permission_manager.check_permission(
                tool_name, requester, tool.metadata
            )
            if not has_permission:
                raise ToolPermissionError(error or "Permission denied")
            
            # Create execution request
            request = ToolExecutionRequest(
                tool_name=tool_name,
                parameters=parameters,
                requester=requester
            )
            
            # Execute tool
            result = await self.execution_engine.execute_tool(tool, request)
            
            if result.success:
                self.framework_stats['successful_requests'] += 1
            else:
                self.framework_stats['failed_requests'] += 1
            
            return result
            
        except Exception as e:
            self.framework_stats['failed_requests'] += 1
            self.logger.error(f"Tool execution failed: {e}")
            raise
    
    async def _load_default_tools(self) -> None:
        """Load default tools into the registry"""
        # This will be implemented in the next subtask
        pass
    
    async def _setup_monitoring(self) -> None:
        """Set up monitoring and logging"""
        # This will be expanded in future subtasks
        pass
    
    def get_available_tools(self, requester: str = "system") -> List[Dict[str, Any]]:
        """Get list of available tools for a requester"""
        available_tools = []
        
        for tool in self.registry.list_tools():
            has_permission, _ = self.permission_manager.check_permission(
                tool.metadata.name, requester, tool.metadata
            )
            
            if has_permission:
                available_tools.append({
                    'name': tool.metadata.name,
                    'description': tool.metadata.description,
                    'type': tool.metadata.tool_type.value,
                    'security_level': tool.metadata.security_level.value,
                    'parameters': [
                        {
                            'name': param.name,
                            'type': param.type,
                            'description': param.description,
                            'required': param.required
                        } for param in tool.metadata.parameters
                    ]
                })
        
        return available_tools
    
    def get_framework_stats(self) -> Dict[str, Any]:
        """Get comprehensive framework statistics"""
        return {
            'framework': self.framework_stats,
            'registry': self.registry.get_registry_stats(),
            'execution': self.execution_engine.get_execution_stats(),
            'permissions': self.permission_manager.get_permissions_summary()
        }
    
    async def cleanup(self) -> None:
        """Clean up framework resources"""
        try:
            self.logger.info("Cleaning up Tool Integration Framework...")
            
            # Cancel active executions
            for execution_id in list(self.execution_engine.active_executions.keys()):
                # In a real implementation, we'd cancel the tasks
                pass
            
            self.initialized = False
            self.logger.info("Tool Integration Framework cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}") 