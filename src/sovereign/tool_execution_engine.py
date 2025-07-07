"""
Enhanced Tool Execution Engine - Secure and robust tool execution with advanced validation
"""

import asyncio
import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from pathlib import Path
import threading
import multiprocessing
import sys
import os
try:
    import resource
except ImportError:
    resource = None  # Windows doesn't have resource module
try:
    import signal
except ImportError:
    signal = None  # Some platforms might not have signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError, ProcessPoolExecutor
import functools
import hashlib
import pickle
from contextlib import contextmanager
import subprocess
import tempfile
import psutil

from .tool_integration_framework import (
    BaseTool, ToolMetadata, ToolParameter, ToolExecutionResult, ToolExecutionRequest,
    ToolSecurityLevel, ToolExecutionMode, ToolType, PermissionManager
)
from .config import Config


class ValidationError(Exception):
    """Parameter validation error"""
    pass


class ExecutionError(Exception):
    """Tool execution error"""
    pass


class SecurityError(Exception):
    """Security violation error"""
    pass


class TimeoutError(Exception):
    """Execution timeout error"""
    pass


class ResourceLimitError(Exception):
    """Resource limit exceeded error"""
    pass


class ParameterType(Enum):
    """Enhanced parameter type definitions"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    FILE_PATH = "file_path"
    URL = "url"
    EMAIL = "email"
    REGEX = "regex"
    JSON = "json"
    DATETIME = "datetime"
    ENUM = "enum"
    BINARY = "binary"


class ExecutionState(Enum):
    """Execution state tracking"""
    QUEUED = "queued"
    VALIDATING = "validating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ValidationRule:
    """Parameter validation rule"""
    parameter_name: str
    validator: Callable[[Any], bool]
    error_message: str
    is_required: bool = True


@dataclass
class ExecutionContext:
    """Execution context with security and monitoring"""
    execution_id: str
    tool_name: str
    parameters: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    security_level: ToolSecurityLevel = ToolSecurityLevel.SAFE
    execution_mode: ToolExecutionMode = ToolExecutionMode.SYNCHRONOUS
    timeout: float = 30.0
    max_memory_mb: int = 100
    max_cpu_percent: float = 50.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    state: ExecutionState = ExecutionState.QUEUED
    error: Optional[str] = None
    result: Optional[ToolExecutionResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionStats:
    """Execution statistics and metrics"""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    timeout_executions: int = 0
    average_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    total_execution_time: float = 0.0
    executions_by_tool: Dict[str, int] = field(default_factory=dict)
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    security_violations: int = 0
    resource_limit_violations: int = 0
    
    def update(self, context: ExecutionContext):
        """Update statistics with execution context"""
        self.total_executions += 1
        
        if context.state == ExecutionState.COMPLETED:
            self.successful_executions += 1
        elif context.state == ExecutionState.FAILED:
            self.failed_executions += 1
        elif context.state == ExecutionState.TIMEOUT:
            self.timeout_executions += 1
        
        # Update execution time stats
        if context.started_at and context.completed_at:
            execution_time = (context.completed_at - context.started_at).total_seconds()
            self.total_execution_time += execution_time
            self.max_execution_time = max(self.max_execution_time, execution_time)
            self.min_execution_time = min(self.min_execution_time, execution_time)
            self.average_execution_time = self.total_execution_time / self.total_executions
        
        # Update tool-specific stats
        self.executions_by_tool[context.tool_name] = self.executions_by_tool.get(context.tool_name, 0) + 1
        
        # Update error stats
        if context.error:
            error_type = type(context.error).__name__ if hasattr(context.error, '__name__') else str(context.error)
            self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1


class ParameterValidator:
    """Advanced parameter validation with comprehensive rules"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.custom_validators: Dict[str, Callable] = {}
        self._setup_builtin_validators()
    
    def _setup_builtin_validators(self):
        """Setup built-in validation functions"""
        import re
        from urllib.parse import urlparse
        
        self.custom_validators.update({
            'email': lambda x: re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', str(x)) is not None,
            'url': lambda x: urlparse(str(x)).scheme in ['http', 'https', 'ftp', 'file'],
            'file_path': lambda x: Path(str(x)).is_file() if Path(str(x)).exists() else len(str(x)) > 0,
            'json': lambda x: self._validate_json(x),
            'regex': lambda x: self._validate_regex(x),
            'positive_int': lambda x: isinstance(x, int) and x > 0,
            'non_negative_int': lambda x: isinstance(x, int) and x >= 0,
            'positive_float': lambda x: isinstance(x, (int, float)) and x > 0,
            'non_empty_string': lambda x: isinstance(x, str) and len(x.strip()) > 0,
            'safe_string': lambda x: self._validate_safe_string(x),
        })
    
    def _validate_json(self, value: Any) -> bool:
        """Validate JSON string"""
        try:
            if isinstance(value, str):
                json.loads(value)
                return True
            return isinstance(value, (dict, list))
        except (json.JSONDecodeError, TypeError):
            return False
    
    def _validate_regex(self, value: Any) -> bool:
        """Validate regex pattern"""
        try:
            import re
            re.compile(str(value))
            return True
        except re.error:
            return False
    
    def _validate_safe_string(self, value: Any) -> bool:
        """Validate string for safety (no injection attempts)"""
        if not isinstance(value, str):
            return False
        
        # Check for common injection patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
            r'import\s+',
            r'__import__',
            r'os\.',
            r'sys\.',
            r'subprocess\.',
            r'shell=True',
            r'rm\s+-rf',
            r'del\s+',
            r';.*rm\s+',
            r'\|.*rm\s+',
            r'&&.*rm\s+',
        ]
        
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        return True
    
    def add_validation_rule(self, tool_name: str, rule: ValidationRule):
        """Add custom validation rule for a tool"""
        if tool_name not in self.validation_rules:
            self.validation_rules[tool_name] = []
        self.validation_rules[tool_name].append(rule)
    
    def validate_parameter(self, param: ToolParameter, value: Any) -> Tuple[bool, str]:
        """Validate a single parameter"""
        try:
            # Check if parameter is required
            if param.required and (value is None or value == ""):
                return False, f"Parameter '{param.name}' is required"
            
            # Skip validation for optional parameters that are None
            if not param.required and value is None:
                return True, ""
            
            # Type validation
            if not self._validate_type(param, value):
                return False, f"Parameter '{param.name}' must be of type {param.type}"
            
            # Range validation for numeric types
            param_type = param.type.lower() if isinstance(param.type, str) else str(param.type).lower()
            if param_type in ["integer", "int", "float", "number"]:
                if not self._validate_range(param, value):
                    return False, f"Parameter '{param.name}' is out of valid range"
            
            # String length validation
            if param_type in ["string", "str"]:
                if not self._validate_string_length(param, value):
                    return False, f"Parameter '{param.name}' length is invalid"
            
            # Array validation
            if param_type in ["array", "list"]:
                if not self._validate_array(param, value):
                    return False, f"Parameter '{param.name}' array is invalid"
            
            # Custom validation (if available)
            if hasattr(param, 'validation_rules') and param.validation_rules:
                for rule in param.validation_rules:
                    if not rule(value):
                        return False, f"Parameter '{param.name}' failed custom validation"
            
            # Built-in validator check
            if param_type in self.custom_validators:
                if not self.custom_validators[param_type](value):
                    return False, f"Parameter '{param.name}' failed {param_type} validation"
            
            return True, ""
            
        except Exception as e:
            self.logger.error(f"Validation error for parameter '{param.name}': {e}")
            return False, f"Validation error: {str(e)}"
    
    def _validate_type(self, param: ToolParameter, value: Any) -> bool:
        """Validate parameter type"""
        param_type = param.type.lower() if isinstance(param.type, str) else str(param.type).lower()
        
        if param_type in ["string", "str"]:
            return isinstance(value, str)
        elif param_type in ["integer", "int"]:
            return isinstance(value, int)
        elif param_type in ["float", "number"]:
            return isinstance(value, (int, float))
        elif param_type in ["boolean", "bool"]:
            return isinstance(value, bool)
        elif param_type in ["array", "list"]:
            return isinstance(value, list)
        elif param_type in ["object", "dict"]:
            return isinstance(value, dict)
        elif param_type == "file_path":
            return isinstance(value, str) and len(value) > 0
        elif param_type == "url":
            return isinstance(value, str)
        elif param_type == "email":
            return isinstance(value, str)
        elif param_type == "regex":
            return isinstance(value, str)
        elif param_type == "json":
            return True  # Can be string or dict/list
        elif param_type == "datetime":
            return isinstance(value, (str, datetime))
        elif param_type == "enum":
            return value in param.allowed_values if hasattr(param, 'allowed_values') and param.allowed_values else True
        elif param_type == "binary":
            return isinstance(value, bytes)
        
        return True
    
    def _validate_range(self, param: ToolParameter, value: Any) -> bool:
        """Validate numeric range"""
        if hasattr(param, 'min_value') and param.min_value is not None:
            if value < param.min_value:
                return False
        
        if hasattr(param, 'max_value') and param.max_value is not None:
            if value > param.max_value:
                return False
        
        return True
    
    def _validate_string_length(self, param: ToolParameter, value: str) -> bool:
        """Validate string length"""
        if hasattr(param, 'min_length') and param.min_length is not None:
            if len(value) < param.min_length:
                return False
        
        if hasattr(param, 'max_length') and param.max_length is not None:
            if len(value) > param.max_length:
                return False
        
        return True
    
    def _validate_array(self, param: ToolParameter, value: list) -> bool:
        """Validate array parameters"""
        if hasattr(param, 'min_items') and param.min_items is not None:
            if len(value) < param.min_items:
                return False
        
        if hasattr(param, 'max_items') and param.max_items is not None:
            if len(value) > param.max_items:
                return False
        
        return True
    
    def validate_all_parameters(self, tool_metadata: ToolMetadata, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate all parameters for a tool"""
        errors = []
        
        # Validate each defined parameter
        for param in tool_metadata.parameters:
            value = parameters.get(param.name)
            is_valid, error_msg = self.validate_parameter(param, value)
            
            if not is_valid:
                errors.append(error_msg)
        
        # Check for unexpected parameters
        expected_params = {param.name for param in tool_metadata.parameters}
        provided_params = set(parameters.keys())
        unexpected_params = provided_params - expected_params
        
        if unexpected_params:
            errors.append(f"Unexpected parameters: {', '.join(unexpected_params)}")
        
        # Apply tool-specific validation rules
        if tool_metadata.name in self.validation_rules:
            for rule in self.validation_rules[tool_metadata.name]:
                param_value = parameters.get(rule.parameter_name)
                try:
                    if rule.is_required or param_value is not None:
                        if not rule.validator(param_value):
                            errors.append(rule.error_message)
                except Exception as e:
                    errors.append(f"Validation rule error for {rule.parameter_name}: {e}")
        
        return len(errors) == 0, errors


class ExecutionSandbox:
    """Secure execution environment with resource limits and isolation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_processes: Dict[str, psutil.Process] = {}
        self.resource_monitors: Dict[str, threading.Thread] = {}
        self.temp_dirs: Dict[str, str] = {}
    
    @contextmanager
    def create_execution_environment(self, context: ExecutionContext):
        """Create isolated execution environment"""
        env_id = context.execution_id
        temp_dir = None
        
        try:
            # Create temporary directory for this execution
            temp_dir = tempfile.mkdtemp(prefix=f"sovereign_tool_{env_id}_")
            self.temp_dirs[env_id] = temp_dir
            
            # Set up resource monitoring
            monitor_thread = threading.Thread(
                target=self._monitor_resources,
                args=(env_id, context),
                daemon=True
            )
            self.resource_monitors[env_id] = monitor_thread
            monitor_thread.start()
            
            self.logger.info(f"Created execution environment {env_id} with temp dir {temp_dir}")
            
            yield {
                'temp_dir': temp_dir,
                'env_id': env_id,
                'context': context
            }
            
        except Exception as e:
            self.logger.error(f"Error in execution environment {env_id}: {e}")
            raise
        finally:
            # Cleanup
            self._cleanup_environment(env_id, temp_dir)
    
    def _monitor_resources(self, env_id: str, context: ExecutionContext):
        """Monitor resource usage during execution"""
        try:
            start_time = time.time()
            
            while time.time() - start_time < context.timeout:
                # Check if execution is still running
                if context.state in [ExecutionState.COMPLETED, ExecutionState.FAILED, ExecutionState.CANCELLED]:
                    break
                
                # Monitor CPU and memory usage
                try:
                    current_process = psutil.Process()
                    cpu_percent = current_process.cpu_percent()
                    memory_mb = current_process.memory_info().rss / 1024 / 1024
                    
                    # Check limits
                    if cpu_percent > context.max_cpu_percent:
                        self.logger.warning(f"CPU limit exceeded for {env_id}: {cpu_percent}%")
                        context.state = ExecutionState.FAILED
                        context.error = f"CPU limit exceeded: {cpu_percent}%"
                        break
                    
                    if memory_mb > context.max_memory_mb:
                        self.logger.warning(f"Memory limit exceeded for {env_id}: {memory_mb}MB")
                        context.state = ExecutionState.FAILED
                        context.error = f"Memory limit exceeded: {memory_mb}MB"
                        break
                    
                except psutil.NoSuchProcess:
                    # Process ended
                    break
                except Exception as e:
                    self.logger.error(f"Error monitoring resources for {env_id}: {e}")
                    break
                
                time.sleep(0.1)  # Check every 100ms
            
            # Check for timeout
            if time.time() - start_time >= context.timeout:
                self.logger.warning(f"Execution timeout for {env_id}")
                context.state = ExecutionState.TIMEOUT
                context.error = f"Execution timeout after {context.timeout} seconds"
                
        except Exception as e:
            self.logger.error(f"Resource monitoring error for {env_id}: {e}")
    
    def _cleanup_environment(self, env_id: str, temp_dir: str):
        """Clean up execution environment"""
        try:
            # Stop resource monitoring
            if env_id in self.resource_monitors:
                monitor_thread = self.resource_monitors[env_id]
                if monitor_thread.is_alive():
                    # Thread will stop when execution completes
                    pass
                del self.resource_monitors[env_id]
            
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                self.logger.info(f"Cleaned up temp directory for {env_id}")
            
            # Remove from tracking
            if env_id in self.temp_dirs:
                del self.temp_dirs[env_id]
            
            if env_id in self.active_processes:
                del self.active_processes[env_id]
                
        except Exception as e:
            self.logger.error(f"Error cleaning up environment {env_id}: {e}")


class ResultFormatter:
    """Format execution results for AI consumption"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def format_result(self, result: ToolExecutionResult, context: ExecutionContext) -> ToolExecutionResult:
        """Format result for optimal AI consumption"""
        try:
            # Create enhanced result
            formatted_result = ToolExecutionResult(
                execution_id=context.execution_id,
                tool_name=context.tool_name,
                success=result.success,
                result=self._format_data(result.result),
                error=result.error,
                execution_time=result.execution_time,
                metadata=self._enhance_metadata(result.metadata, context)
            )
            
            # Add execution context
            formatted_result.metadata.update({
                'execution_id': context.execution_id,
                'tool_name': context.tool_name,
                'security_level': context.security_level.value,
                'execution_mode': context.execution_mode.value,
                'parameters_used': context.parameters,
                'timestamp': context.created_at.isoformat()
            })
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"Error formatting result: {e}")
            # Return original result if formatting fails
            return result
    
    def _format_data(self, data: Any) -> Any:
        """Format data for AI consumption"""
        if isinstance(data, dict):
            return {k: self._format_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._format_data(item) for item in data]
        elif isinstance(data, bytes):
            # Convert bytes to base64 for AI
            import base64
            return {
                'type': 'binary',
                'encoding': 'base64',
                'data': base64.b64encode(data).decode('utf-8')
            }
        elif isinstance(data, datetime):
            return {
                'type': 'datetime',
                'iso': data.isoformat(),
                'timestamp': data.timestamp()
            }
        else:
            return data
    
    def _enhance_metadata(self, metadata: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Enhance metadata with execution context"""
        enhanced = metadata.copy() if metadata else {}
        
        enhanced.update({
            'execution_stats': {
                'duration_ms': (context.completed_at - context.started_at).total_seconds() * 1000 if context.completed_at and context.started_at else 0,
                'state': context.state.value,
                'resource_usage': {
                    'max_memory_mb': context.max_memory_mb,
                    'max_cpu_percent': context.max_cpu_percent,
                    'timeout_seconds': context.timeout
                }
            },
            'security_context': {
                'level': context.security_level.value,
                'user_id': context.user_id,
                'session_id': context.session_id
            }
        })
        
        return enhanced


class EnhancedExecutionEngine:
    """Enhanced execution engine with advanced validation and security"""
    
    def __init__(self, config: Config, permission_manager: PermissionManager):
        self.config = config
        self.permission_manager = permission_manager
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.parameter_validator = ParameterValidator()
        self.execution_sandbox = ExecutionSandbox(config)
        self.result_formatter = ResultFormatter()
        
        # State tracking
        self.execution_contexts: Dict[str, ExecutionContext] = {}
        self.execution_stats = ExecutionStats()
        self.execution_history: List[ExecutionContext] = []
        
        # Thread pools for different execution modes
        self.sync_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sync_tool_")
        self.async_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="async_tool_")
        
        # Result caching
        self.result_cache: Dict[str, Tuple[ToolExecutionResult, datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        self.logger.info("Enhanced execution engine initialized")
    
    def generate_execution_id(self) -> str:
        """Generate unique execution ID"""
        return f"exec_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    
    def create_execution_context(self, tool_name: str, parameters: Dict[str, Any], 
                               user_id: str = None, session_id: str = None,
                               security_level: ToolSecurityLevel = ToolSecurityLevel.SAFE,
                               execution_mode: ToolExecutionMode = ToolExecutionMode.SYNCHRONOUS,
                               **kwargs) -> ExecutionContext:
        """Create execution context with security and monitoring"""
        context = ExecutionContext(
            execution_id=self.generate_execution_id(),
            tool_name=tool_name,
            parameters=parameters,
            user_id=user_id,
            session_id=session_id,
            security_level=security_level,
            execution_mode=execution_mode,
            **kwargs
        )
        
        self.execution_contexts[context.execution_id] = context
        return context
    
    def get_cached_result(self, cache_key: str) -> Optional[ToolExecutionResult]:
        """Get cached result if available and not expired"""
        if cache_key in self.result_cache:
            result, timestamp = self.result_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                self.logger.info(f"Cache hit for {cache_key}")
                return result
            else:
                # Remove expired cache entry
                del self.result_cache[cache_key]
        return None
    
    def cache_result(self, cache_key: str, result: ToolExecutionResult):
        """Cache execution result"""
        self.result_cache[cache_key] = (result, datetime.now())
        
        # Clean up old cache entries
        if len(self.result_cache) > 100:
            # Remove oldest entries
            sorted_cache = sorted(self.result_cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_cache[:50]:  # Keep newest 50
                del self.result_cache[key]
    
    def generate_cache_key(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key for tool execution"""
        # Create deterministic hash of tool name and parameters
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        cache_data = f"{tool_name}:{param_str}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    async def execute_tool_call(self, tool_name: str, parameters: Dict[str, Any], 
                              tool_instance: BaseTool, tool_metadata: ToolMetadata,
                              **context_kwargs) -> ToolExecutionResult:
        """Execute tool call with comprehensive validation and security"""
        context = self.create_execution_context(tool_name, parameters, **context_kwargs)
        
        try:
            self.logger.info(f"Starting tool execution: {tool_name} [{context.execution_id}]")
            context.state = ExecutionState.VALIDATING
            context.started_at = datetime.now()
            
            # Step 1: Permission validation
            if not await self._validate_permissions(context, tool_metadata):
                raise SecurityError(f"Permission denied for tool {tool_name}")
            
            # Step 2: Parameter validation
            if not await self._validate_parameters(context, tool_metadata):
                raise ValidationError(f"Parameter validation failed for tool {tool_name}")
            
            # Step 3: Check cache if applicable
            if hasattr(tool_metadata, 'cacheable') and tool_metadata.cacheable:
                cache_key = self.generate_cache_key(tool_name, parameters)
                cached_result = self.get_cached_result(cache_key)
                if cached_result:
                    context.state = ExecutionState.COMPLETED
                    context.completed_at = datetime.now()
                    return cached_result
            
            # Step 4: Execute tool
            context.state = ExecutionState.EXECUTING
            result = await self._execute_tool_safely(context, tool_instance, tool_metadata)
            
            # Step 5: Format result
            formatted_result = self.result_formatter.format_result(result, context)
            
            # Step 6: Cache result if applicable
            if hasattr(tool_metadata, 'cacheable') and tool_metadata.cacheable and result.success:
                cache_key = self.generate_cache_key(tool_name, parameters)
                self.cache_result(cache_key, formatted_result)
            
            # Step 7: Update context and stats
            context.state = ExecutionState.COMPLETED
            context.completed_at = datetime.now()
            context.result = formatted_result
            
            self.execution_stats.update(context)
            self.execution_history.append(context)
            
            self.logger.info(f"Tool execution completed successfully: {tool_name} [{context.execution_id}]")
            return formatted_result
            
        except Exception as e:
            # Handle execution failure
            error_msg = str(e)
            context.state = ExecutionState.FAILED
            context.error = error_msg
            context.completed_at = datetime.now()
            
            self.execution_stats.update(context)
            self.execution_history.append(context)
            
            self.logger.error(f"Tool execution failed: {tool_name} [{context.execution_id}]: {error_msg}")
            
            # Return error result
            return ToolExecutionResult(
                execution_id=context.execution_id,
                tool_name=tool_name,
                success=False,
                result=None,
                error=error_msg,
                execution_time=(context.completed_at - context.started_at).total_seconds() if context.started_at else 0,
                metadata={
                    'execution_id': context.execution_id,
                    'tool_name': tool_name,
                    'error_type': type(e).__name__,
                    'context': context.__dict__
                }
            )
    
    async def _validate_permissions(self, context: ExecutionContext, tool_metadata: ToolMetadata) -> bool:
        """Validate permissions for tool execution"""
        try:
            # Check if user has permission to execute this tool
            has_permission, permission_error = self.permission_manager.check_permission(
                tool_name=context.tool_name,
                requester=context.user_id or "system",
                tool_metadata=tool_metadata
            )
            
            if not has_permission:
                self.logger.warning(f"Permission denied for user {context.user_id} to execute {context.tool_name}: {permission_error}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating permissions: {e}")
            return False
    
    async def _validate_parameters(self, context: ExecutionContext, tool_metadata: ToolMetadata) -> bool:
        """Validate parameters for tool execution"""
        try:
            is_valid, errors = self.parameter_validator.validate_all_parameters(
                tool_metadata, context.parameters
            )
            
            if not is_valid:
                error_msg = "; ".join(errors)
                context.error = f"Parameter validation failed: {error_msg}"
                self.logger.warning(f"Parameter validation failed for {context.tool_name}: {error_msg}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating parameters: {e}")
            context.error = f"Parameter validation error: {str(e)}"
            return False
    
    async def _execute_tool_safely(self, context: ExecutionContext, tool_instance: BaseTool, 
                                 tool_metadata: ToolMetadata) -> ToolExecutionResult:
        """Execute tool in safe environment with monitoring"""
        with self.execution_sandbox.create_execution_environment(context) as env:
            try:
                # Choose execution strategy based on mode
                if context.execution_mode == ToolExecutionMode.SYNCHRONOUS:
                    return await self._execute_sync(context, tool_instance, env)
                elif context.execution_mode == ToolExecutionMode.ASYNCHRONOUS:
                    return await self._execute_async(context, tool_instance, env)
                else:
                    # Default to synchronous
                    return await self._execute_sync(context, tool_instance, env)
                    
            except asyncio.TimeoutError:
                raise TimeoutError(f"Tool execution timed out after {context.timeout} seconds")
            except Exception as e:
                raise ExecutionError(f"Tool execution failed: {str(e)}")
    
    async def _execute_sync(self, context: ExecutionContext, tool_instance: BaseTool, env: Dict) -> ToolExecutionResult:
        """Execute tool synchronously"""
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                tool_instance.execute(context.parameters),
                timeout=context.timeout
            )
            
            # If result is already a ToolExecutionResult, return it
            if isinstance(result, ToolExecutionResult):
                return result
            
            # Otherwise, wrap raw result in ToolExecutionResult
            return ToolExecutionResult(
                execution_id=context.execution_id,
                tool_name=context.tool_name,
                success=True,
                result=result,
                execution_time=(datetime.now() - context.started_at).total_seconds() if context.started_at else 0
            )
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Synchronous execution timeout")
        except Exception as e:
            raise ExecutionError(f"Synchronous execution failed: {str(e)}")
    
    async def _execute_async(self, context: ExecutionContext, tool_instance: BaseTool, env: Dict) -> ToolExecutionResult:
        """Execute tool asynchronously"""
        try:
            # Execute in async thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.async_executor,
                functools.partial(
                    asyncio.run,
                    tool_instance.execute(context.parameters)
                )
            )
            
            # If result is already a ToolExecutionResult, return it
            if isinstance(result, ToolExecutionResult):
                return result
            
            # Otherwise, wrap raw result in ToolExecutionResult
            return ToolExecutionResult(
                execution_id=context.execution_id,
                tool_name=context.tool_name,
                success=True,
                result=result,
                execution_time=(datetime.now() - context.started_at).total_seconds() if context.started_at else 0
            )
            
        except Exception as e:
            raise ExecutionError(f"Asynchronous execution failed: {str(e)}")
    
    def get_execution_stats(self) -> ExecutionStats:
        """Get current execution statistics"""
        return self.execution_stats
    
    def get_execution_history(self, limit: int = 100) -> List[ExecutionContext]:
        """Get execution history"""
        return self.execution_history[-limit:]
    
    def get_active_executions(self) -> List[ExecutionContext]:
        """Get currently active executions"""
        return [
            context for context in self.execution_contexts.values()
            if context.state in [ExecutionState.QUEUED, ExecutionState.VALIDATING, ExecutionState.EXECUTING]
        ]
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel active execution"""
        if execution_id in self.execution_contexts:
            context = self.execution_contexts[execution_id]
            if context.state in [ExecutionState.QUEUED, ExecutionState.VALIDATING, ExecutionState.EXECUTING]:
                context.state = ExecutionState.CANCELLED
                context.completed_at = datetime.now()
                self.logger.info(f"Execution cancelled: {execution_id}")
                return True
        return False
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Shutdown thread pools
            self.sync_executor.shutdown(wait=True)
            self.async_executor.shutdown(wait=True)
            
            # Cancel active executions
            for context in self.get_active_executions():
                self.cancel_execution(context.execution_id)
            
            # Clear caches
            self.result_cache.clear()
            self.execution_contexts.clear()
            
            self.logger.info("Execution engine cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}") 