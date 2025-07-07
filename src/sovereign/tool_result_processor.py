"""
Tool Result Processor - Advanced result parsing, formatting, and logging system

This module provides comprehensive functionality for parsing tool requests from AI models,
formatting results for AI consumption, and implementing detailed logging for auditing
and monitoring of tool usage within the Sovereign AI system.
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import traceback
import hashlib
import uuid
from pathlib import Path
import asyncio

from .config import Config
from .logger import setup_logger, get_performance_logger
from .tool_integration_framework import (
    ToolExecutionResult, ToolExecutionRequest, ToolMetadata, 
    ToolType, ToolSecurityLevel, ToolExecutionError
)


class LogLevel(Enum):
    """Logging levels for tool operations"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEventType(Enum):
    """Types of audit events"""
    TOOL_EXECUTION = "tool_execution"
    TOOL_REQUEST = "tool_request"
    PERMISSION_CHECK = "permission_check"
    SECURITY_VIOLATION = "security_violation"
    PARAMETER_VALIDATION = "parameter_validation"
    RATE_LIMIT_HIT = "rate_limit_hit"
    SYSTEM_ERROR = "system_error"
    USER_ACTION = "user_action"


@dataclass
class AuditEvent:
    """Audit event record"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.TOOL_EXECUTION
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    tool_name: Optional[str] = None
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    severity: LogLevel = LogLevel.INFO
    execution_id: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'tool_name': self.tool_name,
            'action': self.action,
            'details': self.details,
            'severity': self.severity.value,
            'execution_id': self.execution_id,
            'success': self.success,
            'error_message': self.error_message,
            'duration_ms': self.duration_ms
        }


@dataclass
class ToolCallRequest:
    """Parsed tool call request from AI model"""
    tool_name: str
    parameters: Dict[str, Any]
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    confidence: float = 1.0
    raw_request: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'tool_name': self.tool_name,
            'parameters': self.parameters,
            'request_id': self.request_id,
            'confidence': self.confidence,
            'raw_request': self.raw_request,
            'metadata': self.metadata
        }


@dataclass
class FormattedResult:
    """Formatted result for AI consumption"""
    success: bool
    content: str
    structured_data: Dict[str, Any]
    summary: str
    confidence: float = 1.0
    format_type: str = "standard"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'content': self.content,
            'structured_data': self.structured_data,
            'summary': self.summary,
            'confidence': self.confidence,
            'format_type': self.format_type,
            'metadata': self.metadata
        }


class ToolRequestParser:
    """Parser for extracting tool requests from AI model outputs"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("tool_request_parser")
        
        # Common patterns for tool call detection
        self.patterns = {
            'json_tool_call': re.compile(
                r'(?:```json\s*)?{\s*"tool_name":\s*"([^"]+)"[^}]*"parameters":\s*({[^}]*})[^}]*}(?:\s*```)?',
                re.IGNORECASE | re.DOTALL
            ),
            'function_call': re.compile(
                r'(?:call_tool|execute_tool|use_tool)\s*\(\s*["\']([^"\']+)["\'](?:\s*,\s*({[^}]*}))?\s*\)',
                re.IGNORECASE | re.DOTALL
            ),
            'xml_tool_call': re.compile(
                r'<tool_call>\s*<name>([^<]+)</name>\s*<parameters>([^<]*)</parameters>\s*</tool_call>',
                re.IGNORECASE | re.DOTALL
            ),
            'natural_language': re.compile(
                r'(?:please |can you |i need to |let me )?(search|calculate|read|write|get|find|list|check|delete|create)(?:\s+(?:for|the|a|an))?\s+(.+)',
                re.IGNORECASE
            )
        }
        
        # Tool name mappings for natural language
        self.tool_mappings = {
            'search': 'internet_search',
            'calculate': 'advanced_calculation',
            'read': 'file_access',
            'write': 'file_access',
            'get': 'enhanced_system_info',
            'find': 'internet_search',
            'list': 'file_access',
            'check': 'enhanced_system_info',
            'delete': 'file_access',
            'create': 'file_access'
        }
    
    def parse_tool_request(self, model_output: str) -> List[ToolCallRequest]:
        """
        Parse tool requests from AI model output
        
        Args:
            model_output: Raw output from AI model
            
        Returns:
            List of parsed tool call requests
        """
        requests = []
        
        try:
            # Try JSON format first
            json_requests = self._parse_json_format(model_output)
            requests.extend(json_requests)
            
            # Try function call format
            if not requests:
                function_requests = self._parse_function_format(model_output)
                requests.extend(function_requests)
            
            # Try XML format
            if not requests:
                xml_requests = self._parse_xml_format(model_output)
                requests.extend(xml_requests)
            
            # Try natural language format
            if not requests:
                nl_requests = self._parse_natural_language(model_output)
                requests.extend(nl_requests)
            
            # Log parsing results
            self.logger.info(f"Parsed {len(requests)} tool requests from model output")
            
        except Exception as e:
            self.logger.error(f"Error parsing tool requests: {e}")
            
        return requests
    
    def _parse_json_format(self, text: str) -> List[ToolCallRequest]:
        """Parse JSON format tool calls"""
        requests = []
        
        for match in self.patterns['json_tool_call'].finditer(text):
            try:
                tool_name = match.group(1)
                parameters_text = match.group(2)
                
                # Parse parameters JSON
                parameters = json.loads(parameters_text)
                
                request = ToolCallRequest(
                    tool_name=tool_name,
                    parameters=parameters,
                    confidence=0.9,
                    raw_request=match.group(0),
                    metadata={'format': 'json'}
                )
                requests.append(request)
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON parameters: {e}")
                continue
        
        return requests
    
    def _parse_function_format(self, text: str) -> List[ToolCallRequest]:
        """Parse function call format"""
        requests = []
        
        for match in self.patterns['function_call'].finditer(text):
            try:
                tool_name = match.group(1)
                parameters_text = match.group(2) or "{}"
                
                # Parse parameters
                parameters = json.loads(parameters_text)
                
                request = ToolCallRequest(
                    tool_name=tool_name,
                    parameters=parameters,
                    confidence=0.8,
                    raw_request=match.group(0),
                    metadata={'format': 'function'}
                )
                requests.append(request)
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse function parameters: {e}")
                continue
        
        return requests
    
    def _parse_xml_format(self, text: str) -> List[ToolCallRequest]:
        """Parse XML format tool calls"""
        requests = []
        
        for match in self.patterns['xml_tool_call'].finditer(text):
            try:
                tool_name = match.group(1)
                parameters_text = match.group(2) or "{}"
                
                # Try to parse as JSON, fallback to key-value pairs
                try:
                    parameters = json.loads(parameters_text)
                except json.JSONDecodeError:
                    # Parse as key-value pairs
                    parameters = self._parse_key_value_params(parameters_text)
                
                request = ToolCallRequest(
                    tool_name=tool_name,
                    parameters=parameters,
                    confidence=0.7,
                    raw_request=match.group(0),
                    metadata={'format': 'xml'}
                )
                requests.append(request)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse XML tool call: {e}")
                continue
        
        return requests
    
    def _parse_natural_language(self, text: str) -> List[ToolCallRequest]:
        """Parse natural language tool requests"""
        requests = []
        
        for match in self.patterns['natural_language'].finditer(text):
            try:
                action = match.group(1).lower()
                query = match.group(2).strip()
                
                # Map action to tool name
                tool_name = self.tool_mappings.get(action, 'internet_search')
                
                # Generate parameters based on action
                parameters = self._generate_parameters_from_nl(action, query)
                
                request = ToolCallRequest(
                    tool_name=tool_name,
                    parameters=parameters,
                    confidence=0.6,
                    raw_request=match.group(0),
                    metadata={'format': 'natural_language', 'action': action}
                )
                requests.append(request)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse natural language request: {e}")
                continue
        
        return requests
    
    def _parse_key_value_params(self, params_text: str) -> Dict[str, Any]:
        """Parse key=value parameter format"""
        parameters = {}
        
        # Split by common delimiters
        pairs = re.split(r'[,;\n]', params_text)
        
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                
                # Try to convert to appropriate type
                try:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif value.isdigit():
                        value = int(value)
                    elif '.' in value and value.replace('.', '').isdigit():
                        value = float(value)
                except:
                    pass  # Keep as string
                
                parameters[key] = value
        
        return parameters
    
    def _generate_parameters_from_nl(self, action: str, query: str) -> Dict[str, Any]:
        """Generate parameters from natural language"""
        if action == 'search':
            return {'query': query}
        elif action == 'calculate':
            return {'operation': 'expression', 'expression': query}
        elif action in ['read', 'write', 'list', 'delete', 'create']:
            return {'operation': action, 'path': query}
        elif action in ['get', 'check']:
            return {'category': 'overview'}
        else:
            return {'query': query}


class ResultFormatter:
    """Formatter for tool results optimized for AI consumption"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("result_formatter")
        
        # Formatting templates
        self.templates = {
            'success': "✅ **{tool_name}** executed successfully:\n\n{content}\n\n📊 **Summary:** {summary}",
            'error': "❌ **{tool_name}** failed:\n\n**Error:** {error}\n\n🔍 **Details:** {details}",
            'partial': "⚠️ **{tool_name}** partially successful:\n\n{content}\n\n⚠️ **Issues:** {issues}",
            'data_table': "📋 **{title}**\n\n{table}\n\n📊 **Total:** {count} items",
            'code_block': "```{language}\n{code}\n```",
            'json_data': "```json\n{data}\n```"
        }
    
    def format_result(self, result: ToolExecutionResult, format_type: str = "standard") -> FormattedResult:
        """
        Format tool execution result for AI consumption
        
        Args:
            result: Tool execution result
            format_type: Format type (standard, compact, detailed, json)
            
        Returns:
            Formatted result optimized for AI
        """
        try:
            if format_type == "compact":
                return self._format_compact(result)
            elif format_type == "detailed":
                return self._format_detailed(result)
            elif format_type == "json":
                return self._format_json(result)
            else:
                return self._format_standard(result)
                
        except Exception as e:
            self.logger.error(f"Error formatting result: {e}")
            return self._format_error_fallback(result, str(e))
    
    def _format_standard(self, result: ToolExecutionResult) -> FormattedResult:
        """Standard formatting for general use"""
        if result.success:
            content = self._format_success_content(result)
            summary = self._generate_summary(result)
            
            formatted_content = self.templates['success'].format(
                tool_name=result.tool_name,
                content=content,
                summary=summary
            )
            
            return FormattedResult(
                success=True,
                content=formatted_content,
                structured_data=result.result if isinstance(result.result, dict) else {},
                summary=summary,
                format_type="standard"
            )
        else:
            error_content = self._format_error_content(result)
            
            formatted_content = self.templates['error'].format(
                tool_name=result.tool_name,
                error=result.error or "Unknown error",
                details=error_content
            )
            
            return FormattedResult(
                success=False,
                content=formatted_content,
                structured_data={},
                summary=f"Tool {result.tool_name} failed: {result.error}",
                format_type="standard"
            )
    
    def _format_compact(self, result: ToolExecutionResult) -> FormattedResult:
        """Compact formatting for space-constrained contexts"""
        if result.success:
            summary = self._generate_summary(result, max_length=100)
            content = f"✅ {result.tool_name}: {summary}"
        else:
            content = f"❌ {result.tool_name}: {result.error}"
        
        return FormattedResult(
            success=result.success,
            content=content,
            structured_data=result.result if isinstance(result.result, dict) else {},
            summary=summary if result.success else result.error,
            format_type="compact"
        )
    
    def _format_detailed(self, result: ToolExecutionResult) -> FormattedResult:
        """Detailed formatting with full information"""
        content_parts = []
        
        # Header
        status_icon = "✅" if result.success else "❌"
        content_parts.append(f"{status_icon} **Tool Execution Report: {result.tool_name}**")
        content_parts.append(f"📋 **Execution ID:** {result.execution_id}")
        content_parts.append(f"⏱️ **Duration:** {result.execution_time:.2f}s")
        content_parts.append(f"📅 **Timestamp:** {result.created_at.isoformat()}")
        
        # Result content
        if result.success:
            content_parts.append("\n📊 **Results:**")
            content_parts.append(self._format_success_content(result))
        else:
            content_parts.append(f"\n❌ **Error:** {result.error}")
            content_parts.append(self._format_error_content(result))
        
        # Metadata
        if result.metadata:
            content_parts.append("\n📋 **Metadata:**")
            content_parts.append(json.dumps(result.metadata, indent=2))
        
        # Logs
        if result.logs:
            content_parts.append("\n📝 **Logs:**")
            content_parts.extend(f"  • {log}" for log in result.logs)
        
        formatted_content = "\n".join(content_parts)
        summary = self._generate_summary(result)
        
        return FormattedResult(
            success=result.success,
            content=formatted_content,
            structured_data=result.result if isinstance(result.result, dict) else {},
            summary=summary,
            format_type="detailed"
        )
    
    def _format_json(self, result: ToolExecutionResult) -> FormattedResult:
        """JSON formatting for structured data exchange"""
        data = result.to_dict()
        
        content = self.templates['json_data'].format(
            data=json.dumps(data, indent=2, default=str)
        )
        
        return FormattedResult(
            success=result.success,
            content=content,
            structured_data=data,
            summary=f"JSON data for {result.tool_name} execution",
            format_type="json"
        )
    
    def _format_success_content(self, result: ToolExecutionResult) -> str:
        """Format successful result content"""
        if result.result is None:
            return "Operation completed successfully (no data returned)"
        
        if isinstance(result.result, dict):
            return self._format_dict_content(result.result)
        elif isinstance(result.result, list):
            return self._format_list_content(result.result)
        elif isinstance(result.result, str):
            return result.result
        else:
            return str(result.result)
    
    def _format_dict_content(self, data: Dict[str, Any]) -> str:
        """Format dictionary content"""
        content_parts = []
        
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                content_parts.append(f"**{key}:**")
                content_parts.append(self.templates['json_data'].format(
                    data=json.dumps(value, indent=2, default=str)
                ))
            else:
                content_parts.append(f"**{key}:** {value}")
        
        return "\n".join(content_parts)
    
    def _format_list_content(self, data: List[Any]) -> str:
        """Format list content"""
        if not data:
            return "No items found"
        
        if len(data) == 1:
            return f"Found 1 item:\n• {data[0]}"
        
        content_parts = [f"Found {len(data)} items:"]
        for i, item in enumerate(data[:10]):  # Show first 10 items
            content_parts.append(f"• {item}")
        
        if len(data) > 10:
            content_parts.append(f"... and {len(data) - 10} more items")
        
        return "\n".join(content_parts)
    
    def _format_error_content(self, result: ToolExecutionResult) -> str:
        """Format error content"""
        details = []
        
        if result.metadata:
            details.append(f"Metadata: {json.dumps(result.metadata, indent=2)}")
        
        if result.logs:
            details.append("Logs:")
            details.extend(f"  • {log}" for log in result.logs)
        
        return "\n".join(details) if details else "No additional details available"
    
    def _generate_summary(self, result: ToolExecutionResult, max_length: int = 200) -> str:
        """Generate a concise summary of the result"""
        if not result.success:
            return f"Tool execution failed: {result.error}"
        
        if isinstance(result.result, dict):
            key_count = len(result.result)
            return f"Returned {key_count} data fields in {result.execution_time:.2f}s"
        elif isinstance(result.result, list):
            item_count = len(result.result)
            return f"Returned {item_count} items in {result.execution_time:.2f}s"
        elif isinstance(result.result, str):
            text = result.result[:max_length]
            if len(result.result) > max_length:
                text += "..."
            return f"Returned text content ({len(result.result)} chars) in {result.execution_time:.2f}s"
        else:
            return f"Returned {type(result.result).__name__} in {result.execution_time:.2f}s"
    
    def _format_error_fallback(self, result: ToolExecutionResult, format_error: str) -> FormattedResult:
        """Fallback formatting when primary formatting fails"""
        content = f"❌ **{result.tool_name}** - Formatting Error\n\n**Original Error:** {result.error}\n**Format Error:** {format_error}"
        
        return FormattedResult(
            success=False,
            content=content,
            structured_data={},
            summary=f"Tool {result.tool_name} failed with formatting error",
            format_type="error_fallback"
        )


class ToolAuditLogger:
    """Comprehensive audit logging system for tool operations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("tool_audit")
        
        # Set up audit file logging - use database path as base directory
        base_dir = Path(config.database.db_path).parent
        self.audit_file = base_dir / "logs" / "tool_audit.jsonl"
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_logger = get_performance_logger()
        
        # Audit statistics
        self.stats = {
            'total_events': 0,
            'events_by_type': {},
            'events_by_tool': {},
            'error_count': 0,
            'security_violations': 0,
            'avg_execution_time': 0.0
        }
    
    def log_tool_execution(self, 
                          tool_name: str, 
                          parameters: Dict[str, Any], 
                          result: ToolExecutionResult,
                          user_id: str = "system") -> None:
        """Log tool execution event"""
        event = AuditEvent(
            event_type=AuditEventType.TOOL_EXECUTION,
            user_id=user_id,
            tool_name=tool_name,
            action="execute",
            details={
                'parameters': parameters,
                'result_success': result.success,
                'result_metadata': result.metadata,
                'logs': result.logs
            },
            severity=LogLevel.INFO if result.success else LogLevel.ERROR,
            execution_id=result.execution_id,
            success=result.success,
            error_message=result.error,
            duration_ms=result.execution_time * 1000
        )
        
        self._write_audit_event(event)
    
    def log_tool_request(self, 
                        request: ToolCallRequest, 
                        user_id: str = "system") -> None:
        """Log tool request event"""
        event = AuditEvent(
            event_type=AuditEventType.TOOL_REQUEST,
            user_id=user_id,
            tool_name=request.tool_name,
            action="request",
            details={
                'parameters': request.parameters,
                'confidence': request.confidence,
                'raw_request': request.raw_request,
                'metadata': request.metadata
            },
            severity=LogLevel.INFO,
            execution_id=request.request_id,
            success=True
        )
        
        self._write_audit_event(event)
    
    def log_permission_check(self, 
                           tool_name: str, 
                           user_id: str, 
                           granted: bool, 
                           reason: str = "") -> None:
        """Log permission check event"""
        event = AuditEvent(
            event_type=AuditEventType.PERMISSION_CHECK,
            user_id=user_id,
            tool_name=tool_name,
            action="permission_check",
            details={
                'granted': granted,
                'reason': reason
            },
            severity=LogLevel.INFO if granted else LogLevel.WARNING,
            success=granted,
            error_message=reason if not granted else None
        )
        
        self._write_audit_event(event)
    
    def log_security_violation(self, 
                             tool_name: str, 
                             user_id: str, 
                             violation_type: str, 
                             details: Dict[str, Any]) -> None:
        """Log security violation event"""
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_VIOLATION,
            user_id=user_id,
            tool_name=tool_name,
            action="security_violation",
            details={
                'violation_type': violation_type,
                **details
            },
            severity=LogLevel.CRITICAL,
            success=False,
            error_message=f"Security violation: {violation_type}"
        )
        
        self._write_audit_event(event)
        self.stats['security_violations'] += 1
    
    def log_rate_limit_hit(self, 
                          tool_name: str, 
                          user_id: str, 
                          limit_type: str) -> None:
        """Log rate limit hit event"""
        event = AuditEvent(
            event_type=AuditEventType.RATE_LIMIT_HIT,
            user_id=user_id,
            tool_name=tool_name,
            action="rate_limit_hit",
            details={
                'limit_type': limit_type
            },
            severity=LogLevel.WARNING,
            success=False,
            error_message=f"Rate limit exceeded: {limit_type}"
        )
        
        self._write_audit_event(event)
    
    def log_system_error(self, 
                        error_type: str, 
                        error_message: str, 
                        details: Dict[str, Any]) -> None:
        """Log system error event"""
        event = AuditEvent(
            event_type=AuditEventType.SYSTEM_ERROR,
            action="system_error",
            details={
                'error_type': error_type,
                **details
            },
            severity=LogLevel.ERROR,
            success=False,
            error_message=error_message
        )
        
        self._write_audit_event(event)
    
    def log_user_action(self, 
                       user_id: str, 
                       action: str, 
                       details: Dict[str, Any]) -> None:
        """Log user action event"""
        event = AuditEvent(
            event_type=AuditEventType.USER_ACTION,
            user_id=user_id,
            action=action,
            details=details,
            severity=LogLevel.INFO,
            success=True
        )
        
        self._write_audit_event(event)
    
    def _write_audit_event(self, event: AuditEvent) -> None:
        """Write audit event to file and update statistics"""
        try:
            # Write to audit file
            with open(self.audit_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
            
            # Update statistics
            self.stats['total_events'] += 1
            
            event_type = event.event_type.value
            self.stats['events_by_type'][event_type] = \
                self.stats['events_by_type'].get(event_type, 0) + 1
            
            if event.tool_name:
                self.stats['events_by_tool'][event.tool_name] = \
                    self.stats['events_by_tool'].get(event.tool_name, 0) + 1
            
            if not event.success:
                self.stats['error_count'] += 1
            
            if event.duration_ms:
                # Update average execution time
                current_avg = self.stats['avg_execution_time']
                total_events = self.stats['total_events']
                self.stats['avg_execution_time'] = \
                    (current_avg * (total_events - 1) + event.duration_ms) / total_events
            
            # Log to standard logger
            log_level = getattr(logging, event.severity.value.upper())
            self.logger.log(log_level, 
                          f"Audit: {event.event_type.value} - {event.action} - {event.tool_name or 'N/A'}")
            
        except Exception as e:
            self.logger.error(f"Failed to write audit event: {e}")
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit statistics"""
        return {
            **self.stats,
            'audit_file': str(self.audit_file),
            'file_size_mb': self.audit_file.stat().st_size / (1024 * 1024) if self.audit_file.exists() else 0
        }
    
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit events"""
        events = []
        
        try:
            if self.audit_file.exists():
                with open(self.audit_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # Get last 'limit' lines
                for line in lines[-limit:]:
                    try:
                        event = json.loads(line.strip())
                        events.append(event)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            self.logger.error(f"Failed to read audit events: {e}")
        
        return events
    
    def cleanup_old_events(self, days_to_keep: int = 30) -> int:
        """Clean up old audit events"""
        if not self.audit_file.exists():
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        kept_events = []
        removed_count = 0
        
        try:
            with open(self.audit_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        event_date = datetime.fromisoformat(event['timestamp'])
                        
                        if event_date > cutoff_date:
                            kept_events.append(line)
                        else:
                            removed_count += 1
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
            
            # Rewrite file with kept events
            with open(self.audit_file, 'w', encoding='utf-8') as f:
                f.writelines(kept_events)
            
            self.logger.info(f"Cleaned up {removed_count} old audit events")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup audit events: {e}")
            return 0


class ToolResultProcessor:
    """Main processor class that coordinates parsing, formatting, and logging"""
    
    def __init__(self, config: Config):
        self.config = config
        self.parser = ToolRequestParser(config)
        self.formatter = ResultFormatter(config)
        self.audit_logger = ToolAuditLogger(config)
        self.logger = logging.getLogger("tool_result_processor")
        
        # Processing statistics
        self.stats = {
            'requests_processed': 0,
            'results_formatted': 0,
            'parsing_errors': 0,
            'formatting_errors': 0,
            'avg_processing_time': 0.0
        }
    
    def parse_tool_request(self, model_output: str, user_id: str = "system") -> List[ToolCallRequest]:
        """Parse tool requests from model output with logging"""
        start_time = time.time()
        
        try:
            requests = self.parser.parse_tool_request(model_output)
            
            # Log each request
            for request in requests:
                self.audit_logger.log_tool_request(request, user_id)
            
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time)
            
            return requests
            
        except Exception as e:
            self.logger.error(f"Error processing tool request: {e}")
            self.stats['parsing_errors'] += 1
            self.audit_logger.log_system_error(
                "parsing_error", 
                str(e), 
                {"model_output": model_output[:1000]}
            )
            return []
    
    def format_tool_result(self, 
                          result: ToolExecutionResult, 
                          format_type: str = "standard",
                          user_id: str = "system") -> FormattedResult:
        """Format tool result with logging"""
        start_time = time.time()
        
        try:
            formatted_result = self.formatter.format_result(result, format_type)
            
            # Log the formatting
            self.audit_logger.log_user_action(
                user_id, 
                "format_result", 
                {
                    "tool_name": result.tool_name,
                    "format_type": format_type,
                    "success": result.success
                }
            )
            
            processing_time = time.time() - start_time
            self._update_formatting_stats(processing_time)
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"Error formatting tool result: {e}")
            self.stats['formatting_errors'] += 1
            self.audit_logger.log_system_error(
                "formatting_error", 
                str(e), 
                {"tool_name": result.tool_name, "format_type": format_type}
            )
            return self.formatter._format_error_fallback(result, str(e))
    
    def _update_processing_stats(self, processing_time: float) -> None:
        """Update processing statistics"""
        self.stats['requests_processed'] += 1
        current_avg = self.stats['avg_processing_time']
        count = self.stats['requests_processed']
        self.stats['avg_processing_time'] = \
            (current_avg * (count - 1) + processing_time) / count
    
    def _update_formatting_stats(self, processing_time: float) -> None:
        """Update formatting statistics"""
        self.stats['results_formatted'] += 1
        # Note: We could add formatting-specific timing if needed
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.stats,
            'audit_stats': self.audit_logger.get_audit_statistics()
        }
    
    def get_recent_audit_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit events"""
        return self.audit_logger.get_recent_events(limit)
    
    def cleanup_old_audit_data(self, days_to_keep: int = 30) -> int:
        """Clean up old audit data"""
        return self.audit_logger.cleanup_old_events(days_to_keep)


# Public API functions for easy integration
def parse_tool_request(model_output: str, config: Config = None) -> List[ToolCallRequest]:
    """
    Parse tool requests from AI model output
    
    Args:
        model_output: Raw output from AI model
        config: Optional configuration object
        
    Returns:
        List of parsed tool call requests
    """
    if config is None:
        config = Config()
    
    processor = ToolResultProcessor(config)
    return processor.parse_tool_request(model_output)


def format_tool_result(result: ToolExecutionResult, 
                      format_type: str = "standard",
                      config: Config = None) -> FormattedResult:
    """
    Format tool execution result for AI consumption
    
    Args:
        result: Tool execution result
        format_type: Format type (standard, compact, detailed, json)
        config: Optional configuration object
        
    Returns:
        Formatted result
    """
    if config is None:
        config = Config()
    
    processor = ToolResultProcessor(config)
    return processor.format_tool_result(result, format_type)


def create_audit_logger(config: Config) -> ToolAuditLogger:
    """
    Create a tool audit logger instance
    
    Args:
        config: Configuration object
        
    Returns:
        ToolAuditLogger instance
    """
    return ToolAuditLogger(config) 