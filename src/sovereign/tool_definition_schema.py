"""
Standardized Tool Definition Format - Comprehensive schema and validation system
"""

import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Type, Callable, Tuple
from pathlib import Path

import jsonschema
from jsonschema import validate, ValidationError


class ToolCategory(Enum):
    """Standard tool categories for classification"""
    API = "api"
    FILE_SYSTEM = "file_system"
    DATA_PROCESSING = "data_processing"
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    COMPUTATION = "computation"
    COMMUNICATION = "communication"
    SECURITY = "security"
    MULTIMEDIA = "multimedia"
    DEVELOPMENT = "development"
    CUSTOM = "custom"


class SecurityLevel(Enum):
    """Security access levels for tools"""
    PUBLIC = "public"          # No special permissions required
    PROTECTED = "protected"    # User consent required
    PRIVATE = "private"        # Explicit authorization required
    SYSTEM = "system"          # Administrative privileges required


class CapabilityLevel(Enum):
    """Tool capability classifications"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    NETWORK_ACCESS = "network_access"
    SYSTEM_ACCESS = "system_access"
    FILE_ACCESS = "file_access"
    PRIVILEGED = "privileged"


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
    DATETIME = "datetime"
    JSON = "json"
    BINARY = "binary"
    ENUM = "enum"


@dataclass
class ParameterConstraint:
    """Parameter validation constraints"""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    enum_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    error_message: Optional[str] = None


@dataclass
class ParameterDefinition:
    """Standardized parameter definition"""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default_value: Optional[Any] = None
    constraints: Optional[ParameterConstraint] = None
    examples: Optional[List[Any]] = None
    sensitive: bool = False  # Mark as sensitive for logging/privacy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "required": self.required
        }
        
        if self.default_value is not None:
            result["default_value"] = self.default_value
        if self.examples:
            result["examples"] = self.examples
        if self.sensitive:
            result["sensitive"] = self.sensitive
        if self.constraints:
            constraints_dict = {}
            if self.constraints.min_value is not None:
                constraints_dict["min_value"] = self.constraints.min_value
            if self.constraints.max_value is not None:
                constraints_dict["max_value"] = self.constraints.max_value
            if self.constraints.min_length is not None:
                constraints_dict["min_length"] = self.constraints.min_length
            if self.constraints.max_length is not None:
                constraints_dict["max_length"] = self.constraints.max_length
            if self.constraints.pattern:
                constraints_dict["pattern"] = self.constraints.pattern
            if self.constraints.enum_values:
                constraints_dict["enum_values"] = self.constraints.enum_values
            if self.constraints.error_message:
                constraints_dict["error_message"] = self.constraints.error_message
            if constraints_dict:
                result["constraints"] = constraints_dict
        
        return result


@dataclass
class ErrorDefinition:
    """Standardized error definition"""
    code: str
    message: str
    description: str
    http_status: Optional[int] = None
    retry_after: Optional[int] = None


@dataclass 
class ExampleDefinition:
    """Tool usage example"""
    name: str
    description: str
    input_parameters: Dict[str, Any]
    expected_output: Any
    notes: Optional[str] = None


@dataclass
class ToolDefinition:
    """Comprehensive standardized tool definition"""
    # Required fields
    name: str
    description: str
    parameters: List[ParameterDefinition]
    return_type: str
    
    # Optional metadata
    version: str = "1.0.0"
    author: Optional[str] = None
    category: ToolCategory = ToolCategory.CUSTOM
    security_level: SecurityLevel = SecurityLevel.PROTECTED
    capabilities: List[CapabilityLevel] = field(default_factory=lambda: [CapabilityLevel.READ_ONLY])
    tags: List[str] = field(default_factory=list)
    
    # Documentation
    long_description: Optional[str] = None
    usage_notes: Optional[str] = None
    examples: List[ExampleDefinition] = field(default_factory=list)
    
    # Technical specifications
    timeout_seconds: int = 30
    rate_limit_per_minute: Optional[int] = None
    cacheable: bool = False
    idempotent: bool = False
    
    # Error handling
    possible_errors: List[ErrorDefinition] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    deprecated: bool = False
    deprecation_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category.value,
            "security_level": self.security_level.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "tags": self.tags,
            "parameters": [param.to_dict() for param in self.parameters],
            "return_type": self.return_type,
            "timeout_seconds": self.timeout_seconds,
            "cacheable": self.cacheable,
            "idempotent": self.idempotent,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "deprecated": self.deprecated
        }
        
        # Only include optional fields if they have values
        if self.author:
            result["author"] = self.author
        if self.long_description:
            result["long_description"] = self.long_description
        if self.usage_notes:
            result["usage_notes"] = self.usage_notes
        if self.rate_limit_per_minute:
            result["rate_limit_per_minute"] = self.rate_limit_per_minute
        if self.deprecation_message:
            result["deprecation_message"] = self.deprecation_message
        
        # Handle examples
        if self.examples:
            examples_list = []
            for ex in self.examples:
                example_dict = {
                    "name": ex.name,
                    "description": ex.description,
                    "input_parameters": ex.input_parameters,
                    "expected_output": ex.expected_output
                }
                if ex.notes is not None:
                    example_dict["notes"] = ex.notes
                examples_list.append(example_dict)
            result["examples"] = examples_list
        
        # Handle errors
        if self.possible_errors:
            errors_list = []
            for err in self.possible_errors:
                error_dict = {
                    "code": err.code,
                    "message": err.message,
                    "description": err.description
                }
                if err.http_status is not None:
                    error_dict["http_status"] = err.http_status
                if err.retry_after is not None:
                    error_dict["retry_after"] = err.retry_after
                errors_list.append(error_dict)
            result["possible_errors"] = errors_list
        
        return result


class ToolDefinitionSchema:
    """JSON Schema definition and validation for tool definitions"""
    
    SCHEMA_VERSION = "1.0.0"
    
    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Get JSON schema for parameter definitions"""
        return {
            "type": "object",
            "required": ["name", "type", "description"],
            "properties": {
                "name": {
                    "type": "string",
                    "pattern": "^[a-zA-Z][a-zA-Z0-9_]*$",
                    "minLength": 1,
                    "maxLength": 50
                },
                "type": {
                    "type": "string",
                    "enum": [param_type.value for param_type in ParameterType]
                },
                "description": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500
                },
                "required": {
                    "type": "boolean",
                    "default": True
                },
                "default_value": {},
                "examples": {
                    "type": "array",
                    "items": {}
                },
                "sensitive": {
                    "type": "boolean",
                    "default": False
                },
                "constraints": {
                    "type": "object",
                    "properties": {
                        "min_value": {"type": "number"},
                        "max_value": {"type": "number"},
                        "min_length": {"type": "integer", "minimum": 0},
                        "max_length": {"type": "integer", "minimum": 0},
                        "pattern": {"type": "string"},
                        "enum_values": {"type": "array"},
                        "error_message": {"type": "string"}
                    },
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        }
    
    @classmethod
    def get_example_schema(cls) -> Dict[str, Any]:
        """Get JSON schema for example definitions"""
        return {
            "type": "object",
            "required": ["name", "description", "input_parameters", "expected_output"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "description": {"type": "string", "minLength": 1},
                "input_parameters": {"type": "object"},
                "expected_output": {},
                "notes": {"type": "string"}
            },
            "additionalProperties": False
        }
    
    @classmethod
    def get_error_schema(cls) -> Dict[str, Any]:
        """Get JSON schema for error definitions"""
        return {
            "type": "object",
            "required": ["code", "message", "description"],
            "properties": {
                "code": {"type": "string", "pattern": "^[A-Z_]+$"},
                "message": {"type": "string", "minLength": 1},
                "description": {"type": "string", "minLength": 1},
                "http_status": {"type": "integer", "minimum": 100, "maximum": 599},
                "retry_after": {"type": "integer", "minimum": 0}
            },
            "additionalProperties": False
        }
    
    @classmethod
    def get_tool_schema(cls) -> Dict[str, Any]:
        """Get complete JSON schema for tool definitions"""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Sovereign AI Tool Definition",
            "description": "Standardized format for defining external tools",
            "version": cls.SCHEMA_VERSION,
            "type": "object",
            "required": ["name", "description", "parameters", "return_type"],
            "properties": {
                "name": {
                    "type": "string",
                    "pattern": "^[a-zA-Z][a-zA-Z0-9_]*$",
                    "minLength": 1,
                    "maxLength": 100,
                    "description": "Unique tool identifier"
                },
                "description": {
                    "type": "string",
                    "minLength": 10,
                    "maxLength": 500,
                    "description": "Brief tool description for AI understanding"
                },
                "version": {
                    "type": "string",
                    "pattern": "^\\d+\\.\\d+\\.\\d+$",
                    "default": "1.0.0"
                },
                "author": {
                    "type": "string",
                    "maxLength": 100
                },
                "category": {
                    "type": "string",
                    "enum": [cat.value for cat in ToolCategory],
                    "default": "custom"
                },
                "security_level": {
                    "type": "string",
                    "enum": [level.value for level in SecurityLevel],
                    "default": "protected"
                },
                "capabilities": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [cap.value for cap in CapabilityLevel]
                    },
                    "default": ["read_only"],
                    "uniqueItems": True
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "pattern": "^[a-z0-9_-]+$",
                        "maxLength": 30
                    },
                    "maxItems": 10,
                    "uniqueItems": True
                },
                "parameters": {
                    "type": "array",
                    "items": cls.get_parameter_schema(),
                    "maxItems": 50
                },
                "return_type": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100
                },
                "long_description": {
                    "type": "string",
                    "maxLength": 2000
                },
                "usage_notes": {
                    "type": "string",
                    "maxLength": 1000
                },
                "examples": {
                    "type": "array",
                    "items": cls.get_example_schema(),
                    "maxItems": 20
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                    "default": 30
                },
                "rate_limit_per_minute": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10000
                },
                "cacheable": {
                    "type": "boolean",
                    "default": False
                },
                "idempotent": {
                    "type": "boolean",
                    "default": False
                },
                "possible_errors": {
                    "type": "array",
                    "items": cls.get_error_schema(),
                    "maxItems": 50
                },
                "created_at": {
                    "type": "string",
                    "format": "date-time"
                },
                "updated_at": {
                    "type": "string",
                    "format": "date-time"
                },
                "deprecated": {
                    "type": "boolean",
                    "default": False
                },
                "deprecation_message": {
                    "type": "string",
                    "maxLength": 500
                }
            },
            "additionalProperties": False
        }


class ToolDefinitionValidator:
    """Enhanced validation for tool definitions"""
    
    def __init__(self):
        self.schema = ToolDefinitionSchema.get_tool_schema()
    
    def validate_definition(self, definition: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate tool definition against schema"""
        errors = []
        
        try:
            # Schema validation
            validate(instance=definition, schema=self.schema)
        except ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
            return False, errors
        
        # Custom business logic validation
        try:
            self._validate_parameter_consistency(definition, errors)
            self._validate_security_capabilities(definition, errors)
            self._validate_examples(definition, errors)
            self._validate_naming_conventions(definition, errors)
            self._validate_version_format(definition, errors)
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    def _validate_parameter_consistency(self, definition: Dict[str, Any], errors: List[str]):
        """Validate parameter definitions are consistent"""
        param_names = set()
        
        for param in definition.get("parameters", []):
            name = param.get("name")
            if name in param_names:
                errors.append(f"Duplicate parameter name: {name}")
            param_names.add(name)
            
            # Validate constraints match parameter type
            param_type = param.get("type")
            constraints = param.get("constraints", {})
            
            if param_type in ["integer", "float"] and constraints:
                if "pattern" in constraints:
                    errors.append(f"Pattern constraint not valid for numeric parameter: {name}")
            
            if param_type == "string" and constraints:
                if "min_value" in constraints or "max_value" in constraints:
                    errors.append(f"Numeric constraints not valid for string parameter: {name}")
    
    def _validate_security_capabilities(self, definition: Dict[str, Any], errors: List[str]):
        """Validate security level matches capabilities"""
        security_level = definition.get("security_level", "protected")
        capabilities = definition.get("capabilities", [])
        
        if security_level == "public" and "privileged" in capabilities:
            errors.append("Public tools cannot have privileged capabilities")
        
        if security_level == "system" and "read_only" in capabilities and len(capabilities) == 1:
            errors.append("System-level tools should have more than read-only capabilities")
    
    def _validate_examples(self, definition: Dict[str, Any], errors: List[str]):
        """Validate examples match parameter definitions"""
        parameters = {p["name"]: p for p in definition.get("parameters", [])}
        
        for example in definition.get("examples", []):
            input_params = example.get("input_parameters", {})
            
            # Check required parameters are present
            for param_name, param_def in parameters.items():
                if param_def.get("required", True) and param_name not in input_params:
                    errors.append(f"Example '{example.get('name')}' missing required parameter: {param_name}")
            
            # Check no unexpected parameters
            for param_name in input_params:
                if param_name not in parameters:
                    errors.append(f"Example '{example.get('name')}' has undefined parameter: {param_name}")
    
    def _validate_naming_conventions(self, definition: Dict[str, Any], errors: List[str]):
        """Validate naming follows conventions"""
        name = definition.get("name", "")
        
        # Tool name should be descriptive
        if len(name) < 3:
            errors.append("Tool name should be at least 3 characters")
        
        # Check for common naming issues - only flag if it ends with capital "Tool"
        if name.endswith("Tool"):
            errors.append("Tool name should not end with 'Tool' (implied)")
        
        if " " in name:
            errors.append("Tool name should not contain spaces (use underscores)")
    
    def _validate_version_format(self, definition: Dict[str, Any], errors: List[str]):
        """Validate semantic versioning"""
        version = definition.get("version", "1.0.0")
        
        # Basic semantic version check (already in schema, but more detailed here)
        parts = version.split(".")
        if len(parts) != 3:
            errors.append("Version must follow semantic versioning (major.minor.patch)")
        else:
            try:
                major, minor, patch = map(int, parts)
                if major < 0 or minor < 0 or patch < 0:
                    errors.append("Version numbers must be non-negative")
            except ValueError:
                errors.append("Version parts must be integers")


class ToolBuilder:
    """Fluent builder for creating standardized tool definitions"""
    
    def __init__(self, name: str):
        self.definition = ToolDefinition(
            name=name,
            description="",
            parameters=[],
            return_type="object"
        )
    
    def description(self, desc: str) -> 'ToolBuilder':
        """Set tool description"""
        self.definition.description = desc
        return self
    
    def long_description(self, desc: str) -> 'ToolBuilder':
        """Set detailed description"""
        self.definition.long_description = desc
        return self
    
    def version(self, version: str) -> 'ToolBuilder':
        """Set tool version"""
        self.definition.version = version
        return self
    
    def author(self, author: str) -> 'ToolBuilder':
        """Set tool author"""
        self.definition.author = author
        return self
    
    def category(self, category: ToolCategory) -> 'ToolBuilder':
        """Set tool category"""
        self.definition.category = category
        return self
    
    def security_level(self, level: SecurityLevel) -> 'ToolBuilder':
        """Set security level"""
        self.definition.security_level = level
        return self
    
    def capabilities(self, *capabilities: CapabilityLevel) -> 'ToolBuilder':
        """Set tool capabilities"""
        self.definition.capabilities = list(capabilities)
        return self
    
    def tags(self, *tags: str) -> 'ToolBuilder':
        """Add tags"""
        self.definition.tags.extend(tags)
        return self
    
    def parameter(self, name: str, param_type: ParameterType, description: str,
                  required: bool = True, default_value: Any = None,
                  constraints: Optional[ParameterConstraint] = None,
                  examples: Optional[List[Any]] = None,
                  sensitive: bool = False) -> 'ToolBuilder':
        """Add parameter definition"""
        param = ParameterDefinition(
            name=name,
            type=param_type,
            description=description,
            required=required,
            default_value=default_value,
            constraints=constraints,
            examples=examples,
            sensitive=sensitive
        )
        self.definition.parameters.append(param)
        return self
    
    def return_type(self, return_type: str) -> 'ToolBuilder':
        """Set return type"""
        self.definition.return_type = return_type
        return self
    
    def timeout(self, seconds: int) -> 'ToolBuilder':
        """Set execution timeout"""
        self.definition.timeout_seconds = seconds
        return self
    
    def rate_limit(self, per_minute: int) -> 'ToolBuilder':
        """Set rate limit"""
        self.definition.rate_limit_per_minute = per_minute
        return self
    
    def cacheable(self, cacheable: bool = True) -> 'ToolBuilder':
        """Set caching behavior"""
        self.definition.cacheable = cacheable
        return self
    
    def idempotent(self, idempotent: bool = True) -> 'ToolBuilder':
        """Set idempotent behavior"""
        self.definition.idempotent = idempotent
        return self
    
    def example(self, name: str, description: str, input_params: Dict[str, Any],
                expected_output: Any, notes: Optional[str] = None) -> 'ToolBuilder':
        """Add usage example"""
        example = ExampleDefinition(
            name=name,
            description=description,
            input_parameters=input_params,
            expected_output=expected_output,
            notes=notes
        )
        self.definition.examples.append(example)
        return self
    
    def error(self, code: str, message: str, description: str,
              http_status: Optional[int] = None, retry_after: Optional[int] = None) -> 'ToolBuilder':
        """Add error definition"""
        error = ErrorDefinition(
            code=code,
            message=message,
            description=description,
            http_status=http_status,
            retry_after=retry_after
        )
        self.definition.possible_errors.append(error)
        return self
    
    def usage_notes(self, notes: str) -> 'ToolBuilder':
        """Set usage notes"""
        self.definition.usage_notes = notes
        return self
    
    def deprecate(self, message: Optional[str] = None) -> 'ToolBuilder':
        """Mark as deprecated"""
        self.definition.deprecated = True
        self.definition.deprecation_message = message
        return self
    
    def build(self) -> ToolDefinition:
        """Build and validate the tool definition"""
        if not self.definition.description:
            raise ValueError("Tool description is required")
        
        # Update timestamp
        self.definition.updated_at = datetime.now()
        
        # Validate the definition
        validator = ToolDefinitionValidator()
        definition_dict = self.definition.to_dict()
        is_valid, errors = validator.validate_definition(definition_dict)
        
        if not is_valid:
            raise ValueError(f"Tool definition validation failed: {'; '.join(errors)}")
        
        return self.definition


class ToolTemplateFactory:
    """Factory for creating pre-configured tool templates"""
    
    @staticmethod
    def api_tool(name: str) -> ToolBuilder:
        """Create API tool template"""
        return (ToolBuilder(name)
                .category(ToolCategory.API)
                .capabilities(CapabilityLevel.NETWORK_ACCESS)
                .security_level(SecurityLevel.PROTECTED)
                .parameter("url", ParameterType.URL, "API endpoint URL")
                .parameter("method", ParameterType.ENUM, "HTTP method", 
                          constraints=ParameterConstraint(enum_values=["GET", "POST", "PUT", "DELETE"]))
                .parameter("headers", ParameterType.OBJECT, "HTTP headers", required=False)
                .parameter("body", ParameterType.OBJECT, "Request body", required=False)
                .return_type("object")
                .timeout(30)
                .rate_limit(60))
    
    @staticmethod
    def file_tool(name: str) -> ToolBuilder:
        """Create file system tool template"""
        return (ToolBuilder(name)
                .category(ToolCategory.FILE_SYSTEM)
                .capabilities(CapabilityLevel.FILE_ACCESS)
                .security_level(SecurityLevel.PRIVATE)
                .parameter("file_path", ParameterType.FILE_PATH, "Path to file")
                .return_type("object")
                .timeout(60))
    
    @staticmethod
    def system_tool(name: str) -> ToolBuilder:
        """Create system tool template"""
        return (ToolBuilder(name)
                .category(ToolCategory.SYSTEM)
                .capabilities(CapabilityLevel.SYSTEM_ACCESS)
                .security_level(SecurityLevel.SYSTEM)
                .return_type("object")
                .timeout(30))
    
    @staticmethod
    def computation_tool(name: str) -> ToolBuilder:
        """Create computation tool template"""
        return (ToolBuilder(name)
                .category(ToolCategory.COMPUTATION)
                .capabilities(CapabilityLevel.READ_ONLY)
                .security_level(SecurityLevel.PUBLIC)
                .return_type("number")
                .cacheable()
                .idempotent())
    
    @staticmethod
    def database_tool(name: str) -> ToolBuilder:
        """Create database tool template"""
        return (ToolBuilder(name)
                .category(ToolCategory.DATABASE)
                .capabilities(CapabilityLevel.READ_WRITE)
                .security_level(SecurityLevel.PRIVATE)
                .parameter("connection_string", ParameterType.STRING, "Database connection", sensitive=True)
                .parameter("query", ParameterType.STRING, "SQL query")
                .return_type("array")
                .timeout(120))


class DocumentationGenerator:
    """Generate documentation from tool definitions"""
    
    @staticmethod
    def generate_markdown(definition: ToolDefinition) -> str:
        """Generate markdown documentation for a tool"""
        doc = f"# {definition.name}\n\n"
        doc += f"**Version:** {definition.version}\n\n"
        
        if definition.author:
            doc += f"**Author:** {definition.author}\n\n"
        
        doc += f"**Category:** {definition.category.value}\n\n"
        doc += f"**Security Level:** {definition.security_level.value}\n\n"
        
        if definition.capabilities:
            doc += f"**Capabilities:** {', '.join(cap.value for cap in definition.capabilities)}\n\n"
        
        doc += f"## Description\n\n{definition.description}\n\n"
        
        if definition.long_description:
            doc += f"{definition.long_description}\n\n"
        
        # Parameters section
        if definition.parameters:
            doc += "## Parameters\n\n"
            doc += "| Name | Type | Required | Description |\n"
            doc += "|------|------|----------|-------------|\n"
            
            for param in definition.parameters:
                required = "✓" if param.required else "✗"
                doc += f"| {param.name} | {param.type.value} | {required} | {param.description} |\n"
            doc += "\n"
        
        # Examples section
        if definition.examples:
            doc += "## Examples\n\n"
            for example in definition.examples:
                doc += f"### {example.name}\n\n"
                doc += f"{example.description}\n\n"
                doc += "**Input:**\n```json\n"
                doc += json.dumps(example.input_parameters, indent=2)
                doc += "\n```\n\n"
                doc += "**Output:**\n```json\n"
                doc += json.dumps(example.expected_output, indent=2)
                doc += "\n```\n\n"
                if example.notes:
                    doc += f"**Notes:** {example.notes}\n\n"
        
        # Errors section
        if definition.possible_errors:
            doc += "## Possible Errors\n\n"
            doc += "| Code | Message | Description |\n"
            doc += "|------|---------|-------------|\n"
            for error in definition.possible_errors:
                doc += f"| {error.code} | {error.message} | {error.description} |\n"
            doc += "\n"
        
        # Technical details
        doc += "## Technical Details\n\n"
        doc += f"- **Return Type:** {definition.return_type}\n"
        doc += f"- **Timeout:** {definition.timeout_seconds} seconds\n"
        doc += f"- **Cacheable:** {'Yes' if definition.cacheable else 'No'}\n"
        doc += f"- **Idempotent:** {'Yes' if definition.idempotent else 'No'}\n"
        
        if definition.rate_limit_per_minute:
            doc += f"- **Rate Limit:** {definition.rate_limit_per_minute} calls per minute\n"
        
        if definition.usage_notes:
            doc += f"\n## Usage Notes\n\n{definition.usage_notes}\n"
        
        if definition.deprecated:
            doc += f"\n## ⚠️ Deprecated\n\n"
            if definition.deprecation_message:
                doc += f"{definition.deprecation_message}\n"
            else:
                doc += "This tool is deprecated and may be removed in future versions.\n"
        
        return doc
    
    @staticmethod
    def generate_json_schema(definition: ToolDefinition) -> Dict[str, Any]:
        """Generate JSON schema for tool parameters"""
        schema = {
            "title": f"{definition.name} Parameters",
            "description": definition.description,
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param in definition.parameters:
            param_schema = {
                "type": param.type.value,
                "description": param.description
            }
            
            if param.constraints:
                if param.constraints.min_value is not None:
                    param_schema["minimum"] = param.constraints.min_value
                if param.constraints.max_value is not None:
                    param_schema["maximum"] = param.constraints.max_value
                if param.constraints.min_length is not None:
                    param_schema["minLength"] = param.constraints.min_length
                if param.constraints.max_length is not None:
                    param_schema["maxLength"] = param.constraints.max_length
                if param.constraints.pattern:
                    param_schema["pattern"] = param.constraints.pattern
                if param.constraints.enum_values:
                    param_schema["enum"] = param.constraints.enum_values
            
            if param.default_value is not None:
                param_schema["default"] = param.default_value
            
            if param.examples:
                param_schema["examples"] = param.examples
            
            schema["properties"][param.name] = param_schema
            
            if param.required:
                schema["required"].append(param.name)
        
        return schema 