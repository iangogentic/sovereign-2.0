"""
Test suite for Standardized Tool Definition Format
"""

import pytest
import json
from datetime import datetime
from typing import Dict, Any, List

from src.sovereign.tool_definition_schema import (
    ToolCategory,
    SecurityLevel,
    CapabilityLevel,
    ParameterType,
    ParameterConstraint,
    ParameterDefinition,
    ErrorDefinition,
    ExampleDefinition,
    ToolDefinition,
    ToolDefinitionSchema,
    ToolDefinitionValidator,
    ToolBuilder,
    ToolTemplateFactory,
    DocumentationGenerator
)


class TestParameterDefinition:
    """Test parameter definition functionality"""
    
    def test_basic_parameter_creation(self):
        """Test creating a basic parameter"""
        param = ParameterDefinition(
            name="test_param",
            type=ParameterType.STRING,
            description="Test parameter description"
        )
        
        assert param.name == "test_param"
        assert param.type == ParameterType.STRING
        assert param.description == "Test parameter description"
        assert param.required == True  # Default
        assert param.default_value is None
        assert param.sensitive == False
    
    def test_parameter_with_constraints(self):
        """Test parameter with validation constraints"""
        constraints = ParameterConstraint(
            min_length=5,
            max_length=50,
            pattern="^[a-zA-Z]+$",
            error_message="Must be alphabetic characters only"
        )
        
        param = ParameterDefinition(
            name="username",
            type=ParameterType.STRING,
            description="User name",
            constraints=constraints,
            examples=["alice", "bob123"]
        )
        
        assert param.constraints.min_length == 5
        assert param.constraints.max_length == 50
        assert param.constraints.pattern == "^[a-zA-Z]+$"
        assert param.examples == ["alice", "bob123"]
    
    def test_parameter_to_dict(self):
        """Test parameter serialization to dictionary"""
        constraints = ParameterConstraint(
            min_value=0,
            max_value=100,
            error_message="Must be between 0 and 100"
        )
        
        param = ParameterDefinition(
            name="score",
            type=ParameterType.INTEGER,
            description="Score value",
            required=True,
            default_value=50,
            constraints=constraints,
            sensitive=False
        )
        
        param_dict = param.to_dict()
        
        assert param_dict["name"] == "score"
        assert param_dict["type"] == "integer"
        assert param_dict["description"] == "Score value"
        assert param_dict["required"] == True
        assert param_dict["default_value"] == 50
        assert param_dict["constraints"]["min_value"] == 0
        assert param_dict["constraints"]["max_value"] == 100


class TestToolDefinition:
    """Test tool definition functionality"""
    
    def test_basic_tool_creation(self):
        """Test creating a basic tool definition"""
        param = ParameterDefinition(
            name="message",
            type=ParameterType.STRING,
            description="Message to process"
        )
        
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool for validation",
            parameters=[param],
            return_type="string"
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "A test tool for validation"
        assert len(tool.parameters) == 1
        assert tool.return_type == "string"
        assert tool.version == "1.0.0"  # Default
        assert tool.category == ToolCategory.CUSTOM  # Default
        assert tool.security_level == SecurityLevel.PROTECTED  # Default
    
    def test_comprehensive_tool_creation(self):
        """Test creating a comprehensive tool definition"""
        # Parameters
        param1 = ParameterDefinition(
            name="query",
            type=ParameterType.STRING,
            description="Search query"
        )
        param2 = ParameterDefinition(
            name="limit",
            type=ParameterType.INTEGER,
            description="Result limit",
            required=False,
            default_value=10,
            constraints=ParameterConstraint(min_value=1, max_value=100)
        )
        
        # Example
        example = ExampleDefinition(
            name="Basic search",
            description="Simple search example",
            input_parameters={"query": "python", "limit": 5},
            expected_output={"results": ["result1", "result2"]},
            notes="Returns top matching results"
        )
        
        # Error
        error = ErrorDefinition(
            code="INVALID_QUERY",
            message="Query is invalid",
            description="The provided query contains invalid characters",
            http_status=400
        )
        
        tool = ToolDefinition(
            name="search_api",
            description="Search API for finding information",
            version="2.1.0",
            author="Test Author",
            category=ToolCategory.API,
            security_level=SecurityLevel.PROTECTED,
            capabilities=[CapabilityLevel.NETWORK_ACCESS, CapabilityLevel.READ_ONLY],
            tags=["search", "api", "data"],
            parameters=[param1, param2],
            return_type="object",
            long_description="Comprehensive search API with advanced filtering",
            usage_notes="Rate limited to 100 requests per minute",
            examples=[example],
            timeout_seconds=60,
            rate_limit_per_minute=100,
            cacheable=True,
            idempotent=True,
            possible_errors=[error]
        )
        
        assert tool.version == "2.1.0"
        assert tool.author == "Test Author"
        assert tool.category == ToolCategory.API
        assert len(tool.capabilities) == 2
        assert len(tool.tags) == 3
        assert tool.cacheable == True
        assert len(tool.examples) == 1
        assert len(tool.possible_errors) == 1
    
    def test_tool_to_dict(self):
        """Test tool serialization to dictionary"""
        param = ParameterDefinition(
            name="input",
            type=ParameterType.STRING,
            description="Input data"
        )
        
        tool = ToolDefinition(
            name="simple_tool",
            description="Simple tool for testing",
            parameters=[param],
            return_type="string",
            tags=["test", "simple"]
        )
        
        tool_dict = tool.to_dict()
        
        assert tool_dict["name"] == "simple_tool"
        assert tool_dict["description"] == "Simple tool for testing"
        assert tool_dict["category"] == "custom"
        assert tool_dict["security_level"] == "protected"
        assert len(tool_dict["parameters"]) == 1
        assert tool_dict["parameters"][0]["name"] == "input"
        assert tool_dict["tags"] == ["test", "simple"]


class TestToolDefinitionSchema:
    """Test JSON schema definitions"""
    
    def test_parameter_schema(self):
        """Test parameter schema structure"""
        schema = ToolDefinitionSchema.get_parameter_schema()
        
        assert schema["type"] == "object"
        assert "name" in schema["required"]
        assert "type" in schema["required"]
        assert "description" in schema["required"]
        assert "name" in schema["properties"]
        assert "pattern" in schema["properties"]["name"]
    
    def test_tool_schema(self):
        """Test complete tool schema structure"""
        schema = ToolDefinitionSchema.get_tool_schema()
        
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert schema["title"] == "Sovereign AI Tool Definition"
        assert schema["version"] == "1.0.0"
        assert "name" in schema["required"]
        assert "description" in schema["required"]
        assert "parameters" in schema["required"]
        assert "return_type" in schema["required"]
        
        # Check enum values are properly set
        security_levels = schema["properties"]["security_level"]["enum"]
        assert "public" in security_levels
        assert "protected" in security_levels
        assert "private" in security_levels
        assert "system" in security_levels
    
    def test_example_schema(self):
        """Test example schema structure"""
        schema = ToolDefinitionSchema.get_example_schema()
        
        assert schema["type"] == "object"
        assert "name" in schema["required"]
        assert "description" in schema["required"]
        assert "input_parameters" in schema["required"]
        assert "expected_output" in schema["required"]
    
    def test_error_schema(self):
        """Test error schema structure"""
        schema = ToolDefinitionSchema.get_error_schema()
        
        assert schema["type"] == "object"
        assert "code" in schema["required"]
        assert "message" in schema["required"]
        assert "description" in schema["required"]
        assert schema["properties"]["code"]["pattern"] == "^[A-Z_]+$"


class TestToolDefinitionValidator:
    """Test tool definition validation"""
    
    def test_valid_tool_definition(self):
        """Test validation of a valid tool definition"""
        validator = ToolDefinitionValidator()
        
        definition = {
            "name": "valid_tool",
            "description": "This is a valid tool for testing purposes",
            "version": "1.0.0",
            "category": "computation",
            "security_level": "public",
            "capabilities": ["read_only"],
            "parameters": [
                {
                    "name": "input_value",
                    "type": "integer",
                    "description": "Input value for computation",
                    "required": True
                }
            ],
            "return_type": "number",
            "timeout_seconds": 30,
            "cacheable": True
        }
        
        is_valid, errors = validator.validate_definition(definition)
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_invalid_schema_validation(self):
        """Test validation with schema violations"""
        validator = ToolDefinitionValidator()
        
        # Missing required fields
        definition = {
            "name": "invalid_tool",
            # Missing description, parameters, return_type
        }
        
        is_valid, errors = validator.validate_definition(definition)
        
        assert is_valid == False
        assert len(errors) > 0
        assert "Schema validation failed" in errors[0]
    
    def test_duplicate_parameter_names(self):
        """Test validation catches duplicate parameter names"""
        validator = ToolDefinitionValidator()
        
        definition = {
            "name": "duplicate_param_tool",
            "description": "Tool with duplicate parameter names",
            "parameters": [
                {
                    "name": "duplicate_param",
                    "type": "string",
                    "description": "First parameter"
                },
                {
                    "name": "duplicate_param",
                    "type": "integer",
                    "description": "Second parameter with same name"
                }
            ],
            "return_type": "object"
        }
        
        is_valid, errors = validator.validate_definition(definition)
        
        assert is_valid == False
        assert any("Duplicate parameter name" in error for error in errors)
    
    def test_invalid_constraints_for_type(self):
        """Test validation catches invalid constraints for parameter types"""
        validator = ToolDefinitionValidator()
        
        definition = {
            "name": "invalid_constraints_tool",
            "description": "Tool with invalid parameter constraints",
            "parameters": [
                {
                    "name": "numeric_param",
                    "type": "integer",
                    "description": "Numeric parameter",
                    "constraints": {
                        "pattern": "^[0-9]+$"  # Pattern not valid for numeric
                    }
                }
            ],
            "return_type": "number"
        }
        
        is_valid, errors = validator.validate_definition(definition)
        
        assert is_valid == False
        assert any("Pattern constraint not valid for numeric parameter" in error for error in errors)
    
    def test_security_capability_mismatch(self):
        """Test validation catches security/capability mismatches"""
        validator = ToolDefinitionValidator()
        
        definition = {
            "name": "security_mismatch_tool",
            "description": "Tool with security level and capability mismatch",
            "security_level": "public",
            "capabilities": ["privileged"],  # Public tools shouldn't have privileged capabilities
            "parameters": [],
            "return_type": "object"
        }
        
        is_valid, errors = validator.validate_definition(definition)
        
        assert is_valid == False
        assert any("Public tools cannot have privileged capabilities" in error for error in errors)
    
    def test_example_parameter_validation(self):
        """Test validation of examples against parameter definitions"""
        validator = ToolDefinitionValidator()
        
        definition = {
            "name": "example_validation_tool",
            "description": "Tool for testing example validation",
            "parameters": [
                {
                    "name": "required_param",
                    "type": "string",
                    "description": "Required parameter",
                    "required": True
                }
            ],
            "return_type": "object",
            "examples": [
                {
                    "name": "Invalid example",
                    "description": "Missing required parameter",
                    "input_parameters": {},  # Missing required_param
                    "expected_output": {}
                }
            ]
        }
        
        is_valid, errors = validator.validate_definition(definition)
        
        assert is_valid == False
        assert any("missing required parameter" in error for error in errors)
    
    def test_naming_convention_validation(self):
        """Test naming convention validation"""
        validator = ToolDefinitionValidator()
        
        # Test tool name ending with "Tool"
        definition = {
            "name": "BadTool",  # Should not end with "Tool"
            "description": "Tool with bad naming convention",
            "parameters": [],
            "return_type": "object"
        }
        
        is_valid, errors = validator.validate_definition(definition)
        
        assert is_valid == False
        assert any("should not end with 'Tool'" in error for error in errors)


class TestToolBuilder:
    """Test fluent tool builder functionality"""
    
    def test_basic_builder_usage(self):
        """Test basic builder pattern usage"""
        tool = (ToolBuilder("calculator")
                .description("Simple calculator tool")
                .parameter("operation", ParameterType.STRING, "Math operation to perform")
                .parameter("operand1", ParameterType.FLOAT, "First operand")
                .parameter("operand2", ParameterType.FLOAT, "Second operand")
                .return_type("number")
                .build())
        
        assert tool.name == "calculator"
        assert tool.description == "Simple calculator tool"
        assert len(tool.parameters) == 3
        assert tool.return_type == "number"
    
    def test_comprehensive_builder_usage(self):
        """Test comprehensive builder with all options"""
        constraints = ParameterConstraint(
            min_value=1,
            max_value=100,
            error_message="Must be between 1 and 100"
        )
        
        tool = (ToolBuilder("advanced_search")
                .description("Advanced search API with filtering")
                .long_description("Comprehensive search tool with advanced filtering options")
                .version("2.0.1")
                .author("API Team")
                .category(ToolCategory.API)
                .security_level(SecurityLevel.PROTECTED)
                .capabilities(CapabilityLevel.NETWORK_ACCESS, CapabilityLevel.READ_ONLY)
                .tags("search", "api", "filtering")
                .parameter("query", ParameterType.STRING, "Search query string")
                .parameter("limit", ParameterType.INTEGER, "Result limit", 
                          required=False, default_value=10, constraints=constraints)
                .parameter("category", ParameterType.ENUM, "Search category",
                          required=False, 
                          constraints=ParameterConstraint(enum_values=["web", "images", "news"]))
                .return_type("object")
                .timeout(120)
                .rate_limit(100)
                .cacheable()
                .idempotent()
                .example("Basic search", "Simple search example",
                        {"query": "python programming", "limit": 5},
                        {"results": [{"title": "Python Guide", "url": "..."}]},
                        "Returns top matching results")
                .error("INVALID_QUERY", "Invalid search query", 
                      "The query contains invalid characters", http_status=400)
                .usage_notes("This API is rate limited to 100 requests per minute")
                .build())
        
        assert tool.version == "2.0.1"
        assert tool.author == "API Team"
        assert tool.category == ToolCategory.API
        assert len(tool.capabilities) == 2
        assert len(tool.tags) == 3
        assert len(tool.parameters) == 3
        assert tool.timeout_seconds == 120
        assert tool.rate_limit_per_minute == 100
        assert tool.cacheable == True
        assert tool.idempotent == True
        assert len(tool.examples) == 1
        assert len(tool.possible_errors) == 1
        assert tool.usage_notes is not None
    
    def test_builder_validation_failure(self):
        """Test builder validation catches errors"""
        with pytest.raises(ValueError):
            # Missing description should cause validation failure
            ToolBuilder("invalid_tool").build()
    
    def test_builder_method_chaining(self):
        """Test that all builder methods return self for chaining"""
        builder = ToolBuilder("test_tool")
        
        # All these should return the same builder instance
        assert builder.description("test") is builder
        assert builder.version("1.0.0") is builder
        assert builder.author("test") is builder
        assert builder.category(ToolCategory.CUSTOM) is builder
        assert builder.security_level(SecurityLevel.PUBLIC) is builder
        assert builder.capabilities(CapabilityLevel.READ_ONLY) is builder
        assert builder.tags("test") is builder
        assert builder.return_type("object") is builder
        assert builder.timeout(30) is builder
        assert builder.rate_limit(60) is builder
        assert builder.cacheable() is builder
        assert builder.idempotent() is builder


class TestToolTemplateFactory:
    """Test pre-configured tool templates"""
    
    def test_api_tool_template(self):
        """Test API tool template creation"""
        tool = (ToolTemplateFactory.api_tool("weather_api")
                .description("Weather information API")
                .build())
        
        assert tool.name == "weather_api"
        assert tool.category == ToolCategory.API
        assert CapabilityLevel.NETWORK_ACCESS in tool.capabilities
        assert tool.security_level == SecurityLevel.PROTECTED
        
        # Check default API parameters
        param_names = [p.name for p in tool.parameters]
        assert "url" in param_names
        assert "method" in param_names
        assert "headers" in param_names
        assert "body" in param_names
    
    def test_file_tool_template(self):
        """Test file system tool template creation"""
        tool = (ToolTemplateFactory.file_tool("file_reader")
                .description("Read file contents")
                .build())
        
        assert tool.name == "file_reader"
        assert tool.category == ToolCategory.FILE_SYSTEM
        assert CapabilityLevel.FILE_ACCESS in tool.capabilities
        assert tool.security_level == SecurityLevel.PRIVATE
        
        # Check file parameter
        param_names = [p.name for p in tool.parameters]
        assert "file_path" in param_names
    
    def test_system_tool_template(self):
        """Test system tool template creation"""
        tool = (ToolTemplateFactory.system_tool("system_info")
                .description("Get system information")
                .build())
        
        assert tool.name == "system_info"
        assert tool.category == ToolCategory.SYSTEM
        assert CapabilityLevel.SYSTEM_ACCESS in tool.capabilities
        assert tool.security_level == SecurityLevel.SYSTEM
    
    def test_computation_tool_template(self):
        """Test computation tool template creation"""
        tool = (ToolTemplateFactory.computation_tool("math_calc")
                .description("Mathematical calculations")
                .parameter("expression", ParameterType.STRING, "Math expression")
                .build())
        
        assert tool.name == "math_calc"
        assert tool.category == ToolCategory.COMPUTATION
        assert CapabilityLevel.READ_ONLY in tool.capabilities
        assert tool.security_level == SecurityLevel.PUBLIC
        assert tool.cacheable == True
        assert tool.idempotent == True
        assert tool.return_type == "number"
    
    def test_database_tool_template(self):
        """Test database tool template creation"""
        tool = (ToolTemplateFactory.database_tool("user_query")
                .description("Query user database")
                .build())
        
        assert tool.name == "user_query"
        assert tool.category == ToolCategory.DATABASE
        assert CapabilityLevel.READ_WRITE in tool.capabilities
        assert tool.security_level == SecurityLevel.PRIVATE
        assert tool.timeout_seconds == 120
        
        # Check database parameters
        param_names = [p.name for p in tool.parameters]
        assert "connection_string" in param_names
        assert "query" in param_names
        
        # Connection string should be marked as sensitive
        conn_param = next(p for p in tool.parameters if p.name == "connection_string")
        assert conn_param.sensitive == True


class TestDocumentationGenerator:
    """Test documentation generation functionality"""
    
    def test_markdown_generation(self):
        """Test markdown documentation generation"""
        # Create a comprehensive tool for testing
        param = ParameterDefinition(
            name="search_term",
            type=ParameterType.STRING,
            description="Term to search for"
        )
        
        example = ExampleDefinition(
            name="Basic search",
            description="Simple search example",
            input_parameters={"search_term": "python"},
            expected_output={"results": ["result1", "result2"]},
            notes="Returns matching results"
        )
        
        error = ErrorDefinition(
            code="INVALID_TERM",
            message="Search term is invalid",
            description="The search term contains invalid characters"
        )
        
        tool = ToolDefinition(
            name="search_tool",
            description="Search for information",
            version="1.2.0",
            author="Search Team",
            category=ToolCategory.API,
            security_level=SecurityLevel.PROTECTED,
            capabilities=[CapabilityLevel.NETWORK_ACCESS],
            parameters=[param],
            return_type="object",
            long_description="Comprehensive search tool with advanced features",
            examples=[example],
            possible_errors=[error],
            usage_notes="Rate limited to 100 requests per minute",
            timeout_seconds=60,
            cacheable=True,
            idempotent=True
        )
        
        markdown = DocumentationGenerator.generate_markdown(tool)
        
        # Check key sections are present
        assert "# search_tool" in markdown
        assert "**Version:** 1.2.0" in markdown
        assert "**Author:** Search Team" in markdown
        assert "**Category:** api" in markdown
        assert "**Security Level:** protected" in markdown
        assert "## Description" in markdown
        assert "## Parameters" in markdown
        assert "## Examples" in markdown
        assert "## Possible Errors" in markdown
        assert "## Technical Details" in markdown
        assert "## Usage Notes" in markdown
        
        # Check parameter table
        assert "| search_term | string | ✓ | Term to search for |" in markdown
        
        # Check example section
        assert "### Basic search" in markdown
        assert "**Input:**" in markdown
        assert "**Output:**" in markdown
        assert "**Notes:** Returns matching results" in markdown
        
        # Check error table
        assert "| INVALID_TERM | Search term is invalid |" in markdown
        
        # Check technical details
        assert "- **Return Type:** object" in markdown
        assert "- **Timeout:** 60 seconds" in markdown
        assert "- **Cacheable:** Yes" in markdown
        assert "- **Idempotent:** Yes" in markdown
    
    def test_json_schema_generation(self):
        """Test JSON schema generation for tool parameters"""
        # Create tool with various parameter types and constraints
        param1 = ParameterDefinition(
            name="string_param",
            type=ParameterType.STRING,
            description="String parameter",
            required=True,
            constraints=ParameterConstraint(min_length=1, max_length=100, pattern="^[a-zA-Z]+$"),
            examples=["hello", "world"]
        )
        
        param2 = ParameterDefinition(
            name="number_param",
            type=ParameterType.INTEGER,
            description="Number parameter",
            required=False,
            default_value=42,
            constraints=ParameterConstraint(min_value=0, max_value=1000)
        )
        
        param3 = ParameterDefinition(
            name="enum_param",
            type=ParameterType.ENUM,
            description="Enum parameter",
            required=True,
            constraints=ParameterConstraint(enum_values=["option1", "option2", "option3"])
        )
        
        tool = ToolDefinition(
            name="schema_test_tool",
            description="Tool for testing schema generation",
            parameters=[param1, param2, param3],
            return_type="object"
        )
        
        schema = DocumentationGenerator.generate_json_schema(tool)
        
        # Check schema structure
        assert schema["title"] == "schema_test_tool Parameters"
        assert schema["description"] == "Tool for testing schema generation"
        assert schema["type"] == "object"
        
        # Check properties
        props = schema["properties"]
        
        # String parameter
        assert "string_param" in props
        string_prop = props["string_param"]
        assert string_prop["type"] == "string"
        assert string_prop["minLength"] == 1
        assert string_prop["maxLength"] == 100
        assert string_prop["pattern"] == "^[a-zA-Z]+$"
        assert string_prop["examples"] == ["hello", "world"]
        
        # Number parameter
        assert "number_param" in props
        number_prop = props["number_param"]
        assert number_prop["type"] == "integer"
        assert number_prop["minimum"] == 0
        assert number_prop["maximum"] == 1000
        assert number_prop["default"] == 42
        
        # Enum parameter
        assert "enum_param" in props
        enum_prop = props["enum_param"]
        assert enum_prop["enum"] == ["option1", "option2", "option3"]
        
        # Check required fields
        assert "string_param" in schema["required"]
        assert "enum_param" in schema["required"]
        assert "number_param" not in schema["required"]
    
    def test_deprecated_tool_documentation(self):
        """Test documentation generation for deprecated tools"""
        tool = ToolDefinition(
            name="deprecated_tool",
            description="This tool is deprecated",
            parameters=[],
            return_type="object",
            deprecated=True,
            deprecation_message="Use new_tool instead"
        )
        
        markdown = DocumentationGenerator.generate_markdown(tool)
        
        assert "## ⚠️ Deprecated" in markdown
        assert "Use new_tool instead" in markdown


class TestIntegrationScenarios:
    """Test real-world integration scenarios"""
    
    def test_complete_tool_lifecycle(self):
        """Test complete tool definition, validation, and documentation cycle"""
        # 1. Build tool using builder pattern
        tool = (ToolBuilder("weather_service")
                .description("Get weather information for a location")
                .long_description("Comprehensive weather service with current conditions and forecasts")
                .version("2.1.0")
                .author("Weather Team")
                .category(ToolCategory.API)
                .security_level(SecurityLevel.PROTECTED)
                .capabilities(CapabilityLevel.NETWORK_ACCESS, CapabilityLevel.READ_ONLY)
                .tags("weather", "api", "location")
                .parameter("location", ParameterType.STRING, "Location to get weather for",
                          constraints=ParameterConstraint(min_length=2, max_length=100))
                .parameter("units", ParameterType.ENUM, "Temperature units",
                          required=False, default_value="celsius",
                          constraints=ParameterConstraint(enum_values=["celsius", "fahrenheit"]))
                .parameter("include_forecast", ParameterType.BOOLEAN, "Include 5-day forecast",
                          required=False, default_value=False)
                .return_type("object")
                .timeout(30)
                .rate_limit(1000)
                .cacheable()
                .example("Current weather", "Get current weather for a city",
                        {"location": "New York", "units": "celsius"},
                        {"temperature": 22, "condition": "sunny", "humidity": 65})
                .error("LOCATION_NOT_FOUND", "Location not found",
                      "The specified location could not be found", http_status=404)
                .error("API_LIMIT_EXCEEDED", "API limit exceeded",
                      "Too many requests, try again later", http_status=429, retry_after=60)
                .usage_notes("This API requires an API key and is rate limited")
                .build())
        
        # 2. Validate the tool definition
        validator = ToolDefinitionValidator()
        tool_dict = tool.to_dict()
        is_valid, errors = validator.validate_definition(tool_dict)
        
        assert is_valid == True
        assert len(errors) == 0
        
        # 3. Generate documentation
        markdown = DocumentationGenerator.generate_markdown(tool)
        json_schema = DocumentationGenerator.generate_json_schema(tool)
        
        # Verify documentation quality
        assert "# weather_service" in markdown
        assert "location" in json_schema["properties"]
        assert "units" in json_schema["properties"]
        assert "include_forecast" in json_schema["properties"]
        
        # 4. Verify the tool can be reconstructed from dict
        recreated_tool = ToolDefinition(**{
            k: v for k, v in tool_dict.items() 
            if k in ['name', 'description', 'parameters', 'return_type']
        })
        
        assert recreated_tool.name == tool.name
        assert recreated_tool.description == tool.description
    
    def test_template_customization_workflow(self):
        """Test customizing tool templates for specific use cases"""
        # Start with API template and customize for specific service
        tool = (ToolTemplateFactory.api_tool("github_api")
                .description("GitHub API integration for repository management")
                .author("DevOps Team")
                .tags("github", "repository", "api")
                # Add custom parameters beyond the template defaults
                .parameter("repo_name", ParameterType.STRING, "Repository name")
                .parameter("owner", ParameterType.STRING, "Repository owner")
                .parameter("action", ParameterType.ENUM, "Action to perform",
                          constraints=ParameterConstraint(
                              enum_values=["create", "delete", "update", "get"]
                          ))
                # Override template defaults
                .timeout(120)  # Longer timeout for repo operations
                .rate_limit(5000)  # GitHub API limits
                .example("Get repository", "Retrieve repository information",
                        {"url": "https://api.github.com/repos/owner/repo",
                         "method": "GET", "repo_name": "my-repo", "owner": "user", "action": "get"},
                        {"id": 123, "name": "my-repo", "private": False})
                .build())
        
        # Verify template was properly customized
        assert tool.category == ToolCategory.API  # From template
        assert CapabilityLevel.NETWORK_ACCESS in tool.capabilities  # From template
        assert tool.timeout_seconds == 120  # Customized
        assert tool.rate_limit_per_minute == 5000  # Customized
        
        # Verify custom parameters were added
        param_names = [p.name for p in tool.parameters]
        assert "repo_name" in param_names
        assert "owner" in param_names
        assert "action" in param_names
        # Template parameters should still be there
        assert "url" in param_names
        assert "method" in param_names
    
    def test_validation_with_complex_constraints(self):
        """Test validation with complex parameter constraints"""
        validator = ToolDefinitionValidator()
        
        # Tool with complex validation rules
        definition = {
            "name": "complex_validation_tool",
            "description": "Tool with complex parameter validation rules",
            "parameters": [
                {
                    "name": "email",
                    "type": "string",
                    "description": "User email address",
                    "required": True,
                    "constraints": {
                        "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                        "max_length": 254
                    }
                },
                {
                    "name": "age",
                    "type": "integer",
                    "description": "User age",
                    "required": True,
                    "constraints": {
                        "min_value": 13,
                        "max_value": 120,
                        "error_message": "Age must be between 13 and 120"
                    }
                },
                {
                    "name": "preferences", 
                    "type": "array",
                    "description": "User preferences",
                    "required": False,
                    "constraints": {
                        "min_length": 1,
                        "max_length": 10
                    }
                }
            ],
            "return_type": "object",
            "examples": [
                {
                    "name": "Valid user",
                    "description": "Example with valid user data",
                    "input_parameters": {
                        "email": "user@example.com",
                        "age": 25,
                        "preferences": ["feature1", "feature2"]
                    },
                    "expected_output": {"user_id": 123, "status": "created"}
                }
            ]
        }
        
        is_valid, errors = validator.validate_definition(definition)
        
        # Should be valid - all constraints are appropriate for their types
        assert is_valid == True
        assert len(errors) == 0 