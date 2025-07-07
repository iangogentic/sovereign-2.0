"""
Tool Result Processor Tests - Comprehensive testing for parsing, formatting, and logging
"""

import pytest
import asyncio
import json
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta

from src.sovereign.tool_result_processor import (
    ToolResultProcessor, ToolRequestParser, ResultFormatter, ToolAuditLogger,
    ToolCallRequest, FormattedResult, AuditEvent, AuditEventType, LogLevel,
    parse_tool_request, format_tool_result, create_audit_logger
)
from src.sovereign.tool_integration_framework import (
    ToolExecutionResult, ToolMetadata, ToolType, ToolSecurityLevel, 
    ToolExecutionMode, ToolParameter
)
from src.sovereign.config import Config


class TestToolRequestParser:
    """Test the tool request parser functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return Config()
    
    @pytest.fixture
    def parser(self, config):
        """Create parser instance"""
        return ToolRequestParser(config)
    
    def test_parse_json_format(self, parser):
        """Test parsing JSON format tool calls"""
        model_output = '''
        I need to search for information:
        ```json
        {
            "tool_name": "internet_search",
            "parameters": {
                "query": "Python programming tutorial",
                "max_results": 5
            }
        }
        ```
        '''
        
        requests = parser.parse_tool_request(model_output)
        
        assert len(requests) == 1
        assert requests[0].tool_name == "internet_search"
        assert requests[0].parameters["query"] == "Python programming tutorial"
        assert requests[0].parameters["max_results"] == 5
        assert requests[0].confidence == 0.9
        assert requests[0].metadata["format"] == "json"
    
    def test_parse_function_format(self, parser):
        """Test parsing function call format"""
        model_output = '''
        Let me call the tool:
        call_tool("advanced_calculation", {"operation": "basic", "expression": "2+2"})
        '''
        
        requests = parser.parse_tool_request(model_output)
        
        assert len(requests) == 1
        assert requests[0].tool_name == "advanced_calculation"
        assert requests[0].parameters["operation"] == "basic"
        assert requests[0].confidence == 0.8
        assert requests[0].metadata["format"] == "function"
    
    def test_parse_xml_format(self, parser):
        """Test parsing XML format tool calls"""
        model_output = '''
        <tool_call>
            <name>file_access</name>
            <parameters>{"operation": "read", "path": "/tmp/test.txt"}</parameters>
        </tool_call>
        '''
        
        requests = parser.parse_tool_request(model_output)
        
        assert len(requests) == 1
        assert requests[0].tool_name == "file_access"
        assert requests[0].parameters["operation"] == "read"
        assert requests[0].confidence == 0.7
        assert requests[0].metadata["format"] == "xml"
    
    def test_parse_natural_language(self, parser):
        """Test parsing natural language requests"""
        model_output = "Please search for Python machine learning tutorials"
        
        requests = parser.parse_tool_request(model_output)
        
        assert len(requests) == 1
        assert requests[0].tool_name == "internet_search"
        assert requests[0].parameters["query"] == "Python machine learning tutorials"
        assert requests[0].confidence == 0.6
        assert requests[0].metadata["format"] == "natural_language"
    
    def test_parse_multiple_requests(self, parser):
        """Test parsing multiple tool requests"""
        model_output = '''
        First, let me search:
        {"tool_name": "internet_search", "parameters": {"query": "AI news"}}
        
        Then calculate:
        call_tool("advanced_calculation", {"operation": "basic", "expression": "10*5"})
        '''
        
        requests = parser.parse_tool_request(model_output)
        
        assert len(requests) == 2
        assert requests[0].tool_name == "internet_search"
        assert requests[1].tool_name == "advanced_calculation"
    
    def test_parse_invalid_json(self, parser):
        """Test handling of invalid JSON"""
        model_output = '''
        {"tool_name": "test", "parameters": {invalid json}}
        '''
        
        requests = parser.parse_tool_request(model_output)
        
        # Should handle gracefully and return empty list
        assert len(requests) == 0
    
    def test_parse_empty_input(self, parser):
        """Test handling of empty input"""
        requests = parser.parse_tool_request("")
        assert len(requests) == 0
        
        requests = parser.parse_tool_request("   ")
        assert len(requests) == 0
    
    def test_tool_mapping_accuracy(self, parser):
        """Test accuracy of natural language to tool mapping"""
        test_cases = [
            ("calculate 2+2", "advanced_calculation"),
            ("read the file", "file_access"),
            ("search for news", "internet_search"),
            ("get system info", "enhanced_system_info"),
            ("list files", "file_access"),
            ("delete the file", "file_access")
        ]
        
        for input_text, expected_tool in test_cases:
            requests = parser.parse_tool_request(input_text)
            if requests:  # Some might not match the pattern
                assert requests[0].tool_name == expected_tool


class TestResultFormatter:
    """Test the result formatter functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return Config()
    
    @pytest.fixture
    def formatter(self, config):
        """Create formatter instance"""
        return ResultFormatter(config)
    
    @pytest.fixture
    def success_result(self):
        """Create successful tool execution result"""
        return ToolExecutionResult(
            execution_id="test-123",
            tool_name="test_tool",
            success=True,
            result={
                "data": "test data",
                "count": 5,
                "items": ["item1", "item2", "item3"]
            },
            execution_time=1.2,
            metadata={"version": "1.0"}
        )
    
    @pytest.fixture
    def error_result(self):
        """Create failed tool execution result"""
        return ToolExecutionResult(
            execution_id="test-456",
            tool_name="test_tool",
            success=False,
            error="Connection timeout",
            execution_time=0.5,
            logs=["Attempting connection", "Connection failed"]
        )
    
    def test_format_standard_success(self, formatter, success_result):
        """Test standard formatting for successful results"""
        formatted = formatter.format_result(success_result, "standard")
        
        assert formatted.success == True
        assert "✅" in formatted.content
        assert "test_tool" in formatted.content
        assert "test data" in formatted.content
        assert formatted.format_type == "standard"
        assert len(formatted.structured_data) > 0
    
    def test_format_standard_error(self, formatter, error_result):
        """Test standard formatting for error results"""
        formatted = formatter.format_result(error_result, "standard")
        
        assert formatted.success == False
        assert "❌" in formatted.content
        assert "Connection timeout" in formatted.content
        assert formatted.format_type == "standard"
        assert "Connection timeout" in formatted.summary
    
    def test_format_compact(self, formatter, success_result):
        """Test compact formatting"""
        formatted = formatter.format_result(success_result, "compact")
        
        assert formatted.format_type == "compact"
        assert len(formatted.content) < 150  # Should be compact
        assert "✅" in formatted.content
        assert "test_tool" in formatted.content
    
    def test_format_detailed(self, formatter, success_result):
        """Test detailed formatting"""
        formatted = formatter.format_result(success_result, "detailed")
        
        assert formatted.format_type == "detailed"
        assert "Tool Execution Report" in formatted.content
        assert "test-123" in formatted.content  # execution_id
        assert "1.2" in formatted.content  # execution_time
        assert len(formatted.content) > 200  # Should be detailed
    
    def test_format_json(self, formatter, success_result):
        """Test JSON formatting"""
        formatted = formatter.format_result(success_result, "json")
        
        assert formatted.format_type == "json"
        assert "```json" in formatted.content
        assert "test_tool" in formatted.content
        
        # Should be valid JSON
        json_start = formatted.content.find('{')
        json_end = formatted.content.rfind('}') + 1
        json_content = formatted.content[json_start:json_end]
        parsed = json.loads(json_content)
        assert parsed["tool_name"] == "test_tool"
    
    def test_format_list_result(self, formatter):
        """Test formatting of list results"""
        list_result = ToolExecutionResult(
            execution_id="list-test",
            tool_name="list_tool",
            success=True,
            result=["item1", "item2", "item3", "item4", "item5"],
            execution_time=0.8
        )
        
        formatted = formatter.format_result(list_result, "standard")
        
        assert "Found 5 items" in formatted.content
        assert "item1" in formatted.content
        assert "item5" in formatted.content
    
    def test_format_string_result(self, formatter):
        """Test formatting of string results"""
        string_result = ToolExecutionResult(
            execution_id="string-test",
            tool_name="string_tool",
            success=True,
            result="This is a test string result",
            execution_time=0.3
        )
        
        formatted = formatter.format_result(string_result, "standard")
        
        assert "This is a test string result" in formatted.content
        assert "text content" in formatted.summary
    
    def test_format_large_list(self, formatter):
        """Test formatting of large lists (truncation)"""
        large_list = [f"item_{i}" for i in range(20)]
        large_result = ToolExecutionResult(
            execution_id="large-test",
            tool_name="large_tool",
            success=True,
            result=large_list,
            execution_time=1.5
        )
        
        formatted = formatter.format_result(large_result, "standard")
        
        assert "Found 20 items" in formatted.content
        assert "... and 10 more items" in formatted.content
    
    def test_format_error_fallback(self, formatter, success_result):
        """Test error fallback formatting"""
        # Mock the formatting to raise an exception
        with patch.object(formatter, '_format_standard', side_effect=Exception("Format error")):
            formatted = formatter.format_result(success_result, "standard")
            
            assert formatted.success == False
            assert formatted.format_type == "error_fallback"
            assert "Formatting Error" in formatted.content
            assert "Format error" in formatted.content


class TestToolAuditLogger:
    """Test the audit logging functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration"""
        config = Config()
        config.data_dir = temp_dir
        return config
    
    @pytest.fixture
    def audit_logger(self, config):
        """Create audit logger instance"""
        return ToolAuditLogger(config)
    
    @pytest.fixture
    def sample_result(self):
        """Create sample tool execution result"""
        return ToolExecutionResult(
            execution_id="audit-test",
            tool_name="test_tool",
            success=True,
            result={"data": "test"},
            execution_time=1.0,
            metadata={"version": "1.0"}
        )
    
    def test_log_tool_execution(self, audit_logger, sample_result):
        """Test logging tool execution"""
        parameters = {"test_param": "test_value"}
        
        audit_logger.log_tool_execution("test_tool", parameters, sample_result, "user123")
        
        # Check statistics
        assert audit_logger.stats['total_events'] == 1
        assert audit_logger.stats['events_by_type']['tool_execution'] == 1
        assert audit_logger.stats['events_by_tool']['test_tool'] == 1
        assert audit_logger.stats['error_count'] == 0
    
    def test_log_tool_request(self, audit_logger):
        """Test logging tool request"""
        request = ToolCallRequest(
            tool_name="test_tool",
            parameters={"query": "test"},
            confidence=0.9
        )
        
        audit_logger.log_tool_request(request, "user123")
        
        assert audit_logger.stats['total_events'] == 1
        assert audit_logger.stats['events_by_type']['tool_request'] == 1
    
    def test_log_permission_check(self, audit_logger):
        """Test logging permission check"""
        audit_logger.log_permission_check("test_tool", "user123", True, "Access granted")
        
        assert audit_logger.stats['total_events'] == 1
        assert audit_logger.stats['events_by_type']['permission_check'] == 1
    
    def test_log_security_violation(self, audit_logger):
        """Test logging security violation"""
        audit_logger.log_security_violation(
            "test_tool", 
            "user123", 
            "unauthorized_access",
            {"path": "/etc/passwd"}
        )
        
        assert audit_logger.stats['total_events'] == 1
        assert audit_logger.stats['security_violations'] == 1
    
    def test_log_rate_limit_hit(self, audit_logger):
        """Test logging rate limit hit"""
        audit_logger.log_rate_limit_hit("test_tool", "user123", "requests_per_minute")
        
        assert audit_logger.stats['total_events'] == 1
        assert audit_logger.stats['events_by_type']['rate_limit_hit'] == 1
    
    def test_log_system_error(self, audit_logger):
        """Test logging system error"""
        audit_logger.log_system_error(
            "parsing_error",
            "Failed to parse JSON",
            {"input": "invalid json"}
        )
        
        assert audit_logger.stats['total_events'] == 1
        assert audit_logger.stats['events_by_type']['system_error'] == 1
        assert audit_logger.stats['error_count'] == 1
    
    def test_log_user_action(self, audit_logger):
        """Test logging user action"""
        audit_logger.log_user_action(
            "user123",
            "format_result",
            {"tool_name": "test_tool", "format_type": "standard"}
        )
        
        assert audit_logger.stats['total_events'] == 1
        assert audit_logger.stats['events_by_type']['user_action'] == 1
    
    def test_audit_file_creation(self, audit_logger, sample_result):
        """Test that audit file is created and written to"""
        parameters = {"test": "value"}
        
        audit_logger.log_tool_execution("test_tool", parameters, sample_result)
        
        # Check that file exists and has content
        assert audit_logger.audit_file.exists()
        
        content = audit_logger.audit_file.read_text()
        assert "tool_execution" in content
        assert "test_tool" in content
    
    def test_get_recent_events(self, audit_logger, sample_result):
        """Test getting recent events"""
        # Log multiple events
        for i in range(5):
            audit_logger.log_tool_execution(f"tool_{i}", {}, sample_result)
        
        recent_events = audit_logger.get_recent_events(3)
        
        assert len(recent_events) == 3
        assert all("tool_execution" in event["event_type"] for event in recent_events)
    
    def test_cleanup_old_events(self, audit_logger, sample_result):
        """Test cleanup of old events"""
        # Create an event with old timestamp
        old_event = AuditEvent(
            event_type=AuditEventType.TOOL_EXECUTION,
            timestamp=datetime.now() - timedelta(days=40),
            tool_name="old_tool",
            action="execute",
            success=True
        )
        
        # Write old event manually
        with open(audit_logger.audit_file, 'w') as f:
            f.write(json.dumps(old_event.to_dict()) + '\n')
        
        # Add recent event
        audit_logger.log_tool_execution("new_tool", {}, sample_result)
        
        # Cleanup events older than 30 days
        removed_count = audit_logger.cleanup_old_events(30)
        
        assert removed_count == 1
        
        # Check that only recent event remains
        recent_events = audit_logger.get_recent_events(10)
        assert len(recent_events) == 1
        assert recent_events[0]["tool_name"] == "new_tool"
    
    def test_audit_statistics(self, audit_logger, sample_result):
        """Test audit statistics generation"""
        # Log various events
        audit_logger.log_tool_execution("tool1", {}, sample_result)
        audit_logger.log_tool_execution("tool2", {}, sample_result)
        audit_logger.log_security_violation("tool1", "user", "violation", {})
        
        stats = audit_logger.get_audit_statistics()
        
        assert stats['total_events'] == 3
        assert stats['events_by_tool']['tool1'] == 2  # execution + violation
        assert stats['events_by_tool']['tool2'] == 1
        assert stats['security_violations'] == 1
        assert stats['error_count'] == 0  # security violation is not counted as error
        assert 'audit_file' in stats
        assert 'file_size_mb' in stats


class TestToolResultProcessor:
    """Test the main result processor functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration"""
        config = Config()
        config.data_dir = temp_dir
        return config
    
    @pytest.fixture
    def processor(self, config):
        """Create result processor instance"""
        return ToolResultProcessor(config)
    
    def test_parse_tool_request_with_logging(self, processor):
        """Test parsing with audit logging"""
        model_output = '{"tool_name": "test_tool", "parameters": {"query": "test"}}'
        
        requests = processor.parse_tool_request(model_output, "user123")
        
        assert len(requests) == 1
        assert requests[0].tool_name == "test_tool"
        assert processor.stats['requests_processed'] == 1
    
    def test_format_tool_result_with_logging(self, processor):
        """Test formatting with audit logging"""
        result = ToolExecutionResult(
            execution_id="test",
            tool_name="test_tool",
            success=True,
            result={"data": "test"},
            execution_time=1.0
        )
        
        formatted = processor.format_tool_result(result, "standard", "user123")
        
        assert formatted.success == True
        assert formatted.format_type == "standard"
        assert processor.stats['results_formatted'] == 1
    
    def test_error_handling_in_parsing(self, processor):
        """Test error handling in parsing"""
        # Mock parser to raise exception
        with patch.object(processor.parser, 'parse_tool_request', side_effect=Exception("Parse error")):
            requests = processor.parse_tool_request("test input", "user123")
            
            assert len(requests) == 0
            assert processor.stats['parsing_errors'] == 1
    
    def test_error_handling_in_formatting(self, processor):
        """Test error handling in formatting"""
        result = ToolExecutionResult(
            execution_id="test",
            tool_name="test_tool",
            success=True,
            result={"data": "test"},
            execution_time=1.0
        )
        
        # Mock formatter to raise exception
        with patch.object(processor.formatter, 'format_result', side_effect=Exception("Format error")):
            formatted = processor.format_tool_result(result, "standard", "user123")
            
            assert formatted.success == False
            assert formatted.format_type == "error_fallback"
            assert processor.stats['formatting_errors'] == 1
    
    def test_processing_statistics(self, processor):
        """Test processing statistics tracking"""
        # Process multiple requests
        for i in range(3):
            processor.parse_tool_request(f'{{"tool_name": "tool_{i}", "parameters": {{}}}}')
        
        stats = processor.get_processing_stats()
        
        assert stats['requests_processed'] == 3
        assert stats['parsing_errors'] == 0
        assert stats['avg_processing_time'] > 0
        assert 'audit_stats' in stats
    
    def test_get_recent_audit_events(self, processor):
        """Test getting recent audit events"""
        # Generate some events
        processor.parse_tool_request('{"tool_name": "test", "parameters": {}}')
        
        events = processor.get_recent_audit_events(10)
        
        assert len(events) >= 1
        assert any("tool_request" in event["event_type"] for event in events)
    
    def test_cleanup_old_audit_data(self, processor):
        """Test cleanup of old audit data"""
        # This test depends on the audit logger cleanup functionality
        removed = processor.cleanup_old_audit_data(30)
        
        # Should not fail (might be 0 if no old data)
        assert removed >= 0


class TestPublicAPI:
    """Test the public API functions"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration"""
        config = Config()
        config.data_dir = temp_dir
        return config
    
    def test_parse_tool_request_function(self, config):
        """Test the parse_tool_request public function"""
        model_output = '{"tool_name": "test_tool", "parameters": {"query": "test"}}'
        
        requests = parse_tool_request(model_output, config)
        
        assert len(requests) == 1
        assert requests[0].tool_name == "test_tool"
        assert isinstance(requests[0], ToolCallRequest)
    
    def test_format_tool_result_function(self, config):
        """Test the format_tool_result public function"""
        result = ToolExecutionResult(
            execution_id="test",
            tool_name="test_tool",
            success=True,
            result={"data": "test"},
            execution_time=1.0
        )
        
        formatted = format_tool_result(result, "standard", config)
        
        assert formatted.success == True
        assert formatted.format_type == "standard"
        assert isinstance(formatted, FormattedResult)
    
    def test_create_audit_logger_function(self, config):
        """Test the create_audit_logger public function"""
        audit_logger = create_audit_logger(config)
        
        assert isinstance(audit_logger, ToolAuditLogger)
        assert audit_logger.config == config
    
    def test_public_api_with_default_config(self):
        """Test public API with default configuration"""
        # Should work without providing config
        requests = parse_tool_request('{"tool_name": "test", "parameters": {}}')
        assert isinstance(requests, list)
        
        result = ToolExecutionResult(
            execution_id="test",
            tool_name="test_tool",
            success=True,
            result="test",
            execution_time=1.0
        )
        
        formatted = format_tool_result(result)
        assert isinstance(formatted, FormattedResult)


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration"""
        config = Config()
        config.data_dir = temp_dir
        return config
    
    @pytest.fixture
    def processor(self, config):
        """Create result processor instance"""
        return ToolResultProcessor(config)
    
    def test_complete_workflow(self, processor):
        """Test complete workflow from parsing to formatting"""
        # Simulate AI model output
        model_output = '''
        I need to search for information about Python:
        ```json
        {
            "tool_name": "internet_search",
            "parameters": {
                "query": "Python programming best practices",
                "max_results": 3
            }
        }
        ```
        '''
        
        # Parse the request
        requests = processor.parse_tool_request(model_output, "user123")
        assert len(requests) == 1
        
        # Simulate tool execution result
        execution_result = ToolExecutionResult(
            execution_id="search-123",
            tool_name="internet_search",
            success=True,
            result={
                "query": "Python programming best practices",
                "results": [
                    {
                        "title": "Python Best Practices Guide",
                        "url": "https://example.com/python-guide",
                        "snippet": "Learn Python best practices for clean code"
                    }
                ],
                "total_results": 1
            },
            execution_time=2.1,
            metadata={"region": "us-en"}
        )
        
        # Format the result
        formatted = processor.format_tool_result(execution_result, "standard", "user123")
        
        # Verify the complete workflow
        assert formatted.success == True
        assert "Python Best Practices Guide" in formatted.content
        assert "✅" in formatted.content
        assert "internet_search" in formatted.content
    
    def test_multiple_requests_scenario(self, processor):
        """Test scenario with multiple tool requests"""
        model_output = '''
        I need to do several things:
        1. Search for AI news: {"tool_name": "internet_search", "parameters": {"query": "AI news today"}}
        2. Calculate something: call_tool("advanced_calculation", {"operation": "basic", "expression": "100 * 0.15"})
        3. Check system info: get system overview
        '''
        
        requests = processor.parse_tool_request(model_output, "user123")
        
        # Should find multiple requests
        assert len(requests) >= 2  # At least JSON and function call formats
        
        # Verify different request types
        tool_names = [req.tool_name for req in requests]
        assert "internet_search" in tool_names
        assert "advanced_calculation" in tool_names
    
    def test_error_recovery_scenario(self, processor):
        """Test error recovery in realistic scenario"""
        # Parse request with partial error
        model_output = '''
        Valid request: {"tool_name": "test_tool", "parameters": {"query": "test"}}
        Invalid request: {"tool_name": "bad_tool", "parameters": {invalid json}}
        '''
        
        requests = processor.parse_tool_request(model_output, "user123")
        
        # Should recover and parse the valid request
        assert len(requests) == 1
        assert requests[0].tool_name == "test_tool"
        
        # Should have logged the error
        assert processor.stats['parsing_errors'] == 0  # JSON parsing errors don't increment this
    
    def test_audit_trail_scenario(self, processor):
        """Test complete audit trail for a realistic scenario"""
        # Simulate complete interaction
        model_output = '{"tool_name": "file_access", "parameters": {"operation": "read", "path": "/tmp/test.txt"}}'
        
        # Parse request
        requests = processor.parse_tool_request(model_output, "user123")
        
        # Simulate permission check
        processor.audit_logger.log_permission_check("file_access", "user123", True, "User has file access")
        
        # Simulate tool execution
        result = ToolExecutionResult(
            execution_id="file-read-123",
            tool_name="file_access",
            success=True,
            result={"content": "File content here", "size": 1024},
            execution_time=0.8
        )
        
        # Log execution
        processor.audit_logger.log_tool_execution("file_access", requests[0].parameters, result, "user123")
        
        # Format result
        formatted = processor.format_tool_result(result, "detailed", "user123")
        
        # Verify audit trail
        events = processor.get_recent_audit_events(10)
        event_types = [event["event_type"] for event in events]
        
        assert "tool_request" in event_types
        assert "permission_check" in event_types
        assert "tool_execution" in event_types
        assert "user_action" in event_types  # from formatting
    
    def test_performance_monitoring(self, processor):
        """Test performance monitoring during operations"""
        start_time = time.time()
        
        # Process multiple requests to test performance
        for i in range(10):
            model_output = f'{{"tool_name": "tool_{i}", "parameters": {{"query": "test_{i}"}}}}'
            processor.parse_tool_request(model_output, "user123")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify performance statistics
        stats = processor.get_processing_stats()
        assert stats['requests_processed'] == 10
        assert stats['avg_processing_time'] > 0
        assert stats['avg_processing_time'] < total_time  # Should be less than total time
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, processor):
        """Test concurrent processing scenarios"""
        async def process_request(request_id):
            model_output = f'{{"tool_name": "tool_{request_id}", "parameters": {{"query": "test_{request_id}"}}}}'
            return processor.parse_tool_request(model_output, f"user_{request_id}")
        
        # Process multiple requests concurrently
        tasks = [process_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Verify all requests were processed
        assert len(results) == 5
        assert all(len(result) == 1 for result in results)
        
        # Verify statistics
        stats = processor.get_processing_stats()
        assert stats['requests_processed'] == 5 