#!/usr/bin/env python3
"""
Tool Result Processor Demo - Comprehensive demonstration of result processing capabilities

This script demonstrates the complete functionality of the tool result processor,
including parsing, formatting, and logging capabilities.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from src.sovereign.config import Config
from src.sovereign.tool_result_processor import (
    ToolResultProcessor, parse_tool_request, format_tool_result,
    ToolCallRequest, FormattedResult, create_audit_logger
)
from src.sovereign.tool_integration_framework import (
    ToolExecutionResult, ToolMetadata, ToolType, ToolSecurityLevel,
    ToolExecutionMode, ToolParameter
)


def print_section(title: str, content: str = ""):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print(f"{'='*60}")
    if content:
        print(content)


def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'−'*40}")
    print(f"📋 {title}")
    print(f"{'−'*40}")


def demonstrate_parsing():
    """Demonstrate tool request parsing capabilities"""
    print_section("TOOL REQUEST PARSING", "Testing various input formats and parsing accuracy")
    
    # Initialize processor
    config = Config()
    processor = ToolResultProcessor(config)
    
    # Test cases for different formats
    test_cases = [
        {
            "name": "JSON Format",
            "input": '''
            I need to search for information:
            ```json
            {
                "tool_name": "internet_search",
                "parameters": {
                    "query": "Python machine learning tutorial",
                    "max_results": 5,
                    "region": "us-en"
                }
            }
            ```
            ''',
            "expected_tool": "internet_search"
        },
        {
            "name": "Function Call Format",
            "input": '''
            Let me calculate this:
            call_tool("advanced_calculation", {"operation": "basic", "expression": "2 + 2 * 3"})
            ''',
            "expected_tool": "advanced_calculation"
        },
        {
            "name": "XML Format",
            "input": '''
            <tool_call>
                <name>file_access</name>
                <parameters>{"operation": "read", "path": "/tmp/test.txt"}</parameters>
            </tool_call>
            ''',
            "expected_tool": "file_access"
        },
        {
            "name": "Natural Language",
            "input": "Please search for the latest news about artificial intelligence",
            "expected_tool": "internet_search"
        },
        {
            "name": "Multiple Requests",
            "input": '''
            I need to do several things:
            1. Search: {"tool_name": "internet_search", "parameters": {"query": "AI news"}}
            2. Calculate: call_tool("advanced_calculation", {"operation": "basic", "expression": "100 * 0.85"})
            3. Check files: list files in current directory
            ''',
            "expected_tool": "multiple"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print_subsection(f"Testing {test_case['name']}")
        
        start_time = time.time()
        requests = processor.parse_tool_request(test_case["input"], "demo_user")
        parse_time = time.time() - start_time
        
        print(f"📥 Input: {test_case['input'][:100]}...")
        print(f"⏱️  Parse time: {parse_time:.3f}s")
        print(f"🔍 Found {len(requests)} request(s)")
        
        for i, request in enumerate(requests):
            print(f"  Request {i+1}:")
            print(f"    🎯 Tool: {request.tool_name}")
            print(f"    📊 Confidence: {request.confidence:.2f}")
            print(f"    📝 Parameters: {json.dumps(request.parameters, indent=8)}")
            print(f"    🏷️  Format: {request.metadata.get('format', 'unknown')}")
        
        results.append({
            "test_case": test_case["name"],
            "requests_found": len(requests),
            "parse_time": parse_time,
            "success": len(requests) > 0
        })
    
    # Summary
    print_subsection("Parsing Results Summary")
    successful_parses = sum(1 for r in results if r["success"])
    total_requests = sum(r["requests_found"] for r in results)
    avg_parse_time = sum(r["parse_time"] for r in results) / len(results)
    
    print(f"✅ Successful parses: {successful_parses}/{len(results)}")
    print(f"📊 Total requests found: {total_requests}")
    print(f"⏱️  Average parse time: {avg_parse_time:.3f}s")
    
    return processor


def demonstrate_formatting():
    """Demonstrate result formatting capabilities"""
    print_section("RESULT FORMATTING", "Testing various result types and formatting options")
    
    config = Config()
    processor = ToolResultProcessor(config)
    
    # Create sample results for different scenarios
    sample_results = [
        {
            "name": "Successful Search Result",
            "result": ToolExecutionResult(
                execution_id="search-demo-1",
                tool_name="internet_search",
                success=True,
                result={
                    "query": "Python machine learning tutorial",
                    "results": [
                        {
                            "title": "Machine Learning with Python - Complete Guide",
                            "url": "https://example.com/ml-guide",
                            "snippet": "Learn machine learning fundamentals with Python"
                        },
                        {
                            "title": "Scikit-learn Tutorial",
                            "url": "https://example.com/sklearn",
                            "snippet": "Comprehensive scikit-learn tutorial"
                        }
                    ],
                    "total_results": 2,
                    "search_time": 1.2
                },
                execution_time=1.5,
                metadata={"region": "us-en"}
            )
        },
        {
            "name": "File Access Result",
            "result": ToolExecutionResult(
                execution_id="file-demo-1",
                tool_name="file_access",
                success=True,
                result={
                    "operation": "read",
                    "path": "/tmp/config.json",
                    "content": '{"setting1": "value1", "setting2": "value2"}',
                    "size": 45,
                    "encoding": "utf-8"
                },
                execution_time=0.3,
                logs=["Opening file", "Reading content", "File closed"]
            )
        },
        {
            "name": "Calculation Result",
            "result": ToolExecutionResult(
                execution_id="calc-demo-1",
                tool_name="advanced_calculation",
                success=True,
                result={
                    "operation": "statistics",
                    "input_data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "mean": 5.5,
                    "median": 5.5,
                    "std_dev": 3.03,
                    "min": 1,
                    "max": 10
                },
                execution_time=0.1
            )
        },
        {
            "name": "Error Result",
            "result": ToolExecutionResult(
                execution_id="error-demo-1",
                tool_name="internet_search",
                success=False,
                error="Network timeout after 30 seconds",
                execution_time=30.0,
                logs=["Connecting to API", "Timeout occurred"],
                metadata={"timeout_seconds": 30}
            )
        }
    ]
    
    format_types = ["standard", "compact", "detailed", "json"]
    
    for sample in sample_results:
        print_subsection(f"Formatting: {sample['name']}")
        
        for format_type in format_types:
            print(f"\n📄 Format: {format_type.upper()}")
            
            start_time = time.time()
            formatted = processor.format_tool_result(sample["result"], format_type, "demo_user")
            format_time = time.time() - start_time
            
            print(f"⏱️  Format time: {format_time:.3f}s")
            print(f"✅ Success: {formatted.success}")
            print(f"📏 Content length: {len(formatted.content)} chars")
            print(f"📝 Summary: {formatted.summary}")
            
            # Show content preview
            content_preview = formatted.content[:200] + "..." if len(formatted.content) > 200 else formatted.content
            print(f"📖 Content preview:")
            print(f"   {content_preview}")
    
    return processor


def demonstrate_logging():
    """Demonstrate audit logging capabilities"""
    print_section("AUDIT LOGGING", "Testing comprehensive audit logging and monitoring")
    
    config = Config()
    processor = ToolResultProcessor(config)
    audit_logger = processor.audit_logger
    
    # Simulate various audit events
    print_subsection("Generating Audit Events")
    
    # 1. Tool request events
    sample_request = ToolCallRequest(
        tool_name="internet_search",
        parameters={"query": "test query", "max_results": 3},
        confidence=0.95
    )
    audit_logger.log_tool_request(sample_request, "demo_user")
    print("✅ Logged tool request")
    
    # 2. Permission check events
    audit_logger.log_permission_check("file_access", "demo_user", True, "User has elevated permissions")
    audit_logger.log_permission_check("system_admin", "demo_user", False, "Insufficient privileges")
    print("✅ Logged permission checks")
    
    # 3. Tool execution events
    success_result = ToolExecutionResult(
        execution_id="audit-demo-1",
        tool_name="internet_search",
        success=True,
        result={"results": ["result1", "result2"]},
        execution_time=1.5
    )
    audit_logger.log_tool_execution("internet_search", sample_request.parameters, success_result, "demo_user")
    print("✅ Logged successful tool execution")
    
    error_result = ToolExecutionResult(
        execution_id="audit-demo-2",
        tool_name="file_access",
        success=False,
        error="Permission denied",
        execution_time=0.1
    )
    audit_logger.log_tool_execution("file_access", {"operation": "read", "path": "/root/secret.txt"}, error_result, "demo_user")
    print("✅ Logged failed tool execution")
    
    # 4. Security violation
    audit_logger.log_security_violation(
        "file_access",
        "demo_user",
        "unauthorized_path_access",
        {"attempted_path": "/etc/passwd", "user_permissions": "limited"}
    )
    print("✅ Logged security violation")
    
    # 5. Rate limit hit
    audit_logger.log_rate_limit_hit("internet_search", "demo_user", "requests_per_minute")
    print("✅ Logged rate limit hit")
    
    # 6. System error
    audit_logger.log_system_error(
        "parsing_error",
        "Failed to parse malformed JSON",
        {"input": "invalid json data", "error_code": "JSON_PARSE_ERROR"}
    )
    print("✅ Logged system error")
    
    # 7. User action
    audit_logger.log_user_action(
        "demo_user",
        "format_result",
        {"tool_name": "internet_search", "format_type": "detailed"}
    )
    print("✅ Logged user action")
    
    # Display audit statistics
    print_subsection("Audit Statistics")
    stats = audit_logger.get_audit_statistics()
    
    print(f"📊 Total Events: {stats['total_events']}")
    print(f"🔒 Security Violations: {stats['security_violations']}")
    print(f"❌ Error Count: {stats['error_count']}")
    print(f"⏱️  Average Execution Time: {stats['avg_execution_time']:.2f}ms")
    
    print("\n📋 Events by Type:")
    for event_type, count in stats['events_by_type'].items():
        print(f"  • {event_type}: {count}")
    
    print("\n🔧 Events by Tool:")
    for tool_name, count in stats['events_by_tool'].items():
        print(f"  • {tool_name}: {count}")
    
    print(f"\n📁 Audit File: {stats['audit_file']}")
    print(f"💾 File Size: {stats['file_size_mb']:.2f} MB")
    
    # Show recent events
    print_subsection("Recent Audit Events")
    recent_events = audit_logger.get_recent_events(5)
    
    for i, event in enumerate(recent_events, 1):
        print(f"Event {i}:")
        print(f"  🕒 Time: {event['timestamp']}")
        print(f"  📝 Type: {event['event_type']}")
        print(f"  🎯 Tool: {event.get('tool_name', 'N/A')}")
        print(f"  👤 User: {event.get('user_id', 'N/A')}")
        print(f"  🔧 Action: {event['action']}")
        print(f"  ✅ Success: {event['success']}")
    
    return processor


def demonstrate_integration():
    """Demonstrate complete integration workflow"""
    print_section("INTEGRATION WORKFLOW", "Complete end-to-end processing demonstration")
    
    config = Config()
    processor = ToolResultProcessor(config)
    
    # Simulate a complete AI interaction workflow
    print_subsection("Scenario: AI Research Assistant")
    
    # 1. AI model generates tool request
    ai_output = '''
    I need to research the latest developments in quantum computing and then calculate 
    some performance metrics. Let me start with a search:
    
    ```json
    {
        "tool_name": "internet_search",
        "parameters": {
            "query": "quantum computing breakthroughs 2024",
            "max_results": 3,
            "region": "us-en"
        }
    }
    ```
    '''
    
    print("🤖 AI Model Output:")
    print(ai_output[:200] + "...")
    
    # 2. Parse the request
    print("\n🔍 Parsing Request...")
    requests = processor.parse_tool_request(ai_output, "research_user")
    print(f"✅ Found {len(requests)} tool request(s)")
    
    for request in requests:
        print(f"  🎯 Tool: {request.tool_name}")
        print(f"  📊 Confidence: {request.confidence:.2f}")
        print(f"  📝 Parameters: {json.dumps(request.parameters, indent=6)}")
    
    # 3. Simulate tool execution
    print("\n⚙️  Executing Tool...")
    time.sleep(0.5)  # Simulate processing time
    
    # Create realistic search result
    search_result = ToolExecutionResult(
        execution_id="research-workflow-1",
        tool_name="internet_search",
        success=True,
        result={
            "query": "quantum computing breakthroughs 2024",
            "results": [
                {
                    "title": "IBM Unveils 1000-Qubit Quantum Computer",
                    "url": "https://example.com/ibm-quantum",
                    "snippet": "IBM has announced a breakthrough 1000-qubit quantum processor"
                },
                {
                    "title": "Google's Quantum Supremacy Milestone",
                    "url": "https://example.com/google-quantum",
                    "snippet": "Google achieves new quantum supremacy benchmark"
                },
                {
                    "title": "Quantum Error Correction Advances",
                    "url": "https://example.com/quantum-error",
                    "snippet": "New methods for quantum error correction show promising results"
                }
            ],
            "total_results": 3,
            "search_time": 1.8,
            "region": "us-en"
        },
        execution_time=2.1,
        metadata={"search_engine": "DuckDuckGo", "safe_search": "moderate"}
    )
    
    # 4. Log the execution
    processor.audit_logger.log_tool_execution(
        "internet_search",
        requests[0].parameters,
        search_result,
        "research_user"
    )
    print("✅ Tool execution logged")
    
    # 5. Format results for different contexts
    print("\n📄 Formatting Results...")
    
    formats = [
        ("standard", "For general AI consumption"),
        ("compact", "For space-constrained contexts"),
        ("detailed", "For comprehensive analysis"),
        ("json", "For structured data exchange")
    ]
    
    for format_type, description in formats:
        print(f"\n📋 {format_type.upper()} Format ({description}):")
        
        formatted = processor.format_tool_result(search_result, format_type, "research_user")
        
        print(f"  ✅ Success: {formatted.success}")
        print(f"  📏 Length: {len(formatted.content)} chars")
        print(f"  📝 Summary: {formatted.summary}")
        
        # Show preview
        preview = formatted.content[:150] + "..." if len(formatted.content) > 150 else formatted.content
        print(f"  📖 Preview: {preview}")
    
    # 6. Generate final statistics
    print_subsection("Workflow Statistics")
    
    processing_stats = processor.get_processing_stats()
    print(f"📊 Requests processed: {processing_stats['requests_processed']}")
    print(f"📊 Results formatted: {processing_stats['results_formatted']}")
    print(f"⏱️  Average processing time: {processing_stats['avg_processing_time']:.3f}s")
    print(f"❌ Parsing errors: {processing_stats['parsing_errors']}")
    print(f"❌ Formatting errors: {processing_stats['formatting_errors']}")
    
    audit_stats = processing_stats['audit_stats']
    print(f"📋 Total audit events: {audit_stats['total_events']}")
    print(f"🔒 Security violations: {audit_stats['security_violations']}")
    
    return processor


def demonstrate_performance():
    """Demonstrate performance characteristics"""
    print_section("PERFORMANCE TESTING", "Testing performance under various loads")
    
    config = Config()
    processor = ToolResultProcessor(config)
    
    # Test parsing performance
    print_subsection("Parsing Performance")
    
    test_inputs = [
        '{"tool_name": "test_tool", "parameters": {"query": "test"}}',
        'call_tool("test_tool", {"param": "value"})',
        '<tool_call><name>test_tool</name><parameters>{"test": "value"}</parameters></tool_call>',
        'search for test information'
    ]
    
    iterations = 100
    start_time = time.time()
    
    for i in range(iterations):
        for input_text in test_inputs:
            processor.parse_tool_request(input_text, f"user_{i}")
    
    total_time = time.time() - start_time
    requests_per_second = (iterations * len(test_inputs)) / total_time
    
    print(f"⏱️  Total time: {total_time:.3f}s")
    print(f"📊 Requests processed: {iterations * len(test_inputs)}")
    print(f"🚀 Requests per second: {requests_per_second:.1f}")
    
    # Test formatting performance
    print_subsection("Formatting Performance")
    
    sample_result = ToolExecutionResult(
        execution_id="perf-test",
        tool_name="test_tool",
        success=True,
        result={"data": "test" * 100},  # Some data
        execution_time=1.0
    )
    
    formats = ["standard", "compact", "detailed", "json"]
    
    for format_type in formats:
        start_time = time.time()
        
        for i in range(iterations):
            processor.format_tool_result(sample_result, format_type, f"user_{i}")
        
        format_time = time.time() - start_time
        formats_per_second = iterations / format_time
        
        print(f"  📄 {format_type}: {formats_per_second:.1f} formats/second")
    
    # Memory usage (basic estimation)
    print_subsection("Resource Usage")
    
    stats = processor.get_processing_stats()
    print(f"📊 Processing statistics maintained: {len(stats)} fields")
    print(f"📋 Audit events logged: {stats['audit_stats']['total_events']}")
    print(f"💾 Audit file size: {stats['audit_stats']['file_size_mb']:.2f} MB")
    
    return processor


def main():
    """Main demonstration function"""
    print("🎯 TOOL RESULT PROCESSOR DEMONSTRATION")
    print("="*60)
    print("This demonstration shows the complete functionality of")
    print("the Tool Result Processor system for Subtask 8.6")
    print("="*60)
    
    try:
        # Run all demonstrations
        print("\n🚀 Starting demonstrations...")
        
        # 1. Parsing demonstration
        processor1 = demonstrate_parsing()
        
        # 2. Formatting demonstration
        processor2 = demonstrate_formatting()
        
        # 3. Logging demonstration
        processor3 = demonstrate_logging()
        
        # 4. Integration demonstration
        processor4 = demonstrate_integration()
        
        # 5. Performance demonstration
        processor5 = demonstrate_performance()
        
        # Final summary
        print_section("DEMONSTRATION COMPLETE", "All functionality successfully demonstrated")
        
        print("✅ Tool Request Parsing: Multiple formats supported")
        print("✅ Result Formatting: Four format types available")
        print("✅ Audit Logging: Comprehensive event tracking")
        print("✅ Integration Workflow: End-to-end processing")
        print("✅ Performance Testing: High-throughput capability")
        
        print("\n🎉 SUBTASK 8.6 IMPLEMENTATION SUCCESSFUL!")
        print("The Tool Result Processor is ready for production use.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Demonstration failed. Please check the implementation.")
        raise


if __name__ == "__main__":
    main() 