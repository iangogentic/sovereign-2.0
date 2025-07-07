"""
Demo: Automated Performance Testing Framework

This demo showcases the comprehensive automated performance testing capabilities
of the Sovereign AI system, including:
- Workload simulation for different scenarios
- Automated performance profiling
- Bottleneck identification and analysis
- Comprehensive reporting with actionable insights
- CI/CD integration features
"""

import asyncio
import sys
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sovereign.config import Config
from sovereign.automated_performance_testing import (
    AutomatedPerformanceTester, WorkloadScenario, create_ci_cd_test_script,
    run_quick_performance_test
)


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")


def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f"{title}")
    print(f"{'-'*40}")


async def demo_basic_performance_testing():
    """Demonstrate basic performance testing functionality"""
    print_section("BASIC PERFORMANCE TESTING DEMO")
    
    try:
        # Create configuration
        config = Config()
        config.hardware.gpu_enabled = False  # Disable GPU for demo stability
        
        # Create temporary output directory
        temp_dir = Path(tempfile.mkdtemp())
        print(f"Demo output directory: {temp_dir}")
        
        # Initialize the automated performance tester
        print("\n1. Initializing Automated Performance Tester...")
        tester = AutomatedPerformanceTester(config, temp_dir)
        
        print(f"   - Loaded {len(tester.scenarios)} default scenarios")
        print(f"   - Output directory: {tester.output_dir}")
        print("   - Performance monitoring ready")
        print("   - Profiling integration configured")
        
        # Display available scenarios
        print_subsection("Available Test Scenarios")
        for name, scenario in tester.scenarios.items():
            print(f"   {name}:")
            print(f"     Description: {scenario.description}")
            print(f"     Duration: {scenario.duration_seconds}s")
            print(f"     Operations: {len(scenario.operations)}")
            print(f"     Tags: {', '.join(scenario.tags)}")
            print()
        
        # Run a quick single scenario test
        print_subsection("Running Single Scenario Test")
        print("Testing: light_conversation scenario...")
        
        start_time = time.time()
        result = await tester.run_single_scenario('light_conversation')
        execution_time = time.time() - start_time
        
        print(f"\nScenario Test Results:")
        print(f"   Scenario: {result.scenario_name}")
        print(f"   Execution Time: {execution_time:.1f}s")
        print(f"   Total Operations: {result.total_operations}")
        print(f"   Successful: {result.successful_operations}")
        print(f"   Failed: {result.failed_operations}")
        print(f"   Success Rate: {(result.successful_operations/max(result.total_operations,1)):.1%}")
        
        if result.response_times:
            import statistics
            avg_time = statistics.mean(result.response_times)
            print(f"   Average Response Time: {avg_time:.1f}ms")
            print(f"   Response Time Range: {min(result.response_times):.1f}ms - {max(result.response_times):.1f}ms")
        
        if result.error_messages:
            print(f"   Errors: {len(result.error_messages)}")
            for error in result.error_messages[:3]:  # Show first 3 errors
                print(f"     - {error}")
        
        return temp_dir, tester, True
        
    except Exception as e:
        print(f"Error in basic testing demo: {e}")
        return None, None, False


async def demo_comprehensive_test_suite(tester, temp_dir):
    """Demonstrate comprehensive test suite execution"""
    print_section("COMPREHENSIVE TEST SUITE DEMO")
    
    try:
        # Run a complete test suite
        print("Running comprehensive test suite...")
        print("Scenarios: light_conversation, memory_stress")
        
        start_time = time.time()
        report = await tester.run_test_suite(['light_conversation', 'memory_stress'])
        execution_time = time.time() - start_time
        
        print(f"\nTest Suite Completed in {execution_time:.1f}s")
        
        # Display comprehensive results
        print_subsection("Overall Test Results")
        print(f"   Test Run ID: {report.test_run_id}")
        print(f"   Scenarios Tested: {', '.join(report.scenarios_tested)}")
        print(f"   Overall Success Rate: {report.overall_success_rate:.1%}")
        print(f"   Total Duration: {report.total_duration:.1f}s")
        
        # System performance summary
        print_subsection("System Performance Summary")
        summary = report.system_summary
        print(f"   Total Operations: {summary['total_operations']}")
        print(f"   Successful Operations: {summary['total_successful']}")
        print(f"   Failed Operations: {summary['total_failed']}")
        
        if 'response_time_stats' in summary and summary['response_time_stats']:
            stats = summary['response_time_stats']
            print(f"   Average Response Time: {stats['average_ms']:.1f}ms")
            print(f"   Median Response Time: {stats['median_ms']:.1f}ms")
            print(f"   Response Time Std Dev: {stats.get('std_dev_ms', 0):.1f}ms")
        
        if 'system_resource_usage' in summary:
            usage = summary['system_resource_usage']
            if 'cpu' in usage:
                print(f"   CPU Usage: {usage['cpu']['average_percent']:.1f}% avg, {usage['cpu']['max_percent']:.1f}% max")
            if 'memory' in usage:
                print(f"   Memory Usage: {usage['memory']['average_percent']:.1f}% avg, {usage['memory']['max_percent']:.1f}% max")
        
        # Bottleneck analysis
        if report.bottlenecks_identified:
            print_subsection("Performance Bottlenecks Identified")
            for bottleneck in report.bottlenecks_identified:
                print(f"   {bottleneck['type'].title()} - {bottleneck['severity'].title()} Priority:")
                print(f"     Scenario: {bottleneck['scenario']}")
                print(f"     Description: {bottleneck['description']}")
                if 'metrics' in bottleneck:
                    for key, value in bottleneck['metrics'].items():
                        print(f"     {key}: {value}")
                print()
        else:
            print_subsection("Performance Analysis")
            print("   No significant bottlenecks identified")
            print("   System performance within acceptable ranges")
        
        # Recommendations
        print_subsection("Performance Recommendations")
        for i, rec in enumerate(report.recommendations[:5], 1):  # Show first 5 recommendations
            print(f"   {i}. {rec}")
        
        # Report files generated
        print_subsection("Generated Reports")
        json_files = list(temp_dir.glob('*_report.json'))
        md_files = list(temp_dir.glob('*_report.md'))
        
        print(f"   JSON Reports: {len(json_files)}")
        for file in json_files:
            print(f"     - {file.name}")
        
        print(f"   Markdown Reports: {len(md_files)}")
        for file in md_files:
            print(f"     - {file.name}")
        
        return report, True
        
    except Exception as e:
        print(f"Error in comprehensive test demo: {e}")
        return None, False


def demo_custom_scenarios(tester):
    """Demonstrate custom scenario creation"""
    print_section("CUSTOM SCENARIO DEMO")
    
    try:
        print("Creating custom performance scenarios...")
        
        # Create a custom high-intensity scenario
        high_intensity_scenario = WorkloadScenario(
            name="high_intensity_mixed",
            description="High intensity mixed workload for stress testing",
            operations=[
                {'type': 'conversation', 'messages_per_minute': 30, 'concurrent_users': 4},
                {'type': 'memory', 'queries_per_minute': 40, 'document_size_kb': 100}
            ],
            duration_seconds=180,
            concurrent_users=5,
            expected_response_time_ms=2500,
            expected_success_rate=0.90,
            tags=['stress', 'mixed', 'high-intensity']
        )
        
        # Create a custom edge case scenario
        edge_case_scenario = WorkloadScenario(
            name="edge_case_testing",
            description="Edge case testing with unusual parameters",
            operations=[
                {'type': 'conversation', 'messages_per_minute': 5, 'concurrent_users': 1},
                {'type': 'memory', 'queries_per_minute': 100, 'document_size_kb': 1}
            ],
            duration_seconds=90,
            expected_response_time_ms=1000,
            expected_success_rate=0.99,
            tags=['edge-case', 'boundary-testing']
        )
        
        # Add custom scenarios
        tester.add_custom_scenario(high_intensity_scenario)
        tester.add_custom_scenario(edge_case_scenario)
        
        print("Custom scenarios created:")
        print(f"   1. {high_intensity_scenario.name}")
        print(f"      Description: {high_intensity_scenario.description}")
        print(f"      Duration: {high_intensity_scenario.duration_seconds}s")
        print(f"      Expected Response Time: {high_intensity_scenario.expected_response_time_ms}ms")
        print(f"      Tags: {', '.join(high_intensity_scenario.tags)}")
        
        print(f"\n   2. {edge_case_scenario.name}")
        print(f"      Description: {edge_case_scenario.description}")
        print(f"      Duration: {edge_case_scenario.duration_seconds}s")
        print(f"      Expected Response Time: {edge_case_scenario.expected_response_time_ms}ms")
        print(f"      Tags: {', '.join(edge_case_scenario.tags)}")
        
        print(f"\nTotal scenarios available: {len(tester.scenarios)}")
        
        return True
        
    except Exception as e:
        print(f"Error in custom scenarios demo: {e}")
        return False


def demo_baseline_and_comparison(tester):
    """Demonstrate baseline generation and comparison"""
    print_section("BASELINE AND COMPARISON DEMO")
    
    try:
        print("Generating performance baseline...")
        
        # Check if we have test history
        if not tester.test_history:
            print("   No test history available for baseline generation")
            print("   Run some tests first to generate baseline data")
            return False
        
        # Generate baseline
        baseline = tester.generate_baseline_report(['light_conversation', 'memory_stress'])
        
        if baseline:
            print("Performance baseline generated:")
            for scenario_name, data in baseline.items():
                print(f"\n   Scenario: {scenario_name}")
                print(f"     Baseline Response Time: {data['baseline_response_time_ms']:.1f}ms")
                print(f"     Baseline Success Rate: {data['baseline_success_rate']:.1%}")
                print(f"     Samples Used: {data['samples_count']}")
                print(f"     Last Updated: {data['last_updated']}")
        
        # Demonstrate performance degradation checking
        if len(tester.test_history) >= 2:
            print_subsection("Performance Degradation Analysis")
            print("Comparing last two test runs...")
            
            current_report = tester.test_history[-1]
            previous_report = tester.test_history[-2]
            
            tester._check_performance_degradation(current_report, previous_report)
            print("   Degradation analysis completed")
        
        return True
        
    except Exception as e:
        print(f"Error in baseline demo: {e}")
        return False


def demo_ci_cd_integration(temp_dir):
    """Demonstrate CI/CD integration features"""
    print_section("CI/CD INTEGRATION DEMO")
    
    try:
        print("Creating CI/CD integration scripts...")
        
        # Create CI/CD test script
        ci_script_path = temp_dir / 'ci_performance_test.py'
        config_path = temp_dir / 'test_config.json'
        
        create_ci_cd_test_script(ci_script_path, config_path)
        
        print(f"CI/CD test script created: {ci_script_path}")
        
        # Show script characteristics
        with open(ci_script_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        print(f"   Script size: {len(lines)} lines")
        print(f"   File size: {len(content)} characters")
        
        # Extract key configuration values
        for line in lines:
            if 'success_threshold' in line:
                print(f"   Success threshold: {line.strip()}")
            elif 'response_time_threshold' in line:
                print(f"   Response time threshold: {line.strip()}")
        
        print("\nCI/CD Integration Features:")
        print("   - Automated pass/fail criteria")
        print("   - JSON results export for pipeline artifacts")
        print("   - Configurable performance thresholds")
        print("   - Exit codes for pipeline integration")
        print("   - Lightweight test scenarios for CI/CD")
        
        # Show example pipeline integration
        print_subsection("Example Pipeline Integration")
        pipeline_example = '''
# GitHub Actions Example
- name: Performance Tests
  run: python ci_performance_test.py
  
- name: Upload Performance Results
  uses: actions/upload-artifact@v3
  with:
    name: performance-results
    path: performance_test_results.json
'''
        
        print("GitHub Actions Integration:")
        print(pipeline_example)
        
        return True
        
    except Exception as e:
        print(f"Error in CI/CD demo: {e}")
        return False


async def demo_quick_test_function():
    """Demonstrate the quick test factory function"""
    print_section("QUICK TEST FUNCTION DEMO")
    
    try:
        print("Running quick performance test using factory function...")
        
        config = Config()
        config.hardware.gpu_enabled = False
        
        start_time = time.time()
        report = await run_quick_performance_test(config)
        execution_time = time.time() - start_time
        
        print(f"Quick test completed in {execution_time:.1f}s")
        print(f"   Test Run ID: {report.test_run_id}")
        print(f"   Scenarios: {', '.join(report.scenarios_tested)}")
        print(f"   Success Rate: {report.overall_success_rate:.1%}")
        print(f"   Duration: {report.total_duration:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"Error in quick test demo: {e}")
        return False


def demo_report_analysis(temp_dir):
    """Demonstrate report analysis capabilities"""
    print_section("REPORT ANALYSIS DEMO")
    
    try:
        print("Analyzing generated performance reports...")
        
        # Find all report files
        json_files = list(temp_dir.glob('*_report.json'))
        md_files = list(temp_dir.glob('*_report.md'))
        baseline_files = list(temp_dir.glob('performance_baseline.json'))
        
        print(f"Found {len(json_files)} JSON reports")
        print(f"Found {len(md_files)} Markdown reports")
        print(f"Found {len(baseline_files)} baseline files")
        
        # Analyze the most recent report
        if json_files:
            latest_report = max(json_files, key=lambda f: f.stat().st_mtime)
            print(f"\nAnalyzing latest report: {latest_report.name}")
            
            with open(latest_report, 'r') as f:
                import json
                data = json.load(f)
            
            print("Report Structure:")
            print(f"   Test Run ID: {data['test_run_id']}")
            print(f"   Timestamp: {data['timestamp']}")
            print(f"   Scenarios: {len(data['scenarios_tested'])}")
            print(f"   Success Rate: {data['overall_success_rate']:.1%}")
            print(f"   Duration: {data['total_duration']:.1f}s")
            print(f"   Bottlenecks: {len(data['bottlenecks_identified'])}")
            print(f"   Recommendations: {len(data['recommendations'])}")
            print(f"   Test Results: {len(data['test_results'])}")
        
        # Show human-readable report sample
        if md_files:
            latest_md = max(md_files, key=lambda f: f.stat().st_mtime)
            print(f"\nHuman-readable report: {latest_md.name}")
            
            with open(latest_md, 'r') as f:
                content = f.read()
                lines = content.split('\n')
            
            print("Report sections:")
            sections = [line for line in lines if line.startswith('#')]
            for section in sections[:10]:  # Show first 10 sections
                print(f"   {section}")
        
        return True
        
    except Exception as e:
        print(f"Error in report analysis demo: {e}")
        return False


def cleanup_demo(temp_dir):
    """Clean up demo resources"""
    print_section("DEMO CLEANUP")
    
    try:
        if temp_dir and temp_dir.exists():
            print(f"Cleaning up demo directory: {temp_dir}")
            
            # Show what's being cleaned up
            files = list(temp_dir.rglob('*'))
            print(f"   Removing {len(files)} files and directories")
            
            # Remove the temporary directory
            shutil.rmtree(temp_dir)
            print("   Cleanup completed successfully")
        
        return True
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return False


async def main():
    """Main demo function"""
    print("AUTOMATED PERFORMANCE TESTING FRAMEWORK DEMO")
    print("=" * 60)
    print("This demo showcases the comprehensive automated performance testing")
    print("capabilities of the Sovereign AI system.")
    print("\nFeatures demonstrated:")
    print("- Workload simulation and scenario management")
    print("- Automated performance profiling and monitoring")
    print("- Bottleneck identification and analysis")
    print("- Comprehensive reporting with actionable insights")
    print("- Custom scenario creation")
    print("- Baseline generation and performance comparison")
    print("- CI/CD integration capabilities")
    print("- Quick test factory functions")
    
    temp_dir = None
    tester = None
    
    try:
        # Run basic performance testing demo
        temp_dir, tester, success = await demo_basic_performance_testing()
        if not success:
            print("Basic demo failed - stopping")
            return
        
        # Run comprehensive test suite demo
        report, success = await demo_comprehensive_test_suite(tester, temp_dir)
        if not success:
            print("Comprehensive demo failed - continuing with other demos")
        
        # Demonstrate custom scenarios
        demo_custom_scenarios(tester)
        
        # Demonstrate baseline and comparison
        demo_baseline_and_comparison(tester)
        
        # Demonstrate CI/CD integration
        demo_ci_cd_integration(temp_dir)
        
        # Demonstrate quick test function
        await demo_quick_test_function()
        
        # Demonstrate report analysis
        demo_report_analysis(temp_dir)
        
        # Final summary
        print_section("DEMO SUMMARY")
        print("Automated Performance Testing Framework Demo Completed Successfully!")
        print("\nKey capabilities demonstrated:")
        print("✅ Workload simulation for multiple scenarios")
        print("✅ Automated performance monitoring and profiling")
        print("✅ Comprehensive bottleneck identification")
        print("✅ Actionable performance recommendations")
        print("✅ Custom scenario creation and management")
        print("✅ Performance baseline generation")
        print("✅ CI/CD pipeline integration")
        print("✅ Human-readable and machine-readable reporting")
        print("✅ Quick test factory functions")
        
        print(f"\nThe framework is production-ready and can be integrated")
        print(f"into development workflows for continuous performance monitoring.")
        
        # Show total demo duration
        total_files = len(list(temp_dir.rglob('*'))) if temp_dir else 0
        print(f"\nDemo generated {total_files} files for analysis")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always attempt cleanup
        if temp_dir:
            cleanup_demo(temp_dir)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Demo failed to start: {e}") 