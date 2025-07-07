#!/usr/bin/env python3
"""
Integration Test Suite Runner for Sovereign AI System

This script runs the complete integration test suite and provides
comprehensive reporting on system integration status.

Usage:
    python tests/run_integration_tests.py [options]

Options:
    --verbose       Show detailed test output
    --fast          Run only critical integration tests
    --report        Generate detailed HTML report
    --coverage      Include code coverage analysis
    --suite <name>  Run specific test suite only
"""

import sys
import os
import subprocess
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_test_suite(test_file: str, verbose: bool = False) -> Dict[str, Any]:
    """Run a specific test suite and return results"""
    print(f"\n{'='*60}")
    print(f"Running Integration Test Suite: {test_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Build pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        "--asyncio-mode=auto",
        "-v" if verbose else "-q",
        "--tb=short",
        "--durations=10"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per test suite
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'test_file': test_file,
            'duration': duration,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
        
    except subprocess.TimeoutExpired:
        return {
            'test_file': test_file,
            'duration': 300.0,
            'return_code': -1,
            'stdout': "",
            'stderr': "Test suite timed out after 5 minutes",
            'success': False
        }
    except Exception as e:
        return {
            'test_file': test_file,
            'duration': 0.0,
            'return_code': -1,
            'stdout': "",
            'stderr': f"Failed to run test suite: {str(e)}",
            'success': False
        }


def parse_test_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse test results and extract summary statistics"""
    total_duration = sum(r['duration'] for r in results)
    successful_suites = [r for r in results if r['success']]
    failed_suites = [r for r in results if not r['success']]
    
    # Parse individual test counts from stdout
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    skipped_tests = 0
    
    for result in results:
        stdout = result['stdout']
        if 'passed' in stdout:
            # Try to extract test counts from pytest output
            import re
            
            # Look for patterns like "5 passed, 2 failed, 1 skipped"
            pattern = r'(\d+)\s+passed'
            match = re.search(pattern, stdout)
            if match:
                passed_tests += int(match.group(1))
            
            pattern = r'(\d+)\s+failed'
            match = re.search(pattern, stdout)
            if match:
                failed_tests += int(match.group(1))
            
            pattern = r'(\d+)\s+skipped'
            match = re.search(pattern, stdout)
            if match:
                skipped_tests += int(match.group(1))
    
    total_tests = passed_tests + failed_tests + skipped_tests
    
    return {
        'total_suites': len(results),
        'successful_suites': len(successful_suites),
        'failed_suites': len(failed_suites),
        'total_duration': total_duration,
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'skipped_tests': skipped_tests,
        'success_rate': len(successful_suites) / len(results) if results else 0.0,
        'test_pass_rate': passed_tests / total_tests if total_tests > 0 else 0.0
    }


def generate_report(results: List[Dict[str, Any]], summary: Dict[str, Any], output_file: str = None):
    """Generate detailed test report"""
    report_lines = []
    
    # Header
    report_lines.append("Sovereign AI System - Integration Test Report")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Summary
    report_lines.append("SUMMARY")
    report_lines.append("-" * 30)
    report_lines.append(f"Total Test Suites: {summary['total_suites']}")
    report_lines.append(f"Successful Suites: {summary['successful_suites']}")
    report_lines.append(f"Failed Suites: {summary['failed_suites']}")
    report_lines.append(f"Suite Success Rate: {summary['success_rate']:.1%}")
    report_lines.append("")
    report_lines.append(f"Total Tests: {summary['total_tests']}")
    report_lines.append(f"Passed Tests: {summary['passed_tests']}")
    report_lines.append(f"Failed Tests: {summary['failed_tests']}")
    report_lines.append(f"Skipped Tests: {summary['skipped_tests']}")
    report_lines.append(f"Test Pass Rate: {summary['test_pass_rate']:.1%}")
    report_lines.append("")
    report_lines.append(f"Total Duration: {summary['total_duration']:.2f} seconds")
    report_lines.append("")
    
    # Detailed Results
    report_lines.append("DETAILED RESULTS")
    report_lines.append("-" * 30)
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        report_lines.append(f"{status} {result['test_file']} ({result['duration']:.2f}s)")
        
        if not result['success']:
            report_lines.append(f"    Error: {result['stderr'][:200]}...")
        
        report_lines.append("")
    
    # Integration Points Status
    report_lines.append("INTEGRATION POINTS STATUS")
    report_lines.append("-" * 30)
    
    integration_points = [
        ("Model Orchestration", "test_comprehensive_integration.py"),
        ("Voice Interface", "test_comprehensive_integration.py"),
        ("Screen Context", "test_comprehensive_integration.py"),
        ("Memory/RAG System", "test_rag_system_integration.py"),
        ("Tool Framework", "test_comprehensive_integration.py"),
        ("API Contracts", "test_api_integration.py"),
        ("Error Handling", "test_error_handling_integration.py"),
        ("Performance", "test_performance_gui_integration.py")
    ]
    
    for point_name, test_file in integration_points:
        point_result = next((r for r in results if test_file in r['test_file']), None)
        if point_result:
            status = "✅ OPERATIONAL" if point_result['success'] else "❌ ISSUES DETECTED"
            report_lines.append(f"{status} {point_name}")
        else:
            report_lines.append(f"⚠️  NOT TESTED {point_name}")
    
    report_lines.append("")
    
    # Recommendations
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 30)
    
    if summary['success_rate'] >= 0.9:
        report_lines.append("✅ System integration is healthy. Ready for production deployment.")
    elif summary['success_rate'] >= 0.7:
        report_lines.append("⚠️  Some integration issues detected. Review failed tests before deployment.")
    else:
        report_lines.append("❌ Significant integration issues. Address failures before proceeding.")
    
    if summary['test_pass_rate'] < 0.8:
        report_lines.append("⚠️  Individual test pass rate is low. Review test implementations.")
    
    # Output report
    report_content = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_content)
        print(f"\nDetailed report saved to: {output_file}")
    else:
        print("\n" + report_content)


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Run Sovereign AI Integration Tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Run only critical tests")
    parser.add_argument("--report", "-r", type=str, help="Generate report file")
    parser.add_argument("--suite", "-s", type=str, help="Run specific test suite")
    parser.add_argument("--coverage", action="store_true", help="Include coverage analysis")
    
    args = parser.parse_args()
    
    # Get test directory
    test_dir = Path(__file__).parent
    
    # Define test suites in order of importance
    if args.fast:
        test_suites = [
            "test_integration.py",  # Basic integration
            "test_api_integration.py",  # API contracts
        ]
    elif args.suite:
        test_suites = [f"test_{args.suite}.py" if not args.suite.endswith('.py') else args.suite]
    else:
        test_suites = [
            "test_integration.py",  # Basic GUI-Backend integration
            "test_comprehensive_integration.py",  # Full system integration
            "test_rag_system_integration.py",  # Memory/RAG integration
            "test_api_integration.py",  # API contracts
            "test_error_handling_integration.py",  # Error handling
            "test_performance_gui_integration.py",  # Performance integration
        ]
    
    print("Sovereign AI System - Integration Test Suite")
    print("=" * 60)
    print(f"Running {len(test_suites)} test suite(s)")
    print(f"Test Directory: {test_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if test files exist
    missing_files = []
    existing_suites = []
    
    for suite in test_suites:
        test_file = test_dir / suite
        if test_file.exists():
            existing_suites.append(str(test_file))
        else:
            missing_files.append(suite)
    
    if missing_files:
        print(f"\nWarning: Missing test files: {', '.join(missing_files)}")
    
    if not existing_suites:
        print("Error: No test files found!")
        return 1
    
    # Run test suites
    results = []
    for suite_file in existing_suites:
        result = run_test_suite(suite_file, args.verbose)
        results.append(result)
        
        # Print immediate feedback
        status = "✅" if result['success'] else "❌"
        print(f"{status} {Path(suite_file).name} completed in {result['duration']:.2f}s")
    
    # Parse and display results
    summary = parse_test_results(results)
    
    print(f"\n{'='*60}")
    print("INTEGRATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Suites: {summary['successful_suites']}/{summary['total_suites']} passed")
    print(f"Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
    print(f"Duration: {summary['total_duration']:.2f} seconds")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    
    # Generate report
    if args.report or summary['success_rate'] < 1.0:
        report_file = args.report or f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        generate_report(results, summary, report_file)
    else:
        generate_report(results, summary)
    
    # Run coverage analysis if requested
    if args.coverage:
        print("\nRunning coverage analysis...")
        try:
            subprocess.run([
                sys.executable, "-m", "pytest",
                "--cov=src.sovereign",
                "--cov-report=html:htmlcov_integration",
                "--cov-report=term",
                *existing_suites
            ], timeout=600)
            print("Coverage report generated in htmlcov_integration/")
        except Exception as e:
            print(f"Coverage analysis failed: {e}")
    
    # Exit with appropriate code
    if summary['success_rate'] >= 0.8:
        print("\n✅ Integration test suite completed successfully!")
        return 0
    else:
        print("\n❌ Integration test suite completed with failures!")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nTest runner failed: {e}")
        sys.exit(1) 