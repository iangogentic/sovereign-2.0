"""
Integration Test Orchestration System

This module provides comprehensive orchestration for integration testing,
coordinating environment setup, test execution, validation, and reporting
to ensure reliable and efficient testing workflows.

Features:
- Complete test lifecycle management
- Environment setup and validation
- Test suite orchestration
- Real-time monitoring and reporting
- Automated cleanup and teardown
- Performance benchmarking
- Test result analysis and recommendations
"""

import os
import sys
import time
import json
import asyncio
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from contextlib import asynccontextmanager

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from test_environment_setup import TestEnvironmentManager, create_test_environment
from test_environment_config import (
    TestConfigurationManager, TestEnvironmentType, TestComplexity, 
    get_standard_config
)
from test_environment_validator import (
    EnvironmentValidator, ServiceHealthChecker, validate_environment, 
    check_service_health
)


class TestPhase(Enum):
    """Test execution phases"""
    SETUP = "setup"
    VALIDATION = "validation"
    EXECUTION = "execution"
    CLEANUP = "cleanup"
    REPORTING = "reporting"


class TestResult(Enum):
    """Test execution results"""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class TestSuiteConfig:
    """Configuration for a test suite"""
    name: str
    description: str
    test_files: List[str]
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 0
    parallel_execution: bool = False
    required_services: List[str] = field(default_factory=list)
    environment_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestExecutionResult:
    """Result of test execution"""
    suite_name: str
    result: TestResult
    execution_time_ms: float
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    error_message: Optional[str] = None
    output: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class IntegrationTestReport:
    """Comprehensive integration test report"""
    test_session_id: str
    start_time: float
    end_time: float
    duration_ms: float
    overall_result: TestResult
    environment_config: Dict[str, Any]
    validation_results: Dict[str, Any]
    health_check_results: Dict[str, Any]
    test_suite_results: List[TestExecutionResult]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    artifacts: Dict[str, str] = field(default_factory=dict)


class IntegrationTestOrchestrator:
    """Main orchestrator for integration testing"""
    
    def __init__(self, config_manager: Optional[TestConfigurationManager] = None):
        self.config_manager = config_manager or TestConfigurationManager()
        self.environment_manager = None
        self.current_phase = None
        self.test_session_id = None
        self.start_time = None
        self.results = []
        self.performance_metrics = {}
        self.logger = self._setup_logger()
        
        # Define standard test suites
        self.test_suites = self._define_test_suites()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up orchestrator logger"""
        logger = logging.getLogger("integration_test_orchestrator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _define_test_suites(self) -> Dict[str, TestSuiteConfig]:
        """Define standard test suites"""
        return {
            "comprehensive_integration": TestSuiteConfig(
                name="comprehensive_integration",
                description="Comprehensive integration tests covering all major components",
                test_files=["test_comprehensive_integration.py"],
                timeout_seconds=600,
                required_services=["orchestrator", "memory_manager", "vector_search", "embedding_service"],
                environment_requirements={"max_memory_mb": 4096, "enable_gpu": True}
            ),
            
            "api_integration": TestSuiteConfig(
                name="api_integration",
                description="API contract validation and integration tests",
                test_files=["test_api_integration.py"],
                timeout_seconds=300,
                required_services=["orchestrator", "memory_manager", "vector_search"],
                environment_requirements={"max_memory_mb": 2048}
            ),
            
            "error_handling": TestSuiteConfig(
                name="error_handling",
                description="Error handling and resilience tests",
                test_files=["test_error_handling_integration.py"],
                timeout_seconds=400,
                required_services=["orchestrator", "memory_manager"],
                environment_requirements={"max_memory_mb": 2048}
            ),
            
            "rag_system": TestSuiteConfig(
                name="rag_system",
                description="RAG system integration and performance tests",
                test_files=["test_rag_system_integration.py"],
                timeout_seconds=500,
                required_services=["memory_manager", "vector_search", "embedding_service"],
                environment_requirements={"max_memory_mb": 4096, "enable_gpu": True}
            ),
            
            "performance_gui": TestSuiteConfig(
                name="performance_gui",
                description="Performance monitoring and GUI integration tests",
                test_files=["test_performance_gui_integration.py"],
                timeout_seconds=300,
                required_services=["performance_monitor"],
                environment_requirements={"max_memory_mb": 2048, "enable_performance_monitoring": True}
            ),
            
            "privacy_integration": TestSuiteConfig(
                name="privacy_integration",
                description="Privacy and security integration tests",
                test_files=["test_privacy_integration.py"],
                timeout_seconds=300,
                required_services=["privacy_manager", "consent_manager"],
                environment_requirements={"max_memory_mb": 2048}
            )
        }
    
    async def run_full_integration_test_suite(self, 
                                            env_type: TestEnvironmentType = TestEnvironmentType.INTEGRATION,
                                            complexity: TestComplexity = TestComplexity.STANDARD,
                                            custom_config: Optional[Dict[str, Any]] = None,
                                            selected_suites: Optional[List[str]] = None) -> IntegrationTestReport:
        """Run the complete integration test suite"""
        
        self.test_session_id = f"integration_test_{int(time.time())}"
        self.start_time = time.time()
        
        self.logger.info(f"Starting integration test session: {self.test_session_id}")
        self.logger.info(f"Environment type: {env_type.value}, Complexity: {complexity.value}")
        
        # Get environment configuration
        env_config = self.config_manager.get_config(env_type, complexity, custom_config)
        
        report = IntegrationTestReport(
            test_session_id=self.test_session_id,
            start_time=self.start_time,
            end_time=0,
            duration_ms=0,
            overall_result=TestResult.PASS,
            environment_config=asdict(env_config),
            validation_results={},
            health_check_results={},
            test_suite_results=[],
            performance_metrics={},
            recommendations=[]
        )
        
        try:
            async with self._test_execution_context(env_config) as context:
                # Phase 1: Environment Setup and Validation
                await self._execute_phase(TestPhase.SETUP, report)
                await self._execute_phase(TestPhase.VALIDATION, report)
                
                # Phase 2: Test Execution
                await self._execute_phase(TestPhase.EXECUTION, report, selected_suites)
                
                # Phase 3: Cleanup and Reporting
                await self._execute_phase(TestPhase.CLEANUP, report)
                await self._execute_phase(TestPhase.REPORTING, report)
                
        except Exception as e:
            self.logger.error(f"Integration test session failed: {str(e)}")
            report.overall_result = TestResult.ERROR
            report.recommendations.append(f"Session failed with error: {str(e)}")
        
        finally:
            # Finalize report
            report.end_time = time.time()
            report.duration_ms = (report.end_time - report.start_time) * 1000
            
            self.logger.info(f"Integration test session completed in {report.duration_ms:.1f}ms")
            self.logger.info(f"Overall result: {report.overall_result.value}")
        
        return report
    
    @asynccontextmanager
    async def _test_execution_context(self, env_config):
        """Context manager for test execution lifecycle"""
        try:
            # Setup environment
            self.environment_manager = await create_test_environment(env_config)
            yield self.environment_manager
            
        finally:
            # Cleanup environment
            if self.environment_manager:
                await self.environment_manager.teardown_environment()
                self.environment_manager = None
    
    async def _execute_phase(self, 
                           phase: TestPhase, 
                           report: IntegrationTestReport,
                           selected_suites: Optional[List[str]] = None) -> None:
        """Execute a specific test phase"""
        self.current_phase = phase
        phase_start_time = time.time()
        
        self.logger.info(f"Executing phase: {phase.value}")
        
        try:
            if phase == TestPhase.SETUP:
                await self._phase_setup(report)
            elif phase == TestPhase.VALIDATION:
                await self._phase_validation(report)
            elif phase == TestPhase.EXECUTION:
                await self._phase_execution(report, selected_suites)
            elif phase == TestPhase.CLEANUP:
                await self._phase_cleanup(report)
            elif phase == TestPhase.REPORTING:
                await self._phase_reporting(report)
                
        except Exception as e:
            self.logger.error(f"Phase {phase.value} failed: {str(e)}")
            report.recommendations.append(f"Phase {phase.value} failed: {str(e)}")
            raise
        
        phase_duration = (time.time() - phase_start_time) * 1000
        self.logger.info(f"Phase {phase.value} completed in {phase_duration:.1f}ms")
    
    async def _phase_setup(self, report: IntegrationTestReport) -> None:
        """Execute setup phase"""
        # Environment is already set up by context manager
        # Record environment information
        if self.environment_manager:
            env_info = self.environment_manager.get_environment_info()
            report.environment_config.update(env_info)
            
            self.logger.info(f"Environment setup complete: {env_info['config']['environment_id']}")
            self.logger.info(f"Services initialized: {len(env_info['services'])}")
    
    async def _phase_validation(self, report: IntegrationTestReport) -> None:
        """Execute validation phase"""
        # Validate environment configuration
        validator = EnvironmentValidator(self.environment_manager.config)
        validation_results = await validator.validate_complete_environment()
        report.validation_results = validation_results
        
        if not validation_results["overall_status"]:
            critical_issues = validation_results["critical_issues"]
            if critical_issues > 0:
                raise Exception(f"Environment validation failed with {critical_issues} critical issues")
        
        # Check service health
        health_checker = ServiceHealthChecker(self.environment_manager)
        health_results = await health_checker.check_all_services_health()
        report.health_check_results = health_results
        
        self.logger.info(f"Validation complete: {validation_results['checks_passed']}/{validation_results['checks_performed']} checks passed")
        self.logger.info(f"Service health: {health_results['services_healthy']}/{health_results['services_checked']} services healthy")
    
    async def _phase_execution(self, report: IntegrationTestReport, selected_suites: Optional[List[str]]) -> None:
        """Execute test execution phase"""
        # Determine which test suites to run
        suites_to_run = selected_suites or list(self.test_suites.keys())
        
        self.logger.info(f"Executing {len(suites_to_run)} test suites: {', '.join(suites_to_run)}")
        
        # Execute test suites
        for suite_name in suites_to_run:
            if suite_name in self.test_suites:
                suite_config = self.test_suites[suite_name]
                result = await self._execute_test_suite(suite_config)
                report.test_suite_results.append(result)
                
                # Update overall result
                if result.result == TestResult.FAIL:
                    report.overall_result = TestResult.FAIL
                elif result.result == TestResult.ERROR and report.overall_result == TestResult.PASS:
                    report.overall_result = TestResult.ERROR
            else:
                self.logger.warning(f"Unknown test suite: {suite_name}")
        
        # Analyze results
        total_tests = sum(r.tests_run for r in report.test_suite_results)
        total_passed = sum(r.tests_passed for r in report.test_suite_results)
        total_failed = sum(r.tests_failed for r in report.test_suite_results)
        
        self.logger.info(f"Test execution complete: {total_passed}/{total_tests} tests passed")
        if total_failed > 0:
            self.logger.warning(f"{total_failed} tests failed")
    
    async def _phase_cleanup(self, report: IntegrationTestReport) -> None:
        """Execute cleanup phase"""
        # Collect final performance metrics
        if self.environment_manager:
            env_status = self.environment_manager.health_monitor.get_environment_status()
            report.performance_metrics = env_status.get("metrics", {})
        
        self.logger.info("Cleanup phase completed")
    
    async def _phase_reporting(self, report: IntegrationTestReport) -> None:
        """Execute reporting phase"""
        # Generate recommendations based on results
        report.recommendations.extend(await self._generate_recommendations(report))
        
        # Save test artifacts
        await self._save_test_artifacts(report)
        
        self.logger.info("Reporting phase completed")
    
    async def _execute_test_suite(self, suite_config: TestSuiteConfig) -> TestExecutionResult:
        """Execute a single test suite"""
        self.logger.info(f"Executing test suite: {suite_config.name}")
        start_time = time.time()
        
        result = TestExecutionResult(
            suite_name=suite_config.name,
            result=TestResult.PASS,
            execution_time_ms=0
        )
        
        try:
            # Check service requirements
            await self._check_suite_requirements(suite_config)
            
            # Execute tests using pytest
            test_result = await self._run_pytest_suite(suite_config)
            
            # Parse results
            result.tests_run = test_result.get("tests_run", 0)
            result.tests_passed = test_result.get("tests_passed", 0)
            result.tests_failed = test_result.get("tests_failed", 0)
            result.tests_skipped = test_result.get("tests_skipped", 0)
            result.output = test_result.get("output", "")
            
            # Determine overall result
            if result.tests_failed > 0:
                result.result = TestResult.FAIL
            elif result.tests_run == 0:
                result.result = TestResult.SKIP
                result.error_message = "No tests were run"
            
        except Exception as e:
            result.result = TestResult.ERROR
            result.error_message = str(e)
            self.logger.error(f"Test suite {suite_config.name} failed: {str(e)}")
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        
        self.logger.info(f"Test suite {suite_config.name} completed in {result.execution_time_ms:.1f}ms")
        self.logger.info(f"Result: {result.result.value} - {result.tests_passed}/{result.tests_run} passed")
        
        return result
    
    async def _check_suite_requirements(self, suite_config: TestSuiteConfig) -> None:
        """Check if test suite requirements are met"""
        # Check required services
        for service_name in suite_config.required_services:
            service = self.environment_manager.get_service(service_name)
            if not service:
                raise Exception(f"Required service {service_name} not available")
        
        # Check environment requirements
        env_config = self.environment_manager.config
        for req_key, req_value in suite_config.environment_requirements.items():
            if hasattr(env_config, req_key):
                actual_value = getattr(env_config, req_key)
                if isinstance(req_value, (int, float)) and actual_value < req_value:
                    raise Exception(f"Environment requirement not met: {req_key} = {actual_value}, required >= {req_value}")
                elif isinstance(req_value, bool) and actual_value != req_value:
                    raise Exception(f"Environment requirement not met: {req_key} = {actual_value}, required = {req_value}")
    
    async def _run_pytest_suite(self, suite_config: TestSuiteConfig) -> Dict[str, Any]:
        """Run pytest for a test suite"""
        import subprocess
        import tempfile
        
        # Prepare pytest command
        test_files = [f"tests/{test_file}" for test_file in suite_config.test_files]
        cmd = [
            sys.executable, "-m", "pytest",
            "--tb=short",
            "--no-header",
            "-v",
            f"--timeout={suite_config.timeout_seconds}"
        ] + test_files
        
        # Execute pytest
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=Path(__file__).parent.parent
            )
            
            stdout, _ = await asyncio.wait_for(
                process.communicate(), 
                timeout=suite_config.timeout_seconds
            )
            
            output = stdout.decode('utf-8', errors='replace')
            
            # Parse pytest output to extract metrics
            lines = output.split('\n')
            tests_run = 0
            tests_passed = 0
            tests_failed = 0
            tests_skipped = 0
            
            for line in lines:
                if " passed" in line and " failed" in line:
                    # Parse result line like "5 passed, 2 failed in 10.5s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            tests_passed = int(parts[i-1])
                        elif part == "failed" and i > 0:
                            tests_failed = int(parts[i-1])
                        elif part == "skipped" and i > 0:
                            tests_skipped = int(parts[i-1])
                    tests_run = tests_passed + tests_failed + tests_skipped
                    break
                elif " passed in " in line:
                    # Parse result line like "5 passed in 10.5s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            tests_passed = int(parts[i-1])
                    tests_run = tests_passed
                    break
            
            return {
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "tests_skipped": tests_skipped,
                "output": output,
                "return_code": process.returncode
            }
            
        except asyncio.TimeoutError:
            if process:
                process.terminate()
                await process.wait()
            raise Exception(f"Test suite timed out after {suite_config.timeout_seconds} seconds")
        except Exception as e:
            raise Exception(f"Failed to execute pytest: {str(e)}")
    
    async def _generate_recommendations(self, report: IntegrationTestReport) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Analyze validation results
        if report.validation_results.get("critical_issues", 0) > 0:
            recommendations.append("Address critical validation issues before production deployment")
        
        if report.validation_results.get("warnings", 0) > 3:
            recommendations.append("Review and resolve validation warnings for optimal performance")
        
        # Analyze health check results
        unhealthy_services = report.health_check_results.get("services_unhealthy", 0)
        if unhealthy_services > 0:
            recommendations.append(f"Investigate {unhealthy_services} unhealthy services")
        
        # Analyze test results
        failed_suites = [r for r in report.test_suite_results if r.result == TestResult.FAIL]
        if failed_suites:
            recommendations.append(f"Address failures in {len(failed_suites)} test suites before release")
        
        error_suites = [r for r in report.test_suite_results if r.result == TestResult.ERROR]
        if error_suites:
            recommendations.append(f"Fix execution errors in {len(error_suites)} test suites")
        
        # Performance recommendations
        avg_execution_time = sum(r.execution_time_ms for r in report.test_suite_results) / len(report.test_suite_results) if report.test_suite_results else 0
        if avg_execution_time > 60000:  # 1 minute
            recommendations.append("Consider optimizing test execution time")
        
        # Memory recommendations
        max_memory_used = report.performance_metrics.get("memory_used_mb", 0)
        memory_limit = report.environment_config.get("max_memory_mb", 2048)
        if max_memory_used > memory_limit * 0.9:
            recommendations.append("Consider increasing memory limits for production environment")
        
        return recommendations
    
    async def _save_test_artifacts(self, report: IntegrationTestReport) -> None:
        """Save test artifacts and reports"""
        artifacts_dir = Path("test_artifacts") / report.test_session_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main report
        report_file = artifacts_dir / "integration_test_report.json"
        with open(report_file, 'w') as f:
            # Convert dataclass to dict for JSON serialization
            report_dict = asdict(report)
            json.dump(report_dict, f, indent=2, default=str)
        
        report.artifacts["main_report"] = str(report_file)
        
        # Save individual test outputs
        for suite_result in report.test_suite_results:
            if suite_result.output:
                output_file = artifacts_dir / f"{suite_result.suite_name}_output.txt"
                with open(output_file, 'w') as f:
                    f.write(suite_result.output)
                report.artifacts[f"{suite_result.suite_name}_output"] = str(output_file)
        
        self.logger.info(f"Test artifacts saved to: {artifacts_dir}")


# Convenience functions for common test scenarios
async def run_quick_integration_test() -> IntegrationTestReport:
    """Run a quick integration test with basic configuration"""
    orchestrator = IntegrationTestOrchestrator()
    return await orchestrator.run_full_integration_test_suite(
        env_type=TestEnvironmentType.INTEGRATION,
        complexity=TestComplexity.BASIC,
        selected_suites=["api_integration", "error_handling"]
    )


async def run_comprehensive_integration_test() -> IntegrationTestReport:
    """Run comprehensive integration test with full configuration"""
    orchestrator = IntegrationTestOrchestrator()
    return await orchestrator.run_full_integration_test_suite(
        env_type=TestEnvironmentType.INTEGRATION,
        complexity=TestComplexity.COMPREHENSIVE
    )


async def run_performance_integration_test() -> IntegrationTestReport:
    """Run performance-focused integration test"""
    orchestrator = IntegrationTestOrchestrator()
    return await orchestrator.run_full_integration_test_suite(
        env_type=TestEnvironmentType.PERFORMANCE,
        complexity=TestComplexity.STANDARD,
        selected_suites=["rag_system", "performance_gui"]
    )


if __name__ == "__main__":
    async def main():
        """Main function for testing the orchestrator"""
        print("Starting integration test orchestration...")
        
        # Run quick test
        print("\n=== Running Quick Integration Test ===")
        report = await run_quick_integration_test()
        
        print(f"Test session: {report.test_session_id}")
        print(f"Duration: {report.duration_ms:.1f}ms")
        print(f"Overall result: {report.overall_result.value}")
        print(f"Test suites executed: {len(report.test_suite_results)}")
        
        if report.recommendations:
            print("Recommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")
        
        print(f"Artifacts saved to: {report.artifacts.get('main_report', 'N/A')}")
    
    # Run the test
    asyncio.run(main()) 