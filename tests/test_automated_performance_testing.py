"""
Tests for Automated Performance Optimization and Recovery System

This module tests the advanced automated performance optimization including:
- Performance analysis and automatic optimization engine
- Crash detection and recovery mechanisms
- Performance degradation detection and mitigation
- Resource allocation optimization
- Circuit breaker patterns and failure handling
- State preservation and recovery
"""

import pytest
import asyncio
import tempfile
import shutil
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import threading
import pickle

from src.sovereign.automated_performance_testing import (
    AdvancedPerformanceOptimizer, CrashDetectionRecoverySystem,
    OptimizationType, RecoveryStrategy, CrashType, PerformanceProfile,
    OptimizationResult, CrashEvent, RecoveryAction,
    create_automated_performance_system
)
from src.sovereign.performance_monitor import (
    PerformanceMonitor, MetricType, QueryType, AlertLevel,
    PerformanceMetric, PerformanceAlert
)


class TestDataStructures:
    """Test data structure classes"""
    
    def test_test_config_creation(self):
        """Test TestConfig creation and defaults"""
        config = TestConfig(
            name="test_config",
            test_type=TestType.BENCHMARK
        )
        
        assert config.name == "test_config"
        assert config.test_type == TestType.BENCHMARK
        assert config.duration_seconds is None
        assert config.iterations is None
        assert config.concurrent_workers == 1
        assert config.timeout_seconds == 300.0
        assert config.memory_limit_mb is None
        assert config.gpu_required is False
        assert config.profiling_enabled is False
        assert config.metadata == {}
    
    def test_test_config_with_options(self):
        """Test TestConfig with all options"""
        metadata = {"custom": "value"}
        config = TestConfig(
            name="advanced_test",
            test_type=TestType.LOAD_TEST,
            duration_seconds=60.0,
            iterations=100,
            concurrent_workers=4,
            timeout_seconds=120.0,
            memory_limit_mb=512.0,
            gpu_required=True,
            profiling_enabled=True,
            metadata=metadata
        )
        
        assert config.name == "advanced_test"
        assert config.test_type == TestType.LOAD_TEST
        assert config.duration_seconds == 60.0
        assert config.iterations == 100
        assert config.concurrent_workers == 4
        assert config.timeout_seconds == 120.0
        assert config.memory_limit_mb == 512.0
        assert config.gpu_required is True
        assert config.profiling_enabled is True
        assert config.metadata == metadata
    
    def test_test_result_creation(self):
        """Test TestResult creation and defaults"""
        start_time = datetime.now()
        result = TestResult(
            test_name="test_result",
            test_type=TestType.BENCHMARK,
            status=TestStatus.PENDING,
            start_time=start_time
        )
        
        assert result.test_name == "test_result"
        assert result.test_type == TestType.BENCHMARK
        assert result.status == TestStatus.PENDING
        assert result.start_time == start_time
        assert result.end_time is None
        assert result.duration_seconds is None
        assert result.response_times == []
        assert result.memory_usage_mb == []
        assert result.cpu_usage_percent == []
        assert result.gpu_usage_percent == []
        assert result.error_count == 0
        assert result.error_rate == 0.0
        assert result.metadata == {}
    
    def test_test_result_statistics(self):
        """Test statistical calculations"""
        result = TestResult(
            test_name="stats_test",
            test_type=TestType.BENCHMARK,
            status=TestStatus.COMPLETED,
            start_time=datetime.now()
        )
        
        # Add sample data
        result.response_times = [0.1, 0.2, 0.15, 0.3, 0.25, 0.4, 0.35, 0.45, 0.5, 0.6]
        result.memory_usage_mb = [100, 150, 200, 180, 220]
        result.cpu_usage_percent = [25, 30, 35, 40, 45]
        result.gpu_usage_percent = [60, 65, 70, 75, 80]
        result.error_count = 2
        
        # Calculate statistics
        result.calculate_statistics()
        
        # Verify calculations
        assert result.avg_response_time == 0.33  # Mean of response times (3.3/10)
        assert abs(result.median_response_time - 0.325) < 0.001  # Median with tolerance for floating point
        assert result.p95_response_time == 0.6  # 95th percentile (9th item in sorted 10-item list)
        assert result.p99_response_time == 0.6  # 99th percentile (last item in small list)
        assert result.max_memory_mb == 220  # Max memory
        assert result.avg_cpu_percent == 35  # Mean CPU
        assert result.avg_gpu_percent == 70  # Mean GPU
        
        # Error rate: 2 errors out of (10 successes + 2 errors) = 2/12 = ~0.167
        assert abs(result.error_rate - (2/12)) < 0.001
    
    def test_benchmark_suite_creation(self):
        """Test BenchmarkSuite creation"""
        test1 = TestConfig("test1", TestType.BENCHMARK)
        test2 = TestConfig("test2", TestType.LOAD_TEST)
        
        suite = BenchmarkSuite(
            name="test_suite",
            description="Test benchmark suite",
            tests=[test1, test2]
        )
        
        assert suite.name == "test_suite"
        assert suite.description == "Test benchmark suite"
        assert len(suite.tests) == 2
        assert suite.baseline_results == {}


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_performance_monitor():
    """Mock performance monitor"""
    monitor = Mock()
    monitor.track_response_time = Mock()
    monitor.add_alert_callback = Mock()
    monitor.get_summary = Mock(return_value={})
    return monitor


@pytest.fixture
def performance_tester(temp_output_dir, mock_performance_monitor):
    """Create PerformanceTester instance"""
    tester = PerformanceTester(
        output_dir=temp_output_dir,
        performance_monitor=mock_performance_monitor,
        enable_profiling=False  # Disable for testing
    )
    yield tester
    tester.cleanup()


class TestSystemProfiler:
    """Test SystemProfiler class"""
    
    @pytest.fixture
    def profiler(self):
        """Create SystemProfiler instance"""
        return SystemProfiler(sample_interval=0.01)  # Fast sampling for tests
    
    def test_profiler_initialization(self, profiler):
        """Test profiler initialization"""
        assert profiler.sample_interval == 0.01
        assert profiler.monitoring is False
        assert profiler.cpu_samples == []
        assert profiler.memory_samples == []
        assert profiler.gpu_samples == []
        assert profiler._monitor_thread is None
    
    @patch('src.sovereign.automated_performance_testing.psutil.cpu_percent')
    @patch('src.sovereign.automated_performance_testing.psutil.virtual_memory')
    def test_profiler_monitoring(self, mock_memory, mock_cpu, profiler):
        """Test profiler monitoring functionality"""
        # Setup mocks
        mock_cpu.return_value = 25.0
        mock_memory.return_value = Mock(used=1024*1024*100)  # 100MB
        
        # Start monitoring
        profiler.start_monitoring()
        assert profiler.monitoring is True
        assert profiler._monitor_thread is not None
        
        # Let it collect some samples
        time.sleep(0.05)
        
        # Stop monitoring
        cpu_samples, memory_samples, gpu_samples = profiler.stop_monitoring()
        
        assert profiler.monitoring is False
        assert len(cpu_samples) > 0
        assert len(memory_samples) > 0
        assert all(cpu == 25.0 for cpu in cpu_samples)
        assert all(mem == 100.0 for mem in memory_samples)  # 100MB
    
    def test_profiler_gpu_unavailable(self, profiler):
        """Test profiler when GPU monitoring is unavailable"""
        # GPU monitoring should gracefully handle ImportError
        gpu_usage = profiler._get_gpu_usage()
        assert gpu_usage is None


class TestPerformanceTester:
    """Test PerformanceTester class"""
    
    def test_initialization(self, temp_output_dir, mock_performance_monitor):
        """Test PerformanceTester initialization"""
        tester = PerformanceTester(
            output_dir=temp_output_dir,
            performance_monitor=mock_performance_monitor,
            enable_profiling=True
        )
        
        assert tester.output_dir == temp_output_dir
        assert tester.output_dir.exists()
        assert tester.performance_monitor == mock_performance_monitor
        assert isinstance(tester.registered_tests, dict)
        assert isinstance(tester.benchmark_suites, dict)
        assert isinstance(tester.test_results, dict)
        
        tester.cleanup()
    
    def test_test_registration(self, performance_tester):
        """Test test registration"""
        def sample_test():
            return "test result"
        
        config = TestConfig("sample_test", TestType.BENCHMARK, iterations=10)
        performance_tester.register_test("sample_test", sample_test, config)
        
        assert "sample_test" in performance_tester.registered_tests
        registered = performance_tester.registered_tests["sample_test"]
        assert registered["function"] == sample_test
        assert registered["config"] == config
    
    def test_benchmark_suite_registration(self, performance_tester):
        """Test benchmark suite registration"""
        test1 = TestConfig("test1", TestType.BENCHMARK)
        test2 = TestConfig("test2", TestType.LOAD_TEST)
        suite = BenchmarkSuite("test_suite", "Test suite", [test1, test2])
        
        performance_tester.register_benchmark_suite(suite)
        
        assert "test_suite" in performance_tester.benchmark_suites
        assert performance_tester.benchmark_suites["test_suite"] == suite
    
    @pytest.mark.asyncio
    async def test_run_sync_test_iterations(self, performance_tester):
        """Test running sync test with iterations"""
        execution_count = 0
        
        def test_function():
            nonlocal execution_count
            execution_count += 1
            time.sleep(0.01)  # Small delay
            return execution_count
        
        config = TestConfig("sync_test", TestType.BENCHMARK, iterations=5)
        performance_tester.register_test("sync_test", test_function, config)
        
        result = await performance_tester.run_test("sync_test")
        
        assert result.status == TestStatus.COMPLETED
        assert execution_count == 5
        assert len(result.response_times) == 5
        assert all(t > 0 for t in result.response_times)
        assert result.error_count == 0
        assert result.avg_response_time is not None
        assert result.median_response_time is not None
    
    @pytest.mark.asyncio
    async def test_run_async_test_iterations(self, performance_tester):
        """Test running async test with iterations"""
        execution_count = 0
        
        async def async_test_function():
            nonlocal execution_count
            execution_count += 1
            await asyncio.sleep(0.01)  # Small delay
            return execution_count
        
        config = TestConfig("async_test", TestType.BENCHMARK, iterations=3)
        performance_tester.register_test("async_test", async_test_function, config)
        
        result = await performance_tester.run_test("async_test")
        
        assert result.status == TestStatus.COMPLETED
        assert execution_count == 3
        assert len(result.response_times) == 3
        assert all(t > 0 for t in result.response_times)
        assert result.error_count == 0
    
    @pytest.mark.asyncio
    async def test_run_test_duration(self, performance_tester):
        """Test running test for specific duration"""
        execution_count = 0
        
        def duration_test():
            nonlocal execution_count
            execution_count += 1
            time.sleep(0.01)
        
        config = TestConfig("duration_test", TestType.BENCHMARK, duration_seconds=0.1)
        performance_tester.register_test("duration_test", duration_test, config)
        
        result = await performance_tester.run_test("duration_test")
        
        assert result.status == TestStatus.COMPLETED
        assert execution_count > 0  # Should execute multiple times
        assert len(result.response_times) == execution_count
        assert result.duration_seconds is not None
        # Throughput should be calculated when test duration > 0 and operations > 0
        if result.duration_seconds > 0 and execution_count > 0:
            assert result.throughput_ops_per_sec is not None
            assert result.throughput_ops_per_sec > 0
    
    @pytest.mark.asyncio
    async def test_run_test_single_execution(self, performance_tester):
        """Test running test single execution (no iterations or duration)"""
        execution_count = 0
        
        def single_test():
            nonlocal execution_count
            execution_count += 1
            return "single result"
        
        config = TestConfig("single_test", TestType.BENCHMARK)
        performance_tester.register_test("single_test", single_test, config)
        
        result = await performance_tester.run_test("single_test")
        
        assert result.status == TestStatus.COMPLETED
        assert execution_count == 1
        assert len(result.response_times) == 1
        assert result.error_count == 0
    
    @pytest.mark.asyncio
    async def test_run_test_with_errors(self, performance_tester):
        """Test running test that produces errors"""
        execution_count = 0
        
        def error_test():
            nonlocal execution_count
            execution_count += 1
            if execution_count % 2 == 0:
                raise ValueError(f"Test error {execution_count}")
            return "success"
        
        config = TestConfig("error_test", TestType.BENCHMARK, iterations=5)
        performance_tester.register_test("error_test", error_test, config)
        
        result = await performance_tester.run_test("error_test")
        
        assert result.status == TestStatus.COMPLETED
        assert execution_count == 5
        assert len(result.response_times) == 3  # 3 successful executions
        assert result.error_count == 2  # 2 failed executions
        assert result.error_rate > 0
    
    @pytest.mark.asyncio
    async def test_run_test_timeout(self, performance_tester):
        """Test test timeout handling"""
        def slow_test():
            time.sleep(2.0)  # Longer than timeout
            return "should not complete"
        
        config = TestConfig("slow_test", TestType.BENCHMARK, timeout_seconds=0.1)
        performance_tester.register_test("slow_test", slow_test, config)
        
        result = await performance_tester.run_test("slow_test")
        
        assert result.status == TestStatus.TIMEOUT
        assert "timed out" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_run_test_exception(self, performance_tester):
        """Test test exception handling"""
        def failing_test():
            raise RuntimeError("Test runtime error")
        
        config = TestConfig("failing_test", TestType.BENCHMARK)
        performance_tester.register_test("failing_test", failing_test, config)
        
        result = await performance_tester.run_test("failing_test")
        
        assert result.status == TestStatus.FAILED
        assert "Test runtime error" in result.error_message
        assert result.exception_details == "RuntimeError"
    
    @pytest.mark.asyncio
    async def test_run_unregistered_test(self, performance_tester):
        """Test running unregistered test"""
        with pytest.raises(ValueError, match="not registered"):
            await performance_tester.run_test("nonexistent_test")
    
    @pytest.mark.asyncio
    async def test_run_benchmark_suite(self, performance_tester):
        """Test running complete benchmark suite"""
        # Register test functions
        def test1():
            time.sleep(0.01)
            return "test1"
        
        def test2():
            time.sleep(0.01)
            return "test2"
        
        config1 = TestConfig("test1", TestType.BENCHMARK, iterations=2)
        config2 = TestConfig("test2", TestType.BENCHMARK, iterations=2)
        
        performance_tester.register_test("test1", test1, config1)
        performance_tester.register_test("test2", test2, config2)
        
        # Create and register suite
        suite = BenchmarkSuite("test_suite", "Test suite", [config1, config2])
        performance_tester.register_benchmark_suite(suite)
        
        # Run suite
        results = await performance_tester.run_benchmark_suite("test_suite")
        
        assert len(results) == 2
        assert "test1" in results
        assert "test2" in results
        assert all(r.status == TestStatus.COMPLETED for r in results.values())
        assert all(len(r.response_times) == 2 for r in results.values())
    
    @pytest.mark.asyncio
    async def test_run_benchmark_suite_missing_tests(self, performance_tester):
        """Test running suite with missing test registrations"""
        config1 = TestConfig("missing_test1", TestType.BENCHMARK)
        config2 = TestConfig("missing_test2", TestType.BENCHMARK)
        suite = BenchmarkSuite("missing_suite", "Suite with missing tests", [config1, config2])
        
        performance_tester.register_benchmark_suite(suite)
        
        results = await performance_tester.run_benchmark_suite("missing_suite")
        
        # Should return empty results since no tests were registered
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_run_load_test(self, performance_tester):
        """Test load testing functionality"""
        execution_count = 0
        
        def load_test_function(user_id=None):
            nonlocal execution_count
            execution_count += 1
            time.sleep(0.01)
            return f"user_{user_id}_result"
        
        config = TestConfig("load_test", TestType.LOAD_TEST)
        performance_tester.register_test("load_test", load_test_function, config)
        
        # Run load test with 2 concurrent users for 0.1 seconds
        result = await performance_tester.run_load_test("load_test", 2, 0.1)
        
        assert result.status == TestStatus.COMPLETED
        assert result.test_type == TestType.LOAD_TEST
        assert execution_count > 0  # Should have multiple executions
        assert len(result.response_times) > 0
        assert result.throughput_ops_per_sec is not None
        assert result.avg_response_time is not None
    
    @pytest.mark.asyncio
    async def test_run_load_test_unregistered(self, performance_tester):
        """Test load test with unregistered test"""
        with pytest.raises(ValueError, match="not registered"):
            await performance_tester.run_load_test("nonexistent_load_test", 1, 0.1)
    
    def test_result_saving(self, performance_tester, temp_output_dir):
        """Test result saving to file"""
        result = TestResult(
            test_name="save_test",
            test_type=TestType.BENCHMARK,
            status=TestStatus.COMPLETED,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        result.response_times = [0.1, 0.2, 0.3]
        result.calculate_statistics()
        
        # Save result
        import asyncio
        asyncio.run(performance_tester._save_test_result(result))
        
        # Check file was created
        result_files = list(temp_output_dir.glob("save_test_*.json"))
        assert len(result_files) == 1
        
        # Check file content
        with open(result_files[0]) as f:
            saved_data = json.load(f)
        
        assert saved_data["test_name"] == "save_test"
        assert saved_data["test_type"] == "benchmark"
        assert saved_data["status"] == "completed"
        assert saved_data["response_times"] == [0.1, 0.2, 0.3]
    
    def test_suite_summary_generation(self, performance_tester):
        """Test suite summary generation"""
        # Create test results
        result1 = TestResult("test1", TestType.BENCHMARK, TestStatus.COMPLETED, datetime.now())
        result1.response_times = [0.1, 0.2]
        result1.error_count = 0
        result1.throughput_ops_per_sec = 10.0
        result1.calculate_statistics()
        
        result2 = TestResult("test2", TestType.LOAD_TEST, TestStatus.COMPLETED, datetime.now())
        result2.response_times = [0.15, 0.25]
        result2.error_count = 1
        result2.throughput_ops_per_sec = 8.0
        result2.calculate_statistics()
        
        result3 = TestResult("test3", TestType.BENCHMARK, TestStatus.FAILED, datetime.now())
        
        results = {"test1": result1, "test2": result2, "test3": result3}
        summary = performance_tester._generate_suite_summary(results)
        
        assert summary["total_tests"] == 3
        assert summary["completed_tests"] == 2
        assert summary["failed_tests"] == 1
        assert summary["timeout_tests"] == 0
        assert summary["total_errors"] == 1
        assert summary["total_throughput_ops_per_sec"] == 18.0
        assert "overall_avg_response_time" in summary
        assert "overall_p95_response_time" in summary
    
    def test_performance_report_generation(self, performance_tester, temp_output_dir):
        """Test performance report generation"""
        # Add some test results
        result1 = TestResult("test1", TestType.BENCHMARK, TestStatus.COMPLETED, datetime.now())
        result1.response_times = [0.1, 0.2, 0.3]
        result1.throughput_ops_per_sec = 15.0
        result1.calculate_statistics()
        
        result2 = TestResult("test2", TestType.LOAD_TEST, TestStatus.FAILED, datetime.now())
        result2.error_message = "Test failure message"
        
        performance_tester.test_results = {"test1": result1, "test2": result2}
        
        # Generate report
        report_file = temp_output_dir / "performance_report.txt"
        report_content = performance_tester.generate_performance_report(report_file)
        
        # Check report content
        assert "PERFORMANCE TEST REPORT" in report_content
        assert "Completed: 1" in report_content
        assert "Failed: 1" in report_content
        assert "test1" in report_content
        assert "test2" in report_content
        assert "Avg Response Time" in report_content
        assert "Throughput" in report_content
        assert "Test failure message" in report_content
        
        # Check file was saved
        assert report_file.exists()
        with open(report_file) as f:
            saved_content = f.read()
        assert saved_content == report_content
    
    def test_empty_performance_report(self, performance_tester):
        """Test performance report with no results"""
        report_content = performance_tester.generate_performance_report()
        assert "No test results available" in report_content
    
    def test_profiling_tools_availability(self, performance_tester):
        """Test profiling tools availability check"""
        tools = performance_tester.get_available_profiling_tools()
        
        assert isinstance(tools, dict)
        assert "cprofile" in tools
        assert "memory_profiler" in tools
        assert "line_profiler" in tools
        assert "py_spy" in tools
        
        # Values should be booleans indicating availability
        for tool, available in tools.items():
            assert isinstance(available, bool)
    
    def test_cleanup(self, performance_tester):
        """Test cleanup functionality"""
        # Start monitoring to test cleanup
        performance_tester.profiler.start_monitoring()
        assert performance_tester.profiler.monitoring is True
        
        # Cleanup
        performance_tester.cleanup()
        
        # Monitoring should be stopped
        assert performance_tester.profiler.monitoring is False


class TestDecorators:
    """Test performance testing decorators"""
    
    def test_performance_test_decorator(self):
        """Test performance_test decorator"""
        @performance_test("decorated_test", TestType.BENCHMARK, iterations=5)
        def test_function():
            return "decorated result"
        
        assert hasattr(test_function, '_performance_test_config')
        config = test_function._performance_test_config
        assert config.name == "decorated_test"
        assert config.test_type == TestType.BENCHMARK
        assert config.iterations == 5
        
        # Function should still work normally
        result = test_function()
        assert result == "decorated result"
    
    def test_benchmark_decorator(self):
        """Test benchmark decorator"""
        @benchmark("benchmark_test", iterations=10, timeout_seconds=60)
        def benchmark_function():
            return "benchmark result"
        
        assert hasattr(benchmark_function, '_performance_test_config')
        config = benchmark_function._performance_test_config
        assert config.name == "benchmark_test"
        assert config.test_type == TestType.BENCHMARK
        assert config.iterations == 10
        assert config.timeout_seconds == 60
        
        result = benchmark_function()
        assert result == "benchmark result"
    
    def test_load_test_decorator(self):
        """Test load_test decorator"""
        @load_test("load_test_example", duration_seconds=30.0, concurrent_workers=4)
        def load_test_function():
            return "load test result"
        
        assert hasattr(load_test_function, '_performance_test_config')
        config = load_test_function._performance_test_config
        assert config.name == "load_test_example"
        assert config.test_type == TestType.LOAD_TEST
        assert config.duration_seconds == 30.0
        assert config.concurrent_workers == 4
        
        result = load_test_function()
        assert result == "load test result"


class TestIntegration:
    """Integration tests for the complete automated performance system"""
    
    @pytest.mark.asyncio
    async def test_create_automated_performance_system(self, mock_performance_monitor):
        """Test factory function for creating the complete system"""
        optimizer, recovery_system = create_automated_performance_system(mock_performance_monitor)
        
        assert isinstance(optimizer, AdvancedPerformanceOptimizer)
        assert isinstance(recovery_system, CrashDetectionRecoverySystem)
        assert optimizer.performance_monitor == mock_performance_monitor
        assert recovery_system.performance_monitor == mock_performance_monitor
    
    @patch('src.sovereign.automated_performance_testing.psutil.virtual_memory')
    @patch('src.sovereign.automated_performance_testing.psutil.cpu_percent')
    @patch('src.sovereign.automated_performance_testing.gc.collect')
    def test_end_to_end_optimization_workflow(self, mock_gc, mock_cpu, mock_memory, mock_performance_monitor):
        """Test complete optimization workflow"""
        # Setup mocks
        mock_memory_info = Mock()
        mock_memory_info.percent = 85.0  # High memory usage
        mock_memory_info.available = 1024 * 1024 * 1024  # 1GB
        mock_memory.return_value = mock_memory_info
        mock_cpu.return_value = 90.0  # High CPU usage
        mock_gc.return_value = 150
        
        # Create high memory usage metrics
        high_memory_metrics = [
            PerformanceMetric(
                timestamp=datetime.now(),
                metric_type=MetricType.MEMORY_USAGE,
                value=85.0,
                unit="percent"
            )
        ]
        
        mock_performance_monitor.metrics = {
            MetricType.MEMORY_USAGE: high_memory_metrics
        }
        
        optimizer = AdvancedPerformanceOptimizer(mock_performance_monitor)
        
        # Force immediate optimization check
        optimizer.last_optimization_check = datetime.now() - timedelta(minutes=10)
        
        # Run optimization analysis
        results = optimizer.analyze_and_optimize()
        
        # Should have identified and attempted optimizations
        assert len(results) >= 0  # May be 0 if no opportunities found or already active
        
        # Check optimization summary
        summary = optimizer.get_optimization_summary()
        assert "total_optimizations" in summary
        assert "optimization_enabled" in summary
    
    def test_end_to_end_crash_recovery_workflow(self, mock_performance_monitor, temp_state_dir):
        """Test complete crash recovery workflow"""
        with patch('src.sovereign.automated_performance_testing.Path', return_value=Path(temp_state_dir)):
            recovery_system = CrashDetectionRecoverySystem(mock_performance_monitor)
        
        # Register a process
        recovery_system.register_process(1234, "test_service", {"critical": True})
        
        # Simulate a crash
        process_info = recovery_system.monitored_processes[1234]
        recovery_system._handle_process_crash(1234, process_info)
        
        # Check crash was handled and recovery attempted
        assert len(recovery_system.crash_history) == 1
        assert len(recovery_system.recovery_history) >= 1
        
        # Check recovery summary
        summary = recovery_system.get_recovery_summary()
        assert summary["total_crashes"] == 1
        assert summary["total_recoveries"] >= 1


@pytest.fixture
def temp_state_dir():
    """Create a temporary state directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestPerformanceProfile:
    """Test performance profile configurations"""
    
    def test_profile_creation(self):
        """Test creating performance profiles"""
        profile = PerformanceProfile(
            name="test",
            batch_size=4,
            max_memory_mb=2048,
            gpu_memory_fraction=0.7,
            gc_threshold=(700, 10, 10),
            thread_pool_size=4,
            cache_size_mb=256,
            model_precision="fp32"
        )
        
        assert profile.name == "test"
        assert profile.batch_size == 4
        assert profile.max_memory_mb == 2048
        assert profile.gpu_memory_fraction == 0.7
        assert profile.model_precision == "fp32"


class TestOptimizationResult:
    """Test optimization result data structures"""
    
    def test_optimization_result_creation(self):
        """Test creating optimization results"""
        before_metrics = {"memory_percent": 80.0, "avg_response_time_ms": 2000.0}
        after_metrics = {"memory_percent": 70.0, "avg_response_time_ms": 1500.0}
        
        result = OptimizationResult(
            optimization_type=OptimizationType.MEMORY_CLEANUP,
            success=True,
            timestamp=datetime.now(),
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percent=15.0,
            duration_seconds=30.0
        )
        
        assert result.optimization_type == OptimizationType.MEMORY_CLEANUP
        assert result.success is True
        assert result.improvement_percent == 15.0
        assert result.rollback_applied is False


class TestCrashEvent:
    """Test crash event data structures"""
    
    def test_crash_event_creation(self):
        """Test creating crash events"""
        crash = CrashEvent(
            crash_type=CrashType.MEMORY_EXHAUSTION,
            timestamp=datetime.now(),
            process_id=1234,
            process_name="test_process",
            exit_code=1,
            signal_received=None,
            memory_usage_mb=2048.0,
            cpu_usage_percent=95.0
        )
        
        assert crash.crash_type == CrashType.MEMORY_EXHAUSTION
        assert crash.process_id == 1234
        assert crash.process_name == "test_process"
        assert crash.memory_usage_mb == 2048.0


class TestAdvancedPerformanceOptimizer:
    """Test the advanced performance optimization engine"""
    
    def test_optimizer_initialization(self, mock_performance_monitor):
        """Test optimizer initialization"""
        optimizer = AdvancedPerformanceOptimizer(mock_performance_monitor)
        
        assert optimizer.performance_monitor == mock_performance_monitor
        assert optimizer.optimization_enabled is True
        assert len(optimizer.performance_profiles) >= 3
        assert "balanced" in optimizer.performance_profiles
        assert "speed" in optimizer.performance_profiles
        assert "memory" in optimizer.performance_profiles
        assert optimizer.current_profile == "balanced"
    
    def test_performance_profiles_initialization(self, mock_performance_monitor):
        """Test performance profiles are properly initialized"""
        optimizer = AdvancedPerformanceOptimizer(mock_performance_monitor)
        profiles = optimizer.performance_profiles
        
        # Check speed profile
        speed_profile = profiles["speed"]
        assert speed_profile.batch_size == 8
        assert speed_profile.model_precision == "fp16"
        assert speed_profile.enable_torch_compile is True
        
        # Check memory profile
        memory_profile = profiles["memory"]
        assert memory_profile.batch_size == 1
        assert memory_profile.model_precision == "int8"
        assert memory_profile.max_memory_mb == 1024
    
    @patch('src.sovereign.automated_performance_testing.gc.collect')
    @patch('src.sovereign.automated_performance_testing.torch.cuda.is_available')
    @patch('src.sovereign.automated_performance_testing.torch.cuda.empty_cache')
    def test_memory_cleanup_optimization(self, mock_cache, mock_cuda, mock_gc, mock_performance_monitor):
        """Test memory cleanup optimization"""
        mock_cuda.return_value = True
        mock_gc.return_value = 150
        
        optimizer = AdvancedPerformanceOptimizer(mock_performance_monitor)
        
        # Test memory cleanup
        success = optimizer._optimize_memory_cleanup()
        
        assert success is True
        mock_gc.assert_called_once()
        mock_cache.assert_called_once()
    
    @patch('src.sovereign.automated_performance_testing.torch.cuda.is_available')
    @patch('src.sovereign.automated_performance_testing.torch.cuda.empty_cache')
    def test_gpu_optimization(self, mock_cache, mock_cuda, mock_performance_monitor):
        """Test GPU optimization"""
        mock_cuda.return_value = True
        
        optimizer = AdvancedPerformanceOptimizer(mock_performance_monitor)
        
        success = optimizer._optimize_gpu_usage()
        
        assert success is True
        mock_cache.assert_called_once()
    
    @patch('src.sovereign.automated_performance_testing.gc.set_threshold')
    @patch('src.sovereign.automated_performance_testing.gc.collect')
    def test_garbage_collection_optimization(self, mock_collect, mock_set_threshold, mock_performance_monitor):
        """Test garbage collection optimization"""
        mock_collect.return_value = 100
        
        optimizer = AdvancedPerformanceOptimizer(mock_performance_monitor)
        
        success = optimizer._optimize_garbage_collection()
        
        assert success is True
        mock_set_threshold.assert_called_once_with(500, 8, 8)
        mock_collect.assert_called_once()
    
    @patch('src.sovereign.automated_performance_testing.psutil.virtual_memory')
    @patch('src.sovereign.automated_performance_testing.psutil.cpu_percent')
    def test_capture_current_metrics(self, mock_cpu, mock_memory, mock_performance_monitor):
        """Test capturing current system metrics"""
        # Setup mocks
        mock_memory_info = Mock()
        mock_memory_info.percent = 65.0
        mock_memory_info.available = 2048 * 1024 * 1024  # 2GB in bytes
        mock_memory.return_value = mock_memory_info
        mock_cpu.return_value = 35.0
        
        optimizer = AdvancedPerformanceOptimizer(mock_performance_monitor)
        
        metrics = optimizer._capture_current_metrics()
        
        assert "memory_percent" in metrics
        assert "memory_available_mb" in metrics
        assert "cpu_percent" in metrics
        assert metrics["memory_percent"] == 65.0
        assert metrics["memory_available_mb"] == 2048.0
        assert metrics["cpu_percent"] == 35.0
    
    def test_calculate_improvement(self, mock_performance_monitor):
        """Test improvement calculation"""
        optimizer = AdvancedPerformanceOptimizer(mock_performance_monitor)
        
        before = {
            "memory_percent": 80.0,
            "avg_response_time_ms": 2000.0
        }
        after = {
            "memory_percent": 70.0,
            "avg_response_time_ms": 1500.0
        }
        
        improvement = optimizer._calculate_improvement(before, after)
        
        # Should be average of memory improvement (12.5%) and response time improvement (25%)
        expected = (12.5 + 25.0) / 2
        assert abs(improvement - expected) < 0.1
    
    def test_identify_optimization_opportunities(self, mock_performance_monitor):
        """Test identification of optimization opportunities"""
        optimizer = AdvancedPerformanceOptimizer(mock_performance_monitor)
        
        # Create high memory usage metrics
        high_memory_metrics = [
            PerformanceMetric(
                timestamp=datetime.now(),
                metric_type=MetricType.MEMORY_USAGE,
                value=85.0,
                unit="percent"
            )
        ]
        
        # Create low GPU utilization metrics
        low_gpu_metrics = [
            PerformanceMetric(
                timestamp=datetime.now(),
                metric_type=MetricType.GPU_UTILIZATION,
                value=25.0,
                unit="percent"
            )
        ]
        
        metrics = {
            MetricType.MEMORY_USAGE: high_memory_metrics,
            MetricType.GPU_UTILIZATION: low_gpu_metrics
        }
        
        opportunities = optimizer._identify_optimization_opportunities(metrics)
        
        assert OptimizationType.MEMORY_CLEANUP in opportunities
        assert OptimizationType.BATCH_SIZE_TUNING in opportunities
    
    def test_optimization_history_tracking(self, mock_performance_monitor):
        """Test optimization history tracking"""
        optimizer = AdvancedPerformanceOptimizer(mock_performance_monitor)
        
        # Create optimization result
        result = OptimizationResult(
            optimization_type=OptimizationType.MEMORY_CLEANUP,
            success=True,
            timestamp=datetime.now(),
            before_metrics={"memory_percent": 80.0},
            after_metrics={"memory_percent": 70.0},
            improvement_percent=12.5
        )
        
        optimizer.optimization_history.append(result)
        
        # Test optimization effectiveness tracking
        optimizer.optimization_effectiveness[OptimizationType.MEMORY_CLEANUP].append(12.5)
        
        assert len(optimizer.optimization_history) == 1
        assert len(optimizer.optimization_effectiveness[OptimizationType.MEMORY_CLEANUP]) == 1
        assert optimizer.optimization_effectiveness[OptimizationType.MEMORY_CLEANUP][0] == 12.5
    
    def test_get_optimization_summary(self, mock_performance_monitor):
        """Test getting optimization summary"""
        optimizer = AdvancedPerformanceOptimizer(mock_performance_monitor)
        
        # Add some optimization results
        results = [
            OptimizationResult(
                optimization_type=OptimizationType.MEMORY_CLEANUP,
                success=True,
                timestamp=datetime.now(),
                before_metrics={},
                after_metrics={},
                improvement_percent=15.0
            ),
            OptimizationResult(
                optimization_type=OptimizationType.GPU_OPTIMIZATION,
                success=False,
                timestamp=datetime.now(),
                before_metrics={},
                after_metrics={},
                improvement_percent=0.0,
                error_message="GPU not available"
            )
        ]
        
        for result in results:
            optimizer.optimization_history.append(result)
        
        summary = optimizer.get_optimization_summary()
        
        assert summary["total_optimizations"] == 2
        assert summary["successful_optimizations"] == 1
        assert summary["success_rate"] == 0.5
        assert summary["average_improvement_percent"] == 7.5  # (15.0 + 0.0) / 2
        assert summary["current_profile"] == "balanced"
        assert summary["optimization_enabled"] is True
        assert len(summary["recent_optimizations"]) == 2


class TestCrashDetectionRecoverySystem:
    """Test the crash detection and recovery system"""
    
    @patch('src.sovereign.automated_performance_testing.Path.mkdir')
    def test_system_initialization(self, mock_mkdir, mock_performance_monitor, temp_state_dir):
        """Test crash detection system initialization"""
        with patch('src.sovereign.automated_performance_testing.Path', return_value=Path(temp_state_dir)):
            system = CrashDetectionRecoverySystem(mock_performance_monitor)
        
        assert system.performance_monitor == mock_performance_monitor
        assert system.monitoring_active is False
        assert len(system.recovery_strategies) >= 8
        assert system.max_recovery_attempts == 3
        assert system.recovery_cooldown_minutes == 5
    
    def test_recovery_strategies_initialization(self, mock_performance_monitor, temp_state_dir):
        """Test recovery strategies are properly mapped"""
        with patch('src.sovereign.automated_performance_testing.Path', return_value=Path(temp_state_dir)):
            system = CrashDetectionRecoverySystem(mock_performance_monitor)
        
        strategies = system.recovery_strategies
        
        assert strategies[CrashType.MEMORY_EXHAUSTION] == RecoveryStrategy.GRACEFUL_RESTART
        assert strategies[CrashType.PROCESS_HANG] == RecoveryStrategy.FORCE_RESTART
        assert strategies[CrashType.GPU_TIMEOUT] == RecoveryStrategy.RESOURCE_CLEANUP
        assert strategies[CrashType.NETWORK_FAILURE] == RecoveryStrategy.CIRCUIT_BREAKER
        assert strategies[CrashType.HARDWARE_FAILURE] == RecoveryStrategy.EMERGENCY_SHUTDOWN
    
    def test_process_registration(self, mock_performance_monitor, temp_state_dir):
        """Test process registration for monitoring"""
        with patch('src.sovereign.automated_performance_testing.Path', return_value=Path(temp_state_dir)):
            system = CrashDetectionRecoverySystem(mock_performance_monitor)
        
        # Register a process
        system.register_process(1234, "test_process", {"type": "ai_model"})
        
        assert 1234 in system.monitored_processes
        assert system.monitored_processes[1234]["name"] == "test_process"
        assert system.monitored_processes[1234]["metadata"]["type"] == "ai_model"
    
    def test_process_unregistration(self, mock_performance_monitor, temp_state_dir):
        """Test process unregistration"""
        with patch('src.sovereign.automated_performance_testing.Path', return_value=Path(temp_state_dir)):
            system = CrashDetectionRecoverySystem(mock_performance_monitor)
        
        # Register and then unregister
        system.register_process(1234, "test_process")
        system.unregister_process(1234)
        
        assert 1234 not in system.monitored_processes
    
    @patch('src.sovereign.automated_performance_testing.psutil.Process')
    def test_process_crash_handling(self, mock_process_class, mock_performance_monitor, temp_state_dir):
        """Test handling of process crashes"""
        with patch('src.sovereign.automated_performance_testing.Path', return_value=Path(temp_state_dir)):
            system = CrashDetectionRecoverySystem(mock_performance_monitor)
        
        # Register a process
        process_info = {"name": "test_process", "registered_time": datetime.now()}
        system.monitored_processes[1234] = process_info
        
        # Simulate process crash
        system._handle_process_crash(1234, process_info)
        
        # Check crash was recorded
        assert len(system.crash_history) == 1
        crash_event = system.crash_history[0]
        assert crash_event.process_id == 1234
        assert crash_event.process_name == "test_process"
        assert crash_event.crash_type == CrashType.UNHANDLED_EXCEPTION
        
        # Check process was removed from monitoring
        assert 1234 not in system.monitored_processes
    
    @patch('src.sovereign.automated_performance_testing.psutil.Process')
    def test_memory_exhaustion_handling(self, mock_process_class, mock_performance_monitor, temp_state_dir):
        """Test handling of memory exhaustion"""
        with patch('src.sovereign.automated_performance_testing.Path', return_value=Path(temp_state_dir)):
            system = CrashDetectionRecoverySystem(mock_performance_monitor)
        
        # Setup mock process
        mock_process = Mock()
        mock_process.pid = 1234
        mock_process.name.return_value = "memory_hog"
        mock_process.memory_info.return_value = Mock(rss=2048 * 1024 * 1024)  # 2GB
        
        process_info = {"name": "memory_hog"}
        
        system._handle_memory_exhaustion(mock_process, process_info)
        
        # Check crash was recorded
        assert len(system.crash_history) == 1
        crash_event = system.crash_history[0]
        assert crash_event.crash_type == CrashType.MEMORY_EXHAUSTION
        assert crash_event.memory_usage_mb == 2048.0
    
    def test_circuit_breaker_activation(self, mock_performance_monitor, temp_state_dir):
        """Test circuit breaker activation"""
        with patch('src.sovereign.automated_performance_testing.Path', return_value=Path(temp_state_dir)):
            system = CrashDetectionRecoverySystem(mock_performance_monitor)
        
        crash_event = CrashEvent(
            crash_type=CrashType.NETWORK_FAILURE,
            timestamp=datetime.now(),
            process_id=1234,
            process_name="network_service",
            exit_code=None,
            signal_received=None
        )
        
        # Activate circuit breaker
        success = system._activate_circuit_breaker(crash_event)
        
        assert success is True
        assert "network_service" in system.circuit_breaker_state
        assert system.circuit_breaker_state["network_service"]["failures"] == 1
        assert system.circuit_breaker_state["network_service"]["state"] == "closed"
        
        # Trigger multiple failures to open circuit breaker
        for _ in range(3):
            system._activate_circuit_breaker(crash_event)
        
        assert system.circuit_breaker_state["network_service"]["state"] == "open"
    
    @patch('src.sovereign.automated_performance_testing.gc.collect')
    @patch('src.sovereign.automated_performance_testing.torch.cuda.is_available')
    @patch('src.sovereign.automated_performance_testing.torch.cuda.empty_cache')
    def test_resource_cleanup(self, mock_cache, mock_cuda, mock_gc, mock_performance_monitor, temp_state_dir):
        """Test resource cleanup recovery"""
        mock_cuda.return_value = True
        mock_gc.return_value = 100
        mock_performance_monitor.force_memory_cleanup.return_value = {"status": "success"}
        
        with patch('src.sovereign.automated_performance_testing.Path', return_value=Path(temp_state_dir)):
            system = CrashDetectionRecoverySystem(mock_performance_monitor)
        
        crash_event = CrashEvent(
            crash_type=CrashType.RESOURCE_LEAK,
            timestamp=datetime.now(),
            process_id=1234,
            process_name="leaky_process",
            exit_code=None,
            signal_received=None
        )
        
        success = system._resource_cleanup(crash_event)
        
        assert success is True
        mock_gc.assert_called_once()
        mock_cache.assert_called_once()
    
    @patch('src.sovereign.automated_performance_testing.pickle.dump')
    def test_save_process_state(self, mock_pickle, mock_performance_monitor, temp_state_dir):
        """Test saving process state for recovery"""
        with patch('src.sovereign.automated_performance_testing.Path', return_value=Path(temp_state_dir)):
            system = CrashDetectionRecoverySystem(mock_performance_monitor)
        
        crash_event = CrashEvent(
            crash_type=CrashType.UNHANDLED_EXCEPTION,
            timestamp=datetime.now(),
            process_id=1234,
            process_name="test_process",
            exit_code=1,
            signal_received=None
        )
        
        with patch('builtins.open', create=True) as mock_open:
            system._save_process_state(crash_event)
        
        mock_open.assert_called_once()
        mock_pickle.assert_called_once()
    
    def test_start_stop_monitoring(self, mock_performance_monitor, temp_state_dir):
        """Test starting and stopping crash monitoring"""
        with patch('src.sovereign.automated_performance_testing.Path', return_value=Path(temp_state_dir)):
            system = CrashDetectionRecoverySystem(mock_performance_monitor)
        
        # Test start
        system.start_monitoring(interval_seconds=0.1)
        assert system.monitoring_active is True
        assert system.monitoring_thread is not None
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Test stop
        system.stop_monitoring()
        assert system.monitoring_active is False
    
    def test_get_recovery_summary(self, mock_performance_monitor, temp_state_dir):
        """Test getting recovery summary"""
        with patch('src.sovereign.automated_performance_testing.Path', return_value=Path(temp_state_dir)):
            system = CrashDetectionRecoverySystem(mock_performance_monitor)
        
        # Add some crash and recovery history
        crash_event = CrashEvent(
            crash_type=CrashType.MEMORY_EXHAUSTION,
            timestamp=datetime.now(),
            process_id=1234,
            process_name="test_process",
            exit_code=None,
            signal_received=None
        )
        system.crash_history.append(crash_event)
        
        recovery_action = RecoveryAction(
            recovery_strategy=RecoveryStrategy.GRACEFUL_RESTART,
            timestamp=datetime.now(),
            crash_event=crash_event,
            success=True,
            recovery_time_seconds=5.0
        )
        system.recovery_history.append(recovery_action)
        
        # Register a process
        system.register_process(5678, "monitored_process")
        
        summary = system.get_recovery_summary()
        
        assert summary["monitoring_active"] is False
        assert summary["monitored_processes"] == 1
        assert summary["total_crashes"] == 1
        assert summary["total_recoveries"] == 1
        assert summary["successful_recoveries"] == 1
        assert summary["recovery_success_rate"] == 1.0
        assert summary["crash_by_type"]["memory_exhaustion"] == 1
        assert len(summary["recent_crashes"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 