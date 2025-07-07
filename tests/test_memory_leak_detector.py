"""
Tests for Memory Leak Detection System

This module tests the advanced memory leak detection including:
- Memory leak detection algorithms and patterns
- Resource tracking (files, threads, connections)
- GPU memory leak monitoring  
- Memory profiling and diagnostics
- Integration with performance monitoring
- Alert generation and thresholds
- Memory cleanup and garbage collection
"""

import pytest
import time
import threading
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.sovereign.memory_leak_detector import (
    MemoryLeakDetector, MemoryProfiler, ResourceTracker,
    LeakType, MemoryTrend, AlertLevel,
    MemorySnapshot, LeakDetectionResult, ResourceLeakInfo,
    create_memory_leak_detector
)
from src.sovereign.performance_monitor import PerformanceMonitor, MetricType


@pytest.fixture
def mock_performance_monitor():
    """Create a mock performance monitor"""
    monitor = Mock(spec=PerformanceMonitor)
    monitor._add_alert = Mock()
    return monitor


@pytest.fixture
def temp_log_dir():
    """Create temporary directory for logs"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestMemorySnapshot:
    """Test memory snapshot data structure"""
    
    def test_memory_snapshot_creation(self):
        """Test creating memory snapshots"""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            process_memory_mb=256.5,
            system_memory_percent=45.2,
            gpu_memory_mb=512.0,
            file_descriptor_count=25,
            thread_count=8,
            network_connection_count=3,
            python_objects_count=15000,
            gc_objects_count=120
        )
        
        assert snapshot.process_memory_mb == 256.5
        assert snapshot.system_memory_percent == 45.2
        assert snapshot.gpu_memory_mb == 512.0
        assert snapshot.file_descriptor_count == 25
        assert snapshot.thread_count == 8
        assert snapshot.network_connection_count == 3
        assert snapshot.python_objects_count == 15000
        assert snapshot.gc_objects_count == 120
    
    def test_memory_snapshot_optional_fields(self):
        """Test memory snapshot with optional fields"""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            process_memory_mb=128.0,
            system_memory_percent=30.0
        )
        
        assert snapshot.gpu_memory_mb is None
        assert snapshot.file_descriptor_count == 0
        assert snapshot.thread_count == 0
        assert snapshot.network_connection_count == 0
        assert snapshot.python_objects_count == 0
        assert snapshot.gc_objects_count == 0


class TestLeakDetectionResult:
    """Test leak detection result structure"""
    
    def test_leak_detection_result_creation(self):
        """Test creating leak detection results"""
        result = LeakDetectionResult(
            leak_type=LeakType.MEMORY_GROWTH,
            detected=True,
            severity=AlertLevel.WARNING,
            leak_rate=5.5,
            confidence_score=0.85,
            time_window_minutes=30.0,
            current_value=512.0,
            baseline_value=256.0,
            growth_trend=MemoryTrend.GRADUAL_INCREASE,
            recommended_actions=["Check for memory leaks", "Run garbage collection"]
        )
        
        assert result.leak_type == LeakType.MEMORY_GROWTH
        assert result.detected is True
        assert result.severity == AlertLevel.WARNING
        assert result.leak_rate == 5.5
        assert result.confidence_score == 0.85
        assert result.time_window_minutes == 30.0
        assert result.current_value == 512.0
        assert result.baseline_value == 256.0
        assert result.growth_trend == MemoryTrend.GRADUAL_INCREASE
        assert len(result.recommended_actions) == 2


class TestResourceTracker:
    """Test resource tracking functionality"""
    
    @patch('src.sovereign.memory_leak_detector.psutil')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_resource_tracker_initialization(self, mock_gc, mock_psutil):
        """Test resource tracker initialization"""
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process
        
        tracker = ResourceTracker()
        
        assert tracker.process == mock_process
        assert tracker.baseline_resources is None
    
    @patch('src.sovereign.memory_leak_detector.psutil')
    @patch('src.sovereign.memory_leak_detector.gc')
    @patch('src.sovereign.memory_leak_detector.threading')
    @patch('src.sovereign.memory_leak_detector.torch')
    def test_get_current_resources(self, mock_torch, mock_threading, mock_gc, mock_psutil):
        """Test getting current resource usage"""
        # Setup mocks
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 256 * 1024 * 1024  # 256 MB in bytes
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 45.2
        mock_process.num_fds.return_value = 25
        mock_process.connections.return_value = [1, 2, 3]  # 3 connections
        mock_process.num_threads.return_value = 8
        
        mock_psutil.Process.return_value = mock_process
        mock_gc.get_objects.return_value = [None] * 15000  # 15000 objects
        mock_gc.get_stats.return_value = [None] * 120  # 120 GC stats
        mock_threading.active_count.return_value = 8
        
        # Mock GPU
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 512 * 1024 * 1024  # 512 MB
        
        tracker = ResourceTracker()
        resources = tracker.get_current_resources()
        
        assert resources['memory_mb'] == 256.0
        assert resources['memory_percent'] == 45.2
        assert resources['file_descriptors'] == 25
        assert resources['network_connections'] == 3
        assert resources['threads'] == 8
        assert resources['gpu_memory_mb'] == 512.0
        assert resources['python_objects'] == 15000
    
    @patch('src.sovereign.memory_leak_detector.psutil')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_set_baseline(self, mock_gc, mock_psutil):
        """Test setting resource baseline"""
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 128 * 1024 * 1024  # 128 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 30.0
        mock_process.num_fds.return_value = 15
        mock_process.connections.return_value = [1, 2]
        mock_process.num_threads.return_value = 6
        
        mock_psutil.Process.return_value = mock_process
        mock_gc.get_objects.return_value = [None] * 12000
        mock_gc.get_stats.return_value = [None] * 100
        
        tracker = ResourceTracker()
        tracker.set_baseline()
        
        assert tracker.baseline_resources is not None
        assert tracker.baseline_resources['memory_mb'] == 128.0
        assert tracker.baseline_resources['memory_percent'] == 30.0
    
    @patch('src.sovereign.memory_leak_detector.psutil')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_get_resource_diff(self, mock_gc, mock_psutil):
        """Test calculating resource differences from baseline"""
        mock_process = Mock()
        
        # First call for baseline
        mock_memory_info_baseline = Mock()
        mock_memory_info_baseline.rss = 128 * 1024 * 1024  # 128 MB
        
        # Second call for current
        mock_memory_info_current = Mock()
        mock_memory_info_current.rss = 256 * 1024 * 1024  # 256 MB
        
        mock_process.memory_info.side_effect = [mock_memory_info_baseline, mock_memory_info_current]
        mock_process.memory_percent.side_effect = [30.0, 45.0]
        mock_process.num_fds.side_effect = [15, 25]
        mock_process.connections.side_effect = [[1, 2], [1, 2, 3]]
        mock_process.num_threads.side_effect = [6, 8]
        
        mock_psutil.Process.return_value = mock_process
        mock_gc.get_objects.side_effect = [[None] * 12000, [None] * 15000]
        mock_gc.get_stats.side_effect = [[None] * 100, [None] * 120]
        
        tracker = ResourceTracker()
        tracker.set_baseline()
        diff = tracker.get_resource_diff()
        
        assert diff['memory_mb'] == 128.0  # 256 - 128
        assert diff['memory_percent'] == 15.0  # 45 - 30
        assert diff['file_descriptors'] == 10  # 25 - 15
        assert diff['network_connections'] == 1  # 3 - 2
        assert diff['threads'] == 2  # 8 - 6
        assert diff['python_objects'] == 3000  # 15000 - 12000


class TestMemoryProfiler:
    """Test memory profiling functionality"""
    
    @patch('src.sovereign.memory_leak_detector.tracemalloc')
    def test_memory_profiler_initialization(self, mock_tracemalloc):
        """Test memory profiler initialization"""
        mock_tracemalloc.is_tracing.return_value = False
        
        profiler = MemoryProfiler(enable_tracemalloc=True)
        
        mock_tracemalloc.start.assert_called_once_with(25)
        assert profiler.enable_tracemalloc is True
        assert profiler.tracemalloc_enabled is True
    
    @patch('src.sovereign.memory_leak_detector.tracemalloc')
    def test_memory_profiler_disabled(self, mock_tracemalloc):
        """Test memory profiler with tracemalloc disabled"""
        profiler = MemoryProfiler(enable_tracemalloc=False)
        
        mock_tracemalloc.start.assert_not_called()
        assert profiler.enable_tracemalloc is False
        assert profiler.tracemalloc_enabled is False
    
    @patch('src.sovereign.memory_leak_detector.tracemalloc')
    def test_take_snapshot(self, mock_tracemalloc):
        """Test taking memory snapshots"""
        mock_tracemalloc.is_tracing.return_value = False
        mock_snapshot = Mock()
        mock_tracemalloc.take_snapshot.return_value = mock_snapshot
        
        profiler = MemoryProfiler(enable_tracemalloc=True)
        profiler.tracemalloc_enabled = True  # Force enable for test
        
        snapshot = profiler.take_snapshot()
        
        mock_tracemalloc.take_snapshot.assert_called_once()
        assert snapshot == mock_snapshot
        assert len(profiler.snapshots) == 1
    
    @patch('src.sovereign.memory_leak_detector.tracemalloc')
    def test_compare_snapshots(self, mock_tracemalloc):
        """Test comparing memory snapshots"""
        mock_tracemalloc.is_tracing.return_value = False
        
        # Create mock snapshots
        mock_snapshot1 = Mock()
        mock_snapshot2 = Mock()
        
        # Mock the comparison
        mock_stat = Mock()
        mock_stat.size_diff = 1024  # 1 KB growth
        mock_stat.traceback.format.return_value = ["test_file.py:10"]
        
        mock_snapshot2.compare_to.return_value = [mock_stat]
        
        profiler = MemoryProfiler(enable_tracemalloc=True)
        profiler.tracemalloc_enabled = True
        
        # Manually add snapshots
        time1 = datetime.now() - timedelta(minutes=1)
        time2 = datetime.now()
        profiler.snapshots = [(time1, mock_snapshot1), (time2, mock_snapshot2)]
        
        results = profiler.compare_snapshots(5)
        
        assert len(results) >= 1
        assert "Memory comparison" in results[0]
        assert "+1.0 KB in test_file.py" in results[1]
    
    @patch('src.sovereign.memory_leak_detector.tracemalloc')
    def test_get_top_memory_consumers(self, mock_tracemalloc):
        """Test getting top memory consumers"""
        mock_tracemalloc.is_tracing.return_value = False
        
        # Create mock snapshot
        mock_snapshot = Mock()
        
        # Mock statistics
        mock_stat = Mock()
        mock_stat.size = 2048  # 2 KB
        mock_stat.traceback.format.return_value = ["memory_test.py:15"]
        
        mock_snapshot.statistics.return_value = [mock_stat]
        
        profiler = MemoryProfiler(enable_tracemalloc=True)
        profiler.tracemalloc_enabled = True
        
        # Manually add snapshot
        profiler.snapshots = [(datetime.now(), mock_snapshot)]
        
        results = profiler.get_top_memory_consumers(5)
        
        assert len(results) >= 1
        assert "Top memory consumers" in results[0]
        assert "2.0 KB in memory_test.py" in results[1]


class TestMemoryLeakDetector:
    """Test the main memory leak detector"""
    
    @patch('src.sovereign.memory_leak_detector.psutil.Process')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_memory_leak_detector_initialization(self, mock_gc, mock_process_class, mock_performance_monitor):
        """Test memory leak detector initialization"""
        # Setup complete psutil.Process mock to prevent real network calls
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=256*1024*1024)
        mock_process.memory_percent.return_value = 45.0
        mock_process.num_fds.return_value = 20
        mock_process.connections.return_value = []  # No real connections
        mock_process.num_threads.return_value = 6
        mock_process_class.return_value = mock_process
        
        mock_gc.get_objects.return_value = [None] * 10000
        mock_gc.get_stats.return_value = [None] * 100
        
        detector = MemoryLeakDetector(
            performance_monitor=mock_performance_monitor,
            detection_window_minutes=15.0,
            sampling_interval_seconds=30.0,
            enable_profiling=True
        )
        
        assert detector.performance_monitor == mock_performance_monitor
        assert detector.detection_window_minutes == 15.0
        assert detector.sampling_interval_seconds == 30.0
        assert detector.enable_profiling is True
        assert detector.monitoring_active is False
        assert len(detector.memory_history) == 0
        assert detector.resource_tracker is not None
        assert detector.memory_profiler is not None
    
    @patch('src.sovereign.memory_leak_detector.psutil.Process')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_memory_leak_detector_without_profiling(self, mock_gc, mock_process_class, mock_performance_monitor):
        """Test memory leak detector without profiling"""
        # Setup complete psutil.Process mock to prevent real network calls
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=256*1024*1024)
        mock_process.memory_percent.return_value = 45.0
        mock_process.num_fds.return_value = 20
        mock_process.connections.return_value = []  # No real connections
        mock_process.num_threads.return_value = 6
        mock_process_class.return_value = mock_process
        
        mock_gc.get_objects.return_value = [None] * 10000
        mock_gc.get_stats.return_value = [None] * 100
        
        detector = MemoryLeakDetector(
            performance_monitor=mock_performance_monitor,
            enable_profiling=False
        )
        
        assert detector.memory_profiler is None
    
    @patch('src.sovereign.memory_leak_detector.psutil.Process')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_start_stop_monitoring(self, mock_gc, mock_process_class, mock_performance_monitor):
        """Test starting and stopping monitoring"""
        # Setup complete psutil.Process mock to prevent real network calls
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=256*1024*1024)
        mock_process.memory_percent.return_value = 45.0
        mock_process.num_fds.return_value = 20
        mock_process.connections.return_value = []  # No real connections
        mock_process.num_threads.return_value = 6
        mock_process_class.return_value = mock_process
        
        mock_gc.get_objects.return_value = [None] * 10000
        mock_gc.get_stats.return_value = [None] * 100
        
        detector = MemoryLeakDetector(
            performance_monitor=mock_performance_monitor,
            sampling_interval_seconds=0.1  # Fast for testing
        )
        
        # Test start
        detector.start_monitoring()
        assert detector.monitoring_active is True
        assert detector.monitoring_thread is not None
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Test stop
        detector.stop_monitoring()
        assert detector.monitoring_active is False
    
    @patch('src.sovereign.memory_leak_detector.psutil')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_create_memory_snapshot(self, mock_gc, mock_psutil, mock_performance_monitor):
        """Test creating memory snapshots"""
        # Setup mocks
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 256 * 1024 * 1024  # 256 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 45.0
        mock_process.num_fds.return_value = 20
        mock_process.connections.return_value = [1, 2]
        mock_process.num_threads.return_value = 6
        
        mock_psutil.Process.return_value = mock_process
        mock_gc.get_objects.return_value = [None] * 12000
        mock_gc.get_stats.return_value = [None] * 100
        
        detector = MemoryLeakDetector(performance_monitor=mock_performance_monitor)
        snapshot = detector._create_memory_snapshot()
        
        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.process_memory_mb == 256.0
        assert snapshot.system_memory_percent == 45.0
        assert snapshot.file_descriptor_count == 20
        assert snapshot.network_connection_count == 2
        assert snapshot.thread_count == 6
        assert snapshot.python_objects_count == 12000
    
    @patch('src.sovereign.memory_leak_detector.psutil.Process')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_memory_growth_analysis(self, mock_gc, mock_process_class, mock_performance_monitor):
        """Test memory growth pattern analysis"""
        # Setup complete psutil.Process mock to prevent real network calls
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=256*1024*1024)
        mock_process.memory_percent.return_value = 45.0
        mock_process.num_fds.return_value = 20
        mock_process.connections.return_value = []  # No real connections
        mock_process.num_threads.return_value = 6
        mock_process_class.return_value = mock_process
        
        mock_gc.get_objects.return_value = [None] * 10000
        mock_gc.get_stats.return_value = [None] * 100
        
        detector = MemoryLeakDetector(performance_monitor=mock_performance_monitor)
        
        # Create test history with increasing memory
        base_time = datetime.now() - timedelta(minutes=30)
        history = []
        
        for i in range(5):
            snapshot = MemorySnapshot(
                timestamp=base_time + timedelta(minutes=i*6),
                process_memory_mb=100.0 + (i * 10),  # Increasing by 10 MB each time
                system_memory_percent=30.0 + (i * 2)
            )
            history.append(snapshot)
        
        result = detector._analyze_memory_growth(history)
        
        assert isinstance(result, LeakDetectionResult)
        assert result.leak_type == LeakType.MEMORY_GROWTH
        assert result.current_value == 140.0  # Last value
        assert result.baseline_value == 100.0  # First value
        assert result.leak_rate > 0  # Should detect growth
    
    @patch('src.sovereign.memory_leak_detector.psutil.Process')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_memory_trend_determination(self, mock_gc, mock_process_class, mock_performance_monitor):
        """Test memory trend classification"""
        # Setup complete psutil.Process mock to prevent real network calls
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=256*1024*1024)
        mock_process.memory_percent.return_value = 45.0
        mock_process.num_fds.return_value = 20
        mock_process.connections.return_value = []  # No real connections
        mock_process.num_threads.return_value = 6
        mock_process_class.return_value = mock_process
        
        mock_gc.get_objects.return_value = [None] * 10000
        mock_gc.get_stats.return_value = [None] * 100
        
        detector = MemoryLeakDetector(performance_monitor=mock_performance_monitor)
        
        # Test stable trend
        stable_values = [100.0, 101.0, 99.0, 100.5, 102.0]
        trend = detector._determine_memory_trend(stable_values)
        assert trend == MemoryTrend.STABLE
        
        # Test gradual increase
        gradual_values = [100.0, 105.0, 110.0, 115.0, 118.0]
        trend = detector._determine_memory_trend(gradual_values)
        assert trend == MemoryTrend.GRADUAL_INCREASE
        
        # Test rapid increase
        rapid_values = [100.0, 110.0, 125.0, 140.0, 160.0]
        trend = detector._determine_memory_trend(rapid_values)
        assert trend == MemoryTrend.RAPID_INCREASE
        
        # Test decreasing
        decreasing_values = [100.0, 95.0, 90.0, 85.0, 80.0]
        trend = detector._determine_memory_trend(decreasing_values)
        assert trend == MemoryTrend.DECREASING
    
    @patch('src.sovereign.memory_leak_detector.psutil.Process')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_growth_confidence_calculation(self, mock_gc, mock_process_class, mock_performance_monitor):
        """Test confidence score calculation for growth"""
        # Setup complete psutil.Process mock to prevent real network calls
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=256*1024*1024)
        mock_process.memory_percent.return_value = 45.0
        mock_process.num_fds.return_value = 20
        mock_process.connections.return_value = []  # No real connections
        mock_process.num_threads.return_value = 6
        mock_process_class.return_value = mock_process
        
        mock_gc.get_objects.return_value = [None] * 10000
        mock_gc.get_stats.return_value = [None] * 100
        
        detector = MemoryLeakDetector(performance_monitor=mock_performance_monitor)
        
        # Test consistent small growth - moderate confidence (consistency * magnitude)
        consistent_growth = [100.0, 105.0, 110.0, 115.0, 120.0]
        confidence = detector._calculate_growth_confidence(consistent_growth)
        assert 0.15 < confidence < 0.25  # 1.0 * 0.2 = 0.2
        
        # Test consistent large growth - high confidence
        large_growth = [100.0, 120.0, 140.0, 160.0, 180.0]
        confidence = detector._calculate_growth_confidence(large_growth)
        assert confidence > 0.7  # 1.0 * 0.8 = 0.8
        
        # Test inconsistent values - low confidence
        inconsistent_values = [100.0, 120.0, 90.0, 110.0, 95.0]
        confidence = detector._calculate_growth_confidence(inconsistent_values)
        assert confidence < 0.2
        
        # Test no growth - zero confidence
        no_growth = [100.0, 95.0, 90.0, 85.0, 80.0]
        confidence = detector._calculate_growth_confidence(no_growth)
        assert confidence == 0.0
    
    @patch('src.sovereign.memory_leak_detector.psutil.Process')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_force_garbage_collection(self, mock_gc, mock_process_class, mock_performance_monitor):
        """Test forced garbage collection"""
        # Setup complete psutil.Process mock to prevent real network calls
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=256*1024*1024)
        mock_process.memory_percent.return_value = 45.0
        mock_process.num_fds.return_value = 20
        mock_process.connections.return_value = []  # No real connections
        mock_process.num_threads.return_value = 6
        mock_process_class.return_value = mock_process
        
        mock_gc.get_objects.return_value = [None] * 10000
        mock_gc.get_stats.return_value = [None] * 100
        mock_gc.collect.return_value = 150  # 150 objects collected
        
        detector = MemoryLeakDetector(performance_monitor=mock_performance_monitor)
        
        # Mock resource tracker
        mock_tracker = Mock()
        mock_tracker.get_current_resources.side_effect = [
            {'memory_mb': 256.0, 'python_objects': 15000},  # Before
            {'memory_mb': 240.0, 'python_objects': 14500}   # After
        ]
        detector.resource_tracker = mock_tracker
        
        result = detector.force_garbage_collection()
        
        assert result['objects_collected'] == 150
        assert result['memory_freed_mb'] == 16.0  # 256 - 240
        assert result['objects_freed'] == 500  # 15000 - 14500
        mock_gc.collect.assert_called_once()
    
    @patch('src.sovereign.memory_leak_detector.psutil.Process')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_leak_detection_summary(self, mock_gc, mock_process_class, mock_performance_monitor):
        """Test getting leak detection summary"""
        # Setup complete psutil.Process mock to prevent real network calls
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=256*1024*1024)
        mock_process.memory_percent.return_value = 45.0
        mock_process.num_fds.return_value = 20
        mock_process.connections.return_value = []  # No real connections
        mock_process.num_threads.return_value = 6
        mock_process_class.return_value = mock_process
        
        mock_gc.get_objects.return_value = [None] * 10000
        mock_gc.get_stats.return_value = [None] * 100
        
        detector = MemoryLeakDetector(performance_monitor=mock_performance_monitor)
        
        # Add some test data
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            process_memory_mb=256.0,
            system_memory_percent=45.0,
            file_descriptor_count=20,
            thread_count=6,
            network_connection_count=2
        )
        detector.memory_history.append(snapshot)
        
        summary = detector.get_leak_detection_summary()
        
        assert summary['monitoring_active'] is False
        assert summary['history_points'] == 1
        assert summary['detection_window_minutes'] == detector.detection_window_minutes
        assert summary['latest_snapshot']['memory_mb'] == 256.0
        assert summary['latest_snapshot']['file_descriptors'] == 20
        assert summary['total_alerts'] == 0
    
    @patch('src.sovereign.memory_leak_detector.psutil.Process')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_memory_diagnostic(self, mock_gc, mock_process_class, mock_performance_monitor):
        """Test comprehensive memory diagnostic"""
        # Setup complete psutil.Process mock to prevent real network calls
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=256*1024*1024)
        mock_process.memory_percent.return_value = 45.0
        mock_process.num_fds.return_value = 20
        mock_process.connections.return_value = []  # No real connections
        mock_process.num_threads.return_value = 6
        mock_process_class.return_value = mock_process
        
        mock_gc.get_objects.return_value = [None] * 10000
        mock_gc.get_stats.return_value = [None] * 100
        
        detector = MemoryLeakDetector(performance_monitor=mock_performance_monitor)
        
        # Mock resource tracker
        mock_tracker = Mock()
        mock_tracker.get_current_resources.return_value = {
            'memory_mb': 256.0,
            'memory_percent': 45.0,
            'python_objects': 15000
        }
        mock_tracker.get_resource_diff.return_value = {
            'memory_mb': 56.0,
            'python_objects': 3000
        }
        detector.resource_tracker = mock_tracker
        
        diagnostic = detector.run_memory_diagnostic()
        
        assert 'timestamp' in diagnostic
        assert 'current_resources' in diagnostic
        assert 'resource_diff_from_baseline' in diagnostic
        assert 'gc_stats' in diagnostic
        assert 'leak_detection' in diagnostic
        assert diagnostic['current_resources']['memory_mb'] == 256.0


class TestMemoryLeakIntegration:
    """Test integration with performance monitoring"""
    
    @patch('src.sovereign.memory_leak_detector.psutil')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_performance_monitor_integration(self, mock_gc, mock_psutil):
        """Test memory leak detector integration with performance monitor"""
        # Setup mocks
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 256 * 1024 * 1024
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 45.0
        mock_process.num_fds.return_value = 20
        mock_process.connections.return_value = []
        mock_process.num_threads.return_value = 6
        
        mock_psutil.Process.return_value = mock_process
        mock_gc.get_objects.return_value = [None] * 12000
        mock_gc.get_stats.return_value = [None] * 100
        
        # Create performance monitor with memory leak detection
        with patch('src.sovereign.performance_monitor.HardwareDetector'):
            monitor = PerformanceMonitor(enable_memory_leak_detection=True)
            
            # Test that memory leak detection status is included
            status = monitor.get_current_status()
            
            # Should have memory leak detection in status
            assert 'memory_leak_detection' in status or monitor._memory_leak_detector is None
            
            # Test memory cleanup functionality
            cleanup_result = monitor.force_memory_cleanup()
            assert 'objects_collected' in cleanup_result
            
            # Test memory diagnostic
            diagnostic = monitor.run_memory_diagnostic()
            assert diagnostic is not None
    
    @patch('src.sovereign.memory_leak_detector.psutil.Process')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_create_memory_leak_detector_factory(self, mock_gc, mock_process_class):
        """Test factory function for creating memory leak detector"""
        # Setup complete psutil.Process mock to prevent real network calls
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=256*1024*1024)
        mock_process.memory_percent.return_value = 45.0
        mock_process.num_fds.return_value = 20
        mock_process.connections.return_value = []  # No real connections
        mock_process.num_threads.return_value = 6
        mock_process_class.return_value = mock_process
        
        mock_gc.get_objects.return_value = [None] * 10000
        mock_gc.get_stats.return_value = [None] * 100
        
        mock_monitor = Mock(spec=PerformanceMonitor)
        
        detector = create_memory_leak_detector(mock_monitor)
        
        assert isinstance(detector, MemoryLeakDetector)
        assert detector.performance_monitor == mock_monitor
        assert detector.detection_window_minutes == 30.0
        assert detector.sampling_interval_seconds == 60.0
        assert detector.enable_profiling is True
    
    @patch('src.sovereign.memory_leak_detector.psutil.Process')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_create_memory_leak_detector_without_monitor(self, mock_gc, mock_process_class):
        """Test factory function without performance monitor"""
        # Setup complete psutil.Process mock to prevent real network calls
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=256*1024*1024)
        mock_process.memory_percent.return_value = 45.0
        mock_process.num_fds.return_value = 20
        mock_process.connections.return_value = []  # No real connections
        mock_process.num_threads.return_value = 6
        mock_process_class.return_value = mock_process
        
        mock_gc.get_objects.return_value = [None] * 10000
        mock_gc.get_stats.return_value = [None] * 100
        
        detector = create_memory_leak_detector(None)
        
        assert isinstance(detector, MemoryLeakDetector)
        assert detector.performance_monitor is None


class TestMemoryLeakAlerts:
    """Test memory leak alerting functionality"""
    
    @patch('src.sovereign.memory_leak_detector.psutil.Process')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_leak_alert_generation(self, mock_gc, mock_process_class, mock_performance_monitor):
        """Test generating memory leak alerts"""
        # Setup complete psutil.Process mock to prevent real network calls
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=256*1024*1024)
        mock_process.memory_percent.return_value = 45.0
        mock_process.num_fds.return_value = 20
        mock_process.connections.return_value = []  # No real connections
        mock_process.num_threads.return_value = 6
        mock_process_class.return_value = mock_process
        
        mock_gc.get_objects.return_value = [None] * 10000
        mock_gc.get_stats.return_value = [None] * 100
        
        detector = MemoryLeakDetector(performance_monitor=mock_performance_monitor)
        
        # Create a leak detection result that should trigger an alert
        result = LeakDetectionResult(
            leak_type=LeakType.MEMORY_GROWTH,
            detected=True,
            severity=AlertLevel.WARNING,
            leak_rate=10.0,  # 10 MB/min growth
            confidence_score=0.9,
            time_window_minutes=30.0,
            current_value=300.0,
            baseline_value=200.0,
            growth_trend=MemoryTrend.GRADUAL_INCREASE,
            recommended_actions=["Check for memory leaks", "Monitor usage"]
        )
        
        detector._generate_leak_alert(result)
        
        # Verify alert was added to detector
        assert len(detector.leak_alerts) == 1
        alert = detector.leak_alerts[0]
        
        assert alert.memory_increase_mb == 100.0  # 300 - 200
        assert alert.leak_rate_mb_per_minute == 10.0
        assert alert.timespan_minutes == 30.0
        assert "Check for memory leaks" in alert.recommended_action
        
        # Verify performance monitor was called
        mock_performance_monitor._add_alert.assert_called_once()
    
    @patch('src.sovereign.memory_leak_detector.psutil.Process')
    @patch('src.sovereign.memory_leak_detector.gc')
    def test_multiple_leak_types_detection(self, mock_gc, mock_process_class, mock_performance_monitor):
        """Test detection of multiple leak types"""
        # Setup complete psutil.Process mock to prevent real network calls
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=256*1024*1024)
        mock_process.memory_percent.return_value = 45.0
        mock_process.num_fds.return_value = 20
        mock_process.connections.return_value = []  # No real connections
        mock_process.num_threads.return_value = 6
        mock_process_class.return_value = mock_process
        
        mock_gc.get_objects.return_value = [None] * 10000
        mock_gc.get_stats.return_value = [None] * 100
        
        detector = MemoryLeakDetector(performance_monitor=mock_performance_monitor)
        
        # Create history with multiple resource increases
        base_time = datetime.now() - timedelta(minutes=30)
        history = []
        
        for i in range(5):
            snapshot = MemorySnapshot(
                timestamp=base_time + timedelta(minutes=i*6),
                process_memory_mb=100.0 + (i * 10),  # Memory growth
                system_memory_percent=30.0,
                file_descriptor_count=10 + (i * 2),  # FD growth
                thread_count=5 + i,  # Thread growth
                network_connection_count=2 + i,  # Connection growth
                python_objects_count=10000 + (i * 500)  # Object growth
            )
            history.append(snapshot)
        
        # Manually trigger analysis
        detector.memory_history.extend(history)
        detector._analyze_for_leaks()
        
        # Should have results for multiple leak types
        assert len(detector.detection_results) > 0
        
        # Check that different leak types were analyzed
        result_types = [result.leak_type for result in detector.detection_results.values()]
        assert LeakType.MEMORY_GROWTH in [r for r in result_types if hasattr(r, 'value')]


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 