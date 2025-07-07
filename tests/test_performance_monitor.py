"""
Tests for Performance Monitoring Framework

This test suite covers:
- Performance metric collection and storage
- Response time tracking for different query types
- Voice reliability monitoring
- GPU and system resource monitoring
- Alert generation and threshold checking
- Performance optimization suggestions
- Metric export and cleanup functionality
- Real-time monitoring system
"""

import pytest
import time
import tempfile
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.sovereign.performance_monitor import (
    PerformanceMonitor, MetricType, QueryType, AlertLevel,
    PerformanceMetric, ResponseTimeMetric, VoiceReliabilityMetric,
    PerformanceAlert, PerformanceThresholds, PerformanceOptimizer,
    ResponseTimeTracker, track_response_time
)


class TestMetricTypes:
    """Test metric type definitions and enums"""
    
    def test_metric_types(self):
        """Test all metric types are properly defined"""
        assert MetricType.RESPONSE_TIME.value == "response_time"
        assert MetricType.MEMORY_USAGE.value == "memory_usage"
        assert MetricType.GPU_UTILIZATION.value == "gpu_utilization"
        assert MetricType.VOICE_RELIABILITY.value == "voice_reliability"
    
    def test_query_types(self):
        """Test all query types are properly defined"""
        assert QueryType.TALKER_SIMPLE.value == "talker_simple"
        assert QueryType.THINKER_REASONING.value == "thinker_reasoning"
        assert QueryType.VOICE_RECOGNITION.value == "voice_recognition"
        assert QueryType.RAG_SEARCH.value == "rag_search"
    
    def test_alert_levels(self):
        """Test alert levels are properly defined"""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.EMERGENCY.value == "emergency"


class TestPerformanceMetric:
    """Test performance metric data structures"""
    
    def test_performance_metric_creation(self):
        """Test creating a performance metric"""
        timestamp = datetime.now()
        metric = PerformanceMetric(
            timestamp=timestamp,
            metric_type=MetricType.RESPONSE_TIME,
            value=1500.0,
            unit="ms",
            context={"query": "test"},
            tags={"model": "talker"}
        )
        
        assert metric.timestamp == timestamp
        assert metric.metric_type == MetricType.RESPONSE_TIME
        assert metric.value == 1500.0
        assert metric.unit == "ms"
        assert metric.context["query"] == "test"
        assert metric.tags["model"] == "talker"
    
    def test_response_time_metric(self):
        """Test response time metric structure"""
        start_time = time.time()
        end_time = start_time + 1.5
        
        metric = ResponseTimeMetric(
            query_type=QueryType.TALKER_SIMPLE,
            start_time=start_time,
            end_time=end_time,
            duration_ms=1500.0,
            model_used="gemma2:9b",
            token_count=150,
            success=True
        )
        
        assert metric.query_type == QueryType.TALKER_SIMPLE
        assert metric.duration_ms == 1500.0
        assert metric.model_used == "gemma2:9b"
        assert metric.token_count == 150
        assert metric.success is True
    
    def test_voice_reliability_metric(self):
        """Test voice reliability metric structure"""
        metric = VoiceReliabilityMetric(
            operation_type="recognition",
            success=True,
            duration_ms=2000.0,
            audio_quality_score=0.9,
            confidence_score=0.85
        )
        
        assert metric.operation_type == "recognition"
        assert metric.success is True
        assert metric.duration_ms == 2000.0
        assert metric.audio_quality_score == 0.9
        assert metric.confidence_score == 0.85


class TestPerformanceThresholds:
    """Test performance threshold configuration"""
    
    def test_default_thresholds(self):
        """Test default threshold values"""
        thresholds = PerformanceThresholds()
        
        assert thresholds.talker_max_response_ms == 2000
        assert thresholds.thinker_max_response_ms == 30000
        assert thresholds.max_memory_usage_percent == 85.0
        assert thresholds.min_voice_reliability_percent == 99.0
        assert thresholds.max_gpu_temperature_celsius == 83.0
    
    def test_custom_thresholds(self):
        """Test custom threshold configuration"""
        thresholds = PerformanceThresholds(
            talker_max_response_ms=1500,
            max_memory_usage_percent=80.0,
            max_gpu_temperature_celsius=80.0
        )
        
        assert thresholds.talker_max_response_ms == 1500
        assert thresholds.max_memory_usage_percent == 80.0
        assert thresholds.max_gpu_temperature_celsius == 80.0


@pytest.fixture
def temp_metrics_dir():
    """Create temporary directory for metrics"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture 
def mock_hardware_detector():
    """Mock hardware detector"""
    mock_detector = Mock()
    mock_detector.system_info = Mock()
    mock_detector.system_info.cuda_available = False
    mock_detector.system_info.platform = "test_platform"
    mock_detector.system_info.cpu_count = 8
    mock_detector.system_info.memory_total = 32.0
    mock_detector.gpu_info = None
    return mock_detector


class TestPerformanceOptimizer:
    """Test performance optimization suggestions"""
    
    def test_optimizer_initialization(self, mock_hardware_detector):
        """Test optimizer initialization"""
        optimizer = PerformanceOptimizer(mock_hardware_detector)
        assert optimizer.hardware == mock_hardware_detector
    
    def test_response_time_analysis(self, mock_hardware_detector):
        """Test response time analysis and suggestions"""
        optimizer = PerformanceOptimizer(mock_hardware_detector)
        
        # Create metrics with slow response times
        metrics = [
            PerformanceMetric(
                timestamp=datetime.now(),
                metric_type=MetricType.RESPONSE_TIME,
                value=5000.0,  # 5 seconds - slow
                unit="ms"
            )
        ]
        
        suggestions = optimizer.analyze_performance_data(metrics)
        
        assert len(suggestions) > 0
        assert any("GPU acceleration" in suggestion for suggestion in suggestions)
        assert any("batch size" in suggestion for suggestion in suggestions)
    
    def test_memory_usage_analysis(self, mock_hardware_detector):
        """Test memory usage pattern analysis"""
        optimizer = PerformanceOptimizer(mock_hardware_detector)
        
        # Create metrics showing memory growth
        base_time = datetime.now()
        metrics = []
        
        # Simulate growing memory usage
        for i in range(20):
            metrics.append(PerformanceMetric(
                timestamp=base_time + timedelta(minutes=i),
                metric_type=MetricType.MEMORY_USAGE,
                value=50.0 + (i * 2),  # Growing from 50% to 88%
                unit="percent"
            ))
        
        suggestions = optimizer.analyze_performance_data(metrics)
        
        assert len(suggestions) > 0
        assert any("memory leak" in suggestion.lower() for suggestion in suggestions)


class TestPerformanceMonitor:
    """Test the main PerformanceMonitor class"""
    
    @patch('src.sovereign.performance_monitor.HardwareDetector')
    def test_monitor_initialization(self, mock_hardware_class, temp_metrics_dir, mock_hardware_detector):
        """Test monitor initialization"""
        mock_hardware_class.return_value = mock_hardware_detector
        
        monitor = PerformanceMonitor(
            enable_gpu_monitoring=False,
            enable_real_time_alerts=True,
            metrics_retention_days=7
        )
        
        assert monitor.enable_gpu_monitoring is False
        assert monitor.enable_real_time_alerts is True
        assert monitor.metrics_retention_days == 7
        assert monitor.hardware == mock_hardware_detector
    
    @patch('src.sovereign.performance_monitor.HardwareDetector')
    @patch('src.sovereign.performance_monitor.psutil')
    def test_track_response_time(self, mock_psutil, mock_hardware_class, temp_metrics_dir, mock_hardware_detector):
        """Test response time tracking"""
        # Setup mocks
        mock_hardware_class.return_value = mock_hardware_detector
        
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        
        # Track response time
        start_time = time.time()
        end_time = start_time + 1.5
        
        metric = monitor.track_response_time(
            start_time=start_time,
            end_time=end_time,
            query_type=QueryType.TALKER_SIMPLE,
            model_used="test_model",
            token_count=100,
            success=True
        )
        
        assert metric.query_type == QueryType.TALKER_SIMPLE
        assert metric.duration_ms == 1500.0
        assert metric.model_used == "test_model"
        assert metric.token_count == 100
        assert metric.success is True
        assert metric.memory_peak == 60.0
        
        # Check metric was stored
        assert len(monitor.response_time_metrics) == 1
        assert len(monitor.metrics[MetricType.RESPONSE_TIME]) == 1
    
    @patch('src.sovereign.performance_monitor.HardwareDetector')
    def test_track_voice_reliability(self, mock_hardware_class, temp_metrics_dir, mock_hardware_detector):
        """Test voice reliability tracking"""
        mock_hardware_class.return_value = mock_hardware_detector
        
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        
        # Track successful voice operation
        metric = monitor.track_voice_reliability(
            operation_type="recognition",
            success=True,
            duration_ms=2500.0,
            audio_quality_score=0.95,
            confidence_score=0.88
        )
        
        assert metric.operation_type == "recognition"
        assert metric.success is True
        assert metric.duration_ms == 2500.0
        assert metric.audio_quality_score == 0.95
        assert metric.confidence_score == 0.88
        
        # Check metric was stored
        assert len(monitor.voice_reliability_metrics) == 1
        assert len(monitor.metrics[MetricType.VOICE_RELIABILITY]) == 1
    
    @patch('src.sovereign.performance_monitor.HardwareDetector')
    def test_alert_generation(self, mock_hardware_class, temp_metrics_dir, mock_hardware_detector):
        """Test alert generation for threshold breaches"""
        mock_hardware_class.return_value = mock_hardware_detector
        
        monitor = PerformanceMonitor(enable_gpu_monitoring=False, enable_real_time_alerts=True)
        
        # Manually trigger alert by adding high memory usage metric
        monitor._add_metric(MetricType.MEMORY_USAGE, 90.0, "percent")  # Above 85% threshold
        
        # Check alerts
        monitor._check_alert_conditions()
        
        assert len(monitor.alerts) >= 1
        alert = monitor.alerts[-1]
        assert alert.alert_level == AlertLevel.WARNING
        assert alert.metric_type == MetricType.MEMORY_USAGE
        assert alert.current_value == 90.0
        assert alert.threshold_value == 85.0
    
    @patch('src.sovereign.performance_monitor.HardwareDetector')
    @patch('src.sovereign.performance_monitor.psutil')
    def test_system_metrics_collection(self, mock_psutil, mock_hardware_class, temp_metrics_dir, mock_hardware_detector):
        """Test system metrics collection"""
        # Setup mocks
        mock_hardware_class.return_value = mock_hardware_detector
        
        mock_memory = Mock()
        mock_memory.percent = 70.0
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 45.0
        
        mock_disk_io = Mock()
        mock_disk_io.read_bytes = 1024 * 1024 * 100  # 100MB
        mock_disk_io.write_bytes = 1024 * 1024 * 50   # 50MB
        mock_psutil.disk_io_counters.return_value = mock_disk_io
        
        mock_net_io = Mock()
        mock_net_io.bytes_sent = 1024 * 1024 * 20     # 20MB
        mock_net_io.bytes_recv = 1024 * 1024 * 30     # 30MB
        mock_psutil.net_io_counters.return_value = mock_net_io
        
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        
        # Collect metrics
        monitor._collect_system_metrics()
        
        # Verify metrics were collected
        assert len(monitor.metrics[MetricType.MEMORY_USAGE]) >= 1
        assert len(monitor.metrics[MetricType.CPU_USAGE]) >= 1
        assert len(monitor.metrics[MetricType.DISK_IO]) >= 1
        assert len(monitor.metrics[MetricType.NETWORK_IO]) >= 1
        
        # Verify metric values
        memory_metric = monitor.metrics[MetricType.MEMORY_USAGE][-1]
        assert memory_metric.value == 70.0
        
        cpu_metric = monitor.metrics[MetricType.CPU_USAGE][-1]
        assert cpu_metric.value == 45.0
    
    @patch('src.sovereign.performance_monitor.HardwareDetector')
    def test_performance_summary(self, mock_hardware_class, temp_metrics_dir, mock_hardware_detector):
        """Test performance summary generation"""
        mock_hardware_class.return_value = mock_hardware_detector
        
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        
        # Add some test data
        start_time = time.time()
        monitor.track_response_time(
            start_time=start_time,
            end_time=start_time + 1.2,
            query_type=QueryType.TALKER_SIMPLE,
            model_used="test_model",
            success=True
        )
        
        monitor.track_voice_reliability(
            operation_type="recognition",
            success=True,
            duration_ms=2000.0
        )
        
        # Generate summary
        summary = monitor.get_performance_summary(hours=1)
        
        assert "system_info" in summary
        assert "response_times" in summary
        assert "voice_reliability" in summary
        assert "resource_usage" in summary
        assert "alerts" in summary
        assert "optimization_suggestions" in summary
        
        # Check response times section
        if "talker_simple" in summary["response_times"]:
            talker_stats = summary["response_times"]["talker_simple"]
            assert talker_stats["total_requests"] == 1
            assert talker_stats["successful_requests"] == 1
            assert talker_stats["success_rate"] == 100.0
    
    @patch('src.sovereign.performance_monitor.HardwareDetector')
    def test_metrics_export(self, mock_hardware_class, temp_metrics_dir, mock_hardware_detector):
        """Test metrics export to JSON"""
        mock_hardware_class.return_value = mock_hardware_detector
        
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        
        # Add test data
        start_time = time.time()
        monitor.track_response_time(
            start_time=start_time,
            end_time=start_time + 0.8,
            query_type=QueryType.RAG_SEARCH,
            success=True
        )
        
        # Export metrics
        export_path = monitor.export_metrics(hours=1)
        
        assert Path(export_path).exists()
        
        # Verify export content
        import json
        with open(export_path, 'r') as f:
            data = json.load(f)
        
        assert "time_period_hours" in data
        assert "system_info" in data
        assert data["time_period_hours"] == 1
    
    @patch('src.sovereign.performance_monitor.HardwareDetector')
    def test_current_status(self, mock_hardware_class, temp_metrics_dir, mock_hardware_detector):
        """Test current system status reporting"""
        mock_hardware_class.return_value = mock_hardware_detector
        
        with patch('src.sovereign.performance_monitor.psutil') as mock_psutil:
            mock_memory = Mock()
            mock_memory.percent = 65.0
            mock_psutil.virtual_memory.return_value = mock_memory
            mock_psutil.cpu_percent.return_value = 40.0
            
            mock_disk_usage = Mock()
            mock_disk_usage.percent = 75.0
            mock_psutil.disk_usage.return_value = mock_disk_usage
            
            monitor = PerformanceMonitor(enable_gpu_monitoring=False)
            
            status = monitor.get_current_status()
            
            assert "timestamp" in status
            assert "monitoring_active" in status
            assert "system" in status
            assert status["system"]["memory_percent"] == 65.0
            assert status["system"]["cpu_percent"] == 40.0
            assert status["system"]["disk_usage_percent"] == 75.0
    
    @patch('src.sovereign.performance_monitor.HardwareDetector')
    def test_alert_callbacks(self, mock_hardware_class, temp_metrics_dir, mock_hardware_detector):
        """Test alert callback functionality"""
        mock_hardware_class.return_value = mock_hardware_detector
        
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        
        # Add callback
        callback_called = []
        def test_callback(alert):
            callback_called.append(alert)
        
        monitor.add_alert_callback(test_callback)
        
        # Trigger alert
        alert = PerformanceAlert(
            timestamp=datetime.now(),
            alert_level=AlertLevel.WARNING,
            metric_type=MetricType.MEMORY_USAGE,
            message="Test alert",
            current_value=90.0,
            threshold_value=85.0
        )
        
        monitor._add_alert(alert)
        
        # Verify callback was called
        assert len(callback_called) == 1
        assert callback_called[0] == alert
    
    @patch('src.sovereign.performance_monitor.HardwareDetector')
    def test_cleanup_old_metrics(self, mock_hardware_class, temp_metrics_dir, mock_hardware_detector):
        """Test cleanup of old metrics"""
        mock_hardware_class.return_value = mock_hardware_detector
        
        monitor = PerformanceMonitor(enable_gpu_monitoring=False, metrics_retention_days=1)
        
        # Add old metric (2 days ago)
        old_timestamp = datetime.now() - timedelta(days=2)
        old_metric = PerformanceMetric(
            timestamp=old_timestamp,
            metric_type=MetricType.MEMORY_USAGE,
            value=50.0,
            unit="percent"
        )
        monitor.metrics[MetricType.MEMORY_USAGE].append(old_metric)
        
        # Add recent metric
        recent_metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.MEMORY_USAGE,
            value=60.0,
            unit="percent"
        )
        monitor.metrics[MetricType.MEMORY_USAGE].append(recent_metric)
        
        assert len(monitor.metrics[MetricType.MEMORY_USAGE]) == 2
        
        # Cleanup old metrics
        monitor.cleanup_old_metrics()
        
        # Only recent metric should remain
        assert len(monitor.metrics[MetricType.MEMORY_USAGE]) == 1
        assert monitor.metrics[MetricType.MEMORY_USAGE][0].value == 60.0


class TestResponseTimeTracker:
    """Test response time tracking context manager"""
    
    @patch('src.sovereign.performance_monitor.HardwareDetector')
    def test_response_time_tracker_success(self, mock_hardware_class, temp_metrics_dir, mock_hardware_detector):
        """Test successful response time tracking"""
        mock_hardware_class.return_value = mock_hardware_detector
        
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        
        with ResponseTimeTracker(monitor, QueryType.TALKER_SIMPLE, "test_model") as tracker:
            time.sleep(0.1)  # Simulate some work
            tracker.set_token_count(50)
        
        # Verify metric was recorded
        assert len(monitor.response_time_metrics) == 1
        metric = monitor.response_time_metrics[0]
        assert metric.query_type == QueryType.TALKER_SIMPLE
        assert metric.model_used == "test_model"
        assert metric.token_count == 50
        assert metric.success is True
        assert metric.duration_ms >= 100  # At least 100ms from sleep
    
    @patch('src.sovereign.performance_monitor.HardwareDetector')
    def test_response_time_tracker_error(self, mock_hardware_class, temp_metrics_dir, mock_hardware_detector):
        """Test response time tracking with error"""
        mock_hardware_class.return_value = mock_hardware_detector
        
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        
        try:
            with ResponseTimeTracker(monitor, QueryType.THINKER_REASONING) as tracker:
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected
        
        # Verify metric was recorded with error
        assert len(monitor.response_time_metrics) == 1
        metric = monitor.response_time_metrics[0]
        assert metric.query_type == QueryType.THINKER_REASONING
        assert metric.success is False
        assert "Test error" in metric.error_message


class TestMonitoringIntegration:
    """Integration tests for the monitoring system"""
    
    @patch('src.sovereign.performance_monitor.HardwareDetector')
    @patch('src.sovereign.performance_monitor.psutil')
    def test_continuous_monitoring(self, mock_psutil, mock_hardware_class, temp_metrics_dir, mock_hardware_detector):
        """Test continuous monitoring loop"""
        # Setup mocks
        mock_hardware_class.return_value = mock_hardware_detector
        
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        
        # Start monitoring
        monitor.start_monitoring(interval_seconds=0.1)
        
        # Let it run for a short time
        time.sleep(0.3)
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Verify metrics were collected
        assert len(monitor.metrics[MetricType.MEMORY_USAGE]) >= 2
        assert len(monitor.metrics[MetricType.CPU_USAGE]) >= 2
    
    @patch('src.sovereign.performance_monitor.HardwareDetector')
    def test_decorator_functionality(self, mock_hardware_class, temp_metrics_dir, mock_hardware_detector):
        """Test the response time tracking decorator"""
        mock_hardware_class.return_value = mock_hardware_detector
        
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        
        @track_response_time(monitor, QueryType.RAG_SEARCH, "search_model")
        def test_function():
            time.sleep(0.05)  # Simulate work
            return {"result": "success", "token_count": 25}
        
        result = test_function()
        
        # Verify function result
        assert result["result"] == "success"
        
        # Verify metrics were recorded
        assert len(monitor.response_time_metrics) == 1
        metric = monitor.response_time_metrics[0]
        assert metric.query_type == QueryType.RAG_SEARCH
        assert metric.model_used == "search_model"
        assert metric.token_count == 25
        assert metric.success is True
    
    @patch('src.sovereign.performance_monitor.HardwareDetector')
    def test_context_manager(self, mock_hardware_class, temp_metrics_dir, mock_hardware_detector):
        """Test monitor as context manager"""
        mock_hardware_class.return_value = mock_hardware_detector
        
        with PerformanceMonitor(enable_gpu_monitoring=False) as monitor:
            assert monitor._monitoring_active is False
            
            # Add some metrics
            monitor.track_response_time(
                start_time=time.time(),
                end_time=time.time() + 0.5,
                query_type=QueryType.VOICE_SYNTHESIS,
                success=True
            )
            
            assert len(monitor.response_time_metrics) == 1
        
        # Monitor should be properly closed


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 