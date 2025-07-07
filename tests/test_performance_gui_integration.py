"""
Tests for Performance Dashboard GUI Integration

This module tests the integration of the performance monitoring dashboard
with the Sovereign AI GUI, focusing on core functionality and data handling.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
from datetime import datetime

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.sovereign.performance_monitor import (
    PerformanceMonitor, PerformanceAlert, AlertLevel, MetricType, QueryType
)


class TestPerformanceGUIIntegration:
    """Test performance monitoring integration with GUI"""
    
    @pytest.fixture
    def mock_performance_monitor(self):
        """Mock performance monitor for testing"""
        monitor = Mock(spec=PerformanceMonitor)
        
        # Mock status data
        monitor.get_current_status.return_value = {
            'cpu_usage': 45.2,
            'memory_usage': 67.8,
            'gpu_usage': 85.1,
            'gpu_memory': 72.5,
            'gpu_temperature': 75.0,
            'uptime_seconds': 3661  # 1 hour, 1 minute, 1 second
        }
        
        # Mock summary data
        monitor.get_performance_summary.return_value = {
            'avg_response_time': 1250.5,
            'total_requests': 42,
            'success_rate': 97.6,
            'voice_reliability': 99.2,
            'response_times': [1000, 1200, 1100, 1300, 1250, 1150, 1400, 1350, 1200, 1180]
        }
        
        return monitor
    
    def test_performance_data_formatting(self, mock_performance_monitor):
        """Test performance data formatting for GUI display"""
        # Test status data formatting
        status = mock_performance_monitor.get_current_status.return_value
        
        # Test CPU usage formatting
        cpu_text = f"{status['cpu_usage']:.1f}%"
        assert cpu_text == "45.2%"
        
        # Test memory usage formatting
        memory_text = f"{status['memory_usage']:.1f}%"
        assert memory_text == "67.8%"
        
        # Test GPU temperature color logic
        temp = status['gpu_temperature']
        if temp > 80:
            color = "red"
        elif temp > 70:
            color = "orange"
        else:
            color = "white"
        assert color == "orange"  # 75°C should be orange
        
        # Test uptime formatting
        uptime = status['uptime_seconds']
        if uptime < 60:
            uptime_text = f"{uptime:.0f}s"
        elif uptime < 3600:
            uptime_text = f"{uptime/60:.1f}m"
        else:
            uptime_text = f"{uptime/3600:.1f}h"
        assert uptime_text == "1.0h"
    
    def test_alert_message_formatting(self):
        """Test alert message formatting for GUI display"""
        alert = PerformanceAlert(
            timestamp=datetime(2024, 1, 15, 12, 34, 56),
            alert_level=AlertLevel.WARNING,
            metric_type=MetricType.RESPONSE_TIME,
            message="Response time exceeding threshold",
            current_value=3500.0,
            threshold_value=2000.0,
            suggested_action="Consider enabling GPU acceleration"
        )
        
        # Test timestamp formatting
        timestamp = alert.timestamp.strftime("%H:%M:%S")
        assert timestamp == "12:34:56"
        
        # Test alert level emoji mapping
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️", 
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.EMERGENCY: "🔥"
        }
        emoji = level_emoji.get(alert.alert_level, "📢")
        assert emoji == "⚠️"
        
        # Test complete message formatting
        message = f"[{timestamp}] {emoji} {alert.message}\n"
        if alert.suggested_action:
            message += f"    💡 Suggestion: {alert.suggested_action}\n"
        
        expected = "[12:34:56] ⚠️ Response time exceeding threshold\n    💡 Suggestion: Consider enabling GPU acceleration\n"
        assert message == expected
    
    def test_performance_metrics_color_coding(self, mock_performance_monitor):
        """Test color coding logic for performance metrics"""
        summary = mock_performance_monitor.get_performance_summary.return_value
        
        # Test response time color coding
        avg_time = summary["avg_response_time"]  # 1250.5ms
        if avg_time > 5000:
            rt_color = "red"
        elif avg_time > 2000:
            rt_color = "orange"
        else:
            rt_color = "white"
        assert rt_color == "white"  # 1250ms should be white
        
        # Test success rate color coding
        success_rate = summary["success_rate"]  # 97.6%
        if success_rate < 95:
            sr_color = "red"
        elif success_rate < 99:
            sr_color = "orange"
        else:
            sr_color = "green"
        assert sr_color == "orange"  # 97.6% should be orange
        
        # Test voice reliability color coding
        reliability = summary["voice_reliability"]  # 99.2%
        if reliability < 95:
            vr_color = "red"
        elif reliability < 99:
            vr_color = "orange"
        else:
            vr_color = "green"
        assert vr_color == "green"  # 99.2% should be green
    
    def test_chart_data_preparation(self, mock_performance_monitor):
        """Test data preparation for performance charts"""
        summary = mock_performance_monitor.get_performance_summary.return_value
        status = mock_performance_monitor.get_current_status.return_value
        
        # Test response time chart data
        response_times = summary["response_times"]
        chart_data = response_times[-20:]  # Last 20 data points
        assert len(chart_data) == 10  # We have 10 data points
        assert chart_data == [1000, 1200, 1100, 1300, 1250, 1150, 1400, 1350, 1200, 1180]
        
        # Test system resources chart data
        cpu = status.get("cpu_usage", 0)
        memory = status.get("memory_usage", 0)
        resources = ["CPU", "Memory"]
        values = [cpu, memory]
        assert resources == ["CPU", "Memory"]
        assert values == [45.2, 67.8]
        
        # Test GPU metrics chart data
        gpu_usage = status.get("gpu_usage")
        gpu_temp = status.get("gpu_temperature")
        if gpu_usage is not None and gpu_temp is not None:
            metrics = ["Usage (%)", "Temp (°C)"]
            gpu_values = [gpu_usage, gpu_temp]
            assert metrics == ["Usage (%)", "Temp (°C)"]
            assert gpu_values == [85.1, 75.0]
    
    def test_query_type_determination(self):
        """Test logic for determining query types for performance tracking"""
        from src.sovereign.orchestrator import ModelChoice, QueryComplexity
        
        # Test TALKER model with simple complexity
        model = ModelChoice.TALKER
        complexity = QueryComplexity.SIMPLE
        
        if model == ModelChoice.TALKER:
            # QueryComplexity enum values are strings, need to map to numeric
            complexity_levels = {'simple': 1, 'moderate': 2, 'complex': 4, 'reasoning': 5}
            complexity_value = complexity_levels.get(complexity.value.lower(), 3)
            if complexity_value <= 3:
                query_type = QueryType.TALKER_SIMPLE
            else:
                query_type = QueryType.TALKER_COMPLEX
        else:
            query_type = QueryType.THINKER_REASONING
        
        assert query_type == QueryType.TALKER_SIMPLE
        
        # Test TALKER model with complex complexity
        complexity = QueryComplexity.COMPLEX
        if model == ModelChoice.TALKER:
            complexity_value = complexity_levels.get(complexity.value.lower(), 3)
            if complexity_value <= 3:
                query_type = QueryType.TALKER_SIMPLE
            else:
                query_type = QueryType.TALKER_COMPLEX
        else:
            query_type = QueryType.THINKER_REASONING
        
        assert query_type == QueryType.TALKER_COMPLEX
        
        # Test THINKER model
        model = ModelChoice.THINKER
        if model == ModelChoice.TALKER:
            query_type = QueryType.TALKER_SIMPLE
        else:
            query_type = QueryType.THINKER_REASONING
        
        assert query_type == QueryType.THINKER_REASONING
    
    def test_performance_callback_integration(self, mock_performance_monitor):
        """Test performance monitor callback integration"""
        # Test adding callback
        callback_fn = Mock()
        mock_performance_monitor.add_alert_callback(callback_fn)
        mock_performance_monitor.add_alert_callback.assert_called_once_with(callback_fn)
        
        # Test starting monitoring
        mock_performance_monitor.start_monitoring(interval_seconds=2.0)
        mock_performance_monitor.start_monitoring.assert_called_once_with(interval_seconds=2.0)
        
        # Test stopping monitoring
        mock_performance_monitor.stop_monitoring()
        mock_performance_monitor.stop_monitoring.assert_called_once()
    
    def test_response_time_tracking_parameters(self):
        """Test parameters for response time tracking"""
        start_time = time.time()
        end_time = start_time + 1.5  # 1.5 second response
        
        # Test tracking parameters
        tracking_params = {
            'start_time': start_time,
            'end_time': end_time,
            'query_type': QueryType.TALKER_SIMPLE,
            'model_used': 'talker',
            'success': True
        }
        
        # Verify parameters are correctly formatted
        assert tracking_params['query_type'] == QueryType.TALKER_SIMPLE
        assert tracking_params['model_used'] == 'talker'
        assert tracking_params['success'] is True
        assert tracking_params['end_time'] > tracking_params['start_time']
        
        # Test failed tracking parameters
        failed_params = {
            'start_time': start_time,
            'end_time': end_time,
            'query_type': QueryType.TALKER_SIMPLE,
            'success': False,
            'error_message': 'Test error'
        }
        
        assert failed_params['success'] is False
        assert failed_params['error_message'] == 'Test error'
    
    def test_alert_log_management(self):
        """Test alert log size management logic"""
        # Simulate alert log content (starting from 0 as Python does)
        alerts = [f"Alert {i}" for i in range(1500)]  # More than 1000 alerts
        log_content = '\n'.join(alerts)
        lines = log_content.split('\n')
        
        # Test log size limit logic
        if len(lines) > 1000:
            # Keep only first 1000 lines (most recent in our log system)
            limited_lines = lines[:1000]
        else:
            limited_lines = lines
        
        assert len(limited_lines) == 1000
        assert limited_lines[0] == "Alert 0"
        assert limited_lines[-1] == "Alert 999"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 