"""
Performance Monitoring Framework for Sovereign AI Agent

This module provides comprehensive performance monitoring, including:
- Response time tracking for AI model inference
- Memory usage monitoring with leak detection
- GPU utilization and temperature monitoring  
- Voice interface reliability tracking
- Automated performance alerting and optimization suggestions
- Real-time metrics collection and aggregation
"""

import logging
import time
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import asyncio
from pathlib import Path

from .logger import get_performance_logger, PerformanceTimer
from .hardware import HardwareDetector, GPUInfo, SystemInfo


class MetricType(Enum):
    """Types of performance metrics tracked"""
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage" 
    GPU_UTILIZATION = "gpu_utilization"
    GPU_MEMORY = "gpu_memory"
    GPU_TEMPERATURE = "gpu_temperature"
    VOICE_RELIABILITY = "voice_reliability"
    CPU_USAGE = "cpu_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    MODEL_INFERENCE = "model_inference"
    CONTEXT_WINDOW = "context_window"
    RAG_SEARCH = "rag_search"


class QueryType(Enum):
    """Types of queries for response time tracking"""
    TALKER_SIMPLE = "talker_simple"
    TALKER_COMPLEX = "talker_complex"
    THINKER_REASONING = "thinker_reasoning"
    VOICE_RECOGNITION = "voice_recognition"
    VOICE_SYNTHESIS = "voice_synthesis"
    RAG_SEARCH = "rag_search"
    MEMORY_STORAGE = "memory_storage"
    CONTEXT_RETRIEVAL = "context_retrieval"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    timestamp: datetime
    metric_type: MetricType
    value: float
    unit: str
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ResponseTimeMetric:
    """Specialized metric for response time tracking"""
    query_type: QueryType
    start_time: float
    end_time: float
    duration_ms: float
    model_used: Optional[str] = None
    token_count: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    memory_peak: Optional[float] = None
    gpu_utilization: Optional[float] = None


@dataclass
class VoiceReliabilityMetric:
    """Voice interface reliability tracking"""
    operation_type: str  # "recognition" or "synthesis"
    success: bool
    duration_ms: float
    audio_quality_score: Optional[float] = None
    confidence_score: Optional[float] = None
    error_type: Optional[str] = None
    retry_count: int = 0


@dataclass
class MemoryLeakAlert:
    """Memory leak detection alert"""
    timestamp: datetime
    memory_increase_mb: float
    timespan_minutes: float
    leak_rate_mb_per_minute: float
    process_name: str
    recommended_action: str


@dataclass
class PerformanceAlert:
    """Performance alert notification"""
    timestamp: datetime
    alert_level: AlertLevel
    metric_type: MetricType
    message: str
    current_value: float
    threshold_value: float
    suggested_action: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThresholds:
    """Configurable performance thresholds for alerting"""
    # Response time thresholds (ms)
    talker_max_response_ms: float = 2000  # <2s requirement
    thinker_max_response_ms: float = 30000  # 30s for complex reasoning
    voice_recognition_max_ms: float = 3000
    voice_synthesis_max_ms: float = 5000
    rag_search_max_ms: float = 1000
    
    # Memory thresholds
    max_memory_usage_percent: float = 85.0
    memory_leak_rate_mb_per_min: float = 50.0
    
    # GPU thresholds  
    max_gpu_utilization_percent: float = 95.0
    max_gpu_memory_percent: float = 90.0
    max_gpu_temperature_celsius: float = 83.0
    
    # Voice reliability thresholds
    min_voice_reliability_percent: float = 99.0
    min_voice_confidence_score: float = 0.8
    
    # CPU thresholds
    max_cpu_usage_percent: float = 90.0


class PerformanceOptimizer:
    """Automated performance optimization suggestions"""
    
    def __init__(self, hardware_detector: HardwareDetector):
        self.hardware = hardware_detector
        self.logger = get_performance_logger()
    
    def analyze_performance_data(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Analyze metrics and provide optimization suggestions"""
        suggestions = []
        
        # Group metrics by type
        by_type = defaultdict(list)
        for metric in metrics:
            by_type[metric.metric_type].append(metric)
        
        # Analyze response times
        if MetricType.RESPONSE_TIME in by_type:
            suggestions.extend(self._analyze_response_times(by_type[MetricType.RESPONSE_TIME]))
        
        # Analyze memory usage
        if MetricType.MEMORY_USAGE in by_type:
            suggestions.extend(self._analyze_memory_usage(by_type[MetricType.MEMORY_USAGE]))
        
        # Analyze GPU performance
        if MetricType.GPU_UTILIZATION in by_type:
            suggestions.extend(self._analyze_gpu_performance(by_type[MetricType.GPU_UTILIZATION]))
        
        return suggestions
    
    def _analyze_response_times(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Analyze response time patterns and suggest optimizations"""
        suggestions = []
        
        # Calculate average response time
        avg_time = sum(m.value for m in metrics) / len(metrics)
        
        if avg_time > 3000:  # > 3 seconds
            suggestions.append("Consider enabling GPU acceleration for faster inference")
            suggestions.append("Optimize batch size based on available GPU memory")
            suggestions.append("Consider using quantized models for faster inference")
        
        return suggestions
    
    def _analyze_memory_usage(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Analyze memory usage patterns"""
        suggestions = []
        
        # Check for memory growth trend
        if len(metrics) >= 10:
            recent = metrics[-10:]
            older = metrics[-20:-10] if len(metrics) >= 20 else metrics[:-10]
            
            recent_avg = sum(m.value for m in recent) / len(recent)
            older_avg = sum(m.value for m in older) / len(older)
            
            if recent_avg > older_avg * 1.2:  # 20% increase
                suggestions.append("Potential memory leak detected - consider restarting services")
                suggestions.append("Enable periodic garbage collection")
                suggestions.append("Review model caching strategies")
        
        return suggestions
    
    def _analyze_gpu_performance(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Analyze GPU performance patterns"""
        suggestions = []
        
        avg_utilization = sum(m.value for m in metrics) / len(metrics)
        
        if avg_utilization < 30:
            suggestions.append("GPU underutilized - consider increasing batch size")
            suggestions.append("Enable mixed precision training for better GPU efficiency")
        elif avg_utilization > 95:
            suggestions.append("GPU overutilized - consider reducing batch size")
            suggestions.append("Monitor GPU temperature to prevent thermal throttling")
        
        return suggestions


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for Sovereign AI Agent
    
    Tracks and analyzes:
    - Response times for AI model inference
    - Memory and GPU resource utilization
    - Voice interface reliability
    - System health and stability
    - Automated performance optimization
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 enable_gpu_monitoring: bool = True,
                 enable_real_time_alerts: bool = True,
                 metrics_retention_days: int = 30,
                 enable_memory_leak_detection: bool = True,
                 enable_automated_optimization: bool = True):
        """
        Initialize the performance monitoring system
        
        Args:
            config_path: Optional path to configuration file
            enable_gpu_monitoring: Enable GPU utilization monitoring
            enable_real_time_alerts: Enable real-time alerting
            metrics_retention_days: How long to retain metrics data
            enable_memory_leak_detection: Enable advanced memory leak detection
            enable_automated_optimization: Enable automated performance optimization and crash recovery
        """
        self.logger = get_performance_logger()
        self.debug_logger = logging.getLogger("sovereign.debug")
        
        # Hardware detection
        self.hardware = HardwareDetector()
        self.enable_gpu_monitoring = enable_gpu_monitoring and self.hardware.system_info.cuda_available
        self.enable_memory_leak_detection = enable_memory_leak_detection
        
        # Configuration
        self.thresholds = PerformanceThresholds()
        self.metrics_retention_days = metrics_retention_days
        self.enable_real_time_alerts = enable_real_time_alerts
        
        # Metrics storage
        self.metrics: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.response_time_metrics: deque[ResponseTimeMetric] = deque(maxlen=5000)
        self.voice_reliability_metrics: deque[VoiceReliabilityMetric] = deque(maxlen=1000)
        self.alerts: deque[PerformanceAlert] = deque(maxlen=1000)
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Performance optimizer
        self.optimizer = PerformanceOptimizer(self.hardware)
        
        # Memory leak detector (lazy initialization)
        self._memory_leak_detector = None
        
        # Automated optimization and crash recovery system (lazy initialization)
        self.enable_automated_optimization = enable_automated_optimization
        self._automated_optimizer = None
        self._crash_recovery_system = None
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Metrics file path
        self.metrics_dir = Path("logs/performance")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Performance monitoring system initialized")
        if self.enable_gpu_monitoring:
            self.logger.info("GPU monitoring enabled")
        else:
            self.logger.warning("GPU monitoring disabled - no CUDA available")
        
        if self.enable_memory_leak_detection:
            self.logger.info("Memory leak detection enabled")
        
        if self.enable_automated_optimization:
            self.logger.info("Automated performance optimization and crash recovery enabled")
    
    def start_monitoring(self, interval_seconds: float = 5.0):
        """Start continuous performance monitoring"""
        if self._monitoring_active:
            self.logger.warning("Performance monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True,
            name="PerformanceMonitor"
        )
        self._monitoring_thread.start()
        
        # Start memory leak detection if enabled
        if self.enable_memory_leak_detection:
            self._initialize_memory_leak_detector()
            if self._memory_leak_detector:
                self._memory_leak_detector.start_monitoring()
        
        # Start automated optimization system if enabled
        if self.enable_automated_optimization:
            self._initialize_automated_optimization_system()
            if self._automated_optimizer:
                self._automated_optimizer.start_optimization_monitoring()
            if self._crash_recovery_system:
                self._crash_recovery_system.start_monitoring()
        
        self.logger.info(f"Started performance monitoring (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10.0)
        
        # Stop memory leak detection
        if self._memory_leak_detector:
            self._memory_leak_detector.stop_monitoring()
        
        # Stop automated optimization system
        if self._automated_optimizer:
            self._automated_optimizer.stop_optimization_monitoring()
        if self._crash_recovery_system:
            self._crash_recovery_system.stop_monitoring()
        
        self.logger.info("Stopped performance monitoring")
    
    def _monitoring_loop(self, interval_seconds: float):
        """Main monitoring loop running in separate thread"""
        while self._monitoring_active:
            try:
                self._collect_system_metrics()
                if self.enable_gpu_monitoring:
                    self._collect_gpu_metrics()
                
                # Check for alerts
                if self.enable_real_time_alerts:
                    self._check_alert_conditions()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.debug_logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        timestamp = datetime.now()
        
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self._add_metric(MetricType.MEMORY_USAGE, memory.percent, "percent", timestamp)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self._add_metric(MetricType.CPU_USAGE, cpu_percent, "percent", timestamp)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                read_mb_per_sec = disk_io.read_bytes / (1024*1024) / 5.0  # Approximate per-second rate
                write_mb_per_sec = disk_io.write_bytes / (1024*1024) / 5.0
                self._add_metric(MetricType.DISK_IO, read_mb_per_sec + write_mb_per_sec, "MB/s", timestamp)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                net_mb_per_sec = (net_io.bytes_sent + net_io.bytes_recv) / (1024*1024) / 5.0
                self._add_metric(MetricType.NETWORK_IO, net_mb_per_sec, "MB/s", timestamp)
        
        except Exception as e:
            self.debug_logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_gpu_metrics(self):
        """Collect GPU performance metrics"""
        import torch
        if not torch.cuda.is_available():
            return
        
        timestamp = datetime.now()
        
        try:
            device = torch.cuda.current_device()
            
            # GPU utilization (approximation)
            torch.cuda.synchronize()
            start_time = time.time()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            # GPU memory usage
            memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            
            memory_percent = (memory_reserved / total_memory) * 100
            
            self._add_metric(MetricType.GPU_MEMORY, memory_percent, "percent", timestamp)
            self._add_metric(MetricType.GPU_UTILIZATION, min(gpu_time * 100, 100), "percent", timestamp)
            
            # GPU temperature (if available via nvidia-ml-py)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                self._add_metric(MetricType.GPU_TEMPERATURE, temp, "celsius", timestamp)
            except ImportError:
                pass  # pynvml not available
            except Exception:
                pass  # GPU temperature not available
        
        except Exception as e:
            self.debug_logger.error(f"Error collecting GPU metrics: {e}")
    
    def _add_metric(self, metric_type: MetricType, value: float, unit: str, 
                   timestamp: Optional[datetime] = None, context: Optional[Dict] = None):
        """Add a metric to the monitoring system"""
        if timestamp is None:
            timestamp = datetime.now()
        
        metric = PerformanceMetric(
            timestamp=timestamp,
            metric_type=metric_type,
            value=value,
            unit=unit,
            context=context or {}
        )
        
        with self._lock:
            self.metrics[metric_type].append(metric)
    
    def track_response_time(self, start_time: float, end_time: float, query_type: QueryType,
                          model_used: Optional[str] = None, token_count: Optional[int] = None,
                          success: bool = True, error_message: Optional[str] = None) -> ResponseTimeMetric:
        """
        Track response time for AI model inference
        
        Args:
            start_time: Start timestamp (from time.time())
            end_time: End timestamp (from time.time())
            query_type: Type of query being tracked
            model_used: Name of the model used
            token_count: Number of tokens processed
            success: Whether the operation succeeded
            error_message: Error message if operation failed
        
        Returns:
            ResponseTimeMetric object
        """
        duration_ms = (end_time - start_time) * 1000
        
        # Collect current resource usage
        memory_percent = psutil.virtual_memory().percent
        gpu_utilization = None
        
        if self.enable_gpu_monitoring and torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_utilization = (memory_allocated / total_memory) * 100
            except Exception:
                pass
        
        metric = ResponseTimeMetric(
            query_type=query_type,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            model_used=model_used,
            token_count=token_count,
            success=success,
            error_message=error_message,
            memory_peak=memory_percent,
            gpu_utilization=gpu_utilization
        )
        
        with self._lock:
            self.response_time_metrics.append(metric)
        
        # Add to general metrics for alerting
        self._add_metric(MetricType.RESPONSE_TIME, duration_ms, "ms", 
                        context={"query_type": query_type.value, "success": success})
        
        # Log performance
        if success:
            self.logger.info(f"{query_type.value}: {duration_ms:.1f}ms ({model_used or 'unknown'})")
        else:
            self.logger.warning(f"{query_type.value}: FAILED after {duration_ms:.1f}ms - {error_message}")
        
        # Check response time thresholds
        if self.enable_real_time_alerts:
            self._check_response_time_threshold(metric)
        
        return metric
    
    def track_voice_reliability(self, operation_type: str, success: bool, duration_ms: float,
                              audio_quality_score: Optional[float] = None,
                              confidence_score: Optional[float] = None,
                              error_type: Optional[str] = None,
                              retry_count: int = 0) -> VoiceReliabilityMetric:
        """
        Track voice interface reliability metrics
        
        Args:
            operation_type: "recognition" or "synthesis"
            success: Whether the operation succeeded
            duration_ms: Operation duration in milliseconds
            audio_quality_score: Audio quality score (0-1)
            confidence_score: Recognition confidence score (0-1)
            error_type: Type of error if operation failed
            retry_count: Number of retries attempted
        
        Returns:
            VoiceReliabilityMetric object
        """
        metric = VoiceReliabilityMetric(
            operation_type=operation_type,
            success=success,
            duration_ms=duration_ms,
            audio_quality_score=audio_quality_score,
            confidence_score=confidence_score,
            error_type=error_type,
            retry_count=retry_count
        )
        
        with self._lock:
            self.voice_reliability_metrics.append(metric)
        
        # Add to general metrics
        self._add_metric(MetricType.VOICE_RELIABILITY, 1.0 if success else 0.0, "success_rate",
                        context={"operation_type": operation_type, "retry_count": retry_count})
        
        # Log voice reliability
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"Voice {operation_type}: {status} ({duration_ms:.1f}ms)")
        if not success and error_type:
            self.logger.warning(f"Voice {operation_type} error: {error_type}")
        
        return metric
    
    def _check_response_time_threshold(self, metric: ResponseTimeMetric):
        """Check if response time exceeds thresholds and generate alerts"""
        threshold_ms = None
        
        if metric.query_type == QueryType.TALKER_SIMPLE or metric.query_type == QueryType.TALKER_COMPLEX:
            threshold_ms = self.thresholds.talker_max_response_ms
        elif metric.query_type == QueryType.THINKER_REASONING:
            threshold_ms = self.thresholds.thinker_max_response_ms
        elif metric.query_type == QueryType.VOICE_RECOGNITION:
            threshold_ms = self.thresholds.voice_recognition_max_ms
        elif metric.query_type == QueryType.VOICE_SYNTHESIS:
            threshold_ms = self.thresholds.voice_synthesis_max_ms
        elif metric.query_type == QueryType.RAG_SEARCH:
            threshold_ms = self.thresholds.rag_search_max_ms
        
        if threshold_ms and metric.duration_ms > threshold_ms:
            alert = PerformanceAlert(
                timestamp=datetime.now(),
                alert_level=AlertLevel.WARNING,
                metric_type=MetricType.RESPONSE_TIME,
                message=f"{metric.query_type.value} response time exceeded threshold",
                current_value=metric.duration_ms,
                threshold_value=threshold_ms,
                suggested_action="Consider optimizing model or enabling GPU acceleration",
                context={"query_type": metric.query_type.value, "model": metric.model_used}
            )
            self._add_alert(alert)
    
    def _check_alert_conditions(self):
        """Check all metrics for alert conditions"""
        current_time = datetime.now()
        
        # Check recent metrics (last 5 minutes)
        recent_cutoff = current_time - timedelta(minutes=5)
        
        for metric_type, metric_deque in self.metrics.items():
            if not metric_deque:
                continue
            
            recent_metrics = [m for m in metric_deque if m.timestamp >= recent_cutoff]
            if not recent_metrics:
                continue
            
            avg_value = sum(m.value for m in recent_metrics) / len(recent_metrics)
            
            # Check thresholds
            alert = None
            
            if metric_type == MetricType.MEMORY_USAGE and avg_value > self.thresholds.max_memory_usage_percent:
                alert = PerformanceAlert(
                    timestamp=current_time,
                    alert_level=AlertLevel.WARNING,
                    metric_type=metric_type,
                    message=f"High memory usage: {avg_value:.1f}%",
                    current_value=avg_value,
                    threshold_value=self.thresholds.max_memory_usage_percent,
                    suggested_action="Consider restarting services or clearing caches"
                )
            
            elif metric_type == MetricType.CPU_USAGE and avg_value > self.thresholds.max_cpu_usage_percent:
                alert = PerformanceAlert(
                    timestamp=current_time,
                    alert_level=AlertLevel.WARNING,
                    metric_type=metric_type,
                    message=f"High CPU usage: {avg_value:.1f}%",
                    current_value=avg_value,
                    threshold_value=self.thresholds.max_cpu_usage_percent,
                    suggested_action="Consider reducing concurrent operations"
                )
            
            elif metric_type == MetricType.GPU_TEMPERATURE and avg_value > self.thresholds.max_gpu_temperature_celsius:
                alert = PerformanceAlert(
                    timestamp=current_time,
                    alert_level=AlertLevel.CRITICAL,
                    metric_type=metric_type,
                    message=f"High GPU temperature: {avg_value:.1f}°C",
                    current_value=avg_value,
                    threshold_value=self.thresholds.max_gpu_temperature_celsius,
                    suggested_action="Reduce GPU load immediately to prevent thermal throttling"
                )
            
            if alert:
                self._add_alert(alert)
    
    def _add_alert(self, alert: PerformanceAlert):
        """Add an alert and trigger callbacks"""
        with self._lock:
            self.alerts.append(alert)
        
        # Log alert
        log_method = getattr(self.logger, alert.alert_level.value, self.logger.info)
        log_method(f"ALERT: {alert.message} (current: {alert.current_value}, threshold: {alert.threshold_value})")
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.debug_logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add a callback function to be called when alerts are generated"""
        self.alert_callbacks.append(callback)
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for the specified time period
        
        Args:
            hours: Number of hours to include in summary
        
        Returns:
            Dictionary containing performance statistics and insights
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        summary = {
            "time_period_hours": hours,
            "generated_at": datetime.now().isoformat(),
            "system_info": {
                "platform": self.hardware.system_info.platform,
                "cpu_cores": self.hardware.system_info.cpu_count,
                "total_memory_gb": self.hardware.system_info.memory_total,
                "gpu_available": self.hardware.system_info.cuda_available,
                "gpu_name": self.hardware.gpu_info.name if self.hardware.gpu_info else None
            },
            "response_times": {},
            "voice_reliability": {},
            "resource_usage": {},
            "alerts": [],
            "optimization_suggestions": []
        }
        
        # Response time analysis
        recent_response_times = [m for m in self.response_time_metrics if 
                               datetime.fromtimestamp(m.start_time) >= cutoff_time]
        
        if recent_response_times:
            by_type = defaultdict(list)
            for metric in recent_response_times:
                by_type[metric.query_type].append(metric)
            
            for query_type, metrics in by_type.items():
                durations = [m.duration_ms for m in metrics]
                successes = [m for m in metrics if m.success]
                
                summary["response_times"][query_type.value] = {
                    "total_requests": len(metrics),
                    "successful_requests": len(successes),
                    "success_rate": len(successes) / len(metrics) * 100,
                    "avg_duration_ms": sum(durations) / len(durations),
                    "min_duration_ms": min(durations),
                    "max_duration_ms": max(durations),
                    "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)] if durations else 0
                }
        
        # Voice reliability analysis
        recent_voice_metrics = [m for m in self.voice_reliability_metrics if 
                              m.operation_type in ["recognition", "synthesis"]]
        
        if recent_voice_metrics:
            by_operation = defaultdict(list)
            for metric in recent_voice_metrics:
                by_operation[metric.operation_type].append(metric)
            
            for operation_type, metrics in by_operation.items():
                successes = [m for m in metrics if m.success]
                summary["voice_reliability"][operation_type] = {
                    "total_operations": len(metrics),
                    "successful_operations": len(successes),
                    "reliability_percent": len(successes) / len(metrics) * 100 if metrics else 0,
                    "avg_duration_ms": sum(m.duration_ms for m in metrics) / len(metrics) if metrics else 0
                }
        
        # Resource usage analysis
        for metric_type in [MetricType.MEMORY_USAGE, MetricType.CPU_USAGE, MetricType.GPU_UTILIZATION, MetricType.GPU_MEMORY]:
            recent_metrics = [m for m in self.metrics[metric_type] if m.timestamp >= cutoff_time]
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                summary["resource_usage"][metric_type.value] = {
                    "avg_percent": sum(values) / len(values),
                    "max_percent": max(values),
                    "min_percent": min(values),
                    "current_percent": values[-1] if values else 0
                }
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]
        summary["alerts"] = [
            {
                "timestamp": alert.timestamp.isoformat(),
                "level": alert.alert_level.value,
                "metric": alert.metric_type.value,
                "message": alert.message,
                "suggested_action": alert.suggested_action
            }
            for alert in recent_alerts
        ]
        
        # Optimization suggestions
        recent_metrics_list = []
        for metric_deque in self.metrics.values():
            recent_metrics_list.extend([m for m in metric_deque if m.timestamp >= cutoff_time])
        
        summary["optimization_suggestions"] = self.optimizer.analyze_performance_data(recent_metrics_list)
        
        return summary
    
    def export_metrics(self, filepath: Optional[str] = None, hours: int = 24) -> str:
        """
        Export performance metrics to JSON file
        
        Args:
            filepath: Optional custom file path
            hours: Number of hours of data to export
        
        Returns:
            Path to exported file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(self.metrics_dir / f"performance_export_{timestamp}.json")
        
        summary = self.get_performance_summary(hours)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Exported performance metrics to {filepath}")
        return filepath
    
    def cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = datetime.now() - timedelta(days=self.metrics_retention_days)
        
        with self._lock:
            # Clean up general metrics
            for metric_type, metric_deque in self.metrics.items():
                while metric_deque and metric_deque[0].timestamp < cutoff_time:
                    metric_deque.popleft()
            
            # Clean up response time metrics
            while (self.response_time_metrics and 
                   datetime.fromtimestamp(self.response_time_metrics[0].start_time) < cutoff_time):
                self.response_time_metrics.popleft()
            
            # Clean up alerts
            while self.alerts and self.alerts[0].timestamp < cutoff_time:
                self.alerts.popleft()
        
        self.logger.info(f"Cleaned up metrics older than {self.metrics_retention_days} days")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current real-time system status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": self._monitoring_active,
            "system": {
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(),
                "disk_usage_percent": psutil.disk_usage('/').percent,
            },
            "recent_alerts_count": len([a for a in self.alerts if 
                                      a.timestamp >= datetime.now() - timedelta(hours=1)]),
            "metrics_collected": sum(len(deque) for deque in self.metrics.values())
        }
        
        if self.enable_gpu_monitoring and torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
                total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                
                status["gpu"] = {
                    "memory_allocated_gb": memory_allocated,
                    "memory_total_gb": total_memory,
                    "memory_percent": (memory_allocated / total_memory) * 100,
                    "device_name": torch.cuda.get_device_name(device)
                }
            except Exception as e:
                status["gpu"] = {"error": str(e)}
        
        # Add memory leak detection status if enabled
        if self.enable_memory_leak_detection and self._memory_leak_detector:
            status["memory_leak_detection"] = self._memory_leak_detector.get_leak_detection_summary()
        
        # Add automated optimization status if enabled
        if self.enable_automated_optimization:
            opt_status = self.get_optimization_status()
            recovery_status = self.get_crash_recovery_status()
            
            if opt_status:
                status["automated_optimization"] = opt_status
            if recovery_status:
                status["crash_recovery"] = recovery_status
        
        return status
    
    def _initialize_memory_leak_detector(self):
        """Initialize the memory leak detector (lazy loading)"""
        if self._memory_leak_detector is None:
            try:
                # Import here to avoid circular imports
                from .memory_leak_detector import create_memory_leak_detector
                self._memory_leak_detector = create_memory_leak_detector(self)
                self.logger.info("Memory leak detector initialized")
            except ImportError as e:
                self.logger.error(f"Failed to import memory leak detector: {e}")
                self.enable_memory_leak_detection = False
            except Exception as e:
                self.logger.error(f"Failed to initialize memory leak detector: {e}")
                self.enable_memory_leak_detection = False
    
    def get_memory_leak_status(self) -> Optional[Dict[str, Any]]:
        """Get current memory leak detection status"""
        if not self.enable_memory_leak_detection or not self._memory_leak_detector:
            return None
        
        return self._memory_leak_detector.get_leak_detection_summary()
    
    def force_memory_cleanup(self) -> Dict[str, Any]:
        """Force garbage collection and return cleanup results"""
        if not self.enable_memory_leak_detection or not self._memory_leak_detector:
            # Basic garbage collection
            import gc
            collected = gc.collect()
            return {
                "objects_collected": collected,
                "advanced_cleanup": False,
                "message": "Basic garbage collection performed"
            }
        
        result = self._memory_leak_detector.force_garbage_collection()
        result["advanced_cleanup"] = True
        return result
    
    def run_memory_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive memory diagnostic"""
        if not self.enable_memory_leak_detection or not self._memory_leak_detector:
            return {
                "error": "Memory leak detection not enabled or available",
                "basic_memory_percent": psutil.virtual_memory().percent
            }
        
        return self._memory_leak_detector.run_memory_diagnostic()
    
    def get_memory_profiling_report(self) -> List[str]:
        """Get detailed memory profiling report"""
        if not self.enable_memory_leak_detection or not self._memory_leak_detector:
            return ["Memory leak detection not enabled"]
        
        if not self._memory_leak_detector.memory_profiler:
            return ["Memory profiling not enabled"]
        
        report = []
        report.extend(self._memory_leak_detector.memory_profiler.get_top_memory_consumers(10))
        report.append("")
        report.extend(self._memory_leak_detector.memory_profiler.compare_snapshots(10))
        
        return report
    
    def _initialize_automated_optimization_system(self):
        """Initialize the automated optimization and crash recovery system (lazy loading)"""
        if self._automated_optimizer is None or self._crash_recovery_system is None:
            try:
                # Import here to avoid circular imports
                from .automated_performance_testing import create_automated_performance_system
                self._automated_optimizer, self._crash_recovery_system = create_automated_performance_system(self)
                self.logger.info("Automated optimization and crash recovery system initialized")
            except ImportError as e:
                self.logger.error(f"Failed to import automated optimization system: {e}")
                self.enable_automated_optimization = False
            except Exception as e:
                self.logger.error(f"Failed to initialize automated optimization system: {e}")
                self.enable_automated_optimization = False
    
    def get_optimization_status(self) -> Optional[Dict[str, Any]]:
        """Get current automated optimization status"""
        if not self.enable_automated_optimization or not self._automated_optimizer:
            return None
        
        return self._automated_optimizer.get_optimization_summary()
    
    def get_crash_recovery_status(self) -> Optional[Dict[str, Any]]:
        """Get current crash recovery system status"""
        if not self.enable_automated_optimization or not self._crash_recovery_system:
            return None
        
        return self._crash_recovery_system.get_recovery_summary()
    
    def run_performance_optimization(self) -> Dict[str, Any]:
        """Manually trigger performance optimization analysis"""
        if not self.enable_automated_optimization or not self._automated_optimizer:
            return {
                "error": "Automated optimization not enabled or available",
                "basic_cleanup": False
            }
        
        results = self._automated_optimizer.analyze_and_optimize()
        return {
            "optimization_results": [result.__dict__ for result in results],
            "total_optimizations": len(results),
            "success": True
        }
    
    def register_critical_process(self, process_id: int, process_name: str, metadata: Optional[Dict] = None) -> bool:
        """Register a critical process for crash monitoring and recovery"""
        if not self.enable_automated_optimization or not self._crash_recovery_system:
            return False
        
        self._crash_recovery_system.register_process(process_id, process_name, metadata or {})
        return True
    
    def unregister_process(self, process_id: int) -> bool:
        """Unregister a process from crash monitoring"""
        if not self.enable_automated_optimization or not self._crash_recovery_system:
            return False
        
        self._crash_recovery_system.unregister_process(process_id)
        return True
    
    def get_performance_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance and optimization summary"""
        summary = self.get_performance_summary()
        
        # Add optimization data if available
        if self.enable_automated_optimization:
            opt_status = self.get_optimization_status()
            recovery_status = self.get_crash_recovery_status()
            
            if opt_status:
                summary["automated_optimization"] = opt_status
            if recovery_status:
                summary["crash_recovery"] = recovery_status
        
        return summary
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()


# Convenience decorators and context managers
class ResponseTimeTracker:
    """Context manager for tracking response times"""
    
    def __init__(self, monitor: PerformanceMonitor, query_type: QueryType, 
                 model_name: Optional[str] = None):
        self.monitor = monitor
        self.query_type = query_type
        self.model_name = model_name
        self.start_time = None
        self.token_count = None
        self.error_message = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        success = exc_type is None
        error_message = str(exc_val) if exc_val else None
        
        self.monitor.track_response_time(
            start_time=self.start_time,
            end_time=end_time,
            query_type=self.query_type,
            model_used=self.model_name,
            token_count=self.token_count,
            success=success,
            error_message=error_message
        )
    
    def set_token_count(self, count: int):
        """Set the number of tokens processed"""
        self.token_count = count


def track_response_time(monitor: PerformanceMonitor, query_type: QueryType, model_name: Optional[str] = None):
    """Decorator for tracking function response times"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ResponseTimeTracker(monitor, query_type, model_name) as tracker:
                result = func(*args, **kwargs)
                # Try to extract token count from result if it's a dict
                if isinstance(result, dict) and 'token_count' in result:
                    tracker.set_token_count(result['token_count'])
                return result
        return wrapper
    return decorator 