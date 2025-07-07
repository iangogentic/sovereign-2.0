"""
Advanced Memory and Resource Leak Detection System for Sovereign AI Agent

This module provides comprehensive memory leak detection including:
- Process-level memory tracking and trend analysis
- Memory growth pattern detection using statistical algorithms
- Resource leak detection (file handles, network connections, threads)
- GPU memory leak monitoring
- Memory profiling and diagnostics
- Integration with performance monitoring and alerting system
"""

import gc
import os
import sys
import time
import psutil
import threading
import tracemalloc
import linecache
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Set, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import logging
from pathlib import Path
import weakref

from .logger import get_performance_logger
from .performance_monitor import PerformanceMonitor, MemoryLeakAlert, AlertLevel, PerformanceAlert, MetricType


class LeakType(Enum):
    """Types of resource leaks that can be detected"""
    MEMORY_GROWTH = "memory_growth"
    GPU_MEMORY_LEAK = "gpu_memory_leak"
    FILE_HANDLE_LEAK = "file_handle_leak"
    NETWORK_CONNECTION_LEAK = "network_connection_leak"
    THREAD_LEAK = "thread_leak"
    PYTHON_OBJECT_LEAK = "python_object_leak"
    CACHE_MEMORY_LEAK = "cache_memory_leak"


class MemoryTrend(Enum):
    """Memory usage trend classifications"""
    STABLE = "stable"
    GRADUAL_INCREASE = "gradual_increase"
    RAPID_INCREASE = "rapid_increase"
    OSCILLATING = "oscillating"
    DECREASING = "decreasing"


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point in time"""
    timestamp: datetime
    process_memory_mb: float
    system_memory_percent: float
    gpu_memory_mb: Optional[float] = None
    file_descriptor_count: int = 0
    thread_count: int = 0
    network_connection_count: int = 0
    python_objects_count: int = 0
    gc_objects_count: int = 0


@dataclass
class LeakDetectionResult:
    """Result of leak detection analysis"""
    leak_type: LeakType
    detected: bool
    severity: AlertLevel
    leak_rate: float  # units per minute
    confidence_score: float  # 0.0 to 1.0
    time_window_minutes: float
    current_value: float
    baseline_value: float
    growth_trend: MemoryTrend
    recommended_actions: List[str]
    diagnostic_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceLeakInfo:
    """Information about a specific resource leak"""
    resource_type: str
    resource_id: str
    allocation_time: datetime
    allocation_traceback: Optional[str] = None
    size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryProfiler:
    """Advanced memory profiling and tracking"""
    
    def __init__(self, enable_tracemalloc: bool = True):
        self.enable_tracemalloc = enable_tracemalloc
        self.tracemalloc_enabled = False
        self.snapshots = []
        self.logger = get_performance_logger()
        
        if enable_tracemalloc:
            self._start_tracemalloc()
    
    def _start_tracemalloc(self):
        """Start tracemalloc for detailed memory tracking"""
        try:
            if not tracemalloc.is_tracing():
                tracemalloc.start(25)  # Keep 25 frames in traces
                self.tracemalloc_enabled = True
                self.logger.info("Memory profiling enabled with tracemalloc")
        except Exception as e:
            self.logger.warning(f"Failed to start tracemalloc: {e}")
    
    def take_snapshot(self) -> Optional[Any]:
        """Take a memory snapshot for analysis"""
        if not self.tracemalloc_enabled:
            return None
        
        try:
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append((datetime.now(), snapshot))
            
            # Keep only last 10 snapshots to prevent memory buildup
            if len(self.snapshots) > 10:
                self.snapshots.pop(0)
            
            return snapshot
        except Exception as e:
            self.logger.error(f"Failed to take memory snapshot: {e}")
            return None
    
    def compare_snapshots(self, limit: int = 10) -> List[str]:
        """Compare latest snapshot with previous to find memory growth"""
        if len(self.snapshots) < 2:
            return ["Insufficient snapshots for comparison"]
        
        try:
            current_time, current_snapshot = self.snapshots[-1]
            previous_time, previous_snapshot = self.snapshots[-2]
            
            top_stats = current_snapshot.compare_to(previous_snapshot, 'lineno')
            
            results = []
            results.append(f"Memory comparison ({previous_time.strftime('%H:%M:%S')} → {current_time.strftime('%H:%M:%S')}):")
            
            for stat in top_stats[:limit]:
                if stat.size_diff > 0:  # Only show growing allocations
                    filename = stat.traceback.format()[0].split('/')[-1]
                    results.append(f"  +{stat.size_diff/1024:.1f} KB in {filename}")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Failed to compare snapshots: {e}")
            return [f"Snapshot comparison failed: {e}"]
    
    def get_top_memory_consumers(self, limit: int = 10) -> List[str]:
        """Get top memory consuming code locations"""
        if not self.snapshots:
            return ["No memory snapshots available"]
        
        try:
            _, snapshot = self.snapshots[-1]
            top_stats = snapshot.statistics('lineno')
            
            results = []
            results.append("Top memory consumers:")
            
            for stat in top_stats[:limit]:
                filename = stat.traceback.format()[0].split('/')[-1]
                results.append(f"  {stat.size/1024:.1f} KB in {filename}")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Failed to get memory consumers: {e}")
            return [f"Memory analysis failed: {e}"]


class ResourceTracker:
    """Track system resources for leak detection"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.logger = get_performance_logger()
        self.baseline_resources = None
        self._lock = threading.Lock()
    
    def get_current_resources(self) -> Dict[str, Any]:
        """Get current resource usage"""
        try:
            # Memory info
            memory_info = self.process.memory_info()
            
            # File descriptors (Unix-like systems)
            try:
                file_descriptors = self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
            except:
                file_descriptors = 0
            
            # Network connections
            try:
                connections = len(self.process.connections())
            except:
                connections = 0
            
            # Thread count
            try:
                threads = self.process.num_threads()
            except:
                threads = threading.active_count()
            
            # GPU memory if available
            gpu_memory_mb = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated() / (1024**2)
            except:
                pass
            
            # Python object count
            python_objects = len(gc.get_objects())
            
            return {
                'memory_mb': memory_info.rss / (1024**2),
                'memory_percent': self.process.memory_percent(),
                'file_descriptors': file_descriptors,
                'network_connections': connections,
                'threads': threads,
                'gpu_memory_mb': gpu_memory_mb,
                'python_objects': python_objects,
                'gc_objects': len(gc.get_stats())
            }
        
        except Exception as e:
            self.logger.error(f"Failed to get resource usage: {e}")
            return {}
    
    def set_baseline(self):
        """Set current resource usage as baseline for leak detection"""
        with self._lock:
            self.baseline_resources = self.get_current_resources()
            self.logger.info("Resource baseline established")
    
    def get_resource_diff(self) -> Dict[str, float]:
        """Get difference from baseline resources"""
        if self.baseline_resources is None:
            self.set_baseline()
            return {}
        
        current = self.get_current_resources()
        diff = {}
        
        for key in self.baseline_resources:
            if key in current and current[key] is not None and self.baseline_resources[key] is not None:
                diff[key] = current[key] - self.baseline_resources[key]
        
        return diff


class MemoryLeakDetector:
    """Advanced memory leak detection system"""
    
    def __init__(self, 
                 performance_monitor: Optional[PerformanceMonitor] = None,
                 detection_window_minutes: float = 30.0,
                 sampling_interval_seconds: float = 60.0,
                 enable_profiling: bool = True):
        """
        Initialize memory leak detector
        
        Args:
            performance_monitor: Existing performance monitor to integrate with
            detection_window_minutes: Time window for leak detection analysis
            sampling_interval_seconds: How often to sample memory usage
            enable_profiling: Enable detailed memory profiling
        """
        self.performance_monitor = performance_monitor
        self.detection_window_minutes = detection_window_minutes
        self.sampling_interval_seconds = sampling_interval_seconds
        self.enable_profiling = enable_profiling
        
        # Data storage
        self.memory_history = deque(maxlen=int(detection_window_minutes * 60 / sampling_interval_seconds))
        self.leak_alerts = []
        self.detection_results = {}
        
        # Components
        self.resource_tracker = ResourceTracker()
        self.memory_profiler = MemoryProfiler(enable_profiling) if enable_profiling else None
        
        # State
        self.monitoring_active = False
        self.monitoring_thread = None
        self._lock = threading.Lock()
        
        # Configuration
        self.leak_thresholds = {
            'memory_growth_mb_per_min': 5.0,
            'gpu_memory_growth_mb_per_min': 10.0,
            'file_descriptor_growth_per_min': 2.0,
            'thread_growth_per_min': 1.0,
            'connection_growth_per_min': 1.0,
            'python_object_growth_per_min': 1000.0
        }
        
        self.logger = get_performance_logger()
        self.logger.info("Memory leak detector initialized")
        
        # Set initial baseline
        self.resource_tracker.set_baseline()
    
    def start_monitoring(self):
        """Start continuous memory leak monitoring"""
        if self.monitoring_active:
            self.logger.warning("Memory leak monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Memory leak monitoring started")
    
    def stop_monitoring(self):
        """Stop memory leak monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Memory leak monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Take memory snapshot
                snapshot = self._create_memory_snapshot()
                
                with self._lock:
                    self.memory_history.append(snapshot)
                
                # Analyze for leaks if we have enough data
                if len(self.memory_history) >= 3:
                    self._analyze_for_leaks()
                
                # Take detailed profiling snapshot periodically
                if self.memory_profiler and len(self.memory_history) % 5 == 0:
                    self.memory_profiler.take_snapshot()
                
                time.sleep(self.sampling_interval_seconds)
            
            except Exception as e:
                self.logger.error(f"Error in memory leak monitoring loop: {e}")
                time.sleep(self.sampling_interval_seconds)
    
    def _create_memory_snapshot(self) -> MemorySnapshot:
        """Create a comprehensive memory snapshot"""
        resources = self.resource_tracker.get_current_resources()
        
        return MemorySnapshot(
            timestamp=datetime.now(),
            process_memory_mb=resources.get('memory_mb', 0),
            system_memory_percent=resources.get('memory_percent', 0),
            gpu_memory_mb=resources.get('gpu_memory_mb'),
            file_descriptor_count=resources.get('file_descriptors', 0),
            thread_count=resources.get('threads', 0),
            network_connection_count=resources.get('network_connections', 0),
            python_objects_count=resources.get('python_objects', 0),
            gc_objects_count=resources.get('gc_objects', 0)
        )
    
    def _analyze_for_leaks(self):
        """Analyze memory history for potential leaks"""
        with self._lock:
            history = list(self.memory_history)
        
        if len(history) < 3:
            return
        
        # Analyze different types of potential leaks
        leak_types = [
            ('memory_growth', self._analyze_memory_growth),
            ('gpu_memory_leak', self._analyze_gpu_memory_leak),
            ('file_handle_leak', self._analyze_file_handle_leak),
            ('thread_leak', self._analyze_thread_leak),
            ('network_connection_leak', self._analyze_connection_leak),
            ('python_object_leak', self._analyze_python_object_leak)
        ]
        
        for leak_name, analyzer in leak_types:
            try:
                result = analyzer(history)
                self.detection_results[leak_name] = result
                
                if result.detected and result.severity in [AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                    self._generate_leak_alert(result)
            
            except Exception as e:
                self.logger.error(f"Error analyzing {leak_name}: {e}")
    
    def _analyze_memory_growth(self, history: List[MemorySnapshot]) -> LeakDetectionResult:
        """Analyze memory growth patterns"""
        memory_values = [s.process_memory_mb for s in history]
        timestamps = [s.timestamp for s in history]
        
        if len(memory_values) < 3:
            return LeakDetectionResult(
                leak_type=LeakType.MEMORY_GROWTH,
                detected=False,
                severity=AlertLevel.INFO,
                leak_rate=0.0,
                confidence_score=0.0,
                time_window_minutes=0.0,
                current_value=memory_values[-1] if memory_values else 0,
                baseline_value=memory_values[0] if memory_values else 0,
                growth_trend=MemoryTrend.STABLE,
                recommended_actions=[]
            )
        
        # Calculate growth rate
        time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 60.0  # minutes
        memory_change = memory_values[-1] - memory_values[0]
        leak_rate = memory_change / time_span if time_span > 0 else 0
        
        # Determine trend
        trend = self._determine_memory_trend(memory_values)
        
        # Calculate confidence based on consistency of growth
        confidence = self._calculate_growth_confidence(memory_values)
        
        # Determine if leak is detected
        threshold = self.leak_thresholds['memory_growth_mb_per_min']
        detected = leak_rate > threshold and confidence > 0.7
        
        # Determine severity
        severity = AlertLevel.INFO
        if detected:
            if leak_rate > threshold * 3:
                severity = AlertLevel.CRITICAL
            elif leak_rate > threshold * 2:
                severity = AlertLevel.WARNING
            else:
                severity = AlertLevel.INFO
        
        # Generate recommendations
        recommendations = []
        if detected:
            recommendations.extend([
                "Monitor memory usage patterns over longer periods",
                "Check for unclosed file handles or network connections",
                "Review recent code changes for potential memory leaks",
                "Consider running garbage collection manually: gc.collect()",
                "Enable detailed memory profiling for root cause analysis"
            ])
            
            if leak_rate > threshold * 2:
                recommendations.extend([
                    "Consider restarting the application if memory usage is critical",
                    "Review large data structures and caching mechanisms",
                    "Check for circular references in Python objects"
                ])
        
        return LeakDetectionResult(
            leak_type=LeakType.MEMORY_GROWTH,
            detected=detected,
            severity=severity,
            leak_rate=leak_rate,
            confidence_score=confidence,
            time_window_minutes=time_span,
            current_value=memory_values[-1],
            baseline_value=memory_values[0],
            growth_trend=trend,
            recommended_actions=recommendations,
            diagnostic_data={
                'memory_history': memory_values[-10:],  # Last 10 readings
                'growth_rate_mb_per_min': leak_rate,
                'total_growth_mb': memory_change,
                'analysis_window_minutes': time_span
            }
        )
    
    def _analyze_gpu_memory_leak(self, history: List[MemorySnapshot]) -> LeakDetectionResult:
        """Analyze GPU memory leak patterns"""
        gpu_values = [s.gpu_memory_mb for s in history if s.gpu_memory_mb is not None]
        
        if len(gpu_values) < 3:
            return LeakDetectionResult(
                leak_type=LeakType.GPU_MEMORY_LEAK,
                detected=False,
                severity=AlertLevel.INFO,
                leak_rate=0.0,
                confidence_score=0.0,
                time_window_minutes=0.0,
                current_value=gpu_values[-1] if gpu_values else 0,
                baseline_value=gpu_values[0] if gpu_values else 0,
                growth_trend=MemoryTrend.STABLE,
                recommended_actions=[]
            )
        
        # Calculate GPU memory growth rate
        time_span = self.detection_window_minutes
        gpu_change = gpu_values[-1] - gpu_values[0]
        leak_rate = gpu_change / time_span if time_span > 0 else 0
        
        # Calculate confidence
        confidence = self._calculate_growth_confidence(gpu_values)
        
        # Check threshold
        threshold = self.leak_thresholds['gpu_memory_growth_mb_per_min']
        detected = leak_rate > threshold and confidence > 0.6
        
        # Determine severity
        severity = AlertLevel.INFO
        if detected:
            if leak_rate > threshold * 2:
                severity = AlertLevel.CRITICAL
            else:
                severity = AlertLevel.WARNING
        
        # Recommendations
        recommendations = []
        if detected:
            recommendations.extend([
                "Clear GPU cache: torch.cuda.empty_cache()",
                "Check for tensors not being freed after inference",
                "Review model loading and unloading patterns",
                "Monitor GPU memory during model operations",
                "Consider reducing batch sizes to prevent memory accumulation"
            ])
        
        return LeakDetectionResult(
            leak_type=LeakType.GPU_MEMORY_LEAK,
            detected=detected,
            severity=severity,
            leak_rate=leak_rate,
            confidence_score=confidence,
            time_window_minutes=time_span,
            current_value=gpu_values[-1],
            baseline_value=gpu_values[0],
            growth_trend=self._determine_memory_trend(gpu_values),
            recommended_actions=recommendations,
            diagnostic_data={
                'gpu_memory_history': gpu_values[-10:],
                'gpu_growth_rate_mb_per_min': leak_rate
            }
        )
    
    def _analyze_file_handle_leak(self, history: List[MemorySnapshot]) -> LeakDetectionResult:
        """Analyze file handle leak patterns"""
        fd_values = [s.file_descriptor_count for s in history]
        
        if len(fd_values) < 3:
            return LeakDetectionResult(
                leak_type=LeakType.FILE_HANDLE_LEAK,
                detected=False,
                severity=AlertLevel.INFO,
                leak_rate=0.0,
                confidence_score=0.0,
                time_window_minutes=0.0,
                current_value=fd_values[-1] if fd_values else 0,
                baseline_value=fd_values[0] if fd_values else 0,
                growth_trend=MemoryTrend.STABLE,
                recommended_actions=[]
            )
        
        # Calculate file descriptor growth rate
        time_span = self.detection_window_minutes
        fd_change = fd_values[-1] - fd_values[0]
        leak_rate = fd_change / time_span if time_span > 0 else 0
        
        # Calculate confidence
        confidence = self._calculate_growth_confidence(fd_values)
        
        # Check threshold
        threshold = self.leak_thresholds['file_descriptor_growth_per_min']
        detected = leak_rate > threshold and confidence > 0.8
        
        # Determine severity
        severity = AlertLevel.INFO
        if detected:
            if leak_rate > threshold * 3:
                severity = AlertLevel.CRITICAL
            elif leak_rate > threshold * 2:
                severity = AlertLevel.WARNING
        
        # Recommendations
        recommendations = []
        if detected:
            recommendations.extend([
                "Check for unclosed file handles in recent code",
                "Review file I/O operations for proper cleanup",
                "Use context managers (with statements) for file operations",
                "Check log files and temporary file creation",
                "Monitor file descriptor limits: ulimit -n"
            ])
        
        return LeakDetectionResult(
            leak_type=LeakType.FILE_HANDLE_LEAK,
            detected=detected,
            severity=severity,
            leak_rate=leak_rate,
            confidence_score=confidence,
            time_window_minutes=time_span,
            current_value=fd_values[-1],
            baseline_value=fd_values[0],
            growth_trend=self._determine_memory_trend(fd_values),
            recommended_actions=recommendations,
            diagnostic_data={
                'file_descriptor_history': fd_values[-10:],
                'fd_growth_rate_per_min': leak_rate
            }
        )
    
    def _analyze_thread_leak(self, history: List[MemorySnapshot]) -> LeakDetectionResult:
        """Analyze thread leak patterns"""
        thread_values = [s.thread_count for s in history]
        
        if len(thread_values) < 3:
            return LeakDetectionResult(
                leak_type=LeakType.THREAD_LEAK,
                detected=False,
                severity=AlertLevel.INFO,
                leak_rate=0.0,
                confidence_score=0.0,
                time_window_minutes=0.0,
                current_value=thread_values[-1] if thread_values else 0,
                baseline_value=thread_values[0] if thread_values else 0,
                growth_trend=MemoryTrend.STABLE,
                recommended_actions=[]
            )
        
        # Calculate thread growth rate
        time_span = self.detection_window_minutes
        thread_change = thread_values[-1] - thread_values[0]
        leak_rate = thread_change / time_span if time_span > 0 else 0
        
        # Calculate confidence
        confidence = self._calculate_growth_confidence(thread_values)
        
        # Check threshold
        threshold = self.leak_thresholds['thread_growth_per_min']
        detected = leak_rate > threshold and confidence > 0.8
        
        # Determine severity
        severity = AlertLevel.INFO
        if detected:
            if leak_rate > threshold * 5:
                severity = AlertLevel.CRITICAL
            elif leak_rate > threshold * 2:
                severity = AlertLevel.WARNING
        
        # Recommendations
        recommendations = []
        if detected:
            recommendations.extend([
                "Check for threads that are not being properly joined",
                "Review threading patterns and cleanup",
                "Use thread pools instead of creating individual threads",
                "Check for daemon threads that should be terminated",
                "Monitor threading.active_count() in your application"
            ])
        
        return LeakDetectionResult(
            leak_type=LeakType.THREAD_LEAK,
            detected=detected,
            severity=severity,
            leak_rate=leak_rate,
            confidence_score=confidence,
            time_window_minutes=time_span,
            current_value=thread_values[-1],
            baseline_value=thread_values[0],
            growth_trend=self._determine_memory_trend(thread_values),
            recommended_actions=recommendations,
            diagnostic_data={
                'thread_history': thread_values[-10:],
                'thread_growth_rate_per_min': leak_rate
            }
        )
    
    def _analyze_connection_leak(self, history: List[MemorySnapshot]) -> LeakDetectionResult:
        """Analyze network connection leak patterns"""
        conn_values = [s.network_connection_count for s in history]
        
        if len(conn_values) < 3:
            return LeakDetectionResult(
                leak_type=LeakType.NETWORK_CONNECTION_LEAK,
                detected=False,
                severity=AlertLevel.INFO,
                leak_rate=0.0,
                confidence_score=0.0,
                time_window_minutes=0.0,
                current_value=conn_values[-1] if conn_values else 0,
                baseline_value=conn_values[0] if conn_values else 0,
                growth_trend=MemoryTrend.STABLE,
                recommended_actions=[]
            )
        
        # Calculate connection growth rate
        time_span = self.detection_window_minutes
        conn_change = conn_values[-1] - conn_values[0]
        leak_rate = conn_change / time_span if time_span > 0 else 0
        
        # Calculate confidence
        confidence = self._calculate_growth_confidence(conn_values)
        
        # Check threshold
        threshold = self.leak_thresholds['connection_growth_per_min']
        detected = leak_rate > threshold and confidence > 0.8
        
        # Determine severity
        severity = AlertLevel.INFO
        if detected:
            if leak_rate > threshold * 5:
                severity = AlertLevel.CRITICAL
            elif leak_rate > threshold * 2:
                severity = AlertLevel.WARNING
        
        # Recommendations
        recommendations = []
        if detected:
            recommendations.extend([
                "Check for unclosed network connections",
                "Review HTTP client usage and connection pooling",
                "Use connection managers and proper cleanup",
                "Check for WebSocket or persistent connections",
                "Monitor network connection timeouts"
            ])
        
        return LeakDetectionResult(
            leak_type=LeakType.NETWORK_CONNECTION_LEAK,
            detected=detected,
            severity=severity,
            leak_rate=leak_rate,
            confidence_score=confidence,
            time_window_minutes=time_span,
            current_value=conn_values[-1],
            baseline_value=conn_values[0],
            growth_trend=self._determine_memory_trend(conn_values),
            recommended_actions=recommendations,
            diagnostic_data={
                'connection_history': conn_values[-10:],
                'connection_growth_rate_per_min': leak_rate
            }
        )
    
    def _analyze_python_object_leak(self, history: List[MemorySnapshot]) -> LeakDetectionResult:
        """Analyze Python object leak patterns"""
        obj_values = [s.python_objects_count for s in history]
        
        if len(obj_values) < 3:
            return LeakDetectionResult(
                leak_type=LeakType.PYTHON_OBJECT_LEAK,
                detected=False,
                severity=AlertLevel.INFO,
                leak_rate=0.0,
                confidence_score=0.0,
                time_window_minutes=0.0,
                current_value=obj_values[-1] if obj_values else 0,
                baseline_value=obj_values[0] if obj_values else 0,
                growth_trend=MemoryTrend.STABLE,
                recommended_actions=[]
            )
        
        # Calculate object growth rate
        time_span = self.detection_window_minutes
        obj_change = obj_values[-1] - obj_values[0]
        leak_rate = obj_change / time_span if time_span > 0 else 0
        
        # Calculate confidence
        confidence = self._calculate_growth_confidence(obj_values)
        
        # Check threshold
        threshold = self.leak_thresholds['python_object_growth_per_min']
        detected = leak_rate > threshold and confidence > 0.6
        
        # Determine severity
        severity = AlertLevel.INFO
        if detected:
            if leak_rate > threshold * 3:
                severity = AlertLevel.WARNING
        
        # Recommendations
        recommendations = []
        if detected:
            recommendations.extend([
                "Run garbage collection: gc.collect()",
                "Check for circular references between objects",
                "Review large data structures and caching",
                "Use weak references where appropriate",
                "Enable detailed memory profiling for object tracking"
            ])
        
        return LeakDetectionResult(
            leak_type=LeakType.PYTHON_OBJECT_LEAK,
            detected=detected,
            severity=severity,
            leak_rate=leak_rate,
            confidence_score=confidence,
            time_window_minutes=time_span,
            current_value=obj_values[-1],
            baseline_value=obj_values[0],
            growth_trend=self._determine_memory_trend(obj_values),
            recommended_actions=recommendations,
            diagnostic_data={
                'object_history': obj_values[-10:],
                'object_growth_rate_per_min': leak_rate
            }
        )
    
    def _determine_memory_trend(self, values: List[float]) -> MemoryTrend:
        """Determine the overall trend in memory usage"""
        if len(values) < 3:
            return MemoryTrend.STABLE
        
        # Calculate differences between consecutive readings
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        
        # Analyze trend
        positive_diffs = sum(1 for d in diffs if d > 0)
        negative_diffs = sum(1 for d in diffs if d < 0)
        total_change = values[-1] - values[0]
        
        # Determine trend type
        if abs(total_change) < (values[0] * 0.05):  # Less than 5% change
            return MemoryTrend.STABLE
        elif total_change > 0:
            if total_change > (values[0] * 0.2):  # More than 20% increase
                return MemoryTrend.RAPID_INCREASE
            else:
                return MemoryTrend.GRADUAL_INCREASE
        else:
            return MemoryTrend.DECREASING
    
    def _calculate_growth_confidence(self, values: List[float]) -> float:
        """Calculate confidence score for growth detection (0.0 to 1.0)"""
        if len(values) < 3:
            return 0.0
        
        # Calculate how consistently values are increasing
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        
        if not diffs:
            return 0.0
        
        # Count positive differences (growth)
        positive_count = sum(1 for d in diffs if d > 0)
        consistency = positive_count / len(diffs)
        
        # Factor in magnitude of growth
        total_change = values[-1] - values[0]
        if total_change <= 0:
            return 0.0
        
        # Combine consistency with magnitude for confidence
        magnitude_factor = min(1.0, abs(total_change) / values[0])
        confidence = consistency * magnitude_factor
        
        return min(1.0, confidence)
    
    def _generate_leak_alert(self, result: LeakDetectionResult):
        """Generate and send leak alert"""
        # Create memory leak alert
        alert = MemoryLeakAlert(
            timestamp=datetime.now(),
            memory_increase_mb=result.current_value - result.baseline_value,
            timespan_minutes=result.time_window_minutes,
            leak_rate_mb_per_minute=result.leak_rate,
            process_name=result.leak_type.value,
            recommended_action="; ".join(result.recommended_actions[:3])  # Top 3 recommendations
        )
        
        self.leak_alerts.append(alert)
        
        # Create performance alert for integration
        perf_alert = PerformanceAlert(
            timestamp=datetime.now(),
            alert_level=result.severity,
            metric_type=MetricType.MEMORY_USAGE,
            message=f"{result.leak_type.value} detected: {result.leak_rate:.2f} units/min growth",
            current_value=result.current_value,
            threshold_value=result.baseline_value,
            suggested_action=result.recommended_actions[0] if result.recommended_actions else None,
            context={
                'leak_type': result.leak_type.value,
                'confidence': result.confidence_score,
                'trend': result.growth_trend.value,
                'diagnostic_data': result.diagnostic_data
            }
        )
        
        # Send to performance monitor if available
        if self.performance_monitor:
            self.performance_monitor._add_alert(perf_alert)
        
        # Log the alert
        self.logger.warning(f"Memory leak detected: {result.leak_type.value} - "
                          f"Growth rate: {result.leak_rate:.2f} units/min, "
                          f"Confidence: {result.confidence_score:.2f}")
    
    def get_leak_detection_summary(self) -> Dict[str, Any]:
        """Get summary of current leak detection status"""
        with self._lock:
            history_count = len(self.memory_history)
            latest_snapshot = self.memory_history[-1] if self.memory_history else None
        
        summary = {
            'monitoring_active': self.monitoring_active,
            'history_points': history_count,
            'detection_window_minutes': self.detection_window_minutes,
            'latest_snapshot': {
                'timestamp': latest_snapshot.timestamp.isoformat() if latest_snapshot else None,
                'memory_mb': latest_snapshot.process_memory_mb if latest_snapshot else 0,
                'gpu_memory_mb': latest_snapshot.gpu_memory_mb if latest_snapshot else None,
                'file_descriptors': latest_snapshot.file_descriptor_count if latest_snapshot else 0,
                'threads': latest_snapshot.thread_count if latest_snapshot else 0,
                'connections': latest_snapshot.network_connection_count if latest_snapshot else 0
            },
            'detection_results': {
                leak_type: {
                    'detected': result.detected,
                    'severity': result.severity.value,
                    'leak_rate': result.leak_rate,
                    'confidence': result.confidence_score,
                    'trend': result.growth_trend.value
                }
                for leak_type, result in self.detection_results.items()
            },
            'total_alerts': len(self.leak_alerts),
            'recent_alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'leak_rate': alert.leak_rate_mb_per_minute,
                    'process': alert.process_name,
                    'action': alert.recommended_action
                }
                for alert in self.leak_alerts[-5:]  # Last 5 alerts
            ]
        }
        
        # Add profiling data if available
        if self.memory_profiler:
            summary['profiling'] = {
                'enabled': True,
                'snapshots_taken': len(self.memory_profiler.snapshots),
                'top_consumers': self.memory_profiler.get_top_memory_consumers(5)
            }
        
        return summary
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection and return before/after stats"""
        before_resources = self.resource_tracker.get_current_resources()
        
        # Force garbage collection
        collected = gc.collect()
        
        after_resources = self.resource_tracker.get_current_resources()
        
        result = {
            'objects_collected': collected,
            'memory_before_mb': before_resources.get('memory_mb', 0),
            'memory_after_mb': after_resources.get('memory_mb', 0),
            'memory_freed_mb': before_resources.get('memory_mb', 0) - after_resources.get('memory_mb', 0),
            'python_objects_before': before_resources.get('python_objects', 0),
            'python_objects_after': after_resources.get('python_objects', 0),
            'objects_freed': before_resources.get('python_objects', 0) - after_resources.get('python_objects', 0)
        }
        
        self.logger.info(f"Forced garbage collection: {collected} objects collected, "
                        f"{result['memory_freed_mb']:.2f} MB freed")
        
        return result
    
    def run_memory_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive memory diagnostic"""
        diagnostic = {
            'timestamp': datetime.now().isoformat(),
            'current_resources': self.resource_tracker.get_current_resources(),
            'resource_diff_from_baseline': self.resource_tracker.get_resource_diff(),
            'gc_stats': gc.get_stats(),
            'gc_counts': gc.get_count()
        }
        
        # Add profiling information if available
        if self.memory_profiler:
            diagnostic['memory_profiling'] = {
                'top_consumers': self.memory_profiler.get_top_memory_consumers(10),
                'recent_comparison': self.memory_profiler.compare_snapshots(10)
            }
        
        # Add leak detection summary
        diagnostic['leak_detection'] = self.get_leak_detection_summary()
        
        return diagnostic
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_monitoring()
        if self.memory_profiler and self.memory_profiler.tracemalloc_enabled:
            tracemalloc.stop()
        self.logger.info("Memory leak detector cleaned up")


def create_memory_leak_detector(performance_monitor: Optional[PerformanceMonitor] = None) -> MemoryLeakDetector:
    """Factory function to create a memory leak detector"""
    return MemoryLeakDetector(
        performance_monitor=performance_monitor,
        detection_window_minutes=30.0,
        sampling_interval_seconds=60.0,
        enable_profiling=True
    ) 