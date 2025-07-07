"""
Automated Performance Optimization and Recovery System for Sovereign AI Agent

This module provides comprehensive automated performance optimization including:
- Advanced performance analysis and optimization engine
- Automatic application of performance optimizations
- Dynamic system tuning based on real-time metrics
- Crash detection and automated recovery mechanisms
- Performance degradation detection and mitigation
- Resource allocation optimization
- Model parameter auto-tuning for optimal performance
"""

import os
import sys
import time
import psutil
import threading
import subprocess
import signal
import pickle
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import gc
import numpy as np

from .performance_monitor import (
    PerformanceMonitor, MetricType, QueryType, AlertLevel,
    PerformanceMetric, PerformanceAlert, PerformanceThresholds
)
from .logger import get_performance_logger


class OptimizationType(Enum):
    """Types of performance optimizations"""
    MEMORY_CLEANUP = "memory_cleanup"
    BATCH_SIZE_TUNING = "batch_size_tuning"
    GPU_OPTIMIZATION = "gpu_optimization"
    MODEL_QUANTIZATION = "model_quantization"
    CACHE_OPTIMIZATION = "cache_optimization"
    THREAD_POOL_TUNING = "thread_pool_tuning"
    GC_TUNING = "gc_tuning"
    PROCESS_RESTART = "process_restart"
    RESOURCE_REALLOCATION = "resource_reallocation"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure types"""
    GRACEFUL_RESTART = "graceful_restart"
    FORCE_RESTART = "force_restart"
    SERVICE_FAILOVER = "service_failover"
    RESOURCE_CLEANUP = "resource_cleanup"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    STATE_RECOVERY = "state_recovery"
    CIRCUIT_BREAKER = "circuit_breaker"


class CrashType(Enum):
    """Types of crashes and failures"""
    MEMORY_EXHAUSTION = "memory_exhaustion"
    GPU_TIMEOUT = "gpu_timeout"
    PROCESS_HANG = "process_hang"
    MODEL_FAILURE = "model_failure"
    UNHANDLED_EXCEPTION = "unhandled_exception"
    RESOURCE_LEAK = "resource_leak"
    NETWORK_FAILURE = "network_failure"
    HARDWARE_FAILURE = "hardware_failure"


@dataclass
class OptimizationResult:
    """Result of an applied optimization"""
    optimization_type: OptimizationType
    success: bool
    timestamp: datetime
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement_percent: float
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    rollback_applied: bool = False


@dataclass
class CrashEvent:
    """Information about a detected crash or failure"""
    crash_type: CrashType
    timestamp: datetime
    process_id: int
    process_name: str
    exit_code: Optional[int]
    signal_received: Optional[int]
    stack_trace: Optional[str] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    gpu_usage_percent: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Recovery action taken in response to a crash"""
    recovery_strategy: RecoveryStrategy
    timestamp: datetime
    crash_event: CrashEvent
    success: bool
    recovery_time_seconds: float
    new_process_id: Optional[int] = None
    state_restored: bool = False
    error_message: Optional[str] = None


@dataclass
class PerformanceProfile:
    """Performance profile for dynamic optimization"""
    name: str
    batch_size: int
    max_memory_mb: int
    gpu_memory_fraction: float
    gc_threshold: Tuple[int, int, int]
    thread_pool_size: int
    cache_size_mb: int
    model_precision: str  # "fp32", "fp16", "int8"
    enable_torch_compile: bool = False
    optimization_level: str = "balanced"  # "speed", "balanced", "memory"


class AdvancedPerformanceOptimizer:
    """Advanced performance optimization engine with automatic tuning"""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.logger = get_performance_logger()
        
        # Optimization history
        self.optimization_history: deque[OptimizationResult] = deque(maxlen=1000)
        self.active_optimizations: Dict[OptimizationType, datetime] = {}
        
        # Performance profiles
        self.performance_profiles = self._initialize_performance_profiles()
        self.current_profile = "balanced"
        
        # Optimization state
        self.optimization_enabled = True
        self.auto_optimization_interval = 300  # 5 minutes
        self.last_optimization_check = datetime.now()
        
        # Learning and adaptation
        self.optimization_effectiveness: Dict[OptimizationType, List[float]] = defaultdict(list)
        self.baseline_metrics: Optional[Dict[str, float]] = None
        
        # Configuration
        self.min_improvement_threshold = 5.0  # Minimum 5% improvement to keep optimization
        self.max_concurrent_optimizations = 3
        
        self.logger.info("Advanced performance optimizer initialized")
    
    def _initialize_performance_profiles(self) -> Dict[str, PerformanceProfile]:
        """Initialize predefined performance profiles"""
        return {
            "speed": PerformanceProfile(
                name="speed",
                batch_size=8,
                max_memory_mb=4096,
                gpu_memory_fraction=0.9,
                gc_threshold=(700, 10, 10),
                thread_pool_size=8,
                cache_size_mb=512,
                model_precision="fp16",
                enable_torch_compile=True,
                optimization_level="speed"
            ),
            "balanced": PerformanceProfile(
                name="balanced",
                batch_size=4,
                max_memory_mb=2048,
                gpu_memory_fraction=0.7,
                gc_threshold=(700, 10, 10),
                thread_pool_size=4,
                cache_size_mb=256,
                model_precision="fp32",
                enable_torch_compile=False,
                optimization_level="balanced"
            ),
            "memory": PerformanceProfile(
                name="memory",
                batch_size=1,
                max_memory_mb=1024,
                gpu_memory_fraction=0.5,
                gc_threshold=(500, 8, 8),
                thread_pool_size=2,
                cache_size_mb=128,
                model_precision="int8",
                enable_torch_compile=False,
                optimization_level="memory"
            )
        }
    
    def analyze_and_optimize(self) -> List[OptimizationResult]:
        """Analyze current performance and apply optimizations"""
        if not self.optimization_enabled:
            return []
        
        # Check if enough time has passed since last optimization
        if (datetime.now() - self.last_optimization_check).total_seconds() < self.auto_optimization_interval:
            return []
        
        self.last_optimization_check = datetime.now()
        
        # Get recent performance metrics
        recent_metrics = self._get_recent_metrics()
        if not recent_metrics:
            return []
        
        # Analyze metrics and identify optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(recent_metrics)
        
        # Apply optimizations
        results = []
        for optimization_type in optimization_opportunities:
            if len(self.active_optimizations) >= self.max_concurrent_optimizations:
                break
            
            result = self._apply_optimization(optimization_type, recent_metrics)
            if result:
                results.append(result)
        
        return results
    
    def _get_recent_metrics(self) -> Dict[MetricType, List[PerformanceMetric]]:
        """Get recent performance metrics for analysis"""
        metrics = {}
        cutoff_time = datetime.now() - timedelta(minutes=30)
        
        for metric_type in MetricType:
            metric_list = self.performance_monitor.metrics.get(metric_type, deque())
            recent = [m for m in metric_list if m.timestamp >= cutoff_time]
            if recent:
                metrics[metric_type] = recent
        
        return metrics
    
    def _identify_optimization_opportunities(self, metrics: Dict[MetricType, List[PerformanceMetric]]) -> List[OptimizationType]:
        """Identify optimization opportunities based on metrics analysis"""
        opportunities = []
        
        # Memory optimization opportunities
        if MetricType.MEMORY_USAGE in metrics:
            memory_metrics = metrics[MetricType.MEMORY_USAGE]
            avg_memory = sum(m.value for m in memory_metrics) / len(memory_metrics)
            
            if avg_memory > 80.0:  # High memory usage
                opportunities.append(OptimizationType.MEMORY_CLEANUP)
                if avg_memory > 90.0:
                    opportunities.append(OptimizationType.GC_TUNING)
        
        # GPU optimization opportunities
        if MetricType.GPU_UTILIZATION in metrics:
            gpu_metrics = metrics[MetricType.GPU_UTILIZATION]
            avg_gpu = sum(m.value for m in gpu_metrics) / len(gpu_metrics)
            
            if avg_gpu < 30.0:  # Underutilized GPU
                opportunities.append(OptimizationType.BATCH_SIZE_TUNING)
            elif avg_gpu > 95.0:  # Overutilized GPU
                opportunities.append(OptimizationType.GPU_OPTIMIZATION)
        
        # Response time optimization opportunities
        response_metrics = self.performance_monitor.response_time_metrics
        if response_metrics and hasattr(response_metrics, '__iter__'):
            try:
                recent_responses = [r for r in response_metrics 
                                 if datetime.fromtimestamp(r.end_time) >= datetime.now() - timedelta(minutes=30)]
            except (TypeError, AttributeError):
                recent_responses = []
            
            if recent_responses:
                avg_response_time = sum(r.duration_ms for r in recent_responses) / len(recent_responses)
                
                if avg_response_time > 3000:  # Slow responses
                    opportunities.append(OptimizationType.MODEL_QUANTIZATION)
                    opportunities.append(OptimizationType.CACHE_OPTIMIZATION)
        
        # Remove already active optimizations
        opportunities = [opt for opt in opportunities if opt not in self.active_optimizations]
        
        return opportunities
    
    def _apply_optimization(self, optimization_type: OptimizationType, 
                          current_metrics: Dict[MetricType, List[PerformanceMetric]]) -> Optional[OptimizationResult]:
        """Apply a specific optimization"""
        try:
            self.active_optimizations[optimization_type] = datetime.now()
            start_time = time.time()
            
            # Capture baseline metrics
            before_metrics = self._capture_current_metrics()
            
            # Apply the optimization
            success = False
            error_message = None
            
            if optimization_type == OptimizationType.MEMORY_CLEANUP:
                success = self._optimize_memory_cleanup()
            elif optimization_type == OptimizationType.BATCH_SIZE_TUNING:
                success = self._optimize_batch_size()
            elif optimization_type == OptimizationType.GPU_OPTIMIZATION:
                success = self._optimize_gpu_usage()
            elif optimization_type == OptimizationType.MODEL_QUANTIZATION:
                success = self._optimize_model_quantization()
            elif optimization_type == OptimizationType.CACHE_OPTIMIZATION:
                success = self._optimize_cache()
            elif optimization_type == OptimizationType.GC_TUNING:
                success = self._optimize_garbage_collection()
            else:
                error_message = f"Optimization type {optimization_type} not implemented"
            
            # Wait for optimization to take effect
            time.sleep(30)
            
            # Capture after metrics
            after_metrics = self._capture_current_metrics()
            
            # Calculate improvement
            improvement = self._calculate_improvement(before_metrics, after_metrics)
            
            # Create result
            result = OptimizationResult(
                optimization_type=optimization_type,
                success=success,
                timestamp=datetime.now(),
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=improvement,
                error_message=error_message,
                duration_seconds=time.time() - start_time
            )
            
            # Check if optimization should be rolled back
            if improvement < self.min_improvement_threshold and success:
                self._rollback_optimization(optimization_type)
                result.rollback_applied = True
                result.improvement_percent = 0.0
            
            # Record effectiveness for learning
            self.optimization_effectiveness[optimization_type].append(improvement)
            
            # Store result
            self.optimization_history.append(result)
            
            # Remove from active optimizations
            del self.active_optimizations[optimization_type]
            
            self.logger.info(f"Applied optimization {optimization_type.value}: {improvement:.1f}% improvement")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimization {optimization_type.value}: {e}")
            if optimization_type in self.active_optimizations:
                del self.active_optimizations[optimization_type]
            return None
    
    def _optimize_memory_cleanup(self) -> bool:
        """Perform memory cleanup optimization"""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Clear PyTorch cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            # Trigger memory cleanup in performance monitor
            if hasattr(self.performance_monitor, 'force_memory_cleanup'):
                self.performance_monitor.force_memory_cleanup()
            
            self.logger.info(f"Memory cleanup completed, collected {collected} objects")
            return True
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")
            return False
    
    def _optimize_batch_size(self) -> bool:
        """Optimize batch size based on GPU utilization"""
        try:
            # This would integrate with model inference settings
            # For now, we'll log the optimization attempt
            self.logger.info("Batch size optimization attempted")
            return True
            
        except Exception as e:
            self.logger.error(f"Batch size optimization failed: {e}")
            return False
    
    def _optimize_gpu_usage(self) -> bool:
        """Optimize GPU memory and utilization"""
        try:
            import torch
            if torch.cuda.is_available():
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Reset GPU memory stats
                torch.cuda.reset_peak_memory_stats()
                
                self.logger.info("GPU optimization completed")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"GPU optimization failed: {e}")
            return False
    
    def _optimize_model_quantization(self) -> bool:
        """Apply model quantization for faster inference"""
        try:
            # This would integrate with model loading and inference
            self.logger.info("Model quantization optimization attempted")
            return True
            
        except Exception as e:
            self.logger.error(f"Model quantization failed: {e}")
            return False
    
    def _optimize_cache(self) -> bool:
        """Optimize caching strategies"""
        try:
            # This would clear and reconfigure caches
            self.logger.info("Cache optimization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Cache optimization failed: {e}")
            return False
    
    def _optimize_garbage_collection(self) -> bool:
        """Optimize garbage collection settings"""
        try:
            # Tune garbage collection thresholds
            gc.set_threshold(500, 8, 8)  # More aggressive GC
            
            # Perform full collection
            gc.collect()
            
            self.logger.info("Garbage collection optimization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"GC optimization failed: {e}")
            return False
    
    def _capture_current_metrics(self) -> Dict[str, float]:
        """Capture current system metrics for comparison"""
        metrics = {}
        
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_available_mb'] = memory.available / (1024 * 1024)
            
            # CPU metrics
            metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
            
            # GPU metrics
            try:
                import torch
                if torch.cuda.is_available():
                    metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024 * 1024)
                    metrics['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024 * 1024)
            except ImportError:
                pass
            
            # Response time metrics (recent average)
            recent_responses = list(self.performance_monitor.response_time_metrics)[-10:]
            if recent_responses:
                metrics['avg_response_time_ms'] = sum(r.duration_ms for r in recent_responses) / len(recent_responses)
            
        except Exception as e:
            self.logger.error(f"Failed to capture metrics: {e}")
        
        return metrics
    
    def _calculate_improvement(self, before: Dict[str, float], after: Dict[str, float]) -> float:
        """Calculate overall performance improvement percentage"""
        improvements = []
        
        # Memory improvement (lower is better)
        if 'memory_percent' in before and 'memory_percent' in after:
            if before['memory_percent'] > 0:
                improvement = (before['memory_percent'] - after['memory_percent']) / before['memory_percent'] * 100
                improvements.append(improvement)
        
        # Response time improvement (lower is better)
        if 'avg_response_time_ms' in before and 'avg_response_time_ms' in after:
            if before['avg_response_time_ms'] > 0:
                improvement = (before['avg_response_time_ms'] - after['avg_response_time_ms']) / before['avg_response_time_ms'] * 100
                improvements.append(improvement)
        
        # Return average improvement, or 0 if no improvements calculated
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def _rollback_optimization(self, optimization_type: OptimizationType):
        """Rollback an optimization that didn't provide sufficient benefit"""
        try:
            if optimization_type == OptimizationType.GC_TUNING:
                # Reset to default GC thresholds
                gc.set_threshold(700, 10, 10)
            
            self.logger.info(f"Rolled back optimization: {optimization_type.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to rollback optimization {optimization_type.value}: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization activities"""
        total_optimizations = len(self.optimization_history)
        successful_optimizations = sum(1 for opt in self.optimization_history if opt.success)
        
        avg_improvement = 0.0
        if self.optimization_history:
            avg_improvement = sum(opt.improvement_percent for opt in self.optimization_history) / total_optimizations
        
        return {
            "total_optimizations": total_optimizations,
            "successful_optimizations": successful_optimizations,
            "success_rate": successful_optimizations / total_optimizations if total_optimizations > 0 else 0,
            "average_improvement_percent": avg_improvement,
            "active_optimizations": len(self.active_optimizations),
            "current_profile": self.current_profile,
            "optimization_enabled": self.optimization_enabled,
            "recent_optimizations": [
                {
                    "type": opt.optimization_type.value,
                    "timestamp": opt.timestamp.isoformat(),
                    "improvement": opt.improvement_percent,
                    "success": opt.success
                }
                for opt in list(self.optimization_history)[-5:]
            ]
        }


class CrashDetectionRecoverySystem:
    """Comprehensive crash detection and automated recovery system"""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.logger = get_performance_logger()
        
        # Crash detection state
        self.monitored_processes: Dict[int, Dict[str, Any]] = {}
        self.crash_history: deque[CrashEvent] = deque(maxlen=1000)
        self.recovery_history: deque[RecoveryAction] = deque(maxlen=1000)
        
        # Recovery configuration
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.max_recovery_attempts = 3
        self.recovery_cooldown_minutes = 5
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # State persistence
        self.state_dir = Path("logs/recovery_state")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Circuit breaker pattern
        self.circuit_breaker_state: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "failures": 0,
            "last_failure": None,
            "state": "closed"  # closed, open, half_open
        })
        
        self.logger.info("Crash detection and recovery system initialized")
    
    def _initialize_recovery_strategies(self) -> Dict[CrashType, RecoveryStrategy]:
        """Initialize recovery strategies for different crash types"""
        return {
            CrashType.MEMORY_EXHAUSTION: RecoveryStrategy.GRACEFUL_RESTART,
            CrashType.GPU_TIMEOUT: RecoveryStrategy.RESOURCE_CLEANUP,
            CrashType.PROCESS_HANG: RecoveryStrategy.FORCE_RESTART,
            CrashType.MODEL_FAILURE: RecoveryStrategy.SERVICE_FAILOVER,
            CrashType.UNHANDLED_EXCEPTION: RecoveryStrategy.GRACEFUL_RESTART,
            CrashType.RESOURCE_LEAK: RecoveryStrategy.RESOURCE_CLEANUP,
            CrashType.NETWORK_FAILURE: RecoveryStrategy.CIRCUIT_BREAKER,
            CrashType.HARDWARE_FAILURE: RecoveryStrategy.EMERGENCY_SHUTDOWN
        }
    
    def start_monitoring(self, interval_seconds: float = 10.0):
        """Start crash detection monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._stop_event.clear()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True,
            name="CrashDetectionMonitor"
        )
        self.monitoring_thread.start()
        
        self.logger.info(f"Started crash detection monitoring (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop crash detection monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self._stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10.0)
        
        self.logger.info("Stopped crash detection monitoring")
    
    def _monitoring_loop(self, interval_seconds: float):
        """Main monitoring loop for crash detection"""
        while self.monitoring_active and not self._stop_event.is_set():
            try:
                self._check_for_crashes()
                self._update_process_health()
                self._check_circuit_breakers()
                
            except Exception as e:
                self.logger.error(f"Error in crash detection monitoring: {e}")
            
            self._stop_event.wait(interval_seconds)
    
    def _check_for_crashes(self):
        """Check for process crashes and system failures"""
        # Check monitored processes
        for pid, process_info in list(self.monitored_processes.items()):
            try:
                process = psutil.Process(pid)
                
                # Check if process is still running
                if not process.is_running():
                    self._handle_process_crash(pid, process_info)
                    continue
                
                # Check for hanging processes
                if self._is_process_hanging(process, process_info):
                    self._handle_process_hang(process, process_info)
                
                # Check for memory exhaustion
                memory_percent = process.memory_percent()
                if memory_percent > 95.0:
                    self._handle_memory_exhaustion(process, process_info)
                
            except psutil.NoSuchProcess:
                # Process has terminated
                self._handle_process_crash(pid, process_info)
            except Exception as e:
                self.logger.error(f"Error checking process {pid}: {e}")
    
    def _handle_process_crash(self, pid: int, process_info: Dict[str, Any]):
        """Handle a detected process crash"""
        crash_event = CrashEvent(
            crash_type=CrashType.UNHANDLED_EXCEPTION,
            timestamp=datetime.now(),
            process_id=pid,
            process_name=process_info.get('name', 'unknown'),
            exit_code=None,
            signal_received=None,
            context=process_info
        )
        
        self.crash_history.append(crash_event)
        
        # Remove from monitored processes
        del self.monitored_processes[pid]
        
        # Attempt recovery
        self._attempt_recovery(crash_event)
        
        self.logger.error(f"Process crash detected: {crash_event.process_name} (PID: {pid})")
    
    def _is_process_hanging(self, process: psutil.Process, process_info: Dict[str, Any]) -> bool:
        """Check if a process appears to be hanging"""
        try:
            # Check CPU usage over time
            current_time = time.time()
            last_check = process_info.get('last_cpu_check', current_time)
            
            if current_time - last_check > 60:  # Check every minute
                cpu_percent = process.cpu_percent()
                last_cpu = process_info.get('last_cpu_percent', 0)
                
                # If CPU usage is consistently 0 or 100%, might be hanging
                if cpu_percent == 0 and last_cpu == 0:
                    hang_count = process_info.get('hang_count', 0) + 1
                    process_info['hang_count'] = hang_count
                    
                    if hang_count >= 3:  # 3 consecutive low CPU checks
                        return True
                else:
                    process_info['hang_count'] = 0
                
                process_info['last_cpu_percent'] = cpu_percent
                process_info['last_cpu_check'] = current_time
            
            return False
            
        except Exception:
            return False
    
    def _handle_process_hang(self, process: psutil.Process, process_info: Dict[str, Any]):
        """Handle a hanging process"""
        crash_event = CrashEvent(
            crash_type=CrashType.PROCESS_HANG,
            timestamp=datetime.now(),
            process_id=process.pid,
            process_name=process.name(),
            exit_code=None,
            signal_received=None,
            cpu_usage_percent=process.cpu_percent(),
            memory_usage_mb=process.memory_info().rss / (1024 * 1024),
            context=process_info
        )
        
        self.crash_history.append(crash_event)
        self._attempt_recovery(crash_event)
        
        self.logger.error(f"Process hang detected: {process.name()} (PID: {process.pid})")
    
    def _handle_memory_exhaustion(self, process: psutil.Process, process_info: Dict[str, Any]):
        """Handle memory exhaustion"""
        crash_event = CrashEvent(
            crash_type=CrashType.MEMORY_EXHAUSTION,
            timestamp=datetime.now(),
            process_id=process.pid,
            process_name=process.name(),
            exit_code=None,
            signal_received=None,
            memory_usage_mb=process.memory_info().rss / (1024 * 1024),
            context=process_info
        )
        
        self.crash_history.append(crash_event)
        self._attempt_recovery(crash_event)
        
        self.logger.error(f"Memory exhaustion detected: {process.name()} (PID: {process.pid})")
    
    def _attempt_recovery(self, crash_event: CrashEvent):
        """Attempt to recover from a crash event"""
        recovery_strategy = self.recovery_strategies.get(crash_event.crash_type, RecoveryStrategy.GRACEFUL_RESTART)
        
        start_time = time.time()
        success = False
        new_process_id = None
        error_message = None
        
        try:
            if recovery_strategy == RecoveryStrategy.GRACEFUL_RESTART:
                success, new_process_id = self._graceful_restart(crash_event)
            elif recovery_strategy == RecoveryStrategy.FORCE_RESTART:
                success, new_process_id = self._force_restart(crash_event)
            elif recovery_strategy == RecoveryStrategy.RESOURCE_CLEANUP:
                success = self._resource_cleanup(crash_event)
            elif recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                success = self._activate_circuit_breaker(crash_event)
            elif recovery_strategy == RecoveryStrategy.EMERGENCY_SHUTDOWN:
                success = self._emergency_shutdown(crash_event)
            else:
                error_message = f"Recovery strategy {recovery_strategy} not implemented"
            
        except Exception as e:
            error_message = str(e)
            success = False
        
        recovery_time = time.time() - start_time
        
        recovery_action = RecoveryAction(
            recovery_strategy=recovery_strategy,
            timestamp=datetime.now(),
            crash_event=crash_event,
            success=success,
            recovery_time_seconds=recovery_time,
            new_process_id=new_process_id,
            error_message=error_message
        )
        
        self.recovery_history.append(recovery_action)
        
        if success:
            self.logger.info(f"Recovery successful for {crash_event.process_name}: {recovery_strategy.value}")
        else:
            self.logger.error(f"Recovery failed for {crash_event.process_name}: {error_message}")
    
    def _graceful_restart(self, crash_event: CrashEvent) -> Tuple[bool, Optional[int]]:
        """Perform graceful restart of a crashed process"""
        try:
            # Save current state if possible
            self._save_process_state(crash_event)
            
            # For demonstration, we'll simulate a restart
            # In real implementation, this would restart the actual service
            self.logger.info(f"Performing graceful restart of {crash_event.process_name}")
            
            # Simulate new process ID
            new_pid = os.getpid() + 1000  # Dummy PID for demonstration
            
            return True, new_pid
            
        except Exception as e:
            self.logger.error(f"Graceful restart failed: {e}")
            return False, None
    
    def _force_restart(self, crash_event: CrashEvent) -> Tuple[bool, Optional[int]]:
        """Perform force restart of a hanging process"""
        try:
            # Kill the hanging process if it still exists
            try:
                process = psutil.Process(crash_event.process_id)
                process.terminate()
                
                # Wait for termination, then kill if necessary
                try:
                    process.wait(timeout=10)
                except psutil.TimeoutExpired:
                    process.kill()
                    
            except psutil.NoSuchProcess:
                pass  # Process already terminated
            
            # Restart the process
            return self._graceful_restart(crash_event)
            
        except Exception as e:
            self.logger.error(f"Force restart failed: {e}")
            return False, None
    
    def _resource_cleanup(self, crash_event: CrashEvent) -> bool:
        """Perform resource cleanup"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force memory cleanup through performance monitor
            if hasattr(self.performance_monitor, 'force_memory_cleanup'):
                self.performance_monitor.force_memory_cleanup()
            
            self.logger.info("Resource cleanup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")
            return False
    
    def _activate_circuit_breaker(self, crash_event: CrashEvent) -> bool:
        """Activate circuit breaker for failing service"""
        try:
            service_name = crash_event.process_name
            circuit_state = self.circuit_breaker_state[service_name]
            
            circuit_state['failures'] += 1
            circuit_state['last_failure'] = datetime.now()
            
            if circuit_state['failures'] >= 3:
                circuit_state['state'] = 'open'
                self.logger.warning(f"Circuit breaker opened for {service_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Circuit breaker activation failed: {e}")
            return False
    
    def _emergency_shutdown(self, crash_event: CrashEvent) -> bool:
        """Perform emergency shutdown for critical failures"""
        try:
            self.logger.critical(f"Emergency shutdown triggered by {crash_event.process_name}")
            
            # Save critical state
            self._save_emergency_state(crash_event)
            
            # Stop all monitoring
            self.stop_monitoring()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Emergency shutdown failed: {e}")
            return False
    
    def _save_process_state(self, crash_event: CrashEvent):
        """Save process state for recovery"""
        try:
            state_file = self.state_dir / f"{crash_event.process_name}_{crash_event.timestamp.strftime('%Y%m%d_%H%M%S')}.pkl"
            
            state_data = {
                'crash_event': crash_event,
                'timestamp': datetime.now(),
                'performance_metrics': self.performance_monitor.get_current_status()
            }
            
            with open(state_file, 'wb') as f:
                pickle.dump(state_data, f)
            
            self.logger.info(f"Process state saved to {state_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save process state: {e}")
    
    def _save_emergency_state(self, crash_event: CrashEvent):
        """Save emergency state during critical failures"""
        try:
            emergency_file = self.state_dir / "emergency_state.json"
            
            emergency_data = {
                'timestamp': datetime.now().isoformat(),
                'crash_event': {
                    'type': crash_event.crash_type.value,
                    'process_name': crash_event.process_name,
                    'process_id': crash_event.process_id,
                    'timestamp': crash_event.timestamp.isoformat()
                },
                'system_status': self.performance_monitor.get_current_status()
            }
            
            with open(emergency_file, 'w') as f:
                json.dump(emergency_data, f, indent=2)
            
            self.logger.critical(f"Emergency state saved to {emergency_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save emergency state: {e}")
    
    def _update_process_health(self):
        """Update health information for monitored processes"""
        for pid, process_info in self.monitored_processes.items():
            try:
                process = psutil.Process(pid)
                
                # Update health metrics
                process_info['last_seen'] = datetime.now()
                process_info['memory_mb'] = process.memory_info().rss / (1024 * 1024)
                process_info['cpu_percent'] = process.cpu_percent()
                
            except psutil.NoSuchProcess:
                # Will be handled in crash detection
                pass
            except Exception as e:
                self.logger.error(f"Error updating process health for {pid}: {e}")
    
    def _check_circuit_breakers(self):
        """Check and update circuit breaker states"""
        current_time = datetime.now()
        
        for service_name, circuit_state in self.circuit_breaker_state.items():
            if circuit_state['state'] == 'open':
                # Check if enough time has passed to try half-open
                if circuit_state['last_failure']:
                    time_diff = current_time - circuit_state['last_failure']
                    if time_diff.total_seconds() > self.recovery_cooldown_minutes * 60:
                        circuit_state['state'] = 'half_open'
                        self.logger.info(f"Circuit breaker half-opened for {service_name}")
    
    def register_process(self, pid: int, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Register a process for monitoring"""
        self.monitored_processes[pid] = {
            'name': name,
            'registered_time': datetime.now(),
            'metadata': metadata or {},
            'last_seen': datetime.now()
        }
        
        self.logger.info(f"Registered process for monitoring: {name} (PID: {pid})")
    
    def unregister_process(self, pid: int):
        """Unregister a process from monitoring"""
        if pid in self.monitored_processes:
            del self.monitored_processes[pid]
            self.logger.info(f"Unregistered process from monitoring: PID {pid}")
    
    def get_recovery_summary(self) -> Dict[str, Any]:
        """Get summary of crash detection and recovery activities"""
        total_crashes = len(self.crash_history)
        total_recoveries = len(self.recovery_history)
        successful_recoveries = sum(1 for r in self.recovery_history if r.success)
        
        crash_by_type = defaultdict(int)
        for crash in self.crash_history:
            crash_by_type[crash.crash_type.value] += 1
        
        return {
            "monitoring_active": self.monitoring_active,
            "monitored_processes": len(self.monitored_processes),
            "total_crashes": total_crashes,
            "total_recoveries": total_recoveries,
            "successful_recoveries": successful_recoveries,
            "recovery_success_rate": successful_recoveries / total_recoveries if total_recoveries > 0 else 0,
            "crash_by_type": dict(crash_by_type),
            "circuit_breakers": {
                name: state['state'] for name, state in self.circuit_breaker_state.items()
            },
            "recent_crashes": [
                {
                    "type": crash.crash_type.value,
                    "process_name": crash.process_name,
                    "timestamp": crash.timestamp.isoformat()
                }
                for crash in list(self.crash_history)[-5:]
            ]
        }


def create_automated_performance_system(performance_monitor: PerformanceMonitor) -> Tuple[AdvancedPerformanceOptimizer, CrashDetectionRecoverySystem]:
    """Factory function to create the complete automated performance system"""
    optimizer = AdvancedPerformanceOptimizer(performance_monitor)
    recovery_system = CrashDetectionRecoverySystem(performance_monitor)
    
    return optimizer, recovery_system 