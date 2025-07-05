"""
Logging system for Sovereign AI Agent
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "sovereign",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    debug: bool = False
) -> logging.Logger:
    """
    Set up comprehensive logging for the Sovereign AI Agent
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        debug: Enable debug mode with more detailed logging
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if not debug else logging.DEBUG)
    console_handler.setFormatter(simple_formatter if not debug else detailed_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler to prevent log files from growing too large
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Create logs directory and default log file if none specified
    if not log_file:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        default_log_file = logs_dir / f"sovereign_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            default_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Performance logger for timing critical operations
    perf_logger = logging.getLogger(f"{name}.performance")
    perf_logger.setLevel(logging.INFO)
    
    if not log_file:
        perf_log_file = logs_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.log"
    else:
        perf_log_file = Path(log_file).parent / f"performance_{datetime.now().strftime('%Y%m%d')}.log"
    
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    perf_handler.setFormatter(logging.Formatter(
        '%(asctime)s - PERF - %(message)s'
    ))
    perf_logger.addHandler(perf_handler)
    
    # Don't propagate performance logs to parent logger
    perf_logger.propagate = False
    
    logger.info(f"Logging system initialized - Level: {log_level}")
    if debug:
        logger.debug("Debug mode enabled")
    
    return logger


def get_performance_logger() -> logging.Logger:
    """Get the performance logger for timing critical operations"""
    return logging.getLogger("sovereign.performance")


def get_debug_logger() -> logging.Logger:
    """
    Get a specialized debug logger for detailed debugging and error tracking.
    This logger provides comprehensive tracing for development and troubleshooting.
    """
    debug_logger = logging.getLogger("sovereign.debug")
    
    # Only configure if not already configured
    if not debug_logger.handlers:
        debug_logger.setLevel(logging.DEBUG)
        debug_logger.propagate = False  # Don't propagate to parent loggers
        
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Detailed formatter with full context
        debug_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
        )
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(debug_formatter)
        debug_logger.addHandler(console_handler)
        
        # Dedicated debug file handler
        debug_log_file = logs_dir / f"debug_{datetime.now().strftime('%Y%m%d')}.log"
        debug_file_handler = logging.handlers.RotatingFileHandler(
            debug_log_file,
            maxBytes=20*1024*1024,  # 20MB for detailed debug logs
            backupCount=10
        )
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(debug_formatter)
        debug_logger.addHandler(debug_file_handler)
        
        debug_logger.info("Debug logger initialized for comprehensive error tracking")
    
    return debug_logger


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_performance_logger()
        self.start_time = None
        self.end_time = None
        self.elapsed_time = 0.0
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            self.end_time = datetime.now()
            self.elapsed_time = (self.end_time - self.start_time).total_seconds()
            self.logger.info(f"{self.operation_name}: {self.elapsed_time:.3f}s")


# Convenience function for timing
def time_operation(operation_name: str):
    """Decorator for timing function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceTimer(f"{func.__name__} - {operation_name}"):
                return func(*args, **kwargs)
        return wrapper
    return decorator 