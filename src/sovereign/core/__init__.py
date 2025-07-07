"""
Core Module - Sovereign AI Agent

Ultra-lightweight core components for fast startup and service orchestration.
"""

from .app import CoreApp
from .service_manager import ServiceManager, ServiceStatus

__all__ = ['CoreApp', 'ServiceManager', 'ServiceStatus'] 