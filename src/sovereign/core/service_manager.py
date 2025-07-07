"""
Service Manager - Sovereign AI Agent

Lightweight service orchestration system that manages lazy-loaded services.
No heavy imports - all services are loaded on-demand when first accessed.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Type
from enum import Enum


class ServiceStatus(Enum):
    """Service status enumeration"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing" 
    READY = "ready"
    ERROR = "error"
    STOPPED = "stopped"


class ServiceManager:
    """
    Lightweight Service Manager for Sovereign AI Agent
    
    Manages service lifecycle and lazy loading without importing heavy dependencies.
    Services are only loaded when first accessed to ensure fast startup times.
    """
    
    def __init__(self):
        """Initialize the service manager with minimal overhead"""
        self._services: Dict[str, Any] = {}
        self._service_status: Dict[str, ServiceStatus] = {}
        self._initialized = False
        
        logging.info("ServiceManager created")
    
    def initialize(self):
        """Initialize the service manager (lightweight operation)"""
        if self._initialized:
            logging.warning("ServiceManager already initialized")
            return
        
        # Register placeholder service definitions (no actual loading)
        self._register_service_definitions()
        
        self._initialized = True
        logging.info("ServiceManager initialized - services ready for lazy loading")
    
    def _register_service_definitions(self):
        """Register service definitions without loading them"""
        # Define services that will be available for lazy loading
        service_definitions = [
            "ModelService",      # Will wrap Talker/Thinker models
            "MemoryService",     # Will wrap FAISS + MemoryManager  
            "ToolService",       # Will wrap tool discovery/execution
            "ScreenContextService",  # Will wrap OCR/screen capture
            "VoiceService"       # Will wrap microphone/speaker
        ]
        
        for service_name in service_definitions:
            self._service_status[service_name] = ServiceStatus.UNINITIALIZED
            logging.debug(f"Registered service definition: {service_name}")
    
    def _load_service(self, service_name: str) -> Optional[Any]:
        """
        Load a specific service instance
        
        Args:
            service_name: Name of the service to load
            
        Returns:
            Service instance or None if failed
        """
        logging.info(f"🔄 Loading service: {service_name}")
        
        if service_name == "ModelService":
            # Import ModelService only when needed to avoid heavy imports at startup
            from ..services.model_service import ModelService
            return ModelService()
        
        elif service_name == "MemoryService":
            # Import MemoryService only when needed to avoid heavy imports at startup
            from ..services.memory_service import MemoryService
            return MemoryService()
            
        elif service_name == "ToolService":
            # Import ToolService only when needed to avoid heavy imports at startup
            from ..services.tool_service import ToolService
            return ToolService()
            
        elif service_name == "ScreenContextService":
            # Import ScreenContextService only when needed to avoid heavy imports at startup
            from ..services.screen_context_service import ScreenContextService
            return ScreenContextService()
            
        elif service_name == "VoiceService":
            # Import VoiceService only when needed to avoid heavy imports at startup
            from ..services.voice_service import VoiceService
            return VoiceService()
        
        else:
            logging.error(f"Unknown service: {service_name}")
            return None
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """
        Get a service instance, loading it lazily if needed
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            Service instance or None if not found
        """
        if not self._initialized:
            raise RuntimeError("ServiceManager not initialized")
        
        if service_name not in self._service_status:
            logging.error(f"Unknown service: {service_name}")
            return None
        
        # Check if service is already loaded
        if service_name in self._services:
            return self._services[service_name]
        
        # Lazy load the service
        try:
            service_instance = self._load_service(service_name)
            if service_instance:
                self._services[service_name] = service_instance
                self._service_status[service_name] = ServiceStatus.READY
                logging.info(f"✅ Service {service_name} loaded successfully")
                return service_instance
            else:
                self._service_status[service_name] = ServiceStatus.ERROR
                logging.error(f"❌ Failed to load service {service_name}")
                return None
        except Exception as e:
            self._service_status[service_name] = ServiceStatus.ERROR
            logging.error(f"❌ Error loading service {service_name}: {e}")
            return None
    
    def is_service_ready(self, service_name: str) -> bool:
        """Check if a service is ready for use"""
        status = self._service_status.get(service_name, ServiceStatus.UNINITIALIZED)
        return status == ServiceStatus.READY
    
    def get_service_status(self, service_name: str) -> ServiceStatus:
        """Get the current status of a service"""
        return self._service_status.get(service_name, ServiceStatus.UNINITIALIZED)
    
    def list_services(self) -> Dict[str, ServiceStatus]:
        """Get a dictionary of all services and their statuses"""
        return self._service_status.copy()
    
    def model(self):
        """
        Convenience method to get the ModelService
        
        Returns:
            ModelService instance (loads lazily if needed)
        """
        return self.get_service("ModelService")
    
    def memory(self):
        """
        Convenience method to get the MemoryService
        
        Returns:
            MemoryService instance (loads lazily if needed)
        """
        return self.get_service("MemoryService")
    
    def tool(self):
        """
        Convenience method to get the ToolService
        
        Returns:
            ToolService instance (loads lazily if needed)
        """
        return self.get_service("ToolService")
    
    def screen(self):
        """
        Convenience method to get the ScreenContextService
        
        Returns:
            ScreenContextService instance (loads lazily if needed)
        """
        return self.get_service("ScreenContextService")
    
    def voice(self):
        """
        Convenience method to get the VoiceService
        
        Returns:
            VoiceService instance (loads lazily if needed)
        """
        return self.get_service("VoiceService")
    
    async def shutdown(self):
        """Shutdown all services and cleanup resources"""
        if not self._initialized:
            return
        
        logging.info("Shutting down ServiceManager...")
        
        # Properly shutdown loaded services
        for service_name, service_instance in self._services.items():
            try:
                if hasattr(service_instance, 'close'):
                    if asyncio.iscoroutinefunction(service_instance.close):
                        await service_instance.close()
                    else:
                        service_instance.close()
                logging.info(f"Service {service_name} shut down successfully")
            except Exception as e:
                logging.error(f"Error shutting down service {service_name}: {e}")
            finally:
                self._service_status[service_name] = ServiceStatus.STOPPED
        
        self._services.clear()
        self._initialized = False
        
        logging.info("ServiceManager shutdown complete") 