"""
Core Application - Sovereign AI Agent

Ultra-lightweight core application that orchestrates services via ServiceManager.
Designed for sub-second cold starts with lazy loading of all heavy components.
"""

import logging
from typing import Optional

from .service_manager import ServiceManager


class CoreApp:
    """
    Core Application Shell for Sovereign AI Agent
    
    Provides minimal startup overhead with lazy-loaded services.
    All heavy components (models, FAISS, OCR, etc.) are loaded on-demand.
    """
    
    def __init__(self):
        """Initialize the core application with minimal overhead"""
        self._service_manager: Optional[ServiceManager] = None
        self._running = False
        
        # Configure lightweight logging
        self._setup_logging()
        
        # Create service manager (lightweight placeholder)
        self._service_manager = ServiceManager()
        
        logging.info("CoreApp initialized")
    
    def _setup_logging(self):
        """Setup basic logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def run(self):
        """
        Run the application
        
        For now, this is a minimal implementation that demonstrates
        the service architecture without loading heavy components.
        """
        if self._running:
            logging.warning("Application is already running")
            return
        
        try:
            self._running = True
            logging.info("Starting Sovereign AI Agent services...")
            
            # Initialize service manager
            self._service_manager.initialize()
            
            print("📋 Service Manager initialized")
            print("⚡ All services ready for on-demand loading")
            print("🎯 Ready to receive commands")
            
            # For now, we just print status and exit
            # In future tasks, this will start the full application loop
            
        except Exception as e:
            logging.error(f"Failed to start application: {e}")
            raise
        finally:
            self._running = False
    
    async def stop(self):
        """Stop the application and cleanup resources"""
        if not self._running:
            return
        
        logging.info("Stopping Sovereign AI Agent...")
        
        if self._service_manager:
            await self._service_manager.shutdown()
        
        self._running = False
        logging.info("Application stopped")
    
    @property
    def service_manager(self) -> ServiceManager:
        """Get the service manager instance"""
        return self._service_manager 