"""
Core Application - Sovereign AI Agent

Ultra-lightweight core application that orchestrates services via ServiceManager.
Designed for sub-second cold starts with lazy loading of all heavy components.
"""

import asyncio
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
        Run the application with interactive loop
        
        Presents an interactive prompt for user input and processes queries
        through the ModelService with lazy loading.
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
            print("🎯 Sovereign AI Agent ready!")
            print("💡 Type 'exit' to quit\n")
            
            # Start the interactive loop
            asyncio.run(self._interactive_loop())
            
        except KeyboardInterrupt:
            print("\n👋 Received interrupt signal, shutting down...")
            logging.info("Application interrupted by user")
        except Exception as e:
            logging.error(f"Failed to start application: {e}")
            raise
        finally:
            self._running = False
            print("✅ Sovereign AI Agent stopped")
    
    async def _interactive_loop(self):
        """
        Main interactive loop for processing user queries
        
        Handles user input, integrates with ModelService, and manages
        the conversation flow. Also processes slash commands for feature access.
        """
        while True:
            try:
                # Get user input
                user_input = input("🤖 Sovereign > ").strip()
                
                # Handle exit command
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Handle slash commands for feature access
                if user_input.startswith('/'):
                    await self._handle_slash_command(user_input)
                    continue
                
                # Process query through ModelService
                print("🔄 Processing...")
                
                # Get ModelService from ServiceManager (lazy loading will occur here)
                model_service = self._service_manager.model()
                
                if model_service is None:
                    print("❌ Error: Unable to load ModelService")
                    continue
                
                # Query the AI model
                response = await model_service.query(user_input)
                
                # Display the response
                print(f"\n🤖 {response}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Received interrupt signal, shutting down...")
                break
            except EOFError:
                print("\n👋 End of input, shutting down...")
                break
            except Exception as e:
                logging.error(f"Error in interactive loop: {e}")
                print(f"❌ Error: {e}")
                print("Please try again or type 'exit' to quit.\n")
    
    async def _handle_slash_command(self, command: str):
        """
        Handle slash commands for feature access
        
        Args:
            command: The full command string starting with '/'
        """
        # Help command
        if command == '/help':
            print("📚 Help system: /memory, /tool, /screen")
            return
        
        # Memory commands
        if command.startswith('/memory'):
            print(f"💾 Memory command received: {command}")
            return
        
        # Tool commands  
        if command.startswith('/tool'):
            print(f"🔧 Tool command received: {command}")
            return
        
        # Screen commands
        if command.startswith('/screen'):
            print(f"🖥️ Screen command received: {command}")
            return
        
        # Unknown command
        print(f"❓ Unknown command: {command}")
        print("💡 Type '/help' for available commands")
    
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