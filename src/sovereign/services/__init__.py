"""
Services Module - Sovereign AI Agent

Lazy-loaded service layer for managing AI models and components.
All services implement on-demand loading to ensure fast application startup.
"""

# Remove module-level imports to prevent heavy dependencies during startup
# Services are now imported only when needed via ServiceManager

__all__ = ['ModelService', 'MemoryService', 'ToolService', 'ScreenContextService', 'VoiceService'] 