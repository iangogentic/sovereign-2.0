"""
Tool Service - Lazy-loaded tool discovery, permission, and execution system

Provides on-demand tool management with registry building and execution happening
only on the first tool execution call. Subsequent calls reuse the built registry.
No heavy imports at module level.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolResult:
    """Lightweight container for tool execution results"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_name: str = ""
    execution_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'success': self.success,
            'result': self.result,
            'error': self.error,
            'execution_time': self.execution_time,
            'tool_name': self.tool_name,
            'execution_id': self.execution_id
        }


class ToolService:
    """
    On-demand tool service with lazy loading
    
    Registry building and tool discovery only happens on first execute_tool() call.
    Subsequent calls reuse the existing registry for performance.
    Thread-safe with async locks to prevent double-loading during concurrent access.
    """
    
    def __init__(self):
        """Initialize with minimal overhead - no heavy imports"""
        self._registry = None
        self._execution_engine = None
        self._permission_manager = None
        self._framework = None
        self._initialized = False
        self._loading_lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'registry_load_time': 0.0,
            'first_execution_at': None,
            'last_execution_at': None
        }
        
        logging.info("ToolService created (lazy loading enabled)")
    
    async def _load_components(self):
        """
        Load heavy tool framework components on first use
        
        This is where the actual tool registry building happens.
        Protected by async lock to prevent double-loading.
        """
        async with self._loading_lock:
            if self._initialized:
                return  # Already loaded by another thread
            
            logging.info("🔄 Loading tool registry and framework...")
            start_time = time.perf_counter()
            
            try:
                # Import heavy modules only when needed
                from ..tool_integration_framework import ToolIntegrationFramework
                from ..config import Config
                
                # Create configuration
                config = Config()
                
                # Initialize tool framework
                self._framework = ToolIntegrationFramework(config)
                
                # Initialize framework (loads registry, permission manager, etc.)
                framework_success = await self._framework.initialize()
                if not framework_success:
                    raise RuntimeError("Failed to initialize tool framework")
                
                # Extract components
                self._registry = self._framework.registry
                self._execution_engine = self._framework.execution_engine
                self._permission_manager = self._framework.permission_manager
                
                # Mark as initialized
                self._initialized = True
                
                load_time = time.perf_counter() - start_time
                self._stats['registry_load_time'] = load_time
                
                tool_count = len(self._registry.tools) if self._registry else 0
                logging.info(f"✅ Tool framework loaded: {tool_count} tools in {load_time:.2f}s")
                
            except Exception as e:
                logging.error(f"❌ Failed to load tool framework: {e}")
                raise RuntimeError(f"Tool framework initialization failed: {e}")
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], 
                          requester: str = "system") -> ToolResult:
        """
        Execute a tool with given parameters
        
        On first call, this loads the tool registry and framework.
        Subsequent calls use the cached components.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            requester: ID of the requesting component
            
        Returns:
            ToolResult with execution outcome
        """
        start_time = time.perf_counter()
        
        try:
            # Update statistics (count the execution attempt)
            self._stats['total_executions'] += 1
            if self._stats['first_execution_at'] is None:
                self._stats['first_execution_at'] = datetime.now()
            self._stats['last_execution_at'] = datetime.now()
            
            # Ensure components are loaded (lazy loading)
            if not self._initialized:
                await self._load_components()
            
            # Execute tool through framework
            result = await self._framework.execute_tool(tool_name, parameters, requester)
            
            # Convert to our lightweight result format
            execution_time = time.perf_counter() - start_time
            
            if result.success:
                self._stats['successful_executions'] += 1
                
                return ToolResult(
                    success=True,
                    result=result.result,
                    execution_time=execution_time,
                    tool_name=tool_name,
                    execution_id=result.execution_id
                )
            else:
                self._stats['failed_executions'] += 1
                
                return ToolResult(
                    success=False,
                    error=result.error,
                    execution_time=execution_time,
                    tool_name=tool_name,
                    execution_id=result.execution_id
                )
                
        except Exception as e:
            self._stats['failed_executions'] += 1
            execution_time = time.perf_counter() - start_time
            
            logging.error(f"Tool execution failed: {tool_name} - {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                tool_name=tool_name
            )
    
    async def get_available_tools(self, requester: str = "system") -> List[Dict[str, Any]]:
        """
        Get list of available tools
        
        Loads registry if not already loaded.
        """
        if not self._initialized:
            await self._load_components()
        
        return self._framework.get_available_tools(requester)
    
    def is_registry_loaded(self) -> bool:
        """Check if the tool registry has been loaded"""
        return self._initialized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool service statistics"""
        stats = self._stats.copy()
        
        # Add loaded tool count if registry is available
        if self._registry:
            stats['loaded_tools'] = len(self._registry.tools)
        else:
            stats['loaded_tools'] = 0
        
        # Add framework stats if available
        if self._framework:
            framework_stats = self._framework.get_framework_stats()
            stats.update(framework_stats)
        
        return stats
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get health check information"""
        return {
            'service_name': 'ToolService',
            'status': 'ready' if self._initialized else 'uninitialized',
            'registry_loaded': self._initialized,
            'total_executions': self._stats['total_executions'],
            'components': {
                'registry': self._registry is not None,
                'execution_engine': self._execution_engine is not None,
                'permission_manager': self._permission_manager is not None,
                'framework': self._framework is not None
            }
        }
    
    async def close(self):
        """Cleanup resources"""
        if self._framework:
            try:
                await self._framework.cleanup()
                logging.info("Tool framework cleaned up")
            except Exception as e:
                logging.error(f"Error cleaning up tool framework: {e}")
        
        self._registry = None
        self._execution_engine = None
        self._permission_manager = None
        self._framework = None
        self._initialized = False
        
        logging.info("ToolService closed") 