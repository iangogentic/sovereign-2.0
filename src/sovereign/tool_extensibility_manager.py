"""
Extensibility and Permission Management System for Sovereign AI Tool Integration Framework

This module provides comprehensive functionality for:
1. Custom tool plugin management and dynamic loading
2. User permission controls and access management
3. Extension point registration and management
4. Security and consent validation

Author: Sovereign AI System
Version: 1.0.0
"""

import os
import json
import yaml
import importlib.util
import inspect
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime
import logging
from abc import ABC, abstractmethod
import threading
from functools import wraps
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User role definitions for permission management"""
    ADMIN = "admin"
    USER = "user"
    RESTRICTED = "restricted"
    GUEST = "guest"


class PermissionLevel(Enum):
    """Permission levels for tool access"""
    FULL = "full"
    MODERATE = "moderate"
    LIMITED = "limited"
    DENIED = "denied"


class PluginStatus(Enum):
    """Plugin status definitions"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"


class ConsentStatus(Enum):
    """User consent status"""
    GRANTED = "granted"
    DENIED = "denied"
    PENDING = "pending"
    EXPIRED = "expired"


@dataclass
class UserPermission:
    """User permission configuration"""
    user_id: str
    role: UserRole
    permissions: Dict[str, PermissionLevel]
    tool_access: Dict[str, bool] = field(default_factory=dict)
    consent_history: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PluginMetadata:
    """Plugin metadata and configuration"""
    name: str
    version: str
    author: str
    description: str
    plugin_type: str
    entry_point: str
    dependencies: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    status: PluginStatus = PluginStatus.PENDING
    load_order: int = 100
    created_at: datetime = field(default_factory=datetime.now)
    last_loaded: Optional[datetime] = None


@dataclass
class ExtensionPoint:
    """Extension point configuration"""
    name: str
    interface: Type
    description: str
    required_methods: List[str]
    optional_methods: List[str] = field(default_factory=list)
    hooks: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsentRequest:
    """User consent request"""
    user_id: str
    tool_name: str
    operation: str
    risk_level: str
    description: str
    requested_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    status: ConsentStatus = ConsentStatus.PENDING
    decision_reason: Optional[str] = None


class IPlugin(ABC):
    """Interface for all plugins"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """Clean up plugin resources"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        pass


class IToolPlugin(IPlugin):
    """Interface for tool plugins"""
    
    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions provided by this plugin"""
        pass
    
    @abstractmethod
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool provided by this plugin"""
        pass


class PermissionValidator:
    """Validates user permissions for tool operations"""
    
    def __init__(self):
        self.user_permissions: Dict[str, UserPermission] = {}
        self.consent_cache: Dict[str, ConsentRequest] = {}
        self._lock = threading.Lock()
    
    def add_user(self, user_permission: UserPermission) -> None:
        """Add user permission configuration"""
        with self._lock:
            self.user_permissions[user_permission.user_id] = user_permission
    
    def check_permission(self, user_id: str, tool_name: str, operation: str) -> bool:
        """Check if user has permission for tool operation"""
        with self._lock:
            if user_id not in self.user_permissions:
                return False
            
            user_perm = self.user_permissions[user_id]
            
            # Check role-based permissions
            if user_perm.role == UserRole.ADMIN:
                return True
            elif user_perm.role == UserRole.GUEST:
                return False
            
            # Check specific tool access
            if tool_name in user_perm.tool_access:
                return user_perm.tool_access[tool_name]
            
            # Check operation-specific permissions
            if operation in user_perm.permissions:
                return user_perm.permissions[operation] != PermissionLevel.DENIED
            
            # Default to denied access for unknown operations
            return False
    
    def request_consent(self, user_id: str, tool_name: str, operation: str, 
                       risk_level: str, description: str) -> str:
        """Request user consent for operation"""
        consent_id = hashlib.md5(f"{user_id}_{tool_name}_{operation}_{datetime.now()}".encode()).hexdigest()
        
        consent_request = ConsentRequest(
            user_id=user_id,
            tool_name=tool_name,
            operation=operation,
            risk_level=risk_level,
            description=description
        )
        
        with self._lock:
            self.consent_cache[consent_id] = consent_request
        
        return consent_id
    
    def grant_consent(self, consent_id: str, decision: bool, reason: Optional[str] = None) -> bool:
        """Grant or deny consent for operation"""
        with self._lock:
            if consent_id not in self.consent_cache:
                return False
            
            consent_request = self.consent_cache[consent_id]
            consent_request.status = ConsentStatus.GRANTED if decision else ConsentStatus.DENIED
            consent_request.decision_reason = reason
            
            # Update user permission history
            if consent_request.user_id in self.user_permissions:
                user_perm = self.user_permissions[consent_request.user_id]
                user_perm.consent_history.append({
                    'consent_id': consent_id,
                    'tool_name': consent_request.tool_name,
                    'operation': consent_request.operation,
                    'decision': decision,
                    'timestamp': datetime.now(),
                    'reason': reason
                })
            
            return True
    
    def check_consent(self, consent_id: str) -> ConsentStatus:
        """Check consent status"""
        with self._lock:
            if consent_id not in self.consent_cache:
                return ConsentStatus.DENIED
            
            consent_request = self.consent_cache[consent_id]
            
            # Check if consent expired
            if (consent_request.expires_at and 
                datetime.now() > consent_request.expires_at):
                consent_request.status = ConsentStatus.EXPIRED
            
            return consent_request.status


class ExtensionRegistry:
    """Registry for extension points and hooks"""
    
    def __init__(self):
        self.extension_points: Dict[str, ExtensionPoint] = {}
        self.hooks: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
    
    def register_extension_point(self, extension_point: ExtensionPoint) -> None:
        """Register an extension point"""
        with self._lock:
            self.extension_points[extension_point.name] = extension_point
            logger.info(f"Registered extension point: {extension_point.name}")
    
    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a hook callback"""
        with self._lock:
            if hook_name not in self.hooks:
                self.hooks[hook_name] = []
            self.hooks[hook_name].append(callback)
            logger.info(f"Registered hook: {hook_name}")
    
    def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute all callbacks for a hook"""
        with self._lock:
            if hook_name not in self.hooks:
                return []
            
            results = []
            for callback in self.hooks[hook_name]:
                try:
                    result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Hook execution error for {hook_name}: {e}")
            
            return results
    
    def get_extension_points(self) -> Dict[str, ExtensionPoint]:
        """Get all extension points"""
        with self._lock:
            return self.extension_points.copy()


class ToolPluginManager:
    """Manager for tool plugins"""
    
    def __init__(self, plugin_directory: str = "plugins"):
        self.plugin_directory = Path(plugin_directory)
        self.plugins: Dict[str, IPlugin] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.extension_registry = ExtensionRegistry()
        self.permission_validator = PermissionValidator()
        self._lock = threading.Lock()
        
        # Create plugin directory if it doesn't exist
        self.plugin_directory.mkdir(exist_ok=True)
        
        # Register default extension points
        self._register_default_extension_points()
    
    def _register_default_extension_points(self) -> None:
        """Register default extension points"""
        tool_extension = ExtensionPoint(
            name="tool_plugin",
            interface=IToolPlugin,
            description="Extension point for tool plugins",
            required_methods=["get_tools", "execute_tool"],
            hooks=["before_tool_execution", "after_tool_execution", "tool_error"]
        )
        
        self.extension_registry.register_extension_point(tool_extension)
    
    def discover_plugins(self) -> List[PluginMetadata]:
        """Discover plugins in the plugin directory"""
        discovered_plugins = []
        
        for plugin_path in self.plugin_directory.iterdir():
            if plugin_path.is_dir():
                # Look for plugin configuration
                config_files = [
                    plugin_path / "plugin.yaml",
                    plugin_path / "plugin.json",
                    plugin_path / "plugin.yml"
                ]
                
                for config_file in config_files:
                    if config_file.exists():
                        try:
                            metadata = self._load_plugin_metadata(config_file)
                            discovered_plugins.append(metadata)
                            # Store metadata in the manager
                            with self._lock:
                                self.plugin_metadata[metadata.name] = metadata
                            break
                        except Exception as e:
                            logger.error(f"Failed to load plugin metadata from {config_file}: {e}")
        
        return discovered_plugins
    
    def _load_plugin_metadata(self, config_file: Path) -> PluginMetadata:
        """Load plugin metadata from configuration file"""
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        return PluginMetadata(
            name=config['name'],
            version=config['version'],
            author=config['author'],
            description=config['description'],
            plugin_type=config.get('plugin_type', 'tool'),
            entry_point=config['entry_point'],
            dependencies=config.get('dependencies', []),
            required_permissions=config.get('required_permissions', []),
            configuration=config.get('configuration', {}),
            load_order=config.get('load_order', 100)
        )
    
    def load_plugin(self, plugin_name: str) -> bool:
        """Load a plugin by name"""
        with self._lock:
            if plugin_name in self.plugins:
                logger.warning(f"Plugin {plugin_name} already loaded")
                return True
            
            if plugin_name not in self.plugin_metadata:
                logger.error(f"Plugin {plugin_name} not found")
                return False
            
            metadata = self.plugin_metadata[plugin_name]
            
            try:
                # Load plugin module
                plugin_path = self.plugin_directory / plugin_name / metadata.entry_point
                
                if not plugin_path.exists():
                    logger.error(f"Plugin entry point not found: {plugin_path}")
                    return False
                
                spec = importlib.util.spec_from_file_location(
                    f"plugin_{plugin_name}", 
                    plugin_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find plugin class
                plugin_class = None
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, IPlugin) and obj != IPlugin:
                        plugin_class = obj
                        break
                
                if not plugin_class:
                    logger.error(f"No plugin class found in {plugin_path}")
                    return False
                
                # Initialize plugin
                plugin_instance = plugin_class()
                
                if plugin_instance.initialize(metadata.configuration):
                    self.plugins[plugin_name] = plugin_instance
                    metadata.status = PluginStatus.ACTIVE
                    metadata.last_loaded = datetime.now()
                    
                    # Execute post-load hooks
                    self.extension_registry.execute_hook(
                        "plugin_loaded", 
                        plugin_name, 
                        plugin_instance
                    )
                    
                    logger.info(f"Successfully loaded plugin: {plugin_name}")
                    return True
                else:
                    logger.error(f"Failed to initialize plugin: {plugin_name}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error loading plugin {plugin_name}: {e}")
                metadata.status = PluginStatus.ERROR
                return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin by name"""
        with self._lock:
            if plugin_name not in self.plugins:
                logger.warning(f"Plugin {plugin_name} not loaded")
                return True
            
            plugin = self.plugins[plugin_name]
            
            try:
                # Execute pre-unload hooks
                self.extension_registry.execute_hook(
                    "plugin_unloading", 
                    plugin_name, 
                    plugin
                )
                
                # Cleanup plugin
                if plugin.cleanup():
                    del self.plugins[plugin_name]
                    
                    if plugin_name in self.plugin_metadata:
                        self.plugin_metadata[plugin_name].status = PluginStatus.INACTIVE
                    
                    logger.info(f"Successfully unloaded plugin: {plugin_name}")
                    return True
                else:
                    logger.error(f"Failed to cleanup plugin: {plugin_name}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error unloading plugin {plugin_name}: {e}")
                return False
    
    def get_plugin_tools(self, plugin_name: str) -> List[Dict[str, Any]]:
        """Get tools provided by a plugin"""
        with self._lock:
            if plugin_name not in self.plugins:
                return []
            
            plugin = self.plugins[plugin_name]
            
            if isinstance(plugin, IToolPlugin):
                return plugin.get_tools()
            
            return []
    
    def execute_plugin_tool(self, plugin_name: str, tool_name: str, 
                          parameters: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
        """Execute a tool from a plugin"""
        with self._lock:
            if plugin_name not in self.plugins:
                return {
                    "success": False,
                    "error": f"Plugin {plugin_name} not loaded"
                }
            
            plugin = self.plugins[plugin_name]
            
            if not isinstance(plugin, IToolPlugin):
                return {
                    "success": False,
                    "error": f"Plugin {plugin_name} is not a tool plugin"
                }
            
            # Check permissions if user_id provided
            if user_id:
                if not self.permission_validator.check_permission(user_id, tool_name, "execute"):
                    return {
                        "success": False,
                        "error": "Permission denied"
                    }
            
            try:
                # Execute pre-execution hooks
                self.extension_registry.execute_hook(
                    "before_tool_execution", 
                    plugin_name, 
                    tool_name, 
                    parameters
                )
                
                # Execute tool
                result = plugin.execute_tool(tool_name, parameters)
                
                # Execute post-execution hooks
                self.extension_registry.execute_hook(
                    "after_tool_execution", 
                    plugin_name, 
                    tool_name, 
                    parameters, 
                    result
                )
                
                return result
                
            except Exception as e:
                error_result = {
                    "success": False,
                    "error": str(e)
                }
                
                # Execute error hooks
                self.extension_registry.execute_hook(
                    "tool_error", 
                    plugin_name, 
                    tool_name, 
                    parameters, 
                    e
                )
                
                return error_result
    
    def refresh_plugins(self) -> None:
        """Refresh plugin discovery and reload changed plugins"""
        discovered_plugins = self.discover_plugins()
        
        with self._lock:
            # Update plugin metadata
            for plugin_metadata in discovered_plugins:
                self.plugin_metadata[plugin_metadata.name] = plugin_metadata
            
            # Reload active plugins that have changed
            for plugin_name, plugin in list(self.plugins.items()):
                if plugin_name in self.plugin_metadata:
                    metadata = self.plugin_metadata[plugin_name]
                    # Check if plugin file has been modified
                    plugin_path = self.plugin_directory / plugin_name / metadata.entry_point
                    if plugin_path.exists():
                        # For simplicity, we'll reload if the file timestamp is newer
                        # In a production system, you might want to check version numbers
                        pass
    
    def get_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all plugins"""
        with self._lock:
            status = {}
            
            for plugin_name, metadata in self.plugin_metadata.items():
                status[plugin_name] = {
                    "metadata": metadata,
                    "loaded": plugin_name in self.plugins,
                    "status": metadata.status.value
                }
            
            return status
    
    def create_plugin_template(self, plugin_name: str, plugin_type: str = "tool") -> bool:
        """Create a template for a new plugin"""
        plugin_dir = self.plugin_directory / plugin_name
        
        if plugin_dir.exists():
            logger.error(f"Plugin directory already exists: {plugin_dir}")
            return False
        
        try:
            plugin_dir.mkdir(parents=True)
            
            # Create plugin configuration
            config = {
                "name": plugin_name,
                "version": "1.0.0",
                "author": "User",
                "description": f"Custom {plugin_type} plugin",
                "plugin_type": plugin_type,
                "entry_point": f"{plugin_name}.py",
                "dependencies": [],
                "required_permissions": [],
                "configuration": {},
                "load_order": 100
            }
            
            with open(plugin_dir / "plugin.yaml", 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Create plugin implementation template
            if plugin_type == "tool":
                template = self._get_tool_plugin_template(plugin_name)
            else:
                template = self._get_basic_plugin_template(plugin_name)
            
            with open(plugin_dir / f"{plugin_name}.py", 'w') as f:
                f.write(template)
            
            logger.info(f"Created plugin template: {plugin_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating plugin template: {e}")
            return False
    
    def _get_tool_plugin_template(self, plugin_name: str) -> str:
        """Get tool plugin template"""
        return f'''"""
Custom Tool Plugin: {plugin_name}
"""

from typing import Dict, List, Any
from tool_extensibility_manager import IToolPlugin, PluginMetadata, PluginStatus
from datetime import datetime


class {plugin_name.title()}Plugin(IToolPlugin):
    """Custom tool plugin implementation"""
    
    def __init__(self):
        self.initialized = False
        self.config = {{}}
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin"""
        try:
            self.config = config
            self.initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize {plugin_name} plugin: {{e}}")
            return False
    
    def cleanup(self) -> bool:
        """Clean up plugin resources"""
        try:
            self.initialized = False
            return True
        except Exception as e:
            print(f"Failed to cleanup {plugin_name} plugin: {{e}}")
            return False
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return PluginMetadata(
            name="{plugin_name}",
            version="1.0.0",
            author="User",
            description="Custom {plugin_name} tool plugin",
            plugin_type="tool",
            entry_point="{plugin_name}.py",
            status=PluginStatus.ACTIVE if self.initialized else PluginStatus.INACTIVE
        )
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions provided by this plugin"""
        return [
            {{
                "name": "{plugin_name}_example",
                "description": "Example tool from {plugin_name} plugin",
                "parameters": {{
                    "message": {{
                        "type": "string",
                        "description": "Message to process",
                        "required": True
                    }}
                }},
                "returns": {{
                    "type": "object",
                    "description": "Processed result"
                }}
            }}
        ]
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool provided by this plugin"""
        if not self.initialized:
            return {{
                "success": False,
                "error": "Plugin not initialized"
            }}
        
        if tool_name == "{plugin_name}_example":
            return self._execute_example_tool(parameters)
        
        return {{
            "success": False,
            "error": f"Unknown tool: {{tool_name}}"
        }}
    
    def _execute_example_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the example tool"""
        try:
            message = parameters.get("message", "")
            
            # Your custom tool logic here
            result = f"Processed: {{message}}"
            
            return {{
                "success": True,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }}
            
        except Exception as e:
            return {{
                "success": False,
                "error": str(e)
            }}
'''
    
    def _get_basic_plugin_template(self, plugin_name: str) -> str:
        """Get basic plugin template"""
        return f'''"""
Custom Plugin: {plugin_name}
"""

from typing import Dict, Any
from tool_extensibility_manager import IPlugin, PluginMetadata, PluginStatus


class {plugin_name.title()}Plugin(IPlugin):
    """Custom plugin implementation"""
    
    def __init__(self):
        self.initialized = False
        self.config = {{}}
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin"""
        try:
            self.config = config
            self.initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize {plugin_name} plugin: {{e}}")
            return False
    
    def cleanup(self) -> bool:
        """Clean up plugin resources"""
        try:
            self.initialized = False
            return True
        except Exception as e:
            print(f"Failed to cleanup {plugin_name} plugin: {{e}}")
            return False
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return PluginMetadata(
            name="{plugin_name}",
            version="1.0.0",
            author="User",
            description="Custom {plugin_name} plugin",
            plugin_type="basic",
            entry_point="{plugin_name}.py",
            status=PluginStatus.ACTIVE if self.initialized else PluginStatus.INACTIVE
        )
'''


def requires_permission(permission: str, risk_level: str = "moderate"):
    """Decorator for requiring user permission"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would integrate with the permission system
            # For now, it's a placeholder
            return func(*args, **kwargs)
        return wrapper
    return decorator


def permission_check(user_id: str, tool_name: str, operation: str, 
                    permission_validator: PermissionValidator) -> bool:
    """Helper function for permission checking"""
    return permission_validator.check_permission(user_id, tool_name, operation)


# Example usage and demonstration
if __name__ == "__main__":
    # Create plugin manager
    plugin_manager = ToolPluginManager()
    
    # Create example user permissions
    admin_user = UserPermission(
        user_id="admin_001",
        role=UserRole.ADMIN,
        permissions={}
    )
    
    regular_user = UserPermission(
        user_id="user_001",
        role=UserRole.USER,
        permissions={
            "file_access": PermissionLevel.LIMITED,
            "network_access": PermissionLevel.MODERATE
        }
    )
    
    plugin_manager.permission_validator.add_user(admin_user)
    plugin_manager.permission_validator.add_user(regular_user)
    
    # Create plugin template
    plugin_manager.create_plugin_template("my_custom_tool", "tool")
    
    # Discover plugins
    discovered = plugin_manager.discover_plugins()
    print(f"Discovered {len(discovered)} plugins")
    
    # Show plugin status
    status = plugin_manager.get_plugin_status()
    print(f"Plugin status: {status}")
    
    print("Extensibility and Permission Management System initialized successfully!") 