"""
Tool Discovery Engine - Advanced tool registration and discovery capabilities

This module provides sophisticated tool discovery mechanisms including capability-based
search, intent matching, plugin loading, and dependency resolution for the Sovereign AI
tool integration framework.
"""

import asyncio
import logging
import json
import os
import re
import importlib
import importlib.util
import inspect
import sys
from typing import Dict, List, Optional, Any, Union, Callable, Type, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from collections import defaultdict
import hashlib

from .tool_integration_framework import (
    BaseTool, ToolRegistry, ToolMetadata, ToolParameter, ToolType, 
    ToolSecurityLevel, ToolExecutionMode, ToolException
)
from .config import Config


class DiscoveryMethod(Enum):
    """Methods for discovering tools"""
    NAME_EXACT = "name_exact"
    NAME_FUZZY = "name_fuzzy"
    CAPABILITY = "capability"
    TAG_BASED = "tag_based"
    INTENT_BASED = "intent_based"
    DEPENDENCY = "dependency"
    SEMANTIC = "semantic"


class ToolCapability(Enum):
    """Standard tool capabilities for discovery"""
    TEXT_PROCESSING = "text_processing"
    FILE_OPERATIONS = "file_operations"
    NETWORK_ACCESS = "network_access"
    DATA_ANALYSIS = "data_analysis"
    IMAGE_PROCESSING = "image_processing"
    SYSTEM_OPERATIONS = "system_operations"
    MATHEMATICAL = "mathematical"
    DATABASE_ACCESS = "database_access"
    API_INTEGRATION = "api_integration"
    AUTOMATION = "automation"
    MONITORING = "monitoring"
    SECURITY = "security"


@dataclass
class DiscoveryQuery:
    """Query for tool discovery"""
    query: str
    method: DiscoveryMethod = DiscoveryMethod.SEMANTIC
    capabilities: List[ToolCapability] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    tool_types: List[ToolType] = field(default_factory=list)
    security_levels: List[ToolSecurityLevel] = field(default_factory=list)
    max_results: int = 10
    include_deprecated: bool = False
    require_all_tags: bool = False
    sort_by: str = "relevance"  # relevance, popularity, recent, alphabetical


@dataclass
class DiscoveryResult:
    """Result from tool discovery"""
    tool: BaseTool
    relevance_score: float
    match_reason: str
    capabilities_matched: List[ToolCapability] = field(default_factory=list)
    tags_matched: List[str] = field(default_factory=list)


@dataclass
class PluginManifest:
    """Manifest for tool plugins"""
    name: str
    version: str
    description: str
    author: str
    entry_points: List[str]
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[ToolCapability] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    min_framework_version: str = "1.0.0"
    enabled: bool = True


class ToolRegistrationManager:
    """Advanced tool registration with validation and conflict resolution"""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.logger = logging.getLogger(__name__)
        
        # Registration tracking
        self.registration_history: List[Dict[str, Any]] = []
        self.validation_rules: Dict[str, Callable] = {}
        self.conflict_resolvers: Dict[str, Callable] = {}
        
        # Tool versioning
        self.tool_versions: Dict[str, List[str]] = defaultdict(list)
        self.active_versions: Dict[str, str] = {}
        
        self._setup_default_validation()
        self._setup_default_conflict_resolution()
    
    def register_tool_advanced(self, tool: BaseTool, 
                              replace_existing: bool = False,
                              validate_dependencies: bool = True) -> Tuple[bool, Optional[str]]:
        """Advanced tool registration with comprehensive validation"""
        try:
            tool_name = tool.metadata.name
            
            # Validation phase
            validation_result = self._validate_tool_registration(tool)
            if not validation_result[0]:
                return False, f"Validation failed: {validation_result[1]}"
            
            # Check for conflicts
            if tool_name in self.registry.tools and not replace_existing:
                conflict_resolution = self._resolve_tool_conflict(tool)
                if not conflict_resolution[0]:
                    return False, f"Conflict resolution failed: {conflict_resolution[1]}"
            
            # Dependency validation
            if validate_dependencies:
                dep_validation = self._validate_dependencies(tool)
                if not dep_validation[0]:
                    return False, f"Dependency validation failed: {dep_validation[1]}"
            
            # Version management
            self._manage_tool_version(tool)
            
            # Register the tool
            success = self.registry.register_tool(tool)
            if success:
                self._record_registration(tool, replace_existing)
                self.logger.info(f"Successfully registered tool: {tool_name} v{tool.metadata.version}")
                return True, None
            else:
                return False, "Registry registration failed"
                
        except Exception as e:
            self.logger.error(f"Tool registration failed: {e}")
            return False, str(e)
    
    def _validate_tool_registration(self, tool: BaseTool) -> Tuple[bool, Optional[str]]:
        """Validate tool registration requirements"""
        metadata = tool.metadata
        
        # Required field validation
        required_fields = ['name', 'description', 'version', 'tool_type', 'security_level']
        for field in required_fields:
            if not hasattr(metadata, field) or getattr(metadata, field) is None:
                return False, f"Missing required field: {field}"
        
        # Name validation
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', metadata.name):
            return False, "Tool name must start with letter and contain only letters, numbers, underscores, and hyphens"
        
        # Version validation
        if not re.match(r'^\d+\.\d+\.\d+', metadata.version):
            return False, "Version must follow semantic versioning (e.g., 1.0.0)"
        
        # Parameter validation
        parameter_names = set()
        for param in metadata.parameters:
            if param.name in parameter_names:
                return False, f"Duplicate parameter name: {param.name}"
            parameter_names.add(param.name)
        
        # Custom validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                is_valid, error = rule_func(tool)
                if not is_valid:
                    return False, f"Custom validation '{rule_name}' failed: {error}"
            except Exception as e:
                return False, f"Validation rule '{rule_name}' error: {e}"
        
        return True, None
    
    def _resolve_tool_conflict(self, new_tool: BaseTool) -> Tuple[bool, Optional[str]]:
        """Resolve conflicts when registering tools with existing names"""
        existing_tool = self.registry.get_tool(new_tool.metadata.name)
        if not existing_tool:
            return True, None
        
        # Version comparison
        new_version = self._parse_version(new_tool.metadata.version)
        existing_version = self._parse_version(existing_tool.metadata.version)
        
        if new_version > existing_version:
            self.logger.info(f"Upgrading tool {new_tool.metadata.name} from v{existing_tool.metadata.version} to v{new_tool.metadata.version}")
            return True, None
        elif new_version == existing_version:
            return False, "Tool with same version already exists"
        else:
            return False, f"Newer version already exists (existing: {existing_tool.metadata.version}, new: {new_tool.metadata.version})"
    
    def _validate_dependencies(self, tool: BaseTool) -> Tuple[bool, Optional[str]]:
        """Validate tool dependencies"""
        for dep_name in tool.metadata.dependencies:
            if dep_name not in self.registry.tools:
                return False, f"Missing dependency: {dep_name}"
        
        # Check for circular dependencies
        if self._has_circular_dependency(tool.metadata.name, tool.metadata.dependencies):
            return False, "Circular dependency detected"
        
        return True, None
    
    def _has_circular_dependency(self, tool_name: str, dependencies: List[str], visited: Set[str] = None) -> bool:
        """Check for circular dependencies"""
        if visited is None:
            visited = set()
        
        if tool_name in visited:
            return True
        
        visited.add(tool_name)
        
        for dep_name in dependencies:
            dep_tool = self.registry.get_tool(dep_name)
            if dep_tool and self._has_circular_dependency(dep_name, dep_tool.metadata.dependencies, visited.copy()):
                return True
        
        return False
    
    def _manage_tool_version(self, tool: BaseTool):
        """Manage tool versioning"""
        tool_name = tool.metadata.name
        version = tool.metadata.version
        
        if tool_name not in self.tool_versions:
            self.tool_versions[tool_name] = []
        
        if version not in self.tool_versions[tool_name]:
            self.tool_versions[tool_name].append(version)
            self.tool_versions[tool_name].sort(key=self._parse_version)
        
        self.active_versions[tool_name] = version
    
    def _parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse semantic version string"""
        parts = version.split('.')
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    
    def _record_registration(self, tool: BaseTool, replaced: bool):
        """Record tool registration for audit purposes"""
        record = {
            'tool_name': tool.metadata.name,
            'version': tool.metadata.version,
            'timestamp': datetime.now().isoformat(),
            'replaced_existing': replaced,
            'tool_type': tool.metadata.tool_type.value,
            'security_level': tool.metadata.security_level.value
        }
        self.registration_history.append(record)
        
        # Keep history limited
        if len(self.registration_history) > 1000:
            self.registration_history = self.registration_history[-500:]
    
    def _setup_default_validation(self):
        """Setup default validation rules"""
        self.validation_rules['security_consistency'] = self._validate_security_consistency
        self.validation_rules['parameter_types'] = self._validate_parameter_types
    
    def _setup_default_conflict_resolution(self):
        """Setup default conflict resolution strategies"""
        pass
    
    def _validate_security_consistency(self, tool: BaseTool) -> Tuple[bool, Optional[str]]:
        """Validate security level consistency"""
        metadata = tool.metadata
        
        # High-security tools should require consent
        if metadata.security_level in [ToolSecurityLevel.ELEVATED] and not metadata.requires_consent:
            return False, "Elevated security tools must require consent"
        
        # System tools should have appropriate security level
        if metadata.tool_type == ToolType.SYSTEM and metadata.security_level == ToolSecurityLevel.SAFE:
            return False, "System tools should not have SAFE security level"
        
        return True, None
    
    def _validate_parameter_types(self, tool: BaseTool) -> Tuple[bool, Optional[str]]:
        """Validate parameter type definitions"""
        valid_types = ['string', 'integer', 'number', 'boolean', 'array', 'object']
        
        for param in tool.metadata.parameters:
            if param.type not in valid_types:
                return False, f"Invalid parameter type '{param.type}' for parameter '{param.name}'"
        
        return True, None
    
    def get_registration_history(self, tool_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get registration history"""
        if tool_name:
            return [record for record in self.registration_history if record['tool_name'] == tool_name]
        return self.registration_history.copy()
    
    def get_tool_versions(self, tool_name: str) -> List[str]:
        """Get all versions of a tool"""
        return self.tool_versions.get(tool_name, [])


class ToolDiscoveryEngine:
    """Advanced tool discovery with intelligent search capabilities"""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.logger = logging.getLogger(__name__)
        
        # Discovery state
        self.search_index: Dict[str, Set[str]] = defaultdict(set)
        self.capability_map: Dict[ToolCapability, Set[str]] = defaultdict(set)
        self.usage_stats: Dict[str, int] = defaultdict(int)
        
        # Search algorithms
        self.search_algorithms = {
            DiscoveryMethod.NAME_EXACT: self._search_by_name_exact,
            DiscoveryMethod.NAME_FUZZY: self._search_by_name_fuzzy,
            DiscoveryMethod.CAPABILITY: self._search_by_capability,
            DiscoveryMethod.TAG_BASED: self._search_by_tags,
            DiscoveryMethod.INTENT_BASED: self._search_by_intent,
            DiscoveryMethod.SEMANTIC: self._search_semantic
        }
        
        self._build_search_index()
    
    def discover_tools(self, query: DiscoveryQuery) -> List[DiscoveryResult]:
        """Discover tools using the specified query"""
        try:
            # Get search algorithm
            search_func = self.search_algorithms.get(query.method)
            if not search_func:
                raise ValueError(f"Unknown discovery method: {query.method}")
            
            # Perform search
            results = search_func(query)
            
            # Apply filters
            results = self._apply_filters(results, query)
            
            # Sort results
            results = self._sort_results(results, query.sort_by)
            
            # Limit results
            results = results[:query.max_results]
            
            # Update usage statistics
            for result in results:
                self.usage_stats[result.tool.metadata.name] += 1
            
            self.logger.info(f"Discovery query returned {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Tool discovery failed: {e}")
            return []
    
    def _search_by_name_exact(self, query: DiscoveryQuery) -> List[DiscoveryResult]:
        """Search for tools by exact name match"""
        results = []
        tool = self.registry.get_tool(query.query)
        if tool:
            results.append(DiscoveryResult(
                tool=tool,
                relevance_score=1.0,
                match_reason="Exact name match"
            ))
        return results
    
    def _search_by_name_fuzzy(self, query: DiscoveryQuery) -> List[DiscoveryResult]:
        """Search for tools by fuzzy name matching"""
        results = []
        query_lower = query.query.lower()
        
        for tool in self.registry.tools.values():
            tool_name = tool.metadata.name.lower()
            
            # Calculate similarity score
            score = self._calculate_string_similarity(query_lower, tool_name)
            if score > 0.3:  # Minimum similarity threshold
                results.append(DiscoveryResult(
                    tool=tool,
                    relevance_score=score,
                    match_reason=f"Fuzzy name match (score: {score:.2f})"
                ))
        
        return results
    
    def _search_by_capability(self, query: DiscoveryQuery) -> List[DiscoveryResult]:
        """Search for tools by capability"""
        results = []
        
        for capability in query.capabilities:
            tool_names = self.capability_map.get(capability, set())
            for tool_name in tool_names:
                tool = self.registry.get_tool(tool_name)
                if tool:
                    results.append(DiscoveryResult(
                        tool=tool,
                        relevance_score=0.9,
                        match_reason=f"Capability match: {capability.value}",
                        capabilities_matched=[capability]
                    ))
        
        return results
    
    def _search_by_tags(self, query: DiscoveryQuery) -> List[DiscoveryResult]:
        """Search for tools by tags"""
        results = []
        
        for tool in self.registry.tools.values():
            matched_tags = set(query.tags) & set(tool.metadata.tags)
            
            if matched_tags:
                if query.require_all_tags and len(matched_tags) != len(query.tags):
                    continue
                
                score = len(matched_tags) / len(query.tags) if query.tags else 0
                results.append(DiscoveryResult(
                    tool=tool,
                    relevance_score=score,
                    match_reason=f"Tag match: {', '.join(matched_tags)}",
                    tags_matched=list(matched_tags)
                ))
        
        return results
    
    def _search_by_intent(self, query: DiscoveryQuery) -> List[DiscoveryResult]:
        """Search for tools by intent/description matching"""
        results = []
        query_words = set(query.query.lower().split())
        
        for tool in self.registry.tools.values():
            description_words = set(tool.metadata.description.lower().split())
            
            # Calculate word overlap
            overlap = query_words & description_words
            if overlap:
                score = len(overlap) / len(query_words)
                results.append(DiscoveryResult(
                    tool=tool,
                    relevance_score=score,
                    match_reason=f"Intent match: {', '.join(overlap)}"
                ))
        
        return results
    
    def _search_semantic(self, query: DiscoveryQuery) -> List[DiscoveryResult]:
        """Semantic search combining multiple methods"""
        all_results = []
        
        # Combine multiple search methods
        methods = [
            DiscoveryMethod.NAME_FUZZY,
            DiscoveryMethod.TAG_BASED,
            DiscoveryMethod.INTENT_BASED
        ]
        
        for method in methods:
            method_query = DiscoveryQuery(
                query=query.query,
                method=method,
                tags=query.tags,
                capabilities=query.capabilities
            )
            method_results = self.search_algorithms[method](method_query)
            all_results.extend(method_results)
        
        # Merge and deduplicate results
        tool_scores = defaultdict(list)
        for result in all_results:
            tool_scores[result.tool.metadata.name].append(result)
        
        merged_results = []
        for tool_name, results in tool_scores.items():
            # Average the scores from different methods
            avg_score = sum(r.relevance_score for r in results) / len(results)
            best_result = max(results, key=lambda r: r.relevance_score)
            best_result.relevance_score = avg_score
            best_result.match_reason = f"Semantic match ({len(results)} methods)"
            merged_results.append(best_result)
        
        return merged_results
    
    def _apply_filters(self, results: List[DiscoveryResult], query: DiscoveryQuery) -> List[DiscoveryResult]:
        """Apply filters to discovery results"""
        filtered = results
        
        # Tool type filter
        if query.tool_types:
            filtered = [r for r in filtered if r.tool.metadata.tool_type in query.tool_types]
        
        # Security level filter
        if query.security_levels:
            filtered = [r for r in filtered if r.tool.metadata.security_level in query.security_levels]
        
        # Deprecated filter
        if not query.include_deprecated:
            # Assuming tools have a 'deprecated' field or similar
            filtered = [r for r in filtered if not getattr(r.tool.metadata, 'deprecated', False)]
        
        return filtered
    
    def _sort_results(self, results: List[DiscoveryResult], sort_by: str) -> List[DiscoveryResult]:
        """Sort discovery results"""
        if sort_by == "relevance":
            return sorted(results, key=lambda r: r.relevance_score, reverse=True)
        elif sort_by == "popularity":
            return sorted(results, key=lambda r: self.usage_stats[r.tool.metadata.name], reverse=True)
        elif sort_by == "recent":
            return sorted(results, key=lambda r: r.tool.metadata.last_updated, reverse=True)
        elif sort_by == "alphabetical":
            return sorted(results, key=lambda r: r.tool.metadata.name)
        else:
            return results
    
    def _calculate_string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using simple algorithm"""
        if s1 == s2:
            return 1.0
        
        # Simple character-based similarity
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
        
        # Count common characters
        s1_chars = set(s1)
        s2_chars = set(s2)
        common = s1_chars & s2_chars
        
        return len(common) / max(len(s1_chars), len(s2_chars))
    
    def _build_search_index(self):
        """Build search index for fast discovery"""
        for tool in self.registry.tools.values():
            tool_name = tool.metadata.name
            
            # Index by words in name and description
            words = (tool.metadata.name + " " + tool.metadata.description).lower().split()
            for word in words:
                self.search_index[word].add(tool_name)
            
            # Index by tags
            for tag in tool.metadata.tags:
                self.search_index[tag.lower()].add(tool_name)
            
            # Index by capabilities (inferred from tool type and tags)
            capabilities = self._infer_capabilities(tool.metadata)
            for capability in capabilities:
                self.capability_map[capability].add(tool_name)
    
    def _infer_capabilities(self, metadata: ToolMetadata) -> List[ToolCapability]:
        """Infer tool capabilities from metadata"""
        capabilities = []
        
        # Infer from tool type
        type_capability_map = {
            ToolType.API: [ToolCapability.API_INTEGRATION, ToolCapability.NETWORK_ACCESS],
            ToolType.FILE: [ToolCapability.FILE_OPERATIONS],
            ToolType.SYSTEM: [ToolCapability.SYSTEM_OPERATIONS],
            ToolType.DATABASE: [ToolCapability.DATABASE_ACCESS],
            ToolType.PYTHON: [ToolCapability.DATA_ANALYSIS]
        }
        
        if metadata.tool_type in type_capability_map:
            capabilities.extend(type_capability_map[metadata.tool_type])
        
        # Infer from tags
        tag_capability_map = {
            'math': ToolCapability.MATHEMATICAL,
            'calculation': ToolCapability.MATHEMATICAL,
            'image': ToolCapability.IMAGE_PROCESSING,
            'text': ToolCapability.TEXT_PROCESSING,
            'network': ToolCapability.NETWORK_ACCESS,
            'file': ToolCapability.FILE_OPERATIONS,
            'security': ToolCapability.SECURITY,
            'monitor': ToolCapability.MONITORING,
            'automation': ToolCapability.AUTOMATION
        }
        
        for tag in metadata.tags:
            tag_lower = tag.lower()
            for tag_keyword, capability in tag_capability_map.items():
                if tag_keyword in tag_lower:
                    capabilities.append(capability)
        
        return list(set(capabilities))  # Remove duplicates
    
    def rebuild_index(self):
        """Rebuild the search index"""
        self.search_index.clear()
        self.capability_map.clear()
        self._build_search_index()
        self.logger.info("Search index rebuilt")
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery engine statistics"""
        return {
            'indexed_tools': len(self.registry.tools),
            'search_terms': len(self.search_index),
            'capabilities_mapped': len(self.capability_map),
            'usage_stats': dict(self.usage_stats),
            'top_tools': sorted(self.usage_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        }


class PluginLoader:
    """Plugin loading system for dynamic tool discovery"""
    
    def __init__(self, registration_manager: ToolRegistrationManager):
        self.registration_manager = registration_manager
        self.logger = logging.getLogger(__name__)
        
        # Plugin state
        self.plugin_directories: List[Path] = []
        self.loaded_plugins: Dict[str, PluginManifest] = {}
        self.failed_plugins: Dict[str, str] = {}
    
    def add_plugin_directory(self, directory: Union[str, Path]):
        """Add a directory to search for plugins"""
        path = Path(directory)
        if path.exists() and path.is_dir():
            self.plugin_directories.append(path)
            self.logger.info(f"Added plugin directory: {path}")
        else:
            self.logger.warning(f"Plugin directory does not exist: {path}")
    
    async def discover_and_load_plugins(self) -> Dict[str, Any]:
        """Discover and load all plugins from registered directories"""
        discovered = 0
        loaded = 0
        failed = 0
        
        for directory in self.plugin_directories:
            self.logger.info(f"Discovering plugins in: {directory}")
            
            # Look for plugin manifests
            manifest_files = list(directory.glob("**/plugin.yaml")) + list(directory.glob("**/plugin.yml"))
            
            for manifest_file in manifest_files:
                discovered += 1
                try:
                    success = await self._load_plugin_from_manifest(manifest_file)
                    if success:
                        loaded += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    self.failed_plugins[str(manifest_file)] = str(e)
                    self.logger.error(f"Failed to load plugin {manifest_file}: {e}")
        
        result = {
            'discovered': discovered,
            'loaded': loaded,
            'failed': failed,
            'loaded_plugins': list(self.loaded_plugins.keys()),
            'failed_plugins': dict(self.failed_plugins)
        }
        
        self.logger.info(f"Plugin discovery complete: {result}")
        return result
    
    async def _load_plugin_from_manifest(self, manifest_file: Path) -> bool:
        """Load a plugin from its manifest file"""
        try:
            with open(manifest_file, 'r') as f:
                manifest_data = yaml.safe_load(f)
            
            manifest = PluginManifest(**manifest_data)
            
            if not manifest.enabled:
                self.logger.info(f"Plugin {manifest.name} is disabled, skipping")
                return True
            
            # Load entry points
            plugin_dir = manifest_file.parent
            for entry_point in manifest.entry_points:
                module_path = plugin_dir / entry_point
                if module_path.suffix == '.py':
                    success = await self._load_python_module(module_path, manifest)
                    if not success:
                        return False
            
            self.loaded_plugins[manifest.name] = manifest
            self.logger.info(f"Successfully loaded plugin: {manifest.name} v{manifest.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin manifest {manifest_file}: {e}")
            return False
    
    async def _load_python_module(self, module_path: Path, manifest: PluginManifest) -> bool:
        """Load tools from a Python module"""
        try:
            # Add plugin directory to Python path temporarily
            plugin_dir = str(module_path.parent)
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            
            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for tool classes
                tools_found = 0
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseTool) and 
                        obj != BaseTool):
                        
                        # Instantiate and register the tool
                        try:
                            tool_instance = obj()
                            success, error = self.registration_manager.register_tool_advanced(tool_instance)
                            if success:
                                tools_found += 1
                                self.logger.info(f"Registered tool from plugin: {tool_instance.metadata.name}")
                            else:
                                self.logger.error(f"Failed to register tool {tool_instance.metadata.name}: {error}")
                        except Exception as e:
                            self.logger.error(f"Failed to instantiate tool {name}: {e}")
                
                self.logger.info(f"Loaded {tools_found} tools from {module_path}")
                return tools_found > 0
                
            finally:
                # Remove from path
                if plugin_dir in sys.path:
                    sys.path.remove(plugin_dir)
                    
        except Exception as e:
            self.logger.error(f"Failed to load Python module {module_path}: {e}")
            return False
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginManifest]:
        """Get information about a loaded plugin"""
        return self.loaded_plugins.get(plugin_name)
    
    def list_plugins(self) -> Dict[str, PluginManifest]:
        """List all loaded plugins"""
        return self.loaded_plugins.copy()
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin and its tools"""
        if plugin_name not in self.loaded_plugins:
            return False
        
        # This would require tracking which tools came from which plugin
        # For now, just remove from loaded plugins
        del self.loaded_plugins[plugin_name]
        self.logger.info(f"Unloaded plugin: {plugin_name}")
        return True 