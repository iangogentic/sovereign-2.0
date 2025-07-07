"""
Tests for Tool Discovery Engine

This module contains comprehensive tests for the advanced tool discovery engine,
including registration validation, sophisticated search capabilities, and plugin loading.
"""

import pytest
import asyncio
import tempfile
import yaml
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
from typing import Dict, Any, List

from src.sovereign.tool_discovery_engine import (
    ToolRegistrationManager, ToolDiscoveryEngine, PluginLoader,
    DiscoveryQuery, DiscoveryMethod, ToolCapability, DiscoveryResult,
    PluginManifest
)
from src.sovereign.tool_integration_framework import (
    ToolRegistry, BaseTool, ToolMetadata, ToolParameter, ToolType,
    ToolSecurityLevel, ToolExecutionMode
)
from src.sovereign.tools.example_tools import create_example_tools
from src.sovereign.config import Config


class TestToolRegistrationManager:
    """Test the ToolRegistrationManager class"""
    
    @pytest.fixture
    def registry(self):
        """Create a tool registry"""
        return ToolRegistry()
    
    @pytest.fixture
    def registration_manager(self, registry):
        """Create a registration manager"""
        return ToolRegistrationManager(registry)
    
    @pytest.fixture
    def sample_tool(self):
        """Create a sample tool for testing"""
        class TestTool(BaseTool):
            def __init__(self):
                metadata = ToolMetadata(
                    name="test_tool",
                    description="A test tool for validation",
                    version="1.0.0",
                    tool_type=ToolType.PYTHON,
                    security_level=ToolSecurityLevel.SAFE,
                    execution_mode=ToolExecutionMode.SYNCHRONOUS,
                    parameters=[
                        ToolParameter(
                            name="input",
                            type="string",
                            description="Test input"
                        )
                    ],
                    return_type="string"
                )
                super().__init__(metadata)
            
            async def execute(self, parameters):
                return "test result"
            
            async def validate_parameters(self, parameters):
                return True, None
        
        return TestTool()
    
    def test_registration_manager_creation(self, registration_manager):
        """Test creating a registration manager"""
        assert registration_manager.registry is not None
        assert len(registration_manager.validation_rules) > 0
        assert len(registration_manager.registration_history) == 0
    
    def test_valid_tool_registration(self, registration_manager, sample_tool):
        """Test registering a valid tool"""
        success, error = registration_manager.register_tool_advanced(sample_tool)
        assert success is True
        assert error is None
        assert len(registration_manager.registration_history) == 1
    
    def test_duplicate_registration_without_replace(self, registration_manager, sample_tool):
        """Test registering duplicate tool without replace flag"""
        # Register first time
        success, error = registration_manager.register_tool_advanced(sample_tool)
        assert success is True
        
        # Try to register again without replace
        success, error = registration_manager.register_tool_advanced(sample_tool)
        assert success is False
        assert "same version already exists" in error.lower()
    
    def test_duplicate_registration_with_replace(self, registration_manager, sample_tool):
        """Test registering duplicate tool with replace flag"""
        # Register first time
        registration_manager.register_tool_advanced(sample_tool)
        
        # Register again with replace
        success, error = registration_manager.register_tool_advanced(sample_tool, replace_existing=True)
        assert success is True
        assert error is None
    
    def test_version_upgrade(self, registration_manager):
        """Test tool version upgrade"""
        # Create tool v1.0.0
        class TestTool_v1(BaseTool):
            def __init__(self):
                metadata = ToolMetadata(
                    name="upgrade_test",
                    description="Version upgrade test",
                    version="1.0.0",
                    tool_type=ToolType.PYTHON,
                    security_level=ToolSecurityLevel.SAFE,
                    execution_mode=ToolExecutionMode.SYNCHRONOUS,
                    parameters=[],
                    return_type="string"
                )
                super().__init__(metadata)
            
            async def execute(self, parameters):
                return "v1"
            
            async def validate_parameters(self, parameters):
                return True, None
        
        # Create tool v2.0.0
        class TestTool_v2(BaseTool):
            def __init__(self):
                metadata = ToolMetadata(
                    name="upgrade_test",
                    description="Version upgrade test",
                    version="2.0.0",
                    tool_type=ToolType.PYTHON,
                    security_level=ToolSecurityLevel.SAFE,
                    execution_mode=ToolExecutionMode.SYNCHRONOUS,
                    parameters=[],
                    return_type="string"
                )
                super().__init__(metadata)
            
            async def execute(self, parameters):
                return "v2"
            
            async def validate_parameters(self, parameters):
                return True, None
        
        # Register v1.0.0
        tool_v1 = TestTool_v1()
        success, error = registration_manager.register_tool_advanced(tool_v1)
        assert success is True
        
        # Register v2.0.0 (should upgrade)
        tool_v2 = TestTool_v2()
        success, error = registration_manager.register_tool_advanced(tool_v2)
        assert success is True
        assert error is None
        
        # Check version tracking
        versions = registration_manager.get_tool_versions("upgrade_test")
        assert "1.0.0" in versions
        assert "2.0.0" in versions
    
    def test_invalid_tool_name_validation(self, registration_manager):
        """Test validation of invalid tool names"""
        class InvalidNameTool(BaseTool):
            def __init__(self):
                metadata = ToolMetadata(
                    name="invalid-name!",  # Invalid characters
                    description="Invalid name test",
                    version="1.0.0",
                    tool_type=ToolType.PYTHON,
                    security_level=ToolSecurityLevel.SAFE,
                    execution_mode=ToolExecutionMode.SYNCHRONOUS,
                    parameters=[],
                    return_type="string"
                )
                super().__init__(metadata)
            
            async def execute(self, parameters):
                return "test"
            
            async def validate_parameters(self, parameters):
                return True, None
        
        invalid_tool = InvalidNameTool()
        success, error = registration_manager.register_tool_advanced(invalid_tool)
        assert success is False
        assert "name must start with letter" in error.lower()
    
    def test_invalid_version_validation(self, registration_manager):
        """Test validation of invalid version format"""
        class InvalidVersionTool(BaseTool):
            def __init__(self):
                metadata = ToolMetadata(
                    name="invalid_version_tool",
                    description="Invalid version test",
                    version="invalid.version",  # Invalid format
                    tool_type=ToolType.PYTHON,
                    security_level=ToolSecurityLevel.SAFE,
                    execution_mode=ToolExecutionMode.SYNCHRONOUS,
                    parameters=[],
                    return_type="string"
                )
                super().__init__(metadata)
            
            async def execute(self, parameters):
                return "test"
            
            async def validate_parameters(self, parameters):
                return True, None
        
        invalid_tool = InvalidVersionTool()
        success, error = registration_manager.register_tool_advanced(invalid_tool)
        assert success is False
        assert "semantic versioning" in error.lower()
    
    def test_security_consistency_validation(self, registration_manager):
        """Test security consistency validation"""
        class ElevatedTool(BaseTool):
            def __init__(self):
                metadata = ToolMetadata(
                    name="elevated_tool",
                    description="Elevated tool without consent",
                    version="1.0.0",
                    tool_type=ToolType.SYSTEM,
                    security_level=ToolSecurityLevel.ELEVATED,
                    execution_mode=ToolExecutionMode.SYNCHRONOUS,
                    parameters=[],
                    return_type="string",
                    requires_consent=False  # Should fail validation
                )
                super().__init__(metadata)
            
            async def execute(self, parameters):
                return "test"
            
            async def validate_parameters(self, parameters):
                return True, None
        
        elevated_tool = ElevatedTool()
        success, error = registration_manager.register_tool_advanced(elevated_tool)
        assert success is False
        assert "consent" in error.lower()
    
    def test_dependency_validation(self, registration_manager):
        """Test dependency validation"""
        # Create a tool with missing dependency
        class DependentTool(BaseTool):
            def __init__(self):
                metadata = ToolMetadata(
                    name="dependent_tool",
                    description="Tool with dependencies",
                    version="1.0.0",
                    tool_type=ToolType.PYTHON,
                    security_level=ToolSecurityLevel.SAFE,
                    execution_mode=ToolExecutionMode.SYNCHRONOUS,
                    parameters=[],
                    return_type="string",
                    dependencies=["nonexistent_tool"]
                )
                super().__init__(metadata)
            
            async def execute(self, parameters):
                return "test"
            
            async def validate_parameters(self, parameters):
                return True, None
        
        dependent_tool = DependentTool()
        success, error = registration_manager.register_tool_advanced(dependent_tool)
        assert success is False
        assert "missing dependency" in error.lower()
    
    def test_registration_history_tracking(self, registration_manager, sample_tool):
        """Test registration history tracking"""
        # Register a tool
        registration_manager.register_tool_advanced(sample_tool)
        
        # Check history
        history = registration_manager.get_registration_history()
        assert len(history) == 1
        assert history[0]['tool_name'] == "test_tool"
        assert history[0]['version'] == "1.0.0"
        
        # Check tool-specific history
        tool_history = registration_manager.get_registration_history("test_tool")
        assert len(tool_history) == 1


class TestToolDiscoveryEngine:
    """Test the ToolDiscoveryEngine class"""
    
    @pytest.fixture
    def registry(self):
        """Create a tool registry with example tools"""
        registry = ToolRegistry()
        tools = create_example_tools()
        for tool in tools:
            registry.register_tool(tool)
        return registry
    
    @pytest.fixture
    def discovery_engine(self, registry):
        """Create a discovery engine"""
        return ToolDiscoveryEngine(registry)
    
    def test_discovery_engine_creation(self, discovery_engine):
        """Test creating a discovery engine"""
        assert discovery_engine.registry is not None
        assert len(discovery_engine.search_index) > 0
        assert len(discovery_engine.capability_map) > 0
    
    def test_exact_name_search(self, discovery_engine):
        """Test exact name search"""
        query = DiscoveryQuery(
            query="echo",
            method=DiscoveryMethod.NAME_EXACT
        )
        
        results = discovery_engine.discover_tools(query)
        assert len(results) == 1
        assert results[0].tool.metadata.name == "echo"
        assert results[0].relevance_score == 1.0
    
    def test_fuzzy_name_search(self, discovery_engine):
        """Test fuzzy name search"""
        query = DiscoveryQuery(
            query="calc",
            method=DiscoveryMethod.NAME_FUZZY
        )
        
        results = discovery_engine.discover_tools(query)
        assert len(results) > 0
        # Should find calculator tool
        assert any(r.tool.metadata.name == "calculator" for r in results)
    
    def test_capability_search(self, discovery_engine):
        """Test capability-based search"""
        query = DiscoveryQuery(
            query="math",
            method=DiscoveryMethod.CAPABILITY,
            capabilities=[ToolCapability.MATHEMATICAL]
        )
        
        results = discovery_engine.discover_tools(query)
        assert len(results) > 0
        # Should find calculator tool
        math_tools = [r for r in results if ToolCapability.MATHEMATICAL in r.capabilities_matched]
        assert len(math_tools) > 0
    
    def test_tag_based_search(self, discovery_engine):
        """Test tag-based search"""
        query = DiscoveryQuery(
            query="search",
            method=DiscoveryMethod.TAG_BASED,
            tags=["utility", "test"]
        )
        
        results = discovery_engine.discover_tools(query)
        assert len(results) > 0
        # Should find tools with utility or test tags
        for result in results:
            assert any(tag in result.tool.metadata.tags for tag in ["utility", "test"])
    
    def test_intent_based_search(self, discovery_engine):
        """Test intent-based search"""
        query = DiscoveryQuery(
            query="mathematical operations",
            method=DiscoveryMethod.INTENT_BASED
        )
        
        results = discovery_engine.discover_tools(query)
        assert len(results) > 0
        # Should find calculator tool
        assert any(r.tool.metadata.name == "calculator" for r in results)
    
    def test_semantic_search(self, discovery_engine):
        """Test semantic search (combining multiple methods)"""
        query = DiscoveryQuery(
            query="echo message",
            method=DiscoveryMethod.SEMANTIC
        )
        
        results = discovery_engine.discover_tools(query)
        assert len(results) > 0
        # Should find echo tool with high relevance
        echo_results = [r for r in results if r.tool.metadata.name == "echo"]
        assert len(echo_results) > 0
    
    def test_search_filtering(self, discovery_engine):
        """Test search result filtering"""
        query = DiscoveryQuery(
            query="test",
            method=DiscoveryMethod.SEMANTIC,
            tool_types=[ToolType.PYTHON],
            security_levels=[ToolSecurityLevel.SAFE]
        )
        
        results = discovery_engine.discover_tools(query)
        
        # All results should match filters
        for result in results:
            assert result.tool.metadata.tool_type == ToolType.PYTHON
            assert result.tool.metadata.security_level == ToolSecurityLevel.SAFE
    
    def test_search_sorting(self, discovery_engine):
        """Test search result sorting"""
        query = DiscoveryQuery(
            query="tool",
            method=DiscoveryMethod.SEMANTIC,
            sort_by="alphabetical"
        )
        
        results = discovery_engine.discover_tools(query)
        
        # Results should be sorted alphabetically
        if len(results) > 1:
            tool_names = [r.tool.metadata.name for r in results]
            assert tool_names == sorted(tool_names)
    
    def test_max_results_limiting(self, discovery_engine):
        """Test limiting maximum results"""
        query = DiscoveryQuery(
            query="tool",
            method=DiscoveryMethod.SEMANTIC,
            max_results=2
        )
        
        results = discovery_engine.discover_tools(query)
        assert len(results) <= 2
    
    def test_capability_inference(self, discovery_engine):
        """Test capability inference from tool metadata"""
        # Check that capabilities are properly inferred
        stats = discovery_engine.get_discovery_stats()
        assert stats['capabilities_mapped'] > 0
        
        # Calculator should be mapped to MATHEMATICAL capability
        calc_tools = discovery_engine.capability_map.get(ToolCapability.MATHEMATICAL, set())
        assert "calculator" in calc_tools
    
    def test_usage_statistics_tracking(self, discovery_engine):
        """Test usage statistics tracking"""
        # Perform some searches
        query = DiscoveryQuery(query="echo", method=DiscoveryMethod.NAME_EXACT)
        discovery_engine.discover_tools(query)
        discovery_engine.discover_tools(query)
        
        # Check usage stats
        stats = discovery_engine.get_discovery_stats()
        assert stats['usage_stats']['echo'] == 2
    
    def test_index_rebuilding(self, discovery_engine):
        """Test search index rebuilding"""
        original_index_size = len(discovery_engine.search_index)
        
        # Rebuild index
        discovery_engine.rebuild_index()
        
        # Index should be rebuilt (same size in this case)
        assert len(discovery_engine.search_index) == original_index_size


class TestPluginLoader:
    """Test the PluginLoader class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def registry(self):
        """Create a tool registry"""
        return ToolRegistry()
    
    @pytest.fixture
    def registration_manager(self, registry):
        """Create a registration manager"""
        return ToolRegistrationManager(registry)
    
    @pytest.fixture
    def plugin_loader(self, registration_manager):
        """Create a plugin loader"""
        return PluginLoader(registration_manager)
    
    @pytest.fixture
    def sample_plugin_manifest(self):
        """Create a sample plugin manifest"""
        return {
            'name': 'test_plugin',
            'version': '1.0.0',
            'description': 'A test plugin',
            'author': 'Test Author',
            'entry_points': ['test_tools.py'],
            'dependencies': [],
            'capabilities': ['text_processing'],
            'tags': ['test', 'utility'],
            'enabled': True
        }
    
    @pytest.fixture
    def sample_plugin_tool(self):
        """Create a sample plugin tool code"""
        return '''
from src.sovereign.tool_integration_framework import BaseTool, ToolMetadata, ToolParameter, ToolType, ToolSecurityLevel, ToolExecutionMode

class PluginTestTool(BaseTool):
    def __init__(self):
        metadata = ToolMetadata(
            name="plugin_test_tool",
            description="A tool loaded from plugin",
            version="1.0.0",
            tool_type=ToolType.PYTHON,
            security_level=ToolSecurityLevel.SAFE,
            execution_mode=ToolExecutionMode.SYNCHRONOUS,
            parameters=[
                ToolParameter(
                    name="input",
                    type="string",
                    description="Test input"
                )
            ],
            return_type="string",
            tags=["plugin", "test"]
        )
        super().__init__(metadata)
    
    async def execute(self, parameters):
        return f"Plugin processed: {parameters['input']}"
    
    async def validate_parameters(self, parameters):
        if 'input' not in parameters:
            return False, "Missing input parameter"
        return True, None
'''
    
    def test_plugin_loader_creation(self, plugin_loader):
        """Test creating a plugin loader"""
        assert plugin_loader.registration_manager is not None
        assert len(plugin_loader.plugin_directories) == 0
        assert len(plugin_loader.loaded_plugins) == 0
    
    def test_add_plugin_directory(self, plugin_loader, temp_dir):
        """Test adding plugin directories"""
        plugin_loader.add_plugin_directory(temp_dir)
        assert temp_dir in plugin_loader.plugin_directories
    
    def test_plugin_manifest_creation(self, sample_plugin_manifest):
        """Test creating plugin manifest"""
        manifest = PluginManifest(**sample_plugin_manifest)
        assert manifest.name == "test_plugin"
        assert manifest.version == "1.0.0"
        assert manifest.enabled is True
    
    @pytest.mark.asyncio
    async def test_plugin_loading_disabled(self, plugin_loader, temp_dir, sample_plugin_manifest, sample_plugin_tool):
        """Test loading disabled plugin"""
        # Create plugin directory
        plugin_dir = temp_dir / "test_plugin"
        plugin_dir.mkdir()
        
        # Create disabled manifest
        disabled_manifest = sample_plugin_manifest.copy()
        disabled_manifest['enabled'] = False
        
        manifest_file = plugin_dir / "plugin.yaml"
        with open(manifest_file, 'w') as f:
            yaml.dump(disabled_manifest, f)
        
        # Create tool file
        tool_file = plugin_dir / "test_tools.py"
        with open(tool_file, 'w') as f:
            f.write(sample_plugin_tool)
        
        # Add plugin directory and discover
        plugin_loader.add_plugin_directory(temp_dir)
        result = await plugin_loader.discover_and_load_plugins()
        
        # Plugin should be discovered and marked as loaded (but tools won't be loaded due to disabled status)
        assert result['discovered'] == 1
        assert result['loaded'] == 1
        # Disabled plugins are not added to loaded_plugins dict
        assert len(plugin_loader.loaded_plugins) == 0
    
    @pytest.mark.asyncio
    async def test_plugin_loading_success(self, plugin_loader, temp_dir, sample_plugin_manifest, sample_plugin_tool):
        """Test successful plugin loading"""
        # Create plugin directory
        plugin_dir = temp_dir / "test_plugin"
        plugin_dir.mkdir()
        
        # Create manifest
        manifest_file = plugin_dir / "plugin.yaml"
        with open(manifest_file, 'w') as f:
            yaml.dump(sample_plugin_manifest, f)
        
        # Create tool file
        tool_file = plugin_dir / "test_tools.py"
        with open(tool_file, 'w') as f:
            f.write(sample_plugin_tool)
        
        # Add plugin directory and discover
        plugin_loader.add_plugin_directory(temp_dir)
        result = await plugin_loader.discover_and_load_plugins()
        
        # Plugin should be loaded successfully
        assert result['discovered'] == 1
        assert result['loaded'] == 1
        assert result['failed'] == 0
        assert "test_plugin" in plugin_loader.loaded_plugins
    
    @pytest.mark.asyncio
    async def test_plugin_loading_invalid_manifest(self, plugin_loader, temp_dir):
        """Test loading plugin with invalid manifest"""
        # Create plugin directory
        plugin_dir = temp_dir / "invalid_plugin"
        plugin_dir.mkdir()
        
        # Create invalid manifest (missing required fields)
        invalid_manifest = {'name': 'invalid_plugin'}
        
        manifest_file = plugin_dir / "plugin.yaml"
        with open(manifest_file, 'w') as f:
            yaml.dump(invalid_manifest, f)
        
        # Add plugin directory and discover
        plugin_loader.add_plugin_directory(temp_dir)
        result = await plugin_loader.discover_and_load_plugins()
        
        # Plugin should fail to load
        assert result['discovered'] == 1
        assert result['failed'] == 1
        # Manifest parsing fails early, so no plugin is tracked in failed_plugins
        # The failure is counted in the result though
    
    def test_plugin_info_retrieval(self, plugin_loader):
        """Test retrieving plugin information"""
        # Create mock plugin
        manifest = PluginManifest(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            entry_points=["test.py"]
        )
        plugin_loader.loaded_plugins["test_plugin"] = manifest
        
        # Test retrieval
        info = plugin_loader.get_plugin_info("test_plugin")
        assert info is not None
        assert info.name == "test_plugin"
        
        # Test non-existent plugin
        info = plugin_loader.get_plugin_info("nonexistent")
        assert info is None
    
    def test_plugin_listing(self, plugin_loader):
        """Test listing all plugins"""
        # Create mock plugins
        manifest1 = PluginManifest(name="plugin1", version="1.0.0", description="Plugin 1", author="Author", entry_points=["p1.py"])
        manifest2 = PluginManifest(name="plugin2", version="1.0.0", description="Plugin 2", author="Author", entry_points=["p2.py"])
        
        plugin_loader.loaded_plugins["plugin1"] = manifest1
        plugin_loader.loaded_plugins["plugin2"] = manifest2
        
        # List plugins
        plugins = plugin_loader.list_plugins()
        assert len(plugins) == 2
        assert "plugin1" in plugins
        assert "plugin2" in plugins
    
    def test_plugin_unloading(self, plugin_loader):
        """Test unloading plugins"""
        # Create mock plugin
        manifest = PluginManifest(name="test_plugin", version="1.0.0", description="Test", author="Author", entry_points=["test.py"])
        plugin_loader.loaded_plugins["test_plugin"] = manifest
        
        # Unload plugin
        success = plugin_loader.unload_plugin("test_plugin")
        assert success is True
        assert "test_plugin" not in plugin_loader.loaded_plugins
        
        # Try to unload non-existent plugin
        success = plugin_loader.unload_plugin("nonexistent")
        assert success is False


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios"""
    
    @pytest.fixture
    def full_system(self):
        """Create a complete discovery system"""
        registry = ToolRegistry()
        registration_manager = ToolRegistrationManager(registry)
        discovery_engine = ToolDiscoveryEngine(registry)
        plugin_loader = PluginLoader(registration_manager)
        
        # Register example tools
        tools = create_example_tools()
        for tool in tools:
            registration_manager.register_tool_advanced(tool)
        
        # Rebuild discovery index
        discovery_engine.rebuild_index()
        
        return {
            'registry': registry,
            'registration_manager': registration_manager,
            'discovery_engine': discovery_engine,
            'plugin_loader': plugin_loader
        }
    
    def test_complete_discovery_workflow(self, full_system):
        """Test complete discovery workflow"""
        discovery_engine = full_system['discovery_engine']
        
        # Test various discovery methods
        discovery_methods = [
            ("echo", DiscoveryMethod.NAME_EXACT),
            ("calc", DiscoveryMethod.NAME_FUZZY),
            ("mathematical operations", DiscoveryMethod.INTENT_BASED),
            ("utility tools", DiscoveryMethod.SEMANTIC)
        ]
        
        for query_text, method in discovery_methods:
            query = DiscoveryQuery(query=query_text, method=method)
            results = discovery_engine.discover_tools(query)
            assert len(results) > 0
    
    def test_advanced_search_scenarios(self, full_system):
        """Test advanced search scenarios"""
        discovery_engine = full_system['discovery_engine']
        
        # Test capability-based search
        query = DiscoveryQuery(
            query="math",
            method=DiscoveryMethod.CAPABILITY,
            capabilities=[ToolCapability.MATHEMATICAL]
        )
        results = discovery_engine.discover_tools(query)
        assert any(r.tool.metadata.name == "calculator" for r in results)
        
        # Test filtered search
        query = DiscoveryQuery(
            query="tool",
            method=DiscoveryMethod.SEMANTIC,
            tool_types=[ToolType.PYTHON],
            security_levels=[ToolSecurityLevel.SAFE],
            max_results=3
        )
        results = discovery_engine.discover_tools(query)
        assert len(results) <= 3
        for result in results:
            assert result.tool.metadata.tool_type == ToolType.PYTHON
            assert result.tool.metadata.security_level == ToolSecurityLevel.SAFE
    
    def test_registration_and_discovery_integration(self, full_system):
        """Test integration between registration and discovery"""
        registration_manager = full_system['registration_manager']
        discovery_engine = full_system['discovery_engine']
        
        # Create a new tool
        class IntegrationTool(BaseTool):
            def __init__(self):
                metadata = ToolMetadata(
                    name="integration_tool",
                    description="A tool for integration testing",
                    version="1.0.0",
                    tool_type=ToolType.PYTHON,
                    security_level=ToolSecurityLevel.SAFE,
                    execution_mode=ToolExecutionMode.SYNCHRONOUS,
                    parameters=[],
                    return_type="string",
                    tags=["integration", "test"]
                )
                super().__init__(metadata)
            
            async def execute(self, parameters):
                return "integration result"
            
            async def validate_parameters(self, parameters):
                return True, None
        
        # Register the tool
        tool = IntegrationTool()
        success, error = registration_manager.register_tool_advanced(tool)
        assert success is True
        
        # Rebuild discovery index
        discovery_engine.rebuild_index()
        
        # Search for the tool
        query = DiscoveryQuery(query="integration", method=DiscoveryMethod.NAME_FUZZY)
        results = discovery_engine.discover_tools(query)
        
        # Should find the newly registered tool
        assert any(r.tool.metadata.name == "integration_tool" for r in results)
    
    def test_statistics_and_monitoring(self, full_system):
        """Test statistics and monitoring capabilities"""
        discovery_engine = full_system['discovery_engine']
        registration_manager = full_system['registration_manager']
        
        # Perform some searches to generate usage statistics
        queries = [
            DiscoveryQuery(query="echo", method=DiscoveryMethod.NAME_EXACT),
            DiscoveryQuery(query="calculator", method=DiscoveryMethod.NAME_EXACT),
            DiscoveryQuery(query="echo", method=DiscoveryMethod.NAME_EXACT),  # Duplicate to test usage counting
        ]
        
        for query in queries:
            discovery_engine.discover_tools(query)
        
        # Check discovery statistics
        discovery_stats = discovery_engine.get_discovery_stats()
        assert discovery_stats['indexed_tools'] > 0
        assert 'echo' in discovery_stats['usage_stats']
        assert discovery_stats['usage_stats']['echo'] == 2
        
        # Check registration history
        history = registration_manager.get_registration_history()
        assert len(history) > 0
        
        # Check that all example tools are registered
        assert len(history) >= 5  # At least 5 example tools 