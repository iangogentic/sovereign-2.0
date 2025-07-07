#!/usr/bin/env python3
"""
Comprehensive Demonstration of Extensibility and Permission Management System

This script demonstrates:
1. Plugin creation and management
2. User permission controls and consent workflows
3. Extension point registration and hook execution
4. Custom tool plugin development and execution
5. Security features and access controls
6. Complete end-to-end integration scenarios

Usage: python test_extensibility_demo.py
"""

import os
import sys
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.sovereign.tool_extensibility_manager import (
        ToolPluginManager, PermissionValidator, ExtensionRegistry,
        UserRole, PermissionLevel, PluginStatus, ConsentStatus,
        UserPermission, PluginMetadata, ExtensionPoint, ConsentRequest,
        IPlugin, IToolPlugin, requires_permission, permission_check
    )
except ImportError:
    print("Error: Could not import extensibility modules. Please ensure the modules are in the correct location.")
    sys.exit(1)


class CustomCalculatorPlugin(IToolPlugin):
    """Example custom calculator plugin"""
    
    def __init__(self):
        self.initialized = False
        self.config = {}
    
    def initialize(self, config):
        self.config = config
        self.initialized = True
        print(f"✅ Calculator plugin initialized with config: {config}")
        return True
    
    def cleanup(self):
        self.initialized = False
        print("🧹 Calculator plugin cleaned up")
        return True
    
    def get_metadata(self):
        return PluginMetadata(
            name="custom_calculator",
            version="1.0.0",
            author="Demo Team",
            description="Custom calculator plugin for demonstration",
            plugin_type="tool",
            entry_point="calculator.py",
            status=PluginStatus.ACTIVE if self.initialized else PluginStatus.INACTIVE
        )
    
    def get_tools(self):
        return [
            {
                "name": "advanced_calculator",
                "description": "Advanced calculator with multiple operations",
                "parameters": {
                    "operation": {
                        "type": "string",
                        "description": "Operation to perform (add, subtract, multiply, divide, power)",
                        "required": True
                    },
                    "operand1": {
                        "type": "number",
                        "description": "First operand",
                        "required": True
                    },
                    "operand2": {
                        "type": "number",
                        "description": "Second operand",
                        "required": True
                    }
                },
                "returns": {
                    "type": "object",
                    "description": "Calculation result"
                }
            },
            {
                "name": "statistical_calculator",
                "description": "Statistical operations on data sets",
                "parameters": {
                    "operation": {
                        "type": "string",
                        "description": "Statistical operation (mean, median, mode, std_dev)",
                        "required": True
                    },
                    "data": {
                        "type": "array",
                        "description": "Array of numbers",
                        "required": True
                    }
                },
                "returns": {
                    "type": "object",
                    "description": "Statistical result"
                }
            }
        ]
    
    def execute_tool(self, tool_name, parameters):
        if not self.initialized:
            return {
                "success": False,
                "error": "Plugin not initialized"
            }
        
        if tool_name == "advanced_calculator":
            return self._execute_advanced_calculator(parameters)
        elif tool_name == "statistical_calculator":
            return self._execute_statistical_calculator(parameters)
        
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}"
        }
    
    def _execute_advanced_calculator(self, parameters):
        try:
            operation = parameters.get("operation")
            operand1 = parameters.get("operand1")
            operand2 = parameters.get("operand2")
            
            if operation == "add":
                result = operand1 + operand2
            elif operation == "subtract":
                result = operand1 - operand2
            elif operation == "multiply":
                result = operand1 * operand2
            elif operation == "divide":
                if operand2 == 0:
                    return {
                        "success": False,
                        "error": "Division by zero"
                    }
                result = operand1 / operand2
            elif operation == "power":
                result = operand1 ** operand2
            else:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}"
                }
            
            return {
                "success": True,
                "result": result,
                "operation": operation,
                "operands": [operand1, operand2],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _execute_statistical_calculator(self, parameters):
        try:
            operation = parameters.get("operation")
            data = parameters.get("data", [])
            
            if not data:
                return {
                    "success": False,
                    "error": "No data provided"
                }
            
            if operation == "mean":
                result = sum(data) / len(data)
            elif operation == "median":
                sorted_data = sorted(data)
                n = len(sorted_data)
                if n % 2 == 0:
                    result = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
                else:
                    result = sorted_data[n//2]
            elif operation == "mode":
                from collections import Counter
                counts = Counter(data)
                max_count = max(counts.values())
                result = [k for k, v in counts.items() if v == max_count]
            elif operation == "std_dev":
                mean = sum(data) / len(data)
                variance = sum((x - mean) ** 2 for x in data) / len(data)
                result = variance ** 0.5
            else:
                return {
                    "success": False,
                    "error": f"Unknown statistical operation: {operation}"
                }
            
            return {
                "success": True,
                "result": result,
                "operation": operation,
                "data_size": len(data),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class ExtensibilityDemo:
    """Comprehensive demonstration of extensibility features"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ToolPluginManager(plugin_directory=self.temp_dir)
        self.setup_users()
        self.setup_hooks()
        
        print(f"🚀 Extensibility Demo initialized")
        print(f"📁 Plugin directory: {self.temp_dir}")
        print()
    
    def setup_users(self):
        """Set up test users with different permission levels"""
        print("👥 Setting up user permission system...")
        
        # Admin user - full permissions
        admin_user = UserPermission(
            user_id="admin_001",
            role=UserRole.ADMIN,
            permissions={}
        )
        
        # Regular user - limited permissions
        regular_user = UserPermission(
            user_id="user_001",
            role=UserRole.USER,
            permissions={
                "calculator_access": PermissionLevel.MODERATE,
                "statistical_operations": PermissionLevel.LIMITED
            },
            tool_access={
                "advanced_calculator": True,
                "statistical_calculator": True,
                "sensitive_tool": False
            }
        )
        
        # Restricted user - very limited permissions
        restricted_user = UserPermission(
            user_id="restricted_001",
            role=UserRole.RESTRICTED,
            permissions={
                "basic_calculator": PermissionLevel.LIMITED
            },
            tool_access={
                "advanced_calculator": False,
                "statistical_calculator": False
            }
        )
        
        # Guest user - no permissions
        guest_user = UserPermission(
            user_id="guest_001",
            role=UserRole.GUEST,
            permissions={}
        )
        
        # Add users to validator
        self.manager.permission_validator.add_user(admin_user)
        self.manager.permission_validator.add_user(regular_user)
        self.manager.permission_validator.add_user(restricted_user)
        self.manager.permission_validator.add_user(guest_user)
        
        print(f"✅ Added 4 users: admin, regular, restricted, guest")
        print()
    
    def setup_hooks(self):
        """Set up demonstration hooks"""
        print("🎣 Setting up extension hooks...")
        
        def plugin_loaded_hook(plugin_name, plugin_instance):
            print(f"🔌 Hook: Plugin '{plugin_name}' loaded successfully")
        
        def before_tool_execution_hook(plugin_name, tool_name, parameters):
            print(f"⚡ Hook: Before executing '{tool_name}' from '{plugin_name}'")
        
        def after_tool_execution_hook(plugin_name, tool_name, parameters, result):
            print(f"✨ Hook: After executing '{tool_name}' - Success: {result.get('success', False)}")
        
        def tool_error_hook(plugin_name, tool_name, parameters, error):
            print(f"❌ Hook: Error in '{tool_name}' from '{plugin_name}': {error}")
        
        # Register hooks
        self.manager.extension_registry.register_hook("plugin_loaded", plugin_loaded_hook)
        self.manager.extension_registry.register_hook("before_tool_execution", before_tool_execution_hook)
        self.manager.extension_registry.register_hook("after_tool_execution", after_tool_execution_hook)
        self.manager.extension_registry.register_hook("tool_error", tool_error_hook)
        
        print(f"✅ Registered 4 extension hooks")
        print()
    
    def demonstrate_plugin_creation(self):
        """Demonstrate plugin template creation"""
        print("🛠️  PLUGIN CREATION DEMONSTRATION")
        print("=" * 50)
        
        # Create different types of plugins
        plugins_to_create = [
            ("math_tools", "tool"),
            ("data_processor", "tool"),
            ("utility_plugin", "basic")
        ]
        
        for plugin_name, plugin_type in plugins_to_create:
            print(f"📦 Creating {plugin_type} plugin: {plugin_name}")
            success = self.manager.create_plugin_template(plugin_name, plugin_type)
            
            if success:
                print(f"✅ Successfully created {plugin_name}")
                
                # Show created files
                plugin_dir = Path(self.temp_dir) / plugin_name
                files = list(plugin_dir.glob("*"))
                print(f"   📄 Created files: {[f.name for f in files]}")
            else:
                print(f"❌ Failed to create {plugin_name}")
            
            print()
        
        # Discover all plugins
        print("🔍 Discovering created plugins...")
        discovered = self.manager.discover_plugins()
        
        print(f"✅ Discovered {len(discovered)} plugins:")
        for plugin_meta in discovered:
            print(f"   - {plugin_meta.name} v{plugin_meta.version} by {plugin_meta.author}")
        
        print()
    
    def demonstrate_permission_system(self):
        """Demonstrate permission system"""
        print("🔐 PERMISSION SYSTEM DEMONSTRATION")
        print("=" * 50)
        
        # Test different user permissions
        users_to_test = [
            ("admin_001", "Admin"),
            ("user_001", "Regular User"),
            ("restricted_001", "Restricted User"),
            ("guest_001", "Guest User")
        ]
        
        tools_to_test = [
            ("advanced_calculator", "execute"),
            ("statistical_calculator", "execute"),
            ("sensitive_tool", "execute")
        ]
        
        print("Testing user permissions:")
        for user_id, user_name in users_to_test:
            print(f"\n👤 {user_name} ({user_id}):")
            
            for tool_name, operation in tools_to_test:
                has_permission = self.manager.permission_validator.check_permission(
                    user_id, tool_name, operation
                )
                
                status = "✅ ALLOWED" if has_permission else "❌ DENIED"
                print(f"   {tool_name} ({operation}): {status}")
        
        print()
    
    def demonstrate_consent_workflow(self):
        """Demonstrate consent workflow"""
        print("✋ CONSENT WORKFLOW DEMONSTRATION")
        print("=" * 50)
        
        # Regular user requests access to sensitive tool
        print("📋 User requests consent for sensitive operation...")
        
        consent_id = self.manager.permission_validator.request_consent(
            user_id="user_001",
            tool_name="sensitive_data_processor",
            operation="delete_personal_data",
            risk_level="high",
            description="This operation will permanently delete user's personal data"
        )
        
        print(f"✅ Consent request created: {consent_id}")
        print(f"   Status: {self.manager.permission_validator.check_consent(consent_id).value}")
        
        # Admin reviews and grants consent
        print("\n👨‍💼 Admin reviews and grants consent...")
        
        granted = self.manager.permission_validator.grant_consent(
            consent_id, 
            True, 
            "Approved after user verification and backup confirmation"
        )
        
        if granted:
            print("✅ Consent granted successfully")
            print(f"   New status: {self.manager.permission_validator.check_consent(consent_id).value}")
        else:
            print("❌ Failed to grant consent")
        
        # Check consent history
        user_perm = self.manager.permission_validator.user_permissions["user_001"]
        print(f"\n📈 User consent history: {len(user_perm.consent_history)} entries")
        
        if user_perm.consent_history:
            latest = user_perm.consent_history[-1]
            print(f"   Latest: {latest['tool_name']} - {'Approved' if latest['decision'] else 'Denied'}")
        
        print()
    
    def demonstrate_custom_plugin(self):
        """Demonstrate custom plugin loading and execution"""
        print("🔧 CUSTOM PLUGIN DEMONSTRATION")
        print("=" * 50)
        
        # Register custom plugin directly (simulating loaded plugin)
        print("📝 Registering custom calculator plugin...")
        
        calculator_plugin = CustomCalculatorPlugin()
        success = calculator_plugin.initialize({"precision": 10, "mode": "scientific"})
        
        if success:
            print("✅ Custom calculator plugin initialized")
            
            # Simulate plugin registration
            self.manager.plugins["custom_calculator"] = calculator_plugin
            
            # Get plugin tools
            tools = calculator_plugin.get_tools()
            print(f"🛠️  Plugin provides {len(tools)} tools:")
            for tool in tools:
                print(f"   - {tool['name']}: {tool['description']}")
        
        print()
    
    def demonstrate_plugin_execution(self):
        """Demonstrate plugin tool execution"""
        print("⚡ PLUGIN EXECUTION DEMONSTRATION")
        print("=" * 50)
        
        if "custom_calculator" not in self.manager.plugins:
            print("❌ Custom calculator plugin not available")
            return
        
        # Test advanced calculator
        print("🧮 Testing advanced calculator...")
        
        test_calculations = [
            {"operation": "add", "operand1": 15, "operand2": 25},
            {"operation": "multiply", "operand1": 7, "operand2": 8},
            {"operation": "power", "operand1": 2, "operand2": 10},
            {"operation": "divide", "operand1": 100, "operand2": 0}  # Test error handling
        ]
        
        for calculation in test_calculations:
            result = self.manager.execute_plugin_tool(
                "custom_calculator", 
                "advanced_calculator", 
                calculation,
                "user_001"  # Regular user
            )
            
            if result.get("success"):
                print(f"✅ {calculation['operation']}({calculation['operand1']}, {calculation['operand2']}) = {result['result']}")
            else:
                print(f"❌ {calculation['operation']}({calculation['operand1']}, {calculation['operand2']}) failed: {result['error']}")
        
        print()
        
        # Test statistical calculator
        print("📊 Testing statistical calculator...")
        
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        statistical_tests = [
            {"operation": "mean", "data": test_data},
            {"operation": "median", "data": test_data},
            {"operation": "std_dev", "data": test_data},
            {"operation": "mode", "data": [1, 2, 2, 3, 3, 3, 4, 4, 5]}
        ]
        
        for test in statistical_tests:
            result = self.manager.execute_plugin_tool(
                "custom_calculator",
                "statistical_calculator",
                test,
                "user_001"  # Regular user
            )
            
            if result.get("success"):
                print(f"✅ {test['operation']} = {result['result']}")
            else:
                print(f"❌ {test['operation']} failed: {result['error']}")
        
        print()
    
    def demonstrate_permission_denied(self):
        """Demonstrate permission denied scenarios"""
        print("🚫 PERMISSION DENIED DEMONSTRATION")
        print("=" * 50)
        
        if "custom_calculator" not in self.manager.plugins:
            print("❌ Custom calculator plugin not available")
            return
        
        # Test restricted user trying to access advanced calculator
        print("🔒 Testing restricted user access...")
        
        result = self.manager.execute_plugin_tool(
            "custom_calculator",
            "advanced_calculator",
            {"operation": "add", "operand1": 5, "operand2": 3},
            "restricted_001"  # Restricted user
        )
        
        if result.get("success"):
            print("⚠️  Unexpected: Restricted user was allowed access")
        else:
            print(f"✅ Expected: Access denied - {result['error']}")
        
        # Test guest user trying to access statistical calculator
        print("\n👻 Testing guest user access...")
        
        result = self.manager.execute_plugin_tool(
            "custom_calculator",
            "statistical_calculator",
            {"operation": "mean", "data": [1, 2, 3, 4, 5]},
            "guest_001"  # Guest user
        )
        
        if result.get("success"):
            print("⚠️  Unexpected: Guest user was allowed access")
        else:
            print(f"✅ Expected: Access denied - {result['error']}")
        
        print()
    
    def demonstrate_extension_points(self):
        """Demonstrate extension points and hooks"""
        print("🔗 EXTENSION POINTS DEMONSTRATION")
        print("=" * 50)
        
        # Show registered extension points
        extension_points = self.manager.extension_registry.get_extension_points()
        print(f"📍 Registered extension points: {len(extension_points)}")
        
        for name, point in extension_points.items():
            print(f"   - {name}: {point.description}")
            print(f"     Required methods: {point.required_methods}")
            print(f"     Available hooks: {point.hooks}")
        
        print()
        
        # Test hook execution
        print("🎯 Testing hook execution...")
        
        # Simulate plugin loading to trigger hooks
        self.manager.extension_registry.execute_hook(
            "plugin_loaded", 
            "demo_plugin", 
            {"name": "demo_plugin", "version": "1.0.0"}
        )
        
        # Simulate tool execution to trigger hooks
        self.manager.extension_registry.execute_hook(
            "before_tool_execution",
            "demo_plugin",
            "demo_tool",
            {"param1": "value1"}
        )
        
        self.manager.extension_registry.execute_hook(
            "after_tool_execution",
            "demo_plugin",
            "demo_tool",
            {"param1": "value1"},
            {"success": True, "result": "demo_result"}
        )
        
        print()
    
    def demonstrate_plugin_management(self):
        """Demonstrate plugin management features"""
        print("📦 PLUGIN MANAGEMENT DEMONSTRATION")
        print("=" * 50)
        
        # Show plugin status
        print("📊 Plugin Status Report:")
        status = self.manager.get_plugin_status()
        
        for plugin_name, plugin_info in status.items():
            print(f"   📋 {plugin_name}:")
            print(f"      Status: {plugin_info['status']}")
            print(f"      Loaded: {plugin_info['loaded']}")
            print(f"      Version: {plugin_info['metadata'].version}")
            print(f"      Author: {plugin_info['metadata'].author}")
            print()
        
        # Show active plugins
        print(f"🔌 Active Plugins: {len(self.manager.plugins)}")
        for plugin_name, plugin in self.manager.plugins.items():
            metadata = plugin.get_metadata()
            print(f"   - {plugin_name} v{metadata.version} ({metadata.status.value})")
        
        print()
        
        # Test plugin refresh
        print("🔄 Testing plugin refresh...")
        self.manager.refresh_plugins()
        print("✅ Plugin refresh completed")
        
        print()
    
    def run_comprehensive_demo(self):
        """Run the complete demonstration"""
        print("🎬 COMPREHENSIVE EXTENSIBILITY DEMONSTRATION")
        print("=" * 60)
        print(f"⏰ Started at: {datetime.now()}")
        print()
        
        try:
            # Run all demonstration components
            self.demonstrate_plugin_creation()
            self.demonstrate_permission_system()
            self.demonstrate_consent_workflow()
            self.demonstrate_custom_plugin()
            self.demonstrate_plugin_execution()
            self.demonstrate_permission_denied()
            self.demonstrate_extension_points()
            self.demonstrate_plugin_management()
            
            # Show final statistics
            self.show_final_statistics()
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n🎉 DEMONSTRATION COMPLETED")
        print(f"⏰ Finished at: {datetime.now()}")
        print()
    
    def show_final_statistics(self):
        """Show final demonstration statistics"""
        print("📈 FINAL STATISTICS")
        print("=" * 50)
        
        # Plugin statistics
        discovered_plugins = self.manager.discover_plugins()
        active_plugins = len(self.manager.plugins)
        
        print(f"📦 Plugins:")
        print(f"   - Total discovered: {len(discovered_plugins)}")
        print(f"   - Currently active: {active_plugins}")
        
        # User statistics
        users = self.manager.permission_validator.user_permissions
        print(f"\n👥 Users: {len(users)} registered")
        
        # Count consent requests
        consent_requests = len(self.manager.permission_validator.consent_cache)
        print(f"✋ Consent requests: {consent_requests}")
        
        # Extension statistics
        extension_points = self.manager.extension_registry.get_extension_points()
        hooks = self.manager.extension_registry.hooks
        
        print(f"\n🔗 Extension System:")
        print(f"   - Extension points: {len(extension_points)}")
        print(f"   - Registered hooks: {len(hooks)}")
        
        print(f"\n💾 Data Location: {self.temp_dir}")
        print()
    
    def cleanup(self):
        """Clean up demo resources"""
        print("🧹 Cleaning up demo resources...")
        
        # Cleanup plugins
        for plugin_name in list(self.manager.plugins.keys()):
            self.manager.unload_plugin(plugin_name)
        
        # Remove temporary directory
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print("✅ Temporary files cleaned up")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
        
        print()


def main():
    """Main demonstration function"""
    print("🚀 SOVEREIGN AI EXTENSIBILITY SYSTEM DEMO")
    print("=" * 60)
    
    demo = ExtensibilityDemo()
    
    try:
        demo.run_comprehensive_demo()
    except KeyboardInterrupt:
        print("\n⏸️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main() 