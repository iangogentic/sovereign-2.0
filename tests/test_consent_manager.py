"""
Unit tests for ConsentManager
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import tempfile
import time
import json
from pathlib import Path

from src.sovereign.consent_manager import (
    ConsentManager,
    ConsentMethod,
    ConsentRequest,
    cli_consent_callback,
    gui_consent_callback,
    ConsentLevel,
    PrivacyZoneType,
    MonitoringState,
    ConsentRecord,
    PrivacyZone,
    SensitiveApp
)
from src.sovereign.external_model_connector import (
    RoutingDecision,
    ExternalRoutingCriteria
)
from src.sovereign.config import Config


class TestConsentManager:
    """Test suite for ConsentManager"""
    
    @pytest.fixture
    def routing_decision(self):
        """Create a sample routing decision for testing"""
        return RoutingDecision(
            should_route=True,
            criteria=[ExternalRoutingCriteria.SPECIALIZED_KNOWLEDGE],
            confidence=0.4,
            reasoning="Query requires specialized knowledge"
        )
    
    @pytest.fixture
    def consent_request(self, routing_decision):
        """Create a sample consent request for testing"""
        return ConsentRequest(
            query="What are the latest news about AI?",
            routing_decision=routing_decision,
            provider="openrouter"
        )
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return Config()
    
    @pytest.fixture
    def consent_manager(self, config, temp_dir):
        """Create ConsentManager instance with test data directory"""
        # Patch the data directory to use temp directory
        with patch('src.sovereign.consent_manager.Path.home') as mock_home:
            mock_home.return_value = Path(temp_dir)
            manager = ConsentManager(config)
            yield manager
    
    def test_init_default(self):
        """Test ConsentManager initialization with defaults"""
        manager = ConsentManager()
        assert manager.method == ConsentMethod.CLI
        assert len(manager.consent_history) == 0
        assert manager.gui_consent_callback is None
    
    def test_init_with_method(self):
        """Test ConsentManager initialization with specific method"""
        manager = ConsentManager(ConsentMethod.GUI)
        assert manager.method == ConsentMethod.GUI
    
    def test_set_gui_callback(self):
        """Test setting GUI callback"""
        manager = ConsentManager()
        callback = Mock()
        manager.set_gui_callback(callback)
        assert manager.gui_consent_callback == callback
    
    @pytest.mark.asyncio
    async def test_auto_approve(self, routing_decision):
        """Test auto-approve consent method"""
        manager = ConsentManager(ConsentMethod.AUTO_APPROVE)
        
        result = await manager.request_consent("test query", routing_decision)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_auto_deny(self, routing_decision):
        """Test auto-deny consent method"""
        manager = ConsentManager(ConsentMethod.AUTO_DENY)
        
        result = await manager.request_consent("test query", routing_decision)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cli_consent_approve(self, routing_decision):
        """Test CLI consent with user approval"""
        manager = ConsentManager(ConsentMethod.CLI)
        
        # Mock user input to return 'y'
        with patch('builtins.input', return_value='y'):
            with patch('builtins.print'):  # Suppress print output during test
                result = await manager.request_consent("test query", routing_decision)
                assert result is True
    
    @pytest.mark.asyncio
    async def test_cli_consent_deny(self, routing_decision):
        """Test CLI consent with user denial"""
        manager = ConsentManager(ConsentMethod.CLI)
        
        # Mock user input to return 'n'
        with patch('builtins.input', return_value='n'):
            with patch('builtins.print'):  # Suppress print output during test
                result = await manager.request_consent("test query", routing_decision)
                assert result is False
    
    @pytest.mark.asyncio
    async def test_cli_consent_details_then_approve(self, routing_decision):
        """Test CLI consent with details request then approval"""
        manager = ConsentManager(ConsentMethod.CLI)
        
        # Mock user input to return 'd' first, then 'y'
        with patch('builtins.input', side_effect=['d', 'y']):
            with patch('builtins.print'):  # Suppress print output during test
                result = await manager.request_consent("test query", routing_decision)
                assert result is True
    
    @pytest.mark.asyncio
    async def test_cli_consent_invalid_then_approve(self, routing_decision):
        """Test CLI consent with invalid input then approval"""
        manager = ConsentManager(ConsentMethod.CLI)
        
        # Mock user input to return invalid option first, then 'y'
        with patch('builtins.input', side_effect=['invalid', 'y']):
            with patch('builtins.print'):  # Suppress print output during test
                result = await manager.request_consent("test query", routing_decision)
                assert result is True
    
    @pytest.mark.asyncio
    async def test_cli_consent_keyboard_interrupt(self, routing_decision):
        """Test CLI consent with keyboard interrupt"""
        manager = ConsentManager(ConsentMethod.CLI)
        
        # Mock input to raise KeyboardInterrupt
        with patch('builtins.input', side_effect=KeyboardInterrupt()):
            with patch('builtins.print'):  # Suppress print output during test
                result = await manager.request_consent("test query", routing_decision)
                assert result is False
    
    @pytest.mark.asyncio
    async def test_gui_consent_with_callback(self, routing_decision):
        """Test GUI consent with callback"""
        manager = ConsentManager(ConsentMethod.GUI)
        
        # Mock GUI callback
        callback = AsyncMock(return_value=True)
        manager.set_gui_callback(callback)
        
        result = await manager.request_consent("test query", routing_decision)
        assert result is True
        callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_gui_consent_without_callback_fallback(self, routing_decision):
        """Test GUI consent fallback to CLI when no callback available"""
        manager = ConsentManager(ConsentMethod.GUI)
        
        # No callback set, should fallback to CLI
        with patch('builtins.input', return_value='y'):
            with patch('builtins.print'):  # Suppress print output during test
                result = await manager.request_consent("test query", routing_decision)
                assert result is True
    
    @pytest.mark.asyncio
    async def test_gui_consent_callback_error(self, routing_decision):
        """Test GUI consent with callback error"""
        manager = ConsentManager(ConsentMethod.GUI)
        
        # Mock GUI callback that raises an error
        callback = AsyncMock(side_effect=Exception("GUI error"))
        manager.set_gui_callback(callback)
        
        result = await manager.request_consent("test query", routing_decision)
        assert result is False
    
    def test_format_criteria(self):
        """Test criteria formatting"""
        manager = ConsentManager()
        
        test_cases = [
            (ExternalRoutingCriteria.SPECIALIZED_KNOWLEDGE, "Specialized Knowledge"),
            (ExternalRoutingCriteria.COMPLEX_TOOL_USE, "Complex Tool Use"),
            (ExternalRoutingCriteria.USER_EXPLICIT_REQUEST, "Explicit Request"),
            (ExternalRoutingCriteria.LOCAL_MODEL_FAILURE, "Local Model Limitation"),
            (ExternalRoutingCriteria.RECENT_INFORMATION, "Recent Information"),
        ]
        
        for criteria, expected in test_cases:
            result = manager._format_criteria(criteria)
            assert result == expected
    
    def test_show_detailed_info(self, consent_request):
        """Test detailed info display"""
        manager = ConsentManager()
        
        # Test that the method doesn't raise an error
        with patch('builtins.print'):  # Suppress print output during test
            manager._show_detailed_info(consent_request)
    
    def test_consent_history(self, routing_decision):
        """Test consent history tracking"""
        manager = ConsentManager()
        
        # Initially empty
        history = manager.get_consent_history()
        assert len(history) == 0
        
        # Clear history
        manager.clear_consent_history()
        assert len(manager.consent_history) == 0
    
    @pytest.mark.asyncio
    async def test_cli_consent_callback_function(self):
        """Test the standalone CLI consent callback function"""
        routing_decision = RoutingDecision(
            should_route=True,
            criteria=[ExternalRoutingCriteria.USER_EXPLICIT_REQUEST],
            confidence=0.5,
            reasoning="User explicitly requested external service"
        )
        
        with patch('builtins.input', return_value='y'):
            with patch('builtins.print'):  # Suppress print output during test
                result = await cli_consent_callback("test query", routing_decision)
                assert result is True
    
    @pytest.mark.asyncio
    async def test_gui_consent_callback_function(self):
        """Test the standalone GUI consent callback function"""
        routing_decision = RoutingDecision(
            should_route=True,
            criteria=[ExternalRoutingCriteria.USER_EXPLICIT_REQUEST],
            confidence=0.5,
            reasoning="User explicitly requested external service"
        )
        
        # Should fallback to CLI since no GUI callback is set
        with patch('builtins.input', return_value='n'):
            with patch('builtins.print'):  # Suppress print output during test
                result = await gui_consent_callback("test query", routing_decision)
                assert result is False
    
    @pytest.mark.asyncio
    async def test_unknown_consent_method(self, routing_decision):
        """Test handling of unknown consent method"""
        manager = ConsentManager()
        manager.method = "unknown_method"  # Set invalid method
        
        result = await manager.request_consent("test query", routing_decision)
        assert result is False
    
    def test_initialization(self, consent_manager):
        """Test ConsentManager initialization"""
        assert consent_manager.config is not None
        assert isinstance(consent_manager.consent_records, dict)
        assert isinstance(consent_manager.privacy_zones, dict)
        assert isinstance(consent_manager.sensitive_apps, dict)
        assert consent_manager.monitoring_state == MonitoringState.DISABLED
        
        # Should have default sensitive apps loaded
        assert len(consent_manager.sensitive_apps) > 0
        assert "Password Managers" in consent_manager.sensitive_apps
        assert "Banking & Finance" in consent_manager.sensitive_apps
    
    def test_consent_record_creation(self):
        """Test ConsentRecord creation and validation"""
        record = ConsentRecord(
            consent_id="test_id",
            feature="test_feature",
            consent_level=ConsentLevel.STANDARD,
            granted_timestamp=time.time(),
            expires_timestamp=time.time() + 3600  # 1 hour
        )
        
        assert record.consent_id == "test_id"
        assert record.feature == "test_feature"
        assert record.consent_level == ConsentLevel.STANDARD
        assert record.is_valid()
        assert not record.is_expired()
        
        # Test expired record
        expired_record = ConsentRecord(
            consent_id="expired_id",
            feature="test_feature",
            consent_level=ConsentLevel.STANDARD,
            granted_timestamp=time.time() - 7200,  # 2 hours ago
            expires_timestamp=time.time() - 3600   # 1 hour ago
        )
        
        assert expired_record.is_expired()
        assert not expired_record.is_valid()
        
        # Test denied record
        denied_record = ConsentRecord(
            consent_id="denied_id",
            feature="test_feature",
            consent_level=ConsentLevel.DENIED,
            granted_timestamp=time.time()
        )
        
        assert not denied_record.is_valid()
    
    def test_privacy_zone_creation(self):
        """Test PrivacyZone creation and functionality"""
        zone = PrivacyZone(
            zone_id="test_zone",
            name="Test Zone",
            x=100,
            y=200,
            width=300,
            height=400,
            zone_type=PrivacyZoneType.BLUR
        )
        
        assert zone.zone_id == "test_zone"
        assert zone.name == "Test Zone"
        assert zone.active
        assert zone.zone_type == PrivacyZoneType.BLUR
        
        # Test point containment
        assert zone.contains_point(200, 300)  # Inside
        assert not zone.contains_point(50, 50)   # Outside
        assert zone.contains_point(100, 200)  # Edge (inclusive)
        
        # Test region overlap
        assert zone.overlaps_region(0, 0, 150, 250)     # Overlaps
        assert zone.overlaps_region(50, 50, 100, 100)   # No overlap
        assert zone.overlaps_region(200, 300, 100, 100) # Inside zone
    
    def test_sensitive_app_detection(self):
        """Test SensitiveApp pattern matching"""
        app = SensitiveApp(
            app_name="Test Banking App",
            process_patterns=[r".*bank.*", r"finance\.exe"],
            window_patterns=[r".*Bank.*", r"Online Banking.*"],
            priority=8
        )
        
        assert app.matches_process("mybank.exe")
        assert app.matches_process("finance.exe")
        assert not app.matches_process("notepad.exe")
        
        assert app.matches_window("Chase Bank - Online Banking")
        assert app.matches_window("Bank of America")
        assert not app.matches_window("Notepad")
    
    def test_consent_request_and_management(self, consent_manager):
        """Test consent request and management functionality"""
        # Test requesting consent
        level = consent_manager.request_consent("test_feature", ConsentLevel.STANDARD)
        assert level == ConsentLevel.STANDARD
        
        # Check that consent was recorded
        current = consent_manager.get_current_consent("test_feature")
        assert current is not None
        assert current.consent_level == ConsentLevel.STANDARD
        assert current.feature == "test_feature"
        
        # Test checking consent
        assert consent_manager.has_consent("test_feature", ConsentLevel.BASIC)
        assert consent_manager.has_consent("test_feature", ConsentLevel.STANDARD)
        assert not consent_manager.has_consent("test_feature", ConsentLevel.ENHANCED)
        
        # Test revoking consent
        consent_manager.revoke_consent("test_feature")
        assert not consent_manager.has_consent("test_feature", ConsentLevel.BASIC)
        
        current = consent_manager.get_current_consent("test_feature")
        assert current is None or current.consent_level == ConsentLevel.DENIED
    
    def test_consent_expiration(self, consent_manager):
        """Test consent expiration handling"""
        # Request consent with 1 second expiration
        level = consent_manager.request_consent("temp_feature", ConsentLevel.STANDARD, expires_hours=1/3600)
        assert level == ConsentLevel.STANDARD
        
        # Should be valid initially
        assert consent_manager.has_consent("temp_feature")
        
        # Wait for expiration and check
        time.sleep(1.1)
        
        # Clear expired consents
        consent_manager.clear_expired_consents()
        
        # Should no longer have consent
        assert not consent_manager.has_consent("temp_feature")
    
    def test_privacy_zone_management(self, consent_manager):
        """Test privacy zone creation, modification, and removal"""
        # Add privacy zone
        zone_id = consent_manager.add_privacy_zone(
            "Test Zone", 100, 200, 300, 400, PrivacyZoneType.BLUR
        )
        
        assert zone_id is not None
        assert zone_id in consent_manager.privacy_zones
        
        zone = consent_manager.privacy_zones[zone_id]
        assert zone.name == "Test Zone"
        assert zone.zone_type == PrivacyZoneType.BLUR
        
        # Test getting overlapping zones
        overlapping = consent_manager.get_privacy_zones_for_region(150, 250, 100, 100)
        assert len(overlapping) == 1
        assert overlapping[0].zone_id == zone_id
        
        # Test no overlap
        no_overlap = consent_manager.get_privacy_zones_for_region(0, 0, 50, 50)
        assert len(no_overlap) == 0
        
        # Remove privacy zone
        result = consent_manager.remove_privacy_zone(zone_id)
        assert result is True
        assert zone_id not in consent_manager.privacy_zones
        
        # Try to remove non-existent zone
        result = consent_manager.remove_privacy_zone("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_sensitive_app_scanning(self, consent_manager):
        """Test sensitive application scanning"""
        # Mock psutil to return test processes
        mock_processes = [
            Mock(info={'name': 'notepad.exe'}),
            Mock(info={'name': '1password.exe'}),
            Mock(info={'name': 'chrome.exe'}),
            Mock(info={'name': 'keepass.exe'})
        ]
        
        with patch('psutil.process_iter', return_value=mock_processes):
            detected_apps = await consent_manager.scan_for_sensitive_apps()
            
            # Should detect password managers
            assert len(detected_apps) >= 1
            app_names = [app.app_name for app in detected_apps]
            assert "Password Managers" in app_names
    
    def test_monitoring_state_management(self, consent_manager):
        """Test monitoring state transitions"""
        # Test initial state
        assert consent_manager.monitoring_state == MonitoringState.DISABLED
        
        # Set to active
        consent_manager.set_monitoring_state(MonitoringState.ACTIVE, "test")
        assert consent_manager.monitoring_state == MonitoringState.ACTIVE
        
        # Pause for sensitive app
        consent_manager.pause_monitoring("sensitive_app")
        assert consent_manager.monitoring_state == MonitoringState.PAUSED_SENSITIVE
        
        # Pause for user request
        consent_manager.pause_monitoring("user_request")
        assert consent_manager.monitoring_state == MonitoringState.PAUSED_USER
        
        # Test is_monitoring_allowed
        assert not consent_manager.is_monitoring_allowed()
        
        # Resume (but needs consent)
        consent_manager.request_consent("screen_monitoring", ConsentLevel.BASIC)
        consent_manager.resume_monitoring()
        assert consent_manager.monitoring_state == MonitoringState.ACTIVE
        assert consent_manager.is_monitoring_allowed()
    
    def test_callback_system(self, consent_manager):
        """Test callback system for state and consent changes"""
        state_changes = []
        consent_changes = []
        
        def state_callback(state):
            state_changes.append(state)
        
        def consent_callback(feature, level):
            consent_changes.append((feature, level))
        
        consent_manager.add_state_change_callback(state_callback)
        consent_manager.add_consent_change_callback(consent_callback)
        
        # Test state change callback
        consent_manager.set_monitoring_state(MonitoringState.ACTIVE)
        assert len(state_changes) == 1
        assert state_changes[0] == MonitoringState.ACTIVE
        
        # Test consent change callback
        consent_manager.request_consent("test_feature", ConsentLevel.ENHANCED)
        assert len(consent_changes) == 1
        assert consent_changes[0] == ("test_feature", ConsentLevel.STANDARD)  # Auto-granted level
        
        # Test revoke consent callback
        consent_manager.revoke_consent("test_feature")
        assert len(consent_changes) == 2
        assert consent_changes[1] == ("test_feature", ConsentLevel.DENIED)
    
    def test_consent_summary(self, consent_manager):
        """Test consent summary generation"""
        # Add some consent and privacy zones
        consent_manager.request_consent("feature1", ConsentLevel.STANDARD)
        consent_manager.request_consent("feature2", ConsentLevel.ENHANCED)
        consent_manager.add_privacy_zone("Zone1", 0, 0, 100, 100)
        consent_manager.set_monitoring_state(MonitoringState.ACTIVE)
        
        summary = consent_manager.get_consent_summary()
        
        assert "monitoring_state" in summary
        assert summary["monitoring_state"] == "active"
        assert "active_consents" in summary
        assert "feature1" in summary["active_consents"]
        assert "privacy_zones" in summary
        assert summary["privacy_zones"] == 1
        assert "sensitive_apps" in summary
        assert summary["sensitive_apps"] > 0
    
    def test_data_persistence(self, consent_manager, temp_dir):
        """Test consent and privacy zone data persistence"""
        # Add consent and privacy zone
        consent_manager.request_consent("persistent_feature", ConsentLevel.ENHANCED)
        zone_id = consent_manager.add_privacy_zone("Persistent Zone", 50, 50, 200, 200)
        consent_manager.set_monitoring_state(MonitoringState.ACTIVE)
        
        # Create new manager instance (should load saved data)
        with patch('src.sovereign.consent_manager.Path.home') as mock_home:
            mock_home.return_value = Path(temp_dir)
            new_manager = ConsentManager(Config())
            
            # Check data was loaded
            assert new_manager.has_consent("persistent_feature")
            assert zone_id in new_manager.privacy_zones
            assert new_manager.monitoring_state == MonitoringState.ACTIVE
    
    def test_custom_sensitive_app_addition(self, consent_manager):
        """Test adding custom sensitive application patterns"""
        custom_app = SensitiveApp(
            app_name="Custom Banking App",
            process_patterns=[r"mybank\.exe"],
            window_patterns=[r"MyBank.*"],
            priority=9
        )
        
        consent_manager.add_sensitive_app(custom_app)
        
        assert "Custom Banking App" in consent_manager.sensitive_apps
        assert consent_manager.sensitive_apps["Custom Banking App"].priority == 9
    
    def test_error_handling(self, consent_manager):
        """Test error handling in various scenarios"""
        # Test invalid privacy zone
        zone_id = consent_manager.add_privacy_zone("", -1, -1, 0, 0)
        assert zone_id is not None  # Should still create with validation
        
        # Test getting consent for non-existent feature
        assert not consent_manager.has_consent("nonexistent_feature")
        
        # Test removing non-existent privacy zone
        result = consent_manager.remove_privacy_zone("nonexistent_zone")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cleanup(self, consent_manager):
        """Test resource cleanup"""
        # Add some data
        consent_manager.request_consent("cleanup_test", ConsentLevel.STANDARD, expires_hours=1/3600)
        consent_manager.add_privacy_zone("Cleanup Zone", 0, 0, 100, 100)
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Cleanup should remove expired consents
        await consent_manager.cleanup()
        
        # Expired consent should be gone
        assert not consent_manager.has_consent("cleanup_test")
        
        # Privacy zones should remain
        assert len(consent_manager.privacy_zones) > 0


class TestPrivacyIntegration:
    """Test integration between ConsentManager and other components"""
    
    @pytest.fixture
    def consent_manager(self):
        """Create ConsentManager for integration testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.sovereign.consent_manager.Path.home') as mock_home:
                mock_home.return_value = Path(temp_dir)
                yield ConsentManager(Config())
    
    def test_consent_level_hierarchy(self, consent_manager):
        """Test consent level hierarchy and permissions"""
        # Request different levels and test access
        test_cases = [
            (ConsentLevel.BASIC, [ConsentLevel.BASIC], [ConsentLevel.STANDARD, ConsentLevel.ENHANCED, ConsentLevel.FULL]),
            (ConsentLevel.STANDARD, [ConsentLevel.BASIC, ConsentLevel.STANDARD], [ConsentLevel.ENHANCED, ConsentLevel.FULL]),
            (ConsentLevel.ENHANCED, [ConsentLevel.BASIC, ConsentLevel.STANDARD, ConsentLevel.ENHANCED], [ConsentLevel.FULL]),
            (ConsentLevel.FULL, [ConsentLevel.BASIC, ConsentLevel.STANDARD, ConsentLevel.ENHANCED, ConsentLevel.FULL], [])
        ]
        
        for granted_level, should_allow, should_deny in test_cases:
            # Clear previous consents
            consent_manager.consent_records.clear()
            
            # Grant consent at this level
            consent_manager.request_consent("test_feature", granted_level)
            
            # Test allowed levels
            for allowed_level in should_allow:
                assert consent_manager.has_consent("test_feature", allowed_level), \
                    f"Should have {allowed_level.value} access with {granted_level.value} consent"
            
            # Test denied levels
            for denied_level in should_deny:
                assert not consent_manager.has_consent("test_feature", denied_level), \
                    f"Should not have {denied_level.value} access with {granted_level.value} consent"
    
    @pytest.mark.asyncio
    async def test_sensitive_app_process_detection(self, consent_manager):
        """Test realistic sensitive app process detection"""
        # Create more realistic process mocks
        realistic_processes = [
            Mock(info={'name': 'explorer.exe'}),
            Mock(info={'name': 'chrome.exe'}),
            Mock(info={'name': '1Password 7 - Password Manager.exe'}),
            Mock(info={'name': 'steam.exe'}),
            Mock(info={'name': 'KeePassXC.exe'}),
            Mock(info={'name': 'Bitwarden.exe'}),
            Mock(info={'name': 'vs_code.exe'}),
        ]
        
        with patch('psutil.process_iter', return_value=realistic_processes):
            detected_apps = await consent_manager.scan_for_sensitive_apps()
            
            # Should detect multiple password managers
            app_names = [app.app_name for app in detected_apps]
            assert "Password Managers" in app_names
            
            # Verify the specific apps were detected
            password_manager_app = next(app for app in detected_apps if app.app_name == "Password Managers")
            assert password_manager_app.priority == 10  # High priority
    
    def test_multi_zone_overlap_handling(self, consent_manager):
        """Test handling of multiple overlapping privacy zones"""
        # Create overlapping zones
        zone1_id = consent_manager.add_privacy_zone("Zone 1", 0, 0, 100, 100, PrivacyZoneType.BLUR)
        zone2_id = consent_manager.add_privacy_zone("Zone 2", 50, 50, 100, 100, PrivacyZoneType.FULL_REDACT)
        zone3_id = consent_manager.add_privacy_zone("Zone 3", 200, 200, 100, 100, PrivacyZoneType.SKIP_CAPTURE)
        
        # Test overlapping region
        overlapping = consent_manager.get_privacy_zones_for_region(25, 25, 100, 100)
        assert len(overlapping) == 2  # Should find zone1 and zone2
        
        # Test non-overlapping region
        separate = consent_manager.get_privacy_zones_for_region(150, 150, 50, 50)
        assert len(separate) == 0
        
        # Test region that overlaps with skip-capture zone
        skip_zone = consent_manager.get_privacy_zones_for_region(250, 250, 50, 50)
        assert len(skip_zone) == 1
        assert skip_zone[0].zone_type == PrivacyZoneType.SKIP_CAPTURE
    
    def test_consent_persistence_edge_cases(self, consent_manager):
        """Test edge cases in consent persistence"""
        # Test consent with very long expiration
        far_future = time.time() + (365 * 24 * 3600)  # 1 year
        consent_manager.request_consent("long_term", ConsentLevel.ENHANCED, expires_hours=365*24)
        
        current = consent_manager.get_current_consent("long_term")
        assert current is not None
        assert not current.is_expired()
        
        # Test multiple consents for same feature (should use most recent)
        consent_manager.request_consent("multi_consent", ConsentLevel.BASIC)
        time.sleep(0.1)  # Small delay
        consent_manager.request_consent("multi_consent", ConsentLevel.ENHANCED)
        
        current = consent_manager.get_current_consent("multi_consent")
        assert current.consent_level == ConsentLevel.STANDARD  # Auto-granted level
        
        # Test clearing expired consents with mixed data
        consent_manager.request_consent("expire_test", ConsentLevel.STANDARD, expires_hours=1/3600)
        time.sleep(1.1)
        
        initial_count = len(consent_manager.consent_records)
        consent_manager.clear_expired_consents()
        final_count = len(consent_manager.consent_records)
        
        assert final_count < initial_count  # Should have removed expired consent 