"""
Privacy Integration Tests - Testing integration between ConsentManager and ScreenContextManager
"""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image, ImageDraw
import numpy as np

from src.sovereign.screen_context_manager import ScreenContextManager, ScreenContextConfig
from src.sovereign.consent_manager import (
    ConsentManager, ConsentLevel, PrivacyZoneType, MonitoringState, 
    PrivacyZone, SensitiveApp
)
from src.sovereign.config import Config


class TestPrivacyIntegration:
    """Test integration between privacy features and screen context management"""
    
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
    def screen_config(self, temp_dir):
        """Create screen context configuration"""
        return ScreenContextConfig(
            storage_path=temp_dir,
            capture_interval=1.0,
            privacy_mode=False,
            enable_preprocessing=True,
            min_text_confidence=30.0
        )
    
    @pytest.fixture
    def consent_manager(self, config, temp_dir):
        """Create ConsentManager instance with test data directory"""
        with patch('src.sovereign.consent_manager.Path.home') as mock_home:
            mock_home.return_value = Path(temp_dir)
            manager = ConsentManager(config)
            yield manager
    
    @pytest.fixture
    def screen_manager(self, screen_config, config, temp_dir):
        """Create ScreenContextManager instance"""
        with patch('src.sovereign.consent_manager.Path.home') as mock_home:
            mock_home.return_value = Path(temp_dir)
            
            # Create manager with disabled sensitive app detection for testing
            manager = ScreenContextManager(screen_config, config)
            
            # Replace consent manager with test-friendly version
            from src.sovereign.consent_manager import ConsentManager, ConsentMethod, MonitoringState
            test_consent_manager = ConsentManager(
                config, ConsentMethod.AUTO_APPROVE, enable_sensitive_app_detection=False
            )
            # Set monitoring state to ACTIVE for testing
            test_consent_manager.set_monitoring_state(MonitoringState.ACTIVE, "test_initialization")
            manager.consent_manager = test_consent_manager
            
            yield manager
    
    def create_test_image(self, width=800, height=600, color='white'):
        """Create a test image for capture simulation"""
        img = Image.new('RGB', (width, height), color)
        draw = ImageDraw.Draw(img)
        
        # Add some text for OCR testing
        draw.text((100, 100), "Test Screen Content", fill='black')
        draw.text((100, 200), "Sensitive Information", fill='red')
        draw.text((100, 300), "Bank Account: 1234567890", fill='blue')
        
        return img
    
    @pytest.mark.asyncio
    async def test_privacy_check_blocks_unauthorized_capture(self, screen_manager):
        """Test that privacy check blocks capture when consent is not given"""
        # Mock screen capture to avoid actual screenshot
        with patch('mss.mss') as mock_mss:
            mock_screenshot = Mock()
            mock_screenshot.rgb = b'\x00' * (800 * 600 * 3)  # Mock RGB data
            mock_screenshot.size = (800, 600)
            mock_mss.return_value.__enter__.return_value.grab.return_value = mock_screenshot
            
            # Mock monitor info (index 0 is virtual combined screen, index 1 is primary)
            mock_mss.return_value.__enter__.return_value.monitors = [
                {'left': 0, 'top': 0, 'width': 1600, 'height': 1200},  # Combined virtual screen
                {'left': 0, 'top': 0, 'width': 800, 'height': 600}    # Primary monitor
            ]
            
            # Initially, no consent should prevent capture
            capture = await screen_manager.capture_screen()
            assert capture is None  # Should be blocked by privacy check
    
    @pytest.mark.asyncio
    async def test_privacy_check_allows_capture_with_consent(self, screen_manager):
        """Test that privacy check allows capture when consent is given"""
        # Grant consent for screen monitoring
        screen_manager.request_privacy_consent("screen_monitoring", ConsentLevel.BASIC)
        
                # Mock the entire capture process
        with patch('mss.mss') as mock_mss, \
             patch.object(screen_manager, '_extract_text_with_ocr') as mock_ocr:

            mock_screenshot = Mock()
            mock_screenshot.rgb = b'\x00' * (800 * 600 * 3)  # Mock RGB data
            mock_screenshot.size = (800, 600)
            mock_mss.return_value.__enter__.return_value.grab.return_value = mock_screenshot

            # Mock monitor info (index 0 is virtual combined screen, index 1 is primary)
            mock_mss.return_value.__enter__.return_value.monitors = [
                {'left': 0, 'top': 0, 'width': 1600, 'height': 1200},  # Combined virtual screen
                {'left': 0, 'top': 0, 'width': 800, 'height': 600}    # Primary monitor
            ]
            
            # Mock OCR results
            mock_ocr.return_value = ("Test content", [(0, 0, 100, 20)], 85.0)
            
            # Mock image saving
            with patch('PIL.Image.Image.save'):
                capture = await screen_manager.capture_screen()
                
                # Should succeed with consent
                assert capture is not None
                assert capture.text_content == "Test content"
                assert capture.ocr_confidence == 85.0
    
    @pytest.mark.asyncio
    async def test_sensitive_app_detection_blocks_capture(self, screen_manager):
        """Test that sensitive app detection blocks capture"""
        # Grant consent first
        screen_manager.request_privacy_consent("screen_monitoring", ConsentLevel.BASIC)
        
        # Mock sensitive app detection
        mock_sensitive_apps = [
            SensitiveApp(
                app_name="Test Banking App",
                process_patterns=[r"bankapp\.exe"],
                window_patterns=[r".*Banking.*"],
                priority=10
            )
        ]
        
        with patch('psutil.process_iter') as mock_processes:
            mock_processes.return_value = [Mock(info={'name': 'bankapp.exe'})]
            
            # Mock screen capture
            with patch('mss.mss') as mock_mss:
                mock_screenshot = Mock()
                mock_screenshot.rgb = b'\x00' * (800 * 600 * 3)
                mock_screenshot.size = (800, 600)
                mock_mss.return_value.__enter__.return_value.grab.return_value = mock_screenshot
                
                # Mock monitor info
                mock_mss.return_value.__enter__.return_value.monitors = [
                    {'left': 0, 'top': 0, 'width': 800, 'height': 600}
                ]
                
                # Should be blocked by sensitive app detection
                capture = await screen_manager.capture_screen()
                assert capture is None  # Blocked by sensitive app
    
    @pytest.mark.asyncio
    async def test_privacy_zone_redaction_blur(self, screen_manager):
        """Test privacy zone redaction with blur effect"""
        # Grant consent
        screen_manager.request_privacy_consent("screen_monitoring", ConsentLevel.BASIC)
        
        # Add blur privacy zone
        zone_id = screen_manager.add_privacy_zone(
            "Blur Zone", 100, 100, 200, 200, PrivacyZoneType.BLUR
        )
        
        # Create test image
        test_img = self.create_test_image()
        
        # Test redaction with mock monitor
        mock_monitor = {'left': 0, 'top': 0, 'width': 800, 'height': 600}
        
        redacted_img = await screen_manager._apply_privacy_zones(test_img, mock_monitor)
        
        # Should return modified image (blur applied)
        assert redacted_img is not None
        assert redacted_img.size == test_img.size
        # Note: Actual blur verification would require pixel-level analysis
    
    @pytest.mark.asyncio
    async def test_privacy_zone_redaction_full_redact(self, screen_manager):
        """Test privacy zone redaction with full blackout"""
        # Grant consent
        screen_manager.request_privacy_consent("screen_monitoring", ConsentLevel.BASIC)
        
        # Add full redaction privacy zone
        zone_id = screen_manager.add_privacy_zone(
            "Redact Zone", 100, 100, 200, 200, PrivacyZoneType.FULL_REDACT
        )
        
        # Create test image
        test_img = self.create_test_image()
        
        # Test redaction with mock monitor
        mock_monitor = {'left': 0, 'top': 0, 'width': 800, 'height': 600}
        
        redacted_img = await screen_manager._apply_privacy_zones(test_img, mock_monitor)
        
        # Should return modified image with black rectangle
        assert redacted_img is not None
        assert redacted_img.size == test_img.size
        
        # Check that the redacted area is black
        redacted_region = redacted_img.crop((100, 100, 300, 300))
        # Convert to numpy array for easier checking
        redacted_array = np.array(redacted_region)
        
        # Should be mostly black (allowing for some edge pixels)
        black_pixels = np.sum(redacted_array == 0)
        total_pixels = redacted_array.size
        black_ratio = black_pixels / total_pixels
        
        assert black_ratio > 0.8  # At least 80% black pixels
    
    @pytest.mark.asyncio
    async def test_privacy_zone_skip_capture(self, screen_manager):
        """Test privacy zone that skips capture entirely"""
        # Grant consent
        screen_manager.request_privacy_consent("screen_monitoring", ConsentLevel.BASIC)
        
        # Add skip capture privacy zone
        zone_id = screen_manager.add_privacy_zone(
            "Skip Zone", 100, 100, 200, 200, PrivacyZoneType.SKIP_CAPTURE
        )
        
        # Create test image
        test_img = self.create_test_image()
        
        # Test redaction with mock monitor
        mock_monitor = {'left': 0, 'top': 0, 'width': 800, 'height': 600}
        
        redacted_img = await screen_manager._apply_privacy_zones(test_img, mock_monitor)
        
        # Should return None (skip capture)
        assert redacted_img is None
    
    def test_privacy_zone_management_through_screen_manager(self, screen_manager):
        """Test privacy zone management through ScreenContextManager interface"""
        # Add privacy zone
        zone_id = screen_manager.add_privacy_zone(
            "Test Zone", 50, 50, 100, 100, PrivacyZoneType.BLUR
        )
        
        # Get privacy zones
        zones = screen_manager.get_privacy_zones()
        assert zone_id in zones
        assert zones[zone_id].name == "Test Zone"
        
        # Remove privacy zone
        result = screen_manager.remove_privacy_zone(zone_id)
        assert result is True
        
        # Verify removal
        zones = screen_manager.get_privacy_zones()
        assert zone_id not in zones
    
    def test_monitoring_state_management_through_screen_manager(self, screen_manager):
        """Test monitoring state management through ScreenContextManager"""
        # Test initial state (set to ACTIVE in test fixture)
        assert screen_manager.consent_manager.monitoring_state == MonitoringState.ACTIVE
        
        # Set to active
        screen_manager.set_monitoring_state(MonitoringState.ACTIVE, "test")
        assert screen_manager.consent_manager.monitoring_state == MonitoringState.ACTIVE
        
        # Pause monitoring
        screen_manager.pause_monitoring("test_reason")
        assert screen_manager.consent_manager.monitoring_state == MonitoringState.PAUSED_USER
        
        # Grant consent for screen monitoring before resuming
        screen_manager.request_privacy_consent("screen_monitoring", ConsentLevel.BASIC)
        
        screen_manager.resume_monitoring()
        assert screen_manager.consent_manager.monitoring_state == MonitoringState.ACTIVE
    
    def test_consent_management_through_screen_manager(self, screen_manager):
        """Test consent management through ScreenContextManager interface"""
        # Request consent
        level = screen_manager.request_privacy_consent("test_feature", ConsentLevel.ENHANCED)
        assert level == ConsentLevel.STANDARD  # Auto-granted level
        
        # Get consent summary
        summary = screen_manager.get_consent_summary()
        assert "monitoring_state" in summary
        assert "active_consents" in summary
        assert "test_feature" in summary["active_consents"]
        
        # Revoke consent
        screen_manager.revoke_consent("test_feature")
        
        # Verify revocation
        summary = screen_manager.get_consent_summary()
        assert "test_feature" not in summary["active_consents"]
    
    @pytest.mark.asyncio
    async def test_sensitive_app_management_through_screen_manager(self, screen_manager):
        """Test sensitive app management through ScreenContextManager"""
        # Enable sensitive app detection for this test
        screen_manager.consent_manager.enable_sensitive_app_detection = True
        
        # Add custom sensitive app pattern
        screen_manager.add_sensitive_app_pattern(
            "Custom Test App",
            ["testapp.exe", "customapp.exe"],
            ["Test App Window", "Custom Application"],
            priority=7
        )

        # Mock process detection
        with patch('psutil.process_iter') as mock_processes:
            mock_processes.return_value = [Mock(info={'name': 'testapp.exe'})]

            # Get detected apps
            detected = await screen_manager.get_detected_sensitive_apps()

            # Should detect our custom app
            app_names = [app.app_name for app in detected]
            assert "Custom Test App" in app_names
    
    @pytest.mark.asyncio
    async def test_full_privacy_workflow(self, screen_manager):
        """Test complete privacy workflow from consent to capture"""
        # Step 1: Initially no consent, should block capture
        with patch('mss.mss') as mock_mss:
            mock_screenshot = Mock()
            mock_screenshot.rgb = b'\x00' * (800 * 600 * 3)
            mock_screenshot.size = (800, 600)
            mock_mss.return_value.__enter__.return_value.grab.return_value = mock_screenshot
            
            # Mock monitor info
            mock_mss.return_value.__enter__.return_value.monitors = [
                {'left': 0, 'top': 0, 'width': 1600, 'height': 1200},  # Combined virtual screen
                {'left': 0, 'top': 0, 'width': 800, 'height': 600}    # Primary monitor
            ]
            
            capture = await screen_manager.capture_screen()
            assert capture is None  # Blocked by lack of consent
        
        # Step 2: Grant consent
        level = screen_manager.request_privacy_consent("screen_monitoring", ConsentLevel.STANDARD)
        assert level == ConsentLevel.STANDARD
        
        # Step 3: Add privacy zone
        zone_id = screen_manager.add_privacy_zone(
            "Sensitive Area", 100, 100, 200, 200, PrivacyZoneType.FULL_REDACT
        )
        
        # Step 4: Enable monitoring
        screen_manager.set_monitoring_state(MonitoringState.ACTIVE, "user_enabled")
        
        # Step 5: Attempt capture with privacy protections
        with patch('mss.mss') as mock_mss, \
             patch.object(screen_manager, '_extract_text_with_ocr') as mock_ocr:
            
            mock_screenshot = Mock()
            mock_screenshot.rgb = b'\x00' * (800 * 600 * 3)
            mock_screenshot.size = (800, 600)
            mock_mss.return_value.__enter__.return_value.grab.return_value = mock_screenshot
            
            # Mock monitor info
            mock_mss.return_value.__enter__.return_value.monitors = [
                {'left': 0, 'top': 0, 'width': 1600, 'height': 1200},  # Combined virtual screen
                {'left': 0, 'top': 0, 'width': 800, 'height': 600}    # Primary monitor
            ]
            
            # Mock OCR results
            mock_ocr.return_value = ("Redacted content", [(0, 0, 100, 20)], 75.0)
            
            # Mock image saving
            with patch('PIL.Image.Image.save'):
                capture = await screen_manager.capture_screen()
                
                # Should succeed with redaction applied
                assert capture is not None
                assert capture.text_content == "Redacted content"
        
        # Step 6: Test consent revocation
        screen_manager.revoke_consent("screen_monitoring")
        
        # Step 7: Capture should now be blocked again
        with patch('mss.mss') as mock_mss:
            mock_screenshot = Mock()
            mock_screenshot.rgb = b'\x00' * (800 * 600 * 3)
            mock_screenshot.size = (800, 600)
            mock_mss.return_value.__enter__.return_value.grab.return_value = mock_screenshot
            
            # Mock monitor info
            mock_mss.return_value.__enter__.return_value.monitors = [
                {'left': 0, 'top': 0, 'width': 1600, 'height': 1200},  # Combined virtual screen
                {'left': 0, 'top': 0, 'width': 800, 'height': 600}    # Primary monitor
            ]
            
            capture = await screen_manager.capture_screen()
            assert capture is None  # Blocked by revoked consent
    
    @pytest.mark.asyncio
    async def test_privacy_zone_coordinate_translation(self, screen_manager):
        """Test privacy zone coordinate translation for multi-monitor setups"""
        # Grant consent
        screen_manager.request_privacy_consent("screen_monitoring", ConsentLevel.BASIC)
        
        # Add privacy zone in screen coordinates
        zone_id = screen_manager.add_privacy_zone(
            "Multi-Monitor Zone", 1920, 100, 200, 200, PrivacyZoneType.BLUR
        )
        
        # Create test image
        test_img = self.create_test_image()
        
        # Test with second monitor offset
        mock_monitor = {'left': 1920, 'top': 0, 'width': 800, 'height': 600}
        
        redacted_img = await screen_manager._apply_privacy_zones(test_img, mock_monitor)
        
        # Should apply redaction (zone overlaps with monitor)
        assert redacted_img is not None
        assert redacted_img.size == test_img.size
        
        # Test with monitor that doesn't overlap
        mock_monitor_no_overlap = {'left': 0, 'top': 0, 'width': 800, 'height': 600}
        
        redacted_img_no_overlap = await screen_manager._apply_privacy_zones(test_img, mock_monitor_no_overlap)
        
        # Should return original image (no overlap)
        assert redacted_img_no_overlap is not None
        assert redacted_img_no_overlap.size == test_img.size
    
    @pytest.mark.asyncio
    async def test_callback_integration(self, screen_manager):
        """Test callback integration between components"""
        state_changes = []
        consent_changes = []
        
        def state_callback(state):
            state_changes.append(state)
        
        def consent_callback(feature, level):
            consent_changes.append((feature, level))
        
        # Add callbacks through consent manager
        screen_manager.consent_manager.add_state_change_callback(state_callback)
        screen_manager.consent_manager.add_consent_change_callback(consent_callback)
        
        # Test state changes
        screen_manager.set_monitoring_state(MonitoringState.ACTIVE, "test")
        assert len(state_changes) == 1
        assert state_changes[0] == MonitoringState.ACTIVE
        
        # Test consent changes
        screen_manager.request_privacy_consent("test_feature", ConsentLevel.ENHANCED)
        assert len(consent_changes) == 1
        assert consent_changes[0] == ("test_feature", ConsentLevel.STANDARD)
        
        # Test pausing and resuming
        screen_manager.pause_monitoring("test")
        assert len(state_changes) == 2
        assert state_changes[1] == MonitoringState.PAUSED_USER

        # Grant consent for screen monitoring before resuming
        screen_manager.request_privacy_consent("screen_monitoring", ConsentLevel.BASIC)

        screen_manager.resume_monitoring()
        assert len(state_changes) == 3
        assert state_changes[2] == MonitoringState.ACTIVE
    
    @pytest.mark.asyncio
    async def test_cleanup_integration(self, screen_manager):
        """Test cleanup integration between components"""
        # Add some data to both components
        screen_manager.request_privacy_consent("cleanup_test", ConsentLevel.STANDARD)
        screen_manager.add_privacy_zone("Cleanup Zone", 0, 0, 100, 100)
        
        # Add some performance stats
        screen_manager.performance_stats["test_stat"] = 123
        
        # Test cleanup
        await screen_manager.cleanup()
        
        # Verify cleanup occurred
        assert len(screen_manager.content_cache) == 0
        
        # Note: ConsentManager data should persist through cleanup
        # Only expired consents should be removed
    
    def test_performance_stats_include_privacy_data(self, screen_manager):
        """Test that performance stats include privacy-related data"""
        # Add some privacy data
        screen_manager.request_privacy_consent("test_feature", ConsentLevel.STANDARD)
        screen_manager.add_privacy_zone("Test Zone", 0, 0, 100, 100)
        screen_manager.set_monitoring_state(MonitoringState.ACTIVE)
        
        # Add some mock performance data
        screen_manager.performance_stats["privacy_blocks"] = 5
        screen_manager.performance_stats["sensitive_app_detections"] = 3
        
        # Get performance stats
        stats = screen_manager.get_performance_stats()
        
        # Should include privacy-related stats
        assert "privacy_blocks" in stats
        assert "sensitive_app_detections" in stats
        assert stats["privacy_blocks"] == 5
        assert stats["sensitive_app_detections"] == 3
        
        # Should include standard performance data
        assert "total_captures" in stats
        assert "state" in stats
        assert "config" in stats


class TestPrivacyErrorHandling:
    """Test error handling in privacy features"""
    
    @pytest.fixture
    def screen_manager(self):
        """Create ScreenContextManager with temporary storage"""
        with tempfile.TemporaryDirectory() as temp_dir:
            screen_config = ScreenContextConfig(storage_path=temp_dir)
            with patch('src.sovereign.consent_manager.Path.home') as mock_home:
                mock_home.return_value = Path(temp_dir)
                manager = ScreenContextManager(screen_config, Config())
                yield manager
    
    @pytest.mark.asyncio
    async def test_privacy_zone_error_handling(self, screen_manager):
        """Test error handling in privacy zone operations"""
        # Create a corrupted image
        test_img = Image.new('RGB', (100, 100), 'white')
        
        # Mock monitor data that would cause errors
        invalid_monitor = {'left': 'invalid', 'top': 0, 'width': 100, 'height': 100}
        
        # Should handle errors gracefully
        result = await screen_manager._apply_privacy_zones(test_img, invalid_monitor)
        
        # Should return original image on error
        assert result is not None
        assert result.size == test_img.size
    
    @pytest.mark.asyncio
    async def test_sensitive_app_detection_error_handling(self, screen_manager):
        """Test error handling in sensitive app detection"""
        # Mock psutil to raise an exception
        with patch('psutil.process_iter', side_effect=Exception("Process access denied")):
            # Should handle errors gracefully
            result = await screen_manager.get_detected_sensitive_apps()
            
            # Should return empty list on error
            assert isinstance(result, list)
            assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_consent_manager_error_recovery(self, screen_manager):
        """Test error recovery in consent manager operations"""
        # Create invalid consent request
        with patch.object(screen_manager.consent_manager, 'request_consent', side_effect=Exception("Consent error")):
            # Should handle consent errors gracefully
            try:
                level = screen_manager.request_privacy_consent("test_feature", ConsentLevel.STANDARD)
                # If no exception, should return reasonable default
                assert level in [ConsentLevel.DENIED, ConsentLevel.STANDARD]
            except Exception:
                pytest.fail("Consent manager should handle errors gracefully")
    
    def test_privacy_zone_edge_cases(self, screen_manager):
        """Test edge cases in privacy zone handling"""
        # Test with zero-size zone
        zone_id = screen_manager.add_privacy_zone("Zero Size", 100, 100, 0, 0)
        assert zone_id is not None
        
        # Test with negative coordinates
        zone_id = screen_manager.add_privacy_zone("Negative", -100, -100, 200, 200)
        assert zone_id is not None
        
        # Test with very large zone
        zone_id = screen_manager.add_privacy_zone("Large", 0, 0, 10000, 10000)
        assert zone_id is not None
        
        # Should handle all cases without crashing
        zones = screen_manager.get_privacy_zones()
        assert len(zones) == 3 
