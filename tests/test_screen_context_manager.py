"""
Tests for ScreenContextManager
"""

import asyncio
import pytest
import tempfile
import os
import time
import sqlite3
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path

from PIL import Image
import numpy as np

from src.sovereign.screen_context_manager import (
    ScreenContextManager,
    ScreenContextConfig,
    CaptureState,
    ScreenCapture
)
from src.sovereign.config import Config


class TestScreenContextManager:
    """Test cases for ScreenContextManager"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration"""
        return ScreenContextConfig(
            capture_interval=1.0,
            max_stored_captures=10,
            storage_path=temp_dir,
            privacy_mode=False
        )
    
    @pytest.fixture
    def screen_manager(self, config):
        """Create ScreenContextManager instance"""
        main_config = Config()
        return ScreenContextManager(config, main_config)
    
    @pytest.fixture
    def mock_image(self):
        """Create mock PIL Image"""
        # Create a simple test image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    
    def test_init_default(self):
        """Test initialization with default config"""
        main_config = Config()
        manager = ScreenContextManager(main_config=main_config)
        
        assert manager.config.capture_interval == 4.0
        assert manager.config.max_stored_captures == 100
        assert manager.state == CaptureState.STOPPED
        assert manager.capture_count == 0
        assert isinstance(manager.storage_path, Path)
        assert manager.storage_path.exists()
    
    def test_init_with_config(self, config):
        """Test initialization with custom config"""
        main_config = Config()
        manager = ScreenContextManager(config, main_config)
        
        assert manager.config.capture_interval == 1.0
        assert manager.config.max_stored_captures == 10
        assert manager.config.storage_path == config.storage_path
        assert manager.state == CaptureState.STOPPED
    
    def test_database_initialization(self, screen_manager):
        """Test database initialization"""
        # Database should be created
        assert screen_manager.db_path.exists()
        
        # Check table structure
        conn = sqlite3.connect(str(screen_manager.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='captures'")
        result = cursor.fetchone()
        assert result is not None
        
        # Check index
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_timestamp'")
        result = cursor.fetchone()
        assert result is not None
        
        conn.close()
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, screen_manager):
        """Test successful initialization"""
        with patch.object(screen_manager, '_test_capture', return_value=True):
            result = await screen_manager.initialize()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, screen_manager):
        """Test initialization failure"""
        with patch.object(screen_manager, '_test_capture', return_value=False):
            result = await screen_manager.initialize()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_initialize_with_main_config(self, screen_manager):
        """Test initialization with main config"""
        main_config = Mock()
        main_config.screen_capture_interval = 2.0
        main_config.privacy_mode = True
        
        with patch.object(screen_manager, '_test_capture', return_value=True):
            await screen_manager.initialize(main_config)
            
            assert screen_manager.config.capture_interval == 2.0
            assert screen_manager.config.privacy_mode is True
    
    @pytest.mark.asyncio
    async def test_test_capture_success(self, screen_manager, mock_image):
        """Test successful screen capture test"""
        mock_sct = Mock()
        mock_sct.monitors = [None, {"width": 1920, "height": 1080, "left": 0, "top": 0}]
        mock_screenshot = Mock()
        mock_screenshot.size = (1920, 1080)
        mock_screenshot.rgb = b'fake_rgb_data' * (1920 * 1080 * 3 // 13)  # Approximate size
        mock_sct.grab.return_value = mock_screenshot
        
        with patch('mss.mss') as mock_mss:
            mock_mss.return_value.__enter__.return_value = mock_sct
            with patch('PIL.Image.frombytes', return_value=mock_image):
                result = await screen_manager._test_capture()
                assert result is True
    
    @pytest.mark.asyncio
    async def test_test_capture_failure(self, screen_manager):
        """Test screen capture test failure"""
        with patch('mss.mss') as mock_mss:
            mock_mss.side_effect = Exception("Screen capture failed")
            result = await screen_manager._test_capture()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_capture_screen_success(self, screen_manager, mock_image):
        """Test successful screen capture"""
        mock_sct = Mock()
        mock_sct.monitors = [None, {"width": 1920, "height": 1080, "left": 0, "top": 0}]
        mock_screenshot = Mock()
        mock_screenshot.size = (1920, 1080)
        mock_screenshot.rgb = b'fake_rgb_data' * (1920 * 1080 * 3 // 13)
        mock_sct.grab.return_value = mock_screenshot
        
        with patch('mss.mss') as mock_mss:
            mock_mss.return_value.__enter__.return_value = mock_sct
            with patch('PIL.Image.frombytes', return_value=mock_image):
                with patch.object(screen_manager, '_extract_text_with_ocr', return_value=("test text", [(0, 0, 100, 20)], 85.0)):
                    with patch.object(screen_manager, '_is_sensitive_app_active', return_value=False):
                        with patch.object(screen_manager, '_store_capture'):
                            capture = await screen_manager.capture_screen()
                            
                            assert capture is not None
                            assert isinstance(capture, ScreenCapture)
                            assert capture.text_content == "test text"
                            assert capture.ocr_confidence == 85.0
                            assert len(capture.bounding_boxes) == 1
    
    @pytest.mark.asyncio
    async def test_capture_screen_privacy_mode(self, screen_manager):
        """Test screen capture in privacy mode"""
        screen_manager.config.privacy_mode = True
        
        capture = await screen_manager.capture_screen()
        assert capture is None
    
    @pytest.mark.asyncio
    async def test_capture_screen_sensitive_app(self, screen_manager):
        """Test screen capture with sensitive app active"""
        with patch.object(screen_manager, '_is_sensitive_app_active', return_value=True):
            capture = await screen_manager.capture_screen()
            assert capture is None
    
    @pytest.mark.asyncio
    async def test_capture_screen_duplicate_detection(self, screen_manager, mock_image):
        """Test duplicate capture detection"""
        # First capture
        capture1 = ScreenCapture(
            id="test_id",
            timestamp=time.time(),
            image_path="test_path",
            text_content="test content",
            bounding_boxes=[],
            monitor_info={},
            processing_time=1.0,
            ocr_confidence=80.0
        )
        
        screen_manager.content_cache["test_id"] = capture1
        
        with patch.object(screen_manager, '_generate_capture_id', return_value="test_id"):
            capture2 = await screen_manager.capture_screen()
            
            assert capture2 is not None
            assert capture2.id == "test_id"
            assert screen_manager.performance_stats["cache_hits"] == 1
    
    @pytest.mark.asyncio
    async def test_extract_text_with_ocr_success(self, screen_manager, mock_image):
        """Test successful OCR text extraction"""
        mock_data = {
            'text': ['Hello', 'World', '', 'Test'],
            'conf': [90, 85, 30, 95],
            'left': [10, 50, 0, 100],
            'top': [10, 10, 0, 30],
            'width': [40, 45, 0, 35],
            'height': [20, 20, 0, 20]
        }
        
        with patch('pytesseract.image_to_data', return_value=mock_data):
            with patch.object(screen_manager, '_preprocess_image', return_value=mock_image):
                text, boxes, confidence = await screen_manager._extract_text_with_ocr(mock_image)
                
                assert text == "Hello World Test"
                assert len(boxes) == 3  # Only items with confidence > 30
                assert confidence > 0
    
    @pytest.mark.asyncio
    async def test_extract_text_with_ocr_failure(self, screen_manager, mock_image):
        """Test OCR text extraction failure"""
        with patch('pytesseract.image_to_data', side_effect=Exception("OCR failed")):
            text, boxes, confidence = await screen_manager._extract_text_with_ocr(mock_image)
            
            assert text == ""
            assert boxes == []
            assert confidence == 0.0
    
    def test_preprocess_image_success(self, screen_manager):
        """Test image preprocessing"""
        # Create test image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        processed = screen_manager._preprocess_image(img)
        
        assert processed is not None
        assert processed.mode == 'L'  # Should be grayscale
        # Should be upscaled since original is 100x100
        assert processed.size[0] >= 1000 or processed.size[1] >= 1000
    
    def test_preprocess_image_failure(self, screen_manager, mock_image):
        """Test image preprocessing failure"""
        with patch.object(mock_image, 'convert', side_effect=Exception("Processing failed")):
            processed = screen_manager._preprocess_image(mock_image)
            
            # Should return original image on failure
            assert processed is mock_image
    
    def test_generate_capture_id_success(self, screen_manager, mock_image):
        """Test capture ID generation"""
        capture_id = screen_manager._generate_capture_id(mock_image)
        
        assert isinstance(capture_id, str)
        assert len(capture_id) == 16
        
        # Same image should generate same ID
        capture_id2 = screen_manager._generate_capture_id(mock_image)
        assert capture_id == capture_id2
    
    def test_generate_capture_id_failure(self, screen_manager, mock_image):
        """Test capture ID generation failure"""
        with patch.object(mock_image, 'tobytes', side_effect=Exception("Failed")):
            capture_id = screen_manager._generate_capture_id(mock_image)
            
            assert isinstance(capture_id, str)
            assert len(capture_id) == 16
    
    @pytest.mark.asyncio
    async def test_is_sensitive_app_active(self, screen_manager):
        """Test sensitive application detection"""
        # Currently returns False as placeholder
        result = await screen_manager._is_sensitive_app_active()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_store_capture_success(self, screen_manager):
        """Test successful capture storage"""
        capture = ScreenCapture(
            id="test_id",
            timestamp=time.time(),
            image_path="test_path",
            text_content="test content",
            bounding_boxes=[(0, 0, 100, 20)],
            monitor_info={"width": 1920, "height": 1080},
            processing_time=1.0,
            ocr_confidence=80.0
        )
        
        await screen_manager._store_capture(capture)
        
        # Verify storage
        conn = sqlite3.connect(str(screen_manager.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM captures WHERE id = ?", (capture.id,))
        result = cursor.fetchone()
        conn.close()
        
        assert result is not None
        assert result[0] == capture.id
        assert result[3] == capture.text_content
    
    @pytest.mark.asyncio
    async def test_cleanup_old_captures(self, screen_manager):
        """Test cleanup of old captures"""
        # Set low limit for testing
        screen_manager.config.max_stored_captures = 2
        
        # Add test captures
        for i in range(5):
            capture = ScreenCapture(
                id=f"test_{i}",
                timestamp=time.time() - i,  # Older timestamps
                image_path=f"test_path_{i}",
                text_content=f"test content {i}",
                bounding_boxes=[],
                monitor_info={},
                processing_time=1.0,
                ocr_confidence=80.0
            )
            await screen_manager._store_capture(capture)
        
        # Run cleanup
        await screen_manager._cleanup_old_captures()
        
        # Check remaining captures
        conn = sqlite3.connect(str(screen_manager.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM captures")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 2
    
    @pytest.mark.asyncio
    async def test_start_stop_capture(self, screen_manager):
        """Test starting and stopping capture"""
        # Test start
        await screen_manager.start_capture()
        assert screen_manager.state == CaptureState.RUNNING
        assert screen_manager.capture_task is not None
        
        # Test stop
        await screen_manager.stop_capture()
        assert screen_manager.state == CaptureState.STOPPED
        
        # Small delay to ensure task is cancelled
        await asyncio.sleep(0.1)
    
    @pytest.mark.asyncio
    async def test_start_capture_already_running(self, screen_manager):
        """Test starting capture when already running"""
        await screen_manager.start_capture()
        
        # Try to start again
        await screen_manager.start_capture()
        assert screen_manager.state == CaptureState.RUNNING
        
        await screen_manager.stop_capture()
    
    def test_toggle_capture_on(self, screen_manager):
        """Test toggle capture on (no event loop)"""
        with patch.object(screen_manager, 'start_capture') as mock_start:
            with patch.object(screen_manager.logger, 'warning') as mock_warn:
                screen_manager.toggle_capture(True)
                # Should log warning about no event loop
                mock_warn.assert_called_once()
                # Should not call start_capture since no event loop
                assert not mock_start.called
    
    def test_toggle_capture_off(self, screen_manager):
        """Test toggle capture off (no event loop)"""
        with patch.object(screen_manager, 'stop_capture') as mock_stop:
            with patch.object(screen_manager.logger, 'warning') as mock_warn:
                screen_manager.toggle_capture(False)
                # Should log warning about no event loop
                mock_warn.assert_called_once()
                # Should not call stop_capture since no event loop
                assert not mock_stop.called
    
    @pytest.mark.asyncio
    async def test_toggle_capture_with_event_loop(self, screen_manager):
        """Test toggle capture when event loop is available"""
        with patch.object(screen_manager, 'start_capture') as mock_start:
            with patch.object(screen_manager, 'stop_capture') as mock_stop:
                # Test toggle on
                screen_manager.toggle_capture(True)
                await asyncio.sleep(0.1)  # Let task execute
                
                # Test toggle off  
                screen_manager.toggle_capture(False)
                await asyncio.sleep(0.1)  # Let task execute
    
    def test_get_recent_captures(self, screen_manager):
        """Test getting recent captures"""
        # Add test capture to database
        conn = sqlite3.connect(str(screen_manager.db_path))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO captures (id, timestamp, image_path, text_content, bounding_boxes, monitor_info, processing_time, ocr_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', ("test_id", time.time(), "test_path", "test content", "[]", "{}", 1.0, 80.0))
        conn.commit()
        conn.close()
        
        captures = screen_manager.get_recent_captures(limit=5)
        
        assert len(captures) == 1
        assert captures[0].id == "test_id"
        assert captures[0].text_content == "test content"
    
    def test_get_recent_captures_empty(self, screen_manager):
        """Test getting recent captures when none exist"""
        captures = screen_manager.get_recent_captures()
        assert len(captures) == 0
    
    def test_get_performance_stats(self, screen_manager):
        """Test getting performance statistics"""
        # Set some test data
        screen_manager.capture_count = 5
        screen_manager.last_capture_time = time.time()
        screen_manager.performance_stats["successful_captures"] = 3
        
        stats = screen_manager.get_performance_stats()
        
        assert stats["capture_count"] == 5
        assert stats["successful_captures"] == 3
        assert stats["state"] == "stopped"
        assert "config" in stats
        assert stats["config"]["capture_interval"] == screen_manager.config.capture_interval
    
    @pytest.mark.asyncio
    async def test_cleanup(self, screen_manager):
        """Test cleanup method"""
        # Start capture first
        await screen_manager.start_capture()
        
        # Add some cache data
        screen_manager.content_cache["test"] = Mock()
        
        # Run cleanup
        await screen_manager.cleanup()
        
        assert screen_manager.state == CaptureState.STOPPED
        assert len(screen_manager.content_cache) == 0
    
    @pytest.mark.asyncio
    async def test_capture_loop_integration(self, screen_manager, mock_image):
        """Test capture loop integration"""
        # Mock the capture method to return quickly
        async def mock_capture():
            await asyncio.sleep(0.1)
            return Mock(spec=ScreenCapture)
        
        screen_manager.config.capture_interval = 0.1  # Fast for testing
        
        with patch.object(screen_manager, 'capture_screen', side_effect=mock_capture):
            # Start capture
            await screen_manager.start_capture()
            
            # Let it run briefly
            await asyncio.sleep(0.3)
            
            # Stop capture
            await screen_manager.stop_capture()
            
            # Should have made some captures
            assert screen_manager.capture_count > 0
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = ScreenContextConfig(
            capture_interval=0.5,
            max_stored_captures=1000,
            ocr_language="eng",
            privacy_mode=True
        )
        
        main_config = Config()
        manager = ScreenContextManager(config, main_config)
        
        assert manager.config.capture_interval == 0.5
        assert manager.config.max_stored_captures == 1000
        assert manager.config.ocr_language == "eng"
        assert manager.config.privacy_mode is True 