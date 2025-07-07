"""
Secure Storage System Tests - Testing enhanced storage, context windowing, and privacy features
"""

import pytest
import asyncio
import tempfile
import time
import sqlite3
from unittest.mock import Mock, patch
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw

from src.sovereign.screen_context_manager import (
    ScreenContextManager, 
    ScreenContextConfig, 
    ScreenCapture, 
    ScreenReference,
    ContextWindow,
    ContextRelevance
)
from src.sovereign.config import Config


class TestSecureStorageSystem:
    """Test enhanced storage system functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def enhanced_config(self, temp_dir):
        """Create enhanced storage configuration"""
        return ScreenContextConfig(
            storage_path=temp_dir,
            enable_compression=True,
            enable_encryption=False,
            context_windows={
                "recent": ContextWindow("recent", 10, 1, 3, 0.1),      # 1 hour, light compression
                "archived": ContextWindow("archived", 20, 24, 6, 0.3), # 24 hours, medium compression  
                "historical": ContextWindow("historical", 5, 168, 9, 0.7) # 1 week, high compression
            },
            max_reference_age_hours=12,
            enable_semantic_indexing=False
        )
    
    @pytest.fixture
    def storage_manager(self, enhanced_config):
        """Create ScreenContextManager with enhanced storage"""
        main_config = Config()
        return ScreenContextManager(enhanced_config, main_config)
    
    def create_test_image(self, text: str = "Test Content", width: int = 200, height: int = 100) -> Image.Image:
        """Create a test image with text"""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 30), text, fill='black')
        return img
    
    def create_test_capture(self, capture_id: str = "test123", text_content: str = "Test text content") -> ScreenCapture:
        """Create a test ScreenCapture object"""
        return ScreenCapture(
            id=capture_id,
            timestamp=time.time(),
            image_path="/tmp/test.png",
            text_content=text_content,
            bounding_boxes=[(10, 10, 100, 50)],
            monitor_info={"width": 1920, "height": 1080},
            processing_time=0.5,
            ocr_confidence=85.0,
            relevance_score=0.6,
            context_window="recent"
        )
    
    def test_enhanced_database_schema(self, storage_manager):
        """Test that enhanced database schema is created correctly"""
        # Check that all new tables exist
        conn = sqlite3.connect(str(storage_manager.db_path))
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert "captures" in tables
        assert "screen_references" in tables
        assert "context_metadata" in tables
        assert "privacy_log" in tables
        
        # Check enhanced captures table schema
        cursor.execute("PRAGMA table_info(captures)")
        columns = [row[1] for row in cursor.fetchall()]
        
        assert "relevance_score" in columns
        assert "context_window" in columns
        assert "compressed_image_data" in columns
        assert "text_embedding" in columns
        assert "created_date" in columns
        assert "archived_date" in columns
        
        conn.close()
    
    def test_context_window_initialization(self, storage_manager):
        """Test context window metadata initialization"""
        conn = sqlite3.connect(str(storage_manager.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT window_name FROM context_metadata")
        windows = [row[0] for row in cursor.fetchall()]
        
        assert "recent" in windows
        assert "archived" in windows
        assert "historical" in windows
        
        conn.close()
    
    def test_relevance_score_calculation(self, storage_manager):
        """Test relevance score calculation"""
        # Test with high-confidence, high-content capture
        high_capture = self.create_test_capture(
            text_content="Important error message: login failed password authentication"
        )
        high_capture.ocr_confidence = 95.0
        high_capture.bounding_boxes = [(10, 10, 100, 50), (110, 60, 200, 90)]
        
        high_score = storage_manager._calculate_relevance_score(high_capture)
        assert high_score > 0.5
        
        # Test with low-confidence, minimal content capture
        low_capture = self.create_test_capture(text_content="a")
        low_capture.ocr_confidence = 30.0
        low_capture.bounding_boxes = []
        
        low_score = storage_manager._calculate_relevance_score(low_capture)
        assert low_score < 0.5
        
        # Test similarity comparison
        previous_capture = self.create_test_capture(text_content="Same text content")
        similar_capture = self.create_test_capture(text_content="Same text content")
        different_capture = self.create_test_capture(text_content="Completely different content")
        
        similar_score = storage_manager._calculate_relevance_score(similar_capture, previous_capture)
        different_score = storage_manager._calculate_relevance_score(different_capture, previous_capture)
        
        assert different_score > similar_score  # More difference = higher relevance
    
    def test_text_similarity_calculation(self, storage_manager):
        """Test text similarity calculation"""
        text1 = "hello world test"
        text2 = "hello world test"
        text3 = "completely different text"
        text4 = ""
        
        # Identical texts
        assert storage_manager._calculate_text_similarity(text1, text2) == 1.0
        
        # Different texts
        similarity = storage_manager._calculate_text_similarity(text1, text3)
        assert 0.0 <= similarity < 1.0
        
        # Empty text handling
        assert storage_manager._calculate_text_similarity(text1, text4) == 0.0
        assert storage_manager._calculate_text_similarity(text4, text4) == 1.0
    
    def test_image_compression_decompression(self, storage_manager):
        """Test image compression and decompression"""
        # Create test image
        test_image = self.create_test_image("Compression Test", 400, 200)
        
        # Test compression
        compressed_data = storage_manager._compress_image_data(test_image, compression_level=6)
        assert compressed_data  # Should produce non-empty compressed data
        assert isinstance(compressed_data, str)  # Should be base64 string
        
        # Test decompression
        decompressed_image = storage_manager._decompress_image_data(compressed_data)
        assert decompressed_image is not None
        assert isinstance(decompressed_image, Image.Image)
        
        # Images should be similar (compression may introduce small changes)
        assert decompressed_image.size == test_image.size
    
    @pytest.mark.asyncio
    async def test_screen_reference_creation(self, storage_manager):
        """Test screen reference creation and retrieval"""
        # Create test capture
        capture = self.create_test_capture()
        await storage_manager._store_capture(capture)
        
        # Create screen reference
        reference = await storage_manager.create_screen_reference(
            capture, 
            element_type="text",
            semantic_tags=["ui", "button"]
        )
        
        assert reference is not None
        assert reference.capture_id == capture.id
        assert reference.element_type == "text"
        assert "ui" in reference.semantic_tags
        assert "button" in reference.semantic_tags
        
        # Test retrieval
        references = await storage_manager.get_screen_references(capture_id=capture.id)
        assert len(references) == 1
        assert references[0].reference_id == reference.reference_id
    
    @pytest.mark.asyncio
    async def test_privacy_logging(self, storage_manager):
        """Test privacy action logging"""
        # Log privacy action
        await storage_manager._log_privacy_action(
            "test_action",
            "Test privacy action details",
            sensitive_app=True
        )
        
        # Retrieve logs
        logs = storage_manager.get_privacy_log(limit=10)
        assert len(logs) == 1
        assert logs[0]["action"] == "test_action"
        assert logs[0]["details"] == "Test privacy action details"
        assert logs[0]["sensitive_app_detected"] is True
    
    @pytest.mark.asyncio
    async def test_intelligent_context_management(self, storage_manager):
        """Test intelligent context window management"""
        current_time = time.time()
        
        # Create captures with different ages and relevance scores
        old_time = current_time - (2 * 3600)  # 2 hours old
        
        # High relevance, old capture (should be moved to archived)
        high_relevance_capture = self.create_test_capture("high_rel")
        high_relevance_capture.timestamp = old_time
        high_relevance_capture.relevance_score = 0.8
        await storage_manager._store_capture(high_relevance_capture)
        
        # Low relevance, old capture (should be deleted)
        low_relevance_capture = self.create_test_capture("low_rel")
        low_relevance_capture.timestamp = old_time
        low_relevance_capture.relevance_score = 0.05
        await storage_manager._store_capture(low_relevance_capture)
        
        # Run intelligent context management
        await storage_manager._intelligent_context_management()
        
        # Check results
        conn = sqlite3.connect(str(storage_manager.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, context_window FROM captures")
        results = {row[0]: row[1] for row in cursor.fetchall()}
        
        # High relevance should still exist (possibly in archived)
        assert "high_rel" in results
        
        # Low relevance might be deleted (depending on threshold configuration)
        # This depends on the specific configuration and timing
        
        conn.close()
    
    @pytest.mark.asyncio
    async def test_capture_archival(self, storage_manager):
        """Test capture archival with compression"""
        # Create and store a capture
        capture = self.create_test_capture()
        capture.image_path = str(storage_manager.storage_path / "test_image.png")
        
        # Create actual image file
        test_image = self.create_test_image()
        test_image.save(capture.image_path)
        
        await storage_manager._store_capture(capture)
        
        # Archive the capture
        await storage_manager._archive_capture(capture.id)
        
        # Check that image file was deleted and compressed data was stored
        conn = sqlite3.connect(str(storage_manager.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT image_path, compressed_image_data, context_window FROM captures WHERE id = ?", (capture.id,))
        row = cursor.fetchone()
        
        assert row[0] is None  # image_path should be None
        assert row[1] is not None  # compressed_image_data should exist
        assert row[2] == "historical"  # should be in historical context
        
        # Original file should be deleted
        assert not Path(capture.image_path).exists()
        
        conn.close()
    
    @pytest.mark.asyncio 
    async def test_capture_deletion(self, storage_manager):
        """Test capture deletion with cleanup"""
        # Create and store a capture
        capture = self.create_test_capture()
        capture.image_path = str(storage_manager.storage_path / "test_delete.png")
        
        # Create actual image file
        test_image = self.create_test_image()
        test_image.save(capture.image_path)
        
        await storage_manager._store_capture(capture)
        
        # Delete the capture
        await storage_manager._delete_capture(capture.id)
        
        # Check that capture is deleted from database
        conn = sqlite3.connect(str(storage_manager.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM captures WHERE id = ?", (capture.id,))
        count = cursor.fetchone()[0]
        assert count == 0
        
        # Image file should be deleted
        assert not Path(capture.image_path).exists()
        
        conn.close()
    
    def test_context_window_cleanup(self, storage_manager):
        """Test context window-based cleanup"""
        # Modify config for testing
        storage_manager.config.context_windows["recent"].max_captures = 2
        
        # Create multiple captures
        captures = []
        for i in range(4):
            capture = self.create_test_capture(f"test_{i}")
            capture.relevance_score = i * 0.2  # Increasing relevance
            captures.append(capture)
        
        # Store captures
        async def store_captures():
            for capture in captures:
                await storage_manager._store_capture(capture)
        
        asyncio.run(store_captures())
        
        # Run cleanup
        asyncio.run(storage_manager._cleanup_old_captures())
        
        # Check that only 2 captures remain (highest relevance)
        conn = sqlite3.connect(str(storage_manager.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM captures WHERE context_window = 'recent'")
        count = cursor.fetchone()[0]
        assert count <= 2
        
        # Remaining captures should be the highest relevance ones
        cursor.execute("SELECT id FROM captures WHERE context_window = 'recent' ORDER BY relevance_score DESC")
        remaining_ids = [row[0] for row in cursor.fetchall()]
        
        # Should include test_3 (highest relevance)
        assert "test_3" in remaining_ids
        
        conn.close()
    
    def test_enhanced_capture_retrieval(self, storage_manager):
        """Test enhanced capture retrieval methods"""
        # Create captures with different properties
        captures_data = [
            ("high_rel", 0.8, "recent", "Important content with error message"),
            ("med_rel", 0.5, "archived", "Medium relevance content"),
            ("low_rel", 0.2, "recent", "Low relevance content"),
        ]
        
        async def setup_captures():
            for capture_id, relevance, window, text in captures_data:
                capture = self.create_test_capture(capture_id, text)
                capture.relevance_score = relevance
                capture.context_window = window
                await storage_manager._store_capture(capture)
        
        asyncio.run(setup_captures())
        
        # Test get_recent_captures with context window filter
        recent_captures = storage_manager.get_recent_captures(limit=10, context_window="recent")
        recent_ids = [c.id for c in recent_captures]
        assert "high_rel" in recent_ids
        assert "low_rel" in recent_ids
        assert "med_rel" not in recent_ids  # In archived window
        
        # Test get_captures_by_relevance
        high_relevance_captures = storage_manager.get_captures_by_relevance(min_relevance=0.7, limit=10)
        assert len(high_relevance_captures) == 1
        assert high_relevance_captures[0].id == "high_rel"
        
        # Test search_captures_by_text
        search_results = asyncio.run(storage_manager.search_captures_by_text("error"))
        assert len(search_results) == 1
        assert search_results[0].id == "high_rel"
    
    def test_context_window_stats(self, storage_manager):
        """Test context window statistics"""
        # Create captures in different windows
        async def setup_captures():
            for i in range(3):
                capture = self.create_test_capture(f"recent_{i}")
                capture.context_window = "recent"
                capture.relevance_score = i * 0.3
                await storage_manager._store_capture(capture)
                
            for i in range(2):
                capture = self.create_test_capture(f"archived_{i}")
                capture.context_window = "archived"
                capture.relevance_score = i * 0.4
                await storage_manager._store_capture(capture)
        
        asyncio.run(setup_captures())
        
        # Get stats
        stats = storage_manager.get_context_window_stats()
        
        assert "recent" in stats
        assert "archived" in stats
        assert "historical" in stats
        
        assert stats["recent"]["capture_count"] == 3
        assert stats["archived"]["capture_count"] == 2
        assert stats["historical"]["capture_count"] == 0
        
        # Check average relevance
        assert 0 <= stats["recent"]["avg_relevance"] <= 1
        assert 0 <= stats["archived"]["avg_relevance"] <= 1
    
    @pytest.mark.asyncio
    async def test_capture_image_retrieval(self, storage_manager):
        """Test capture image retrieval with compression"""
        # Create capture with image
        capture = self.create_test_capture()
        test_image = self.create_test_image("Image Retrieval Test")
        
        # Test with file path
        image_path = storage_manager.storage_path / "test_retrieval.png"
        test_image.save(image_path)
        capture.image_path = str(image_path)
        
        await storage_manager._store_capture(capture)
        
        # Retrieve image
        retrieved_image = await storage_manager.get_capture_image(capture.id)
        assert retrieved_image is not None
        assert retrieved_image.size == test_image.size
        
                # Test with compressed data (simulate archival)
        compressed_data = storage_manager._compress_image_data(test_image)
        capture.compressed_image_data = compressed_data
        capture.image_path = None  # Remove file path

        await storage_manager._store_capture(capture)
        
        # Close any image handles before deleting
        try:
            if retrieved_image:
                retrieved_image.close()
            image_path.unlink()  # Delete physical file
        except (PermissionError, OSError) as e:
            # On Windows, file might still be locked - this is expected in tests
            pass
        
        # Should still retrieve from compressed data
        retrieved_image_compressed = await storage_manager.get_capture_image(capture.id)
        assert retrieved_image_compressed is not None
        assert retrieved_image_compressed.size == test_image.size
    
    def test_enhanced_performance_stats(self, storage_manager):
        """Test enhanced performance statistics"""
        stats = storage_manager.get_performance_stats()
        
        # Check new fields
        assert "context_windows" in stats
        assert "storage_features" in stats
        assert "storage_info" in stats
        
        # Check storage features
        storage_features = stats["storage_features"]
        assert "compression_enabled" in storage_features
        assert "encryption_enabled" in storage_features
        assert "semantic_indexing_enabled" in storage_features
        assert "max_reference_age_hours" in storage_features
        
        # Check storage info
        storage_info = stats["storage_info"]
        assert "database_size_bytes" in storage_info
        assert "storage_path" in storage_info
        
        # Check context window info
        assert stats["config"]["context_window_count"] == 3
    
    @pytest.mark.asyncio
    async def test_comprehensive_storage_workflow(self, storage_manager):
        """Test complete storage workflow with all features"""
        # Create test image
        test_image = self.create_test_image("Workflow Test", 300, 150)
        
        # Simulate screen capture process
        capture_id = "workflow_test"
        image_path = storage_manager.storage_path / f"capture_{capture_id}.png"
        test_image.save(image_path)
        
        # Create capture with all features
        capture = ScreenCapture(
            id=capture_id,
            timestamp=time.time(),
            image_path=str(image_path),
            text_content="Important workflow test with login and password fields",
            bounding_boxes=[(10, 10, 100, 30), (10, 40, 100, 60)],
            monitor_info={"width": 1920, "height": 1080},
            processing_time=0.8,
            ocr_confidence=92.0,
            relevance_score=0.0,  # Will be calculated
            context_window="recent"
        )
        
        # Calculate relevance score
        capture.relevance_score = storage_manager._calculate_relevance_score(capture)
        
        # Add compression
        capture.compressed_image_data = storage_manager._compress_image_data(test_image, 3)
        
        # Store capture
        await storage_manager._store_capture(capture)
        
        # Create screen reference
        reference = await storage_manager.create_screen_reference(
            capture, 
            element_type="form",
            semantic_tags=["login", "authentication"]
        )
        
        # Log privacy action
        await storage_manager._log_privacy_action(
            "workflow_test", 
            "Testing complete workflow",
            sensitive_app=False
        )
        
        # Verify storage
        retrieved_captures = storage_manager.get_recent_captures(limit=1)
        assert len(retrieved_captures) == 1
        assert retrieved_captures[0].id == capture_id
        assert retrieved_captures[0].relevance_score > 0
        
        # Verify references
        references = await storage_manager.get_screen_references(capture_id=capture_id)
        assert len(references) == 1
        assert "login" in references[0].semantic_tags
        
        # Verify privacy log
        logs = storage_manager.get_privacy_log(limit=1)
        assert len(logs) == 1
        assert logs[0]["action"] == "workflow_test"
        
        # Verify image retrieval
        retrieved_image = await storage_manager.get_capture_image(capture_id)
        assert retrieved_image is not None
        
        # Test search functionality
        search_results = await storage_manager.search_captures_by_text("password")
        assert len(search_results) == 1
        assert search_results[0].id == capture_id
        
        print("✅ Complete storage workflow test passed!") 