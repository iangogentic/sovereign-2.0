"""
Screen Context Manager - Handles real-time screen capture and OCR text extraction
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import sqlite3
import json
import tempfile
import os
import shutil
import zlib
import base64
from datetime import datetime, timedelta

# Third-party imports
import mss
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .config import Config
from .consent_manager import ConsentManager, MonitoringState, ConsentLevel, PrivacyZone, PrivacyZoneType


class CaptureState(Enum):
    """Screen capture states"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class ContextRelevance(Enum):
    """Context relevance levels for intelligent archival"""
    CRITICAL = "critical"      # User interactions, important changes
    HIGH = "high"             # Significant text changes, new applications
    MEDIUM = "medium"         # Minor text updates, scrolling
    LOW = "low"              # Minimal changes, static content


@dataclass
class ScreenCapture:
    """Represents a screen capture with metadata"""
    id: str
    timestamp: float
    image_path: str
    text_content: str
    bounding_boxes: List[Tuple[int, int, int, int]]
    monitor_info: Dict[str, Any]
    processing_time: float
    ocr_confidence: float = 0.0
    relevance_score: float = 0.0
    context_window: str = "recent"  # recent, archived, or historical
    compressed_image_data: Optional[str] = None  # Base64 encoded compressed image
    text_embedding: Optional[List[float]] = None  # For semantic search (future)


@dataclass 
class ScreenReference:
    """Reference to specific screen elements for AI interaction"""
    reference_id: str
    capture_id: str
    element_type: str  # text, image, ui_element, region
    bounding_box: Tuple[int, int, int, int]
    text_content: str
    confidence: float
    semantic_tags: List[str] = field(default_factory=list)
    

@dataclass
class ContextWindow:
    """Defines a context window for storage management"""
    name: str
    max_captures: int
    max_age_hours: int
    compression_level: int = 6  # 0-9, higher = more compression
    relevance_threshold: float = 0.3
    

@dataclass
class ScreenContextConfig:
    """Configuration for screen context system"""
    capture_interval: float = 4.0  # seconds
    max_stored_captures: int = 100
    ocr_language: str = "eng"
    enable_preprocessing: bool = True
    min_text_confidence: float = 30.0
    storage_path: str = ""
    privacy_mode: bool = False
    tesseract_path: str = ""  # Custom Tesseract executable path
    auto_disable_apps: List[str] = field(default_factory=lambda: [
        "1password", "keepass", "bitwarden", "lastpass", "banking", "wallet"
    ])
    
    # Enhanced storage configuration
    enable_compression: bool = True
    enable_encryption: bool = False  # Future enhancement
    context_windows: Dict[str, ContextWindow] = field(default_factory=lambda: {
        "recent": ContextWindow("recent", 50, 24, 3, 0.1),      # Last 24 hours, light compression
        "archived": ContextWindow("archived", 200, 168, 6, 0.3), # Last week, medium compression  
        "historical": ContextWindow("historical", 100, 720, 9, 0.7) # Last month, high compression
    })
    max_reference_age_hours: int = 72  # How long to keep screen references
    enable_semantic_indexing: bool = False  # For future AI integration


class ScreenContextManager:
    """
    Manages real-time screen capture and OCR text extraction
    """
    
    def __init__(self, config: Optional[ScreenContextConfig] = None, main_config: Optional[Config] = None):
        self.config = config or ScreenContextConfig()
        self.main_config = main_config  # Store reference to main config
        self.logger = logging.getLogger(__name__)
        self.state = CaptureState.STOPPED
        
        # Core components
        self.capture_task: Optional[asyncio.Task] = None
        self.last_capture_time = 0.0
        self.capture_count = 0
        
        # Storage
        self.storage_path = Path(self.config.storage_path) if self.config.storage_path else Path.home() / ".sovereign" / "screen_context"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Privacy and consent management
        self.consent_manager = ConsentManager(self.main_config)
        self.last_sensitive_app_scan = 0.0
        self.sensitive_app_scan_interval = 5.0  # Scan every 5 seconds
        
        # Performance tracking
        self.performance_stats = {
            "total_captures": 0,
            "successful_captures": 0,
            "failed_captures": 0,
            "avg_capture_time": 0.0,
            "avg_ocr_time": 0.0,
            "total_text_extracted": 0,
            "cache_hits": 0,
            "privacy_blocks": 0,
            "sensitive_app_detections": 0,
        }
        
        # Cache for duplicate detection
        self.content_cache: Dict[str, ScreenCapture] = {}
        
        # Initialize database
        self.db_path = self.storage_path / "captures.db"
        self._init_database()
        
        # Setup Tesseract
        self._setup_tesseract()
        
        # Initialize context windows
        self._init_context_windows()
        
        # Setup consent callbacks
        self._setup_consent_callbacks()
        
        self.logger.info(f"ScreenContextManager initialized with interval {self.config.capture_interval}s")
    
    def _setup_consent_callbacks(self):
        """Setup consent manager callbacks"""
        def on_state_change(state: MonitoringState):
            """Handle monitoring state changes"""
            self.logger.info(f"Monitoring state changed to: {state.value}")
            if state == MonitoringState.PAUSED_SENSITIVE:
                self.performance_stats["sensitive_app_detections"] += 1
            elif state in [MonitoringState.PAUSED_PRIVACY, MonitoringState.PAUSED_USER]:
                self.performance_stats["privacy_blocks"] += 1
        
        def on_consent_change(feature: str, level: ConsentLevel):
            """Handle consent level changes"""
            self.logger.info(f"Consent for {feature} changed to: {level.value}")
            if feature == "screen_monitoring":
                if level == ConsentLevel.DENIED:
                    # Stop monitoring if consent is revoked
                    asyncio.create_task(self.stop_capture())
        
        self.consent_manager.add_state_change_callback(on_state_change)
        self.consent_manager.add_consent_change_callback(on_consent_change)
    
    def _setup_tesseract(self):
        """Setup Tesseract OCR executable path"""
        try:
            import pytesseract
            # Check main config first (priority)
            if self.main_config and self.main_config.screen_capture.tesseract_cmd_path:
                if os.path.isfile(self.main_config.screen_capture.tesseract_cmd_path):
                    pytesseract.pytesseract.tesseract_cmd = self.main_config.screen_capture.tesseract_cmd_path
                    self.logger.info(f"Using configured Tesseract path: {self.main_config.screen_capture.tesseract_cmd_path}")
                    return
                else:
                    self.logger.warning(f"Configured Tesseract path not found: {self.main_config.screen_capture.tesseract_cmd_path}")
            
            # Fallback to local config
            if self.config.tesseract_path:
                if os.path.isfile(self.config.tesseract_path):
                    pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_path
                    self.logger.info(f"Using local Tesseract path: {self.config.tesseract_path}")
                    return
                else:
                    self.logger.warning(f"Local Tesseract path not found: {self.config.tesseract_path}")
            
            # Try to find Tesseract in common Windows locations
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Tesseract-OCR\tesseract.exe",
                r"C:\tools\tesseract\tesseract.exe",
            ]
            
            for path in common_paths:
                if os.path.isfile(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    self.logger.info(f"Found Tesseract at: {path}")
                    return
            
            # Try to find Tesseract in PATH
            tesseract_path = shutil.which("tesseract")
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                self.logger.info(f"Found Tesseract in PATH: {tesseract_path}")
                return
            
            # If we get here, Tesseract wasn't found
            self.logger.error("Tesseract OCR not found. Please install Tesseract or set tesseract_cmd_path in config.")
            self.logger.error("Download from: https://github.com/tesseract-ocr/tesseract")
            
        except Exception as e:
            self.logger.error(f"Error setting up Tesseract: {e}")
    
    def _test_tesseract(self) -> bool:
        """Test if Tesseract is working properly"""
        try:
            import pytesseract
            # Create a simple test image
            test_img = Image.new('RGB', (200, 50), color='white')
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(test_img)
            
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            
            draw.text((10, 15), "TEST", fill='black', font=font)
            
            # Try to extract text
            text = pytesseract.image_to_string(test_img)
            self.logger.info(f"Tesseract test successful - extracted: '{text.strip()}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Tesseract test failed: {e}")
            return False
    
    def _init_database(self):
        """Initialize SQLite database for storing captures"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Enhanced captures table with new fields
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS captures (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    image_path TEXT,
                    text_content TEXT,
                    bounding_boxes TEXT,
                    monitor_info TEXT,
                    processing_time REAL,
                    ocr_confidence REAL,
                    relevance_score REAL DEFAULT 0.0,
                    context_window TEXT DEFAULT 'recent',
                    compressed_image_data TEXT,
                    text_embedding TEXT,
                    created_date TEXT DEFAULT (datetime('now')),
                    archived_date TEXT
                )
            ''')
            
            # MIGRATION: Add missing columns to existing tables
            self._migrate_database_schema(cursor)
            
            # Screen references table for AI interaction
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS screen_references (
                    reference_id TEXT PRIMARY KEY,
                    capture_id TEXT,
                    element_type TEXT,
                    bounding_box TEXT,
                    text_content TEXT,
                    confidence REAL,
                    semantic_tags TEXT,
                    created_timestamp REAL,
                    FOREIGN KEY (capture_id) REFERENCES captures (id) ON DELETE CASCADE
                )
            ''')
            
            # Context management table for intelligent archival
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS context_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    window_name TEXT,
                    capture_count INTEGER,
                    last_cleanup REAL,
                    total_size_bytes INTEGER,
                    compression_ratio REAL
                )
            ''')
            
            # Privacy audit log for compliance
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS privacy_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    action TEXT,
                    details TEXT,
                    sensitive_app_detected BOOLEAN DEFAULT 0
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON captures(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_context_window ON captures(context_window)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_relevance_score ON captures(relevance_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_date ON captures(created_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_reference_capture ON screen_references(capture_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_reference_timestamp ON screen_references(created_timestamp)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def _migrate_database_schema(self, cursor: sqlite3.Cursor):
        """Migrate existing database schema to add missing columns"""
        try:
            # Check if captures table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='captures'")
            if cursor.fetchone():
                # Table exists, check for missing columns
                cursor.execute("PRAGMA table_info(captures)")
                existing_columns = [row[1] for row in cursor.fetchall()]
                
                # Add missing columns if they don't exist
                required_columns = {
                    'relevance_score': 'REAL DEFAULT 0.0',
                    'context_window': "TEXT DEFAULT 'recent'",
                    'compressed_image_data': 'TEXT',
                    'text_embedding': 'TEXT',
                    'created_date': "TEXT DEFAULT (datetime('now'))",
                    'archived_date': 'TEXT'
                }
                
                for column_name, column_definition in required_columns.items():
                    if column_name not in existing_columns:
                        self.logger.info(f"Adding missing column: {column_name}")
                        cursor.execute(f"ALTER TABLE captures ADD COLUMN {column_name} {column_definition}")
                        
        except Exception as e:
            self.logger.error(f"Database schema migration failed: {e}")
            # Don't raise here to allow initialization to continue
    
    async def initialize(self, config: Optional[Config] = None):
        """Initialize the screen context manager"""
        try:
            if config:
                # Update configuration from main config
                self.config.capture_interval = getattr(config, 'screen_capture_interval', self.config.capture_interval)
                self.config.privacy_mode = getattr(config, 'privacy_mode', self.config.privacy_mode)
            
            # Test screen capture capability
            test_capture = await self._test_capture()
            if test_capture:
                self.logger.info("Screen capture test successful")
                return True
            else:
                self.logger.error("Screen capture test failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def _test_capture(self) -> bool:
        """Test screen capture functionality"""
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor
                screenshot = sct.grab(monitor)
                
                # Convert to PIL Image
                img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
                
                # Basic validation
                if img.size[0] > 0 and img.size[1] > 0:
                    return True
                    
        except Exception as e:
            self.logger.error(f"Screen capture test failed: {e}")
            
        return False
    
    async def capture_screen(self) -> Optional[ScreenCapture]:
        """Capture the current screen and extract text"""
        start_time = time.time()
        
        try:
            self.performance_stats["total_captures"] += 1
            
            # Privacy and consent checks
            if not await self._privacy_check_before_capture():
                return None
            
            # Check if privacy mode is enabled
            if self.config.privacy_mode:
                self.logger.debug("Privacy mode enabled, skipping capture")
                return None
            
            # Check for sensitive applications
            if await self._is_sensitive_app_active():
                self.logger.debug("Sensitive application detected, skipping capture")
                return None
            
            # Capture screen
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor
                screenshot = sct.grab(monitor)
                
                # Convert to PIL Image
                img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
                
                # Apply privacy zone redaction
                img = await self._apply_privacy_zones(img, monitor)
                if img is None:  # Privacy zone requested to skip capture
                    return None
                
                # Generate unique ID for this capture
                capture_id = self._generate_capture_id(img)
                
                # Check for duplicates
                if capture_id in self.content_cache:
                    self.performance_stats["cache_hits"] += 1
                    self.logger.debug("Duplicate capture detected, using cached version")
                    return self.content_cache[capture_id]
                
                # Save screenshot
                image_path = self.storage_path / f"capture_{capture_id}.png"
                img.save(image_path)
                
                # Extract text with OCR
                text_content, bounding_boxes, ocr_confidence = await self._extract_text_with_ocr(img)
                
                # Create capture object
                capture = ScreenCapture(
                    id=capture_id,
                    timestamp=time.time(),
                    image_path=str(image_path),
                    text_content=text_content,
                    bounding_boxes=bounding_boxes,
                    monitor_info={
                        "width": monitor["width"],
                        "height": monitor["height"],
                        "left": monitor["left"],
                        "top": monitor["top"]
                    },
                    processing_time=time.time() - start_time,
                    ocr_confidence=ocr_confidence,
                    relevance_score=0.0,  # Will be calculated below
                    context_window="recent"
                )
                
                # Calculate relevance score after capture creation
                previous_capture = None
                if len(self.content_cache) > 0:
                    previous_capture = list(self.content_cache.values())[-1]
                capture.relevance_score = self._calculate_relevance_score(capture, previous_capture)
                
                # Add compression data if enabled
                if self.config.enable_compression:
                    capture.compressed_image_data = self._compress_image_data(img, 
                        self.config.context_windows["recent"].compression_level)
                
                # Store in database
                await self._store_capture(capture)
                
                # Create screen reference for AI interaction
                if capture.text_content.strip():  # Only create reference if text was extracted
                    await self.create_screen_reference(capture, element_type="text")
                
                # Cache for duplicate detection
                self.content_cache[capture_id] = capture
                
                # Clean up old captures and manage context windows
                await self._cleanup_old_captures()
                await self._intelligent_context_management()
                
                # Log privacy action
                await self._log_privacy_action("screen_captured", f"Capture {capture_id} created with relevance {capture.relevance_score:.2f}")
                
                self.performance_stats["successful_captures"] += 1
                self.performance_stats["avg_capture_time"] = (
                    self.performance_stats["avg_capture_time"] * (self.performance_stats["successful_captures"] - 1) +
                    capture.processing_time
                ) / self.performance_stats["successful_captures"]
                
                self.logger.debug(f"Screen capture completed in {capture.processing_time:.2f}s")
                return capture
                
        except Exception as e:
            self.performance_stats["failed_captures"] += 1
            self.logger.error(f"Screen capture failed: {e}")
            return None
    
    async def _extract_text_with_ocr(self, image: Image.Image) -> Tuple[str, List[Tuple[int, int, int, int]], float]:
        """Extract text from image using OCR"""
        ocr_start_time = time.time()
        
        try:
            import pytesseract
            # Preprocess image if enabled
            if self.config.enable_preprocessing:
                image = self._preprocess_image(image)
            
            # Extract text with bounding boxes
            try:
                # Get detailed OCR data
                data = pytesseract.image_to_data(
                    image,
                    lang=self.config.ocr_language,
                    output_type=pytesseract.Output.DICT
                )
                
                # Extract text and bounding boxes
                text_parts = []
                bounding_boxes = []
                confidences = []
                
                for i, text in enumerate(data['text']):
                    confidence = int(data['conf'][i])
                    if confidence > self.config.min_text_confidence and text.strip():
                        text_parts.append(text.strip())
                        bounding_boxes.append((
                            data['left'][i],
                            data['top'][i],
                            data['width'][i],
                            data['height'][i]
                        ))
                        confidences.append(confidence)
                
                # Combine text
                text_content = ' '.join(text_parts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                # Update performance stats
                ocr_time = time.time() - ocr_start_time
                self.performance_stats["avg_ocr_time"] = (
                    self.performance_stats["avg_ocr_time"] * self.performance_stats["total_text_extracted"] +
                    ocr_time
                ) / (self.performance_stats["total_text_extracted"] + 1)
                
                self.performance_stats["total_text_extracted"] += len(text_parts)
                
                return text_content, bounding_boxes, avg_confidence
                
            except Exception as e:
                self.logger.error(f"OCR extraction failed: {e}")
                return "", [], 0.0
                
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return "", [], 0.0
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Apply slight sharpening
            image = image.filter(ImageFilter.SHARPEN)
            
            # Scale up small images
            if image.size[0] < 1000 or image.size[1] < 1000:
                scale_factor = max(1000 / image.size[0], 1000 / image.size[1])
                new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
                image = image.resize(new_size, Image.LANCZOS)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def _generate_capture_id(self, image: Image.Image) -> str:
        """Generate unique ID for capture based on image content"""
        try:
            # Create hash of image data
            img_bytes = image.tobytes()
            hash_obj = hashlib.md5(img_bytes)
            return hash_obj.hexdigest()[:16]
            
        except Exception:
            # Fallback to timestamp-based ID, padded to 16 chars
            timestamp_id = str(int(time.time() * 1000))
            return timestamp_id.zfill(16)[-16:]
    
    async def _privacy_check_before_capture(self) -> bool:
        """Comprehensive privacy check before capturing screen"""
        try:
            # Check if monitoring is allowed by consent manager
            if not self.consent_manager.is_monitoring_allowed():
                self.logger.debug("Monitoring not allowed by consent manager")
                return False
            
            # Check for consent
            if not self.consent_manager.has_consent("screen_monitoring", ConsentLevel.BASIC):
                # Request consent for screen monitoring
                granted_level = self.consent_manager.request_privacy_consent("screen_monitoring", ConsentLevel.BASIC)
                if granted_level == ConsentLevel.DENIED:
                    self.logger.debug("Screen monitoring consent denied")
                    return False
            
            # Periodically scan for sensitive apps
            current_time = time.time()
            if current_time - self.last_sensitive_app_scan > self.sensitive_app_scan_interval:
                detected_apps = await self.consent_manager.scan_for_sensitive_apps()
                if detected_apps:
                    self.logger.warning(f"Detected {len(detected_apps)} sensitive apps: {[app.app_name for app in detected_apps]}")
                    self.consent_manager.pause_monitoring("sensitive_app")
                    return False
                self.last_sensitive_app_scan = current_time
            
            # Check if currently paused for sensitive apps
            if self.consent_manager.monitoring_state == MonitoringState.PAUSED_SENSITIVE:
                # Try to resume if no sensitive apps detected
                detected_apps = await self.consent_manager.scan_for_sensitive_apps()
                if not detected_apps:
                    self.consent_manager.resume_monitoring()
                else:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Privacy check failed: {e}")
            return False
    
    async def _is_sensitive_app_active(self) -> bool:
        """Check if a sensitive application is currently active"""
        try:
            detected_apps = await self.consent_manager.scan_for_sensitive_apps()
            return len(detected_apps) > 0
        except Exception as e:
            self.logger.error(f"Sensitive app detection failed: {e}")
            return False
    
    async def _apply_privacy_zones(self, image: Image.Image, monitor: Dict[str, Any]) -> Image.Image:
        """Apply privacy zone redaction to captured image"""
        try:
            # Get privacy zones that overlap with the capture area
            privacy_zones = self.consent_manager.get_privacy_zones_for_region(
                monitor["left"], monitor["top"], 
                monitor["width"], monitor["height"]
            )
            
            if not privacy_zones:
                return image
            
            # Create a copy to modify
            redacted_image = image.copy()
            
            from PIL import ImageDraw, ImageFilter
            draw = ImageDraw.Draw(redacted_image)
            
            for zone in privacy_zones:
                # Adjust coordinates relative to monitor
                rel_x = zone.x - monitor["left"]
                rel_y = zone.y - monitor["top"]
                
                # Ensure coordinates are within image bounds
                rel_x = max(0, min(rel_x, image.width))
                rel_y = max(0, min(rel_y, image.height))
                rel_width = min(zone.width, image.width - rel_x)
                rel_height = min(zone.height, image.height - rel_y)
                
                if rel_width <= 0 or rel_height <= 0:
                    continue
                
                # Apply redaction based on zone type
                if zone.zone_type == PrivacyZoneType.FULL_REDACT:
                    # Complete blackout
                    draw.rectangle(
                        [rel_x, rel_y, rel_x + rel_width, rel_y + rel_height],
                        fill='black'
                    )
                elif zone.zone_type == PrivacyZoneType.BLUR:
                    # Extract region, blur it, and paste back
                    region = redacted_image.crop((rel_x, rel_y, rel_x + rel_width, rel_y + rel_height))
                    blurred_region = region.filter(ImageFilter.GaussianBlur(radius=15))
                    redacted_image.paste(blurred_region, (rel_x, rel_y))
                elif zone.zone_type == PrivacyZoneType.SKIP_CAPTURE:
                    # This zone should prevent capture entirely
                    self.logger.info(f"Skipping capture due to privacy zone: {zone.name}")
                    return None
            
            self.logger.debug(f"Applied {len(privacy_zones)} privacy zone redactions")
            return redacted_image
            
        except Exception as e:
            self.logger.error(f"Privacy zone application failed: {e}")
            return image
    
    async def _store_capture(self, capture: ScreenCapture):
        """Store capture in database"""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO captures 
                (id, timestamp, image_path, text_content, bounding_boxes, monitor_info, processing_time, ocr_confidence,
                 relevance_score, context_window, compressed_image_data, text_embedding, created_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                capture.id,
                capture.timestamp,
                capture.image_path,
                capture.text_content,
                json.dumps(capture.bounding_boxes),
                json.dumps(capture.monitor_info),
                capture.processing_time,
                capture.ocr_confidence,
                capture.relevance_score,
                capture.context_window,
                capture.compressed_image_data,
                json.dumps(capture.text_embedding) if capture.text_embedding else None,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Database storage failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def _init_context_windows(self):
        """Initialize context window metadata in database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            for window_name, window_config in self.config.context_windows.items():
                cursor.execute('''
                    INSERT OR IGNORE INTO context_metadata 
                    (window_name, capture_count, last_cleanup, total_size_bytes, compression_ratio)
                    VALUES (?, 0, ?, 0, 0.0)
                ''', (window_name, time.time()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Context window initialization failed: {e}")
    
    def _calculate_relevance_score(self, capture: ScreenCapture, previous_capture: Optional[ScreenCapture] = None) -> float:
        """Calculate relevance score for intelligent archival"""
        try:
            score = 0.0
            
            # Base score from OCR confidence
            score += min(capture.ocr_confidence / 100.0, 0.3)
            
            # Text content analysis
            if capture.text_content:
                text_length = len(capture.text_content.strip())
                score += min(text_length / 1000.0, 0.2)  # More text = potentially more relevant
                
                # Check for important keywords
                important_keywords = ['error', 'warning', 'login', 'password', 'submit', 'save', 'delete']
                keyword_count = sum(1 for keyword in important_keywords if keyword.lower() in capture.text_content.lower())
                score += keyword_count * 0.1
            
            # Compare with previous capture for change detection
            if previous_capture:
                text_similarity = self._calculate_text_similarity(capture.text_content, previous_capture.text_content)
                # Higher difference = higher relevance
                score += (1.0 - text_similarity) * 0.3
            
            # Bounding box analysis (more UI elements = potentially more relevant)
            if capture.bounding_boxes:
                score += min(len(capture.bounding_boxes) / 20.0, 0.2)
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Relevance score calculation failed: {e}")
            return 0.5  # Default moderate relevance
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        try:
            # Handle empty string cases first
            if not text1 and not text2:
                return 1.0  # Both empty = identical
            if not text1 or not text2:
                return 0.0  # One empty, one not = completely different
            
            # Simple word-based similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _compress_image_data(self, image: Image.Image, compression_level: int = 6) -> str:
        """Compress image data for efficient storage"""
        try:
            if not self.config.enable_compression:
                return ""
            
            # Convert image to bytes
            import io
            buffer = io.BytesIO()
            
            # Save as PNG with compression
            image.save(buffer, format='PNG', optimize=True, compress_level=compression_level)
            buffer.seek(0)
            
            # Compress with zlib
            compressed_data = zlib.compress(buffer.getvalue(), compression_level)
            
            # Encode as base64 for database storage
            return base64.b64encode(compressed_data).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Image compression failed: {e}")
            return ""
    
    def _decompress_image_data(self, compressed_data: str) -> Optional[Image.Image]:
        """Decompress image data from storage"""
        try:
            if not compressed_data:
                return None
            
            # Decode from base64
            compressed_bytes = base64.b64decode(compressed_data.encode('utf-8'))
            
            # Decompress with zlib
            decompressed_data = zlib.decompress(compressed_bytes)
            
            # Convert back to PIL Image
            import io
            buffer = io.BytesIO(decompressed_data)
            return Image.open(buffer)
            
        except Exception as e:
            self.logger.error(f"Image decompression failed: {e}")
            return None
    
    async def create_screen_reference(self, capture: ScreenCapture, element_type: str = "text", 
                                    semantic_tags: List[str] = None) -> Optional[ScreenReference]:
        """Create a screen reference for AI interaction"""
        try:
            reference_id = f"ref_{capture.id}_{int(time.time())}"
            
            # For now, create a reference to the entire capture
            # In the future, this could be enhanced to reference specific regions
            bounding_box = (0, 0, 
                          capture.monitor_info.get('width', 1920), 
                          capture.monitor_info.get('height', 1080))
            
            reference = ScreenReference(
                reference_id=reference_id,
                capture_id=capture.id,
                element_type=element_type,
                bounding_box=bounding_box,
                text_content=capture.text_content,
                confidence=capture.ocr_confidence,
                semantic_tags=semantic_tags or []
            )
            
            # Store in database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO screen_references 
                (reference_id, capture_id, element_type, bounding_box, text_content, confidence, semantic_tags, created_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                reference.reference_id,
                reference.capture_id,
                reference.element_type,
                json.dumps(reference.bounding_box),
                reference.text_content,
                reference.confidence,
                json.dumps(reference.semantic_tags),
                time.time()
            ))
            
            conn.commit()
            conn.close()
            
            return reference
            
        except Exception as e:
            self.logger.error(f"Screen reference creation failed: {e}")
            return None
    
    async def get_screen_references(self, capture_id: Optional[str] = None, 
                                  element_type: Optional[str] = None,
                                  max_age_hours: Optional[int] = None) -> List[ScreenReference]:
        """Get screen references for AI interaction"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            query = "SELECT * FROM screen_references WHERE 1=1"
            params = []
            
            if capture_id:
                query += " AND capture_id = ?"
                params.append(capture_id)
            
            if element_type:
                query += " AND element_type = ?"
                params.append(element_type)
            
            if max_age_hours is None:
                max_age_hours = self.config.max_reference_age_hours
            
            cutoff_time = time.time() - (max_age_hours * 3600)
            query += " AND created_timestamp > ?"
            params.append(cutoff_time)
            
            query += " ORDER BY created_timestamp DESC"
            
            cursor.execute(query, params)
            references = []
            
            for row in cursor.fetchall():
                reference = ScreenReference(
                    reference_id=row[0],
                    capture_id=row[1],
                    element_type=row[2],
                    bounding_box=tuple(json.loads(row[3])),
                    text_content=row[4],
                    confidence=row[5],
                    semantic_tags=json.loads(row[6])
                )
                references.append(reference)
            
            conn.close()
            return references
            
        except Exception as e:
            self.logger.error(f"Failed to get screen references: {e}")
            return []
    
    async def _log_privacy_action(self, action: str, details: str, sensitive_app: bool = False):
        """Log privacy-related actions for audit trail"""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO privacy_log (timestamp, action, details, sensitive_app_detected)
                VALUES (?, ?, ?, ?)
            ''', (time.time(), action, details, sensitive_app))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Privacy logging failed: {e}")
        finally:
            if conn:
                conn.close()
    
    async def _intelligent_context_management(self):
        """Manage context windows intelligently based on relevance and age"""
        try:
            current_time = time.time()
            
            for window_name, window_config in self.config.context_windows.items():
                # Get captures in this window
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Check captures that need to be moved or archived
                cutoff_time = current_time - (window_config.max_age_hours * 3600)
                
                cursor.execute('''
                    SELECT id, relevance_score, timestamp FROM captures 
                    WHERE context_window = ? AND timestamp < ?
                    ORDER BY relevance_score DESC, timestamp DESC
                ''', (window_name, cutoff_time))
                
                old_captures = cursor.fetchall()
                
                # Archive high-relevance captures, delete low-relevance ones
                for capture_id, relevance_score, timestamp in old_captures:
                    if relevance_score >= window_config.relevance_threshold:
                        # Move to next context window or archive
                        next_window = self._get_next_context_window(window_name)
                        if next_window:
                            cursor.execute('''
                                UPDATE captures SET context_window = ?, archived_date = ?
                                WHERE id = ?
                            ''', (next_window, datetime.now().isoformat(), capture_id))
                        else:
                            # Final archive with high compression
                            await self._archive_capture(capture_id)
                    else:
                        # Delete low-relevance capture
                        await self._delete_capture(capture_id)
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            self.logger.error(f"Intelligent context management failed: {e}")
    
    def _get_next_context_window(self, current_window: str) -> Optional[str]:
        """Get the next context window for archival"""
        window_order = ["recent", "archived", "historical"]
        try:
            current_index = window_order.index(current_window)
            if current_index < len(window_order) - 1:
                return window_order[current_index + 1]
        except ValueError:
            pass
        return None
    
    async def _archive_capture(self, capture_id: str):
        """Archive a capture with maximum compression"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get capture data
            cursor.execute('SELECT * FROM captures WHERE id = ?', (capture_id,))
            row = cursor.fetchone()
            
            if row and row[2]:  # Has image_path
                # Load and compress image
                image_path = Path(row[2])
                if image_path.exists():
                    image = Image.open(image_path)
                    compressed_data = self._compress_image_data(image, compression_level=9)
                    
                    # Update database with compressed data and remove file path
                    cursor.execute('''
                        UPDATE captures 
                        SET compressed_image_data = ?, image_path = NULL, context_window = 'historical'
                        WHERE id = ?
                    ''', (compressed_data, capture_id))
                    
                    # Delete physical image file
                    image_path.unlink()
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Capture archival failed: {e}")
    
    async def _delete_capture(self, capture_id: str):
        """Permanently delete a capture and its files"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get image path before deletion
            cursor.execute('SELECT image_path FROM captures WHERE id = ?', (capture_id,))
            row = cursor.fetchone()
            
            if row and row[0]:
                image_path = Path(row[0])
                if image_path.exists():
                    image_path.unlink()
            
            # Delete from database (CASCADE will handle references)
            cursor.execute('DELETE FROM captures WHERE id = ?', (capture_id,))
            
            conn.commit()
            conn.close()
            
            await self._log_privacy_action("capture_deleted", f"Capture {capture_id} deleted")
            
        except Exception as e:
            self.logger.error(f"Capture deletion failed: {e}")
    
    async def _cleanup_old_captures(self):
        """Clean up old captures to maintain storage limits per context window"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Cleanup each context window separately
            for window_name, window_config in self.config.context_windows.items():
                # Get count of captures in this window
                cursor.execute('''
                    SELECT COUNT(*) FROM captures WHERE context_window = ?
                ''', (window_name,))
                count = cursor.fetchone()[0]
                
                if count > window_config.max_captures:
                    # Delete oldest captures in this window (lowest relevance first, then oldest)
                    cursor.execute('''
                        SELECT id, image_path FROM captures 
                        WHERE context_window = ? 
                        ORDER BY relevance_score ASC, timestamp ASC 
                        LIMIT ?
                    ''', (window_name, count - window_config.max_captures))
                    
                    captures_to_delete = cursor.fetchall()
                    
                    for capture_id, image_path in captures_to_delete:
                        # Delete image file if it exists
                        if image_path:
                            try:
                                os.remove(image_path)
                            except Exception:
                                pass  # File might already be deleted
                        
                        # Delete capture from database
                        cursor.execute('DELETE FROM captures WHERE id = ?', (capture_id,))
                        
                        await self._log_privacy_action("capture_cleanup", f"Deleted capture {capture_id} from {window_name}")
            
            # Also cleanup old screen references
            cutoff_time = time.time() - (self.config.max_reference_age_hours * 3600)
            cursor.execute('''
                DELETE FROM screen_references WHERE created_timestamp < ?
            ''', (cutoff_time,))
            
            # Update context metadata
            for window_name in self.config.context_windows.keys():
                cursor.execute('''
                    UPDATE context_metadata 
                    SET last_cleanup = ?, capture_count = (
                        SELECT COUNT(*) FROM captures WHERE context_window = ?
                    )
                    WHERE window_name = ?
                ''', (time.time(), window_name, window_name))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    async def start_capture(self):
        """Start continuous screen capture"""
        if self.state == CaptureState.RUNNING:
            self.logger.warning("Screen capture already running")
            return
        
        self.state = CaptureState.RUNNING
        self.capture_task = asyncio.create_task(self._capture_loop())
        self.logger.info("Screen capture started")
    
    async def stop_capture(self):
        """Stop continuous screen capture"""
        if self.state != CaptureState.RUNNING:
            return
        
        self.state = CaptureState.STOPPED
        
        if self.capture_task:
            self.capture_task.cancel()
            try:
                await self.capture_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Screen capture stopped")
    
    async def _capture_loop(self):
        """Main capture loop"""
        while self.state == CaptureState.RUNNING:
            try:
                # Capture screen
                capture = await self.capture_screen()
                
                if capture:
                    self.capture_count += 1
                    self.last_capture_time = time.time()
                
                # Wait for next capture interval
                await asyncio.sleep(self.config.capture_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Capture loop error: {e}")
                await asyncio.sleep(self.config.capture_interval)
    
    def toggle_capture(self, enabled: bool):
        """Toggle screen capture on/off"""
        try:
            # Try to use the current event loop
            loop = asyncio.get_running_loop()
            if enabled:
                asyncio.create_task(self.start_capture())
            else:
                asyncio.create_task(self.stop_capture())
        except RuntimeError:
            # No event loop running, this should be called from async context
            self.logger.warning("toggle_capture called outside async context - use await start_capture()/stop_capture() instead")
    
    def get_recent_captures(self, limit: int = 10, context_window: str = None) -> List[ScreenCapture]:
        """Get recent screen captures from specified context window"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            query = '''
                SELECT id, timestamp, image_path, text_content, bounding_boxes, monitor_info, 
                       processing_time, ocr_confidence, relevance_score, context_window, 
                       compressed_image_data, text_embedding, created_date, archived_date
                FROM captures
            '''
            params = []
            
            if context_window:
                query += ' WHERE context_window = ?'
                params.append(context_window)
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            
            captures = []
            for row in cursor.fetchall():
                capture = ScreenCapture(
                    id=row[0],
                    timestamp=row[1],
                    image_path=row[2],
                    text_content=row[3],
                    bounding_boxes=json.loads(row[4]) if row[4] else [],
                    monitor_info=json.loads(row[5]) if row[5] else {},
                    processing_time=row[6],
                    ocr_confidence=row[7],
                    relevance_score=row[8] if row[8] is not None else 0.0,
                    context_window=row[9] if row[9] else "recent",
                    compressed_image_data=row[10],
                    text_embedding=json.loads(row[11]) if row[11] else None
                )
                captures.append(capture)
            
            conn.close()
            return captures
            
        except Exception as e:
            self.logger.error(f"Failed to get recent captures: {e}")
            return []
    
    def get_captures_by_relevance(self, min_relevance: float = 0.5, limit: int = 20) -> List[ScreenCapture]:
        """Get captures with relevance score above threshold"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, timestamp, image_path, text_content, bounding_boxes, monitor_info, 
                       processing_time, ocr_confidence, relevance_score, context_window, 
                       compressed_image_data, text_embedding, created_date, archived_date
                FROM captures
                WHERE relevance_score >= ?
                ORDER BY relevance_score DESC, timestamp DESC
                LIMIT ?
            ''', (min_relevance, limit))
            
            captures = []
            for row in cursor.fetchall():
                capture = ScreenCapture(
                    id=row[0],
                    timestamp=row[1],
                    image_path=row[2],
                    text_content=row[3],
                    bounding_boxes=json.loads(row[4]) if row[4] else [],
                    monitor_info=json.loads(row[5]) if row[5] else {},
                    processing_time=row[6],
                    ocr_confidence=row[7],
                    relevance_score=row[8] if row[8] is not None else 0.0,
                    context_window=row[9] if row[9] else "recent",
                    compressed_image_data=row[10],
                    text_embedding=json.loads(row[11]) if row[11] else None
                )
                captures.append(capture)
            
            conn.close()
            return captures
            
        except Exception as e:
            self.logger.error(f"Failed to get captures by relevance: {e}")
            return []
    
    async def search_captures_by_text(self, search_term: str, limit: int = 10) -> List[ScreenCapture]:
        """Search captures by text content"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, timestamp, image_path, text_content, bounding_boxes, monitor_info, 
                       processing_time, ocr_confidence, relevance_score, context_window, 
                       compressed_image_data, text_embedding, created_date, archived_date
                FROM captures
                WHERE text_content LIKE ?
                ORDER BY relevance_score DESC, timestamp DESC
                LIMIT ?
            ''', (f'%{search_term}%', limit))
            
            captures = []
            for row in cursor.fetchall():
                capture = ScreenCapture(
                    id=row[0],
                    timestamp=row[1],
                    image_path=row[2],
                    text_content=row[3],
                    bounding_boxes=json.loads(row[4]) if row[4] else [],
                    monitor_info=json.loads(row[5]) if row[5] else {},
                    processing_time=row[6],
                    ocr_confidence=row[7],
                    relevance_score=row[8] if row[8] is not None else 0.0,
                    context_window=row[9] if row[9] else "recent",
                    compressed_image_data=row[10],
                    text_embedding=json.loads(row[11]) if row[11] else None
                )
                captures.append(capture)
            
            conn.close()
            return captures
            
        except Exception as e:
            self.logger.error(f"Failed to search captures: {e}")
            return []
    
    def get_context_window_stats(self) -> Dict[str, Any]:
        """Get statistics for each context window"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            stats = {}
            for window_name in self.config.context_windows.keys():
                cursor.execute('''
                    SELECT 
                        COUNT(*) as count,
                        AVG(relevance_score) as avg_relevance,
                        MIN(timestamp) as oldest_timestamp,
                        MAX(timestamp) as newest_timestamp
                    FROM captures
                    WHERE context_window = ?
                ''', (window_name,))
                
                row = cursor.fetchone()
                if row:
                    stats[window_name] = {
                        "capture_count": row[0],
                        "avg_relevance": row[1] if row[1] else 0.0,
                        "oldest_timestamp": row[2],
                        "newest_timestamp": row[3],
                        "age_hours": (time.time() - row[2]) / 3600 if row[2] else 0
                    }
            
            conn.close()
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get context window stats: {e}")
            return {}
    
    async def get_capture_image(self, capture_id: str) -> Optional[Image.Image]:
        """Get the image for a specific capture, decompressing if necessary"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT image_path, compressed_image_data FROM captures WHERE id = ?
            ''', (capture_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            image_path, compressed_data = row
            
            # Try to load from file first
            if image_path:
                file_path = Path(image_path)
                if file_path.exists():
                    return Image.open(file_path)
            
            # Try to decompress from database
            if compressed_data:
                return self._decompress_image_data(compressed_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get capture image: {e}")
            return None
    
    def get_privacy_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get privacy action log"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, action, details, sensitive_app_detected
                FROM privacy_log
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            logs = []
            for row in cursor.fetchall():
                logs.append({
                    "timestamp": row[0],
                    "action": row[1],
                    "details": row[2],
                    "sensitive_app_detected": bool(row[3])
                })
            
            conn.close()
            return logs
            
        except Exception as e:
            self.logger.error(f"Failed to get privacy log: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        context_stats = self.get_context_window_stats()
        
        stats.update({
            "state": self.state.value,
            "capture_count": self.capture_count,
            "last_capture_time": self.last_capture_time,
            "cache_size": len(self.content_cache),
            "context_windows": context_stats,
            "storage_features": {
                "compression_enabled": self.config.enable_compression,
                "encryption_enabled": self.config.enable_encryption,
                "semantic_indexing_enabled": self.config.enable_semantic_indexing,
                "max_reference_age_hours": self.config.max_reference_age_hours
            },
            "config": {
                "capture_interval": self.config.capture_interval,
                "max_stored_captures": self.config.max_stored_captures,
                "privacy_mode": self.config.privacy_mode,
                "context_window_count": len(self.config.context_windows)
            }
        })
        
        # Add database size info
        try:
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            stats["storage_info"] = {
                "database_size_bytes": db_size,
                "storage_path": str(self.storage_path)
            }
        except Exception:
            stats["storage_info"] = {"database_size_bytes": 0, "storage_path": str(self.storage_path)}
        
        return stats
    
    # Privacy and Consent Management Methods
    
    def add_privacy_zone(self, name: str, x: int, y: int, width: int, height: int,
                        zone_type: PrivacyZoneType = PrivacyZoneType.BLUR) -> str:
        """Add a privacy zone to protect sensitive screen areas"""
        return self.consent_manager.add_privacy_zone(name, x, y, width, height, zone_type)
    
    def remove_privacy_zone(self, zone_id: str) -> bool:
        """Remove a privacy zone"""
        return self.consent_manager.remove_privacy_zone(zone_id)
    
    def get_privacy_zones(self) -> Dict[str, PrivacyZone]:
        """Get all configured privacy zones"""
        return self.consent_manager.privacy_zones.copy()
    
    def set_monitoring_state(self, state: MonitoringState, reason: str = ""):
        """Set the monitoring state"""
        self.consent_manager.set_monitoring_state(state, reason)
    
    def pause_monitoring(self, reason: str = "user_request"):
        """Pause screen monitoring"""
        self.consent_manager.pause_monitoring(reason)
    
    def resume_monitoring(self):
        """Resume screen monitoring if consent allows"""
        self.consent_manager.resume_monitoring()
    
    def get_consent_summary(self) -> Dict[str, Any]:
        """Get summary of current privacy and consent status"""
        return self.consent_manager.get_consent_summary()
    
    def revoke_consent(self, feature: str):
        """Revoke consent for a specific feature"""
        self.consent_manager.revoke_consent(feature)
    
    def request_privacy_consent(self, feature: str, required_level: ConsentLevel = ConsentLevel.STANDARD,
                               expires_hours: Optional[int] = None) -> ConsentLevel:
        """Request user consent for a feature"""
        return self.consent_manager.request_privacy_consent(feature, required_level, expires_hours)
    
    async def get_detected_sensitive_apps(self) -> List:
        """Get currently detected sensitive applications"""
        return await self.consent_manager.scan_for_sensitive_apps()
    
    def add_sensitive_app_pattern(self, app_name: str, process_patterns: List[str], 
                                 window_patterns: List[str], priority: int = 5):
        """Add custom sensitive application detection pattern"""
        from .consent_manager import SensitiveApp
        app = SensitiveApp(
            app_name=app_name,
            process_patterns=process_patterns,
            window_patterns=window_patterns,
            priority=priority
        )
        self.consent_manager.add_sensitive_app(app)
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            await self.stop_capture()
            
            # Cleanup consent manager
            await self.consent_manager.cleanup()
            
            # Clear caches
            self.content_cache.clear()
            
            self.logger.info("ScreenContextManager cleaned up")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}") 