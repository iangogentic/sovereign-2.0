"""
Screen Context Service - Lazy-loaded screen capture and OCR system

Provides on-demand screen capture and text extraction with heavy OCR libraries
loaded only on first capture request. Subsequent calls reuse the initialized components.
No heavy imports at module level.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ScreenCaptureResult:
    """Lightweight container for screen capture results"""
    success: bool
    capture_id: Optional[str] = None
    text_content: str = ""
    bounding_boxes: List[Tuple[int, int, int, int]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    ocr_confidence: float = 0.0
    
    def __post_init__(self):
        if self.bounding_boxes is None:
            self.bounding_boxes = []


class ScreenContextService:
    """
    On-demand screen context service with lazy loading
    
    Screen capture and OCR initialization only happens on first capture_and_analyse() call.
    Subsequent calls reuse the existing components for performance.
    Thread-safe with async locks to prevent double-loading during concurrent access.
    """
    
    def __init__(self):
        """Initialize with minimal overhead - no heavy imports"""
        self._screen_manager = None
        self._initialized = False
        self._loading_lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            'total_captures': 0,
            'successful_captures': 0,
            'failed_captures': 0,
            'initialization_time': 0.0,
            'avg_capture_time': 0.0,
            'first_capture_at': None,
            'last_capture_at': None
        }
        
        logging.info("ScreenContextService created (lazy loading enabled)")
    
    async def _load_components(self):
        """
        Load heavy screen capture and OCR components on first use
        
        This is where the actual screen manager initialization happens.
        Protected by async lock to prevent double-loading.
        """
        async with self._loading_lock:
            if self._initialized:
                return  # Already loaded by another thread
            
            logging.info("🔄 Loading screen capture and OCR components...")
            start_time = time.perf_counter()
            
            try:
                # Import heavy modules only when needed
                from ..screen_context_manager import ScreenContextManager, ScreenContextConfig
                from ..config import Config
                
                # Create configuration
                main_config = Config()
                screen_config = ScreenContextConfig(
                    capture_interval=main_config.screen_capture.capture_interval,
                    max_stored_captures=main_config.screen_capture.max_screenshot_history,
                    ocr_language="eng",
                    enable_preprocessing=True,
                    min_text_confidence=30.0,
                    privacy_mode=not main_config.screen_capture.enabled,
                    tesseract_path=main_config.screen_capture.tesseract_cmd_path or "",
                )
                
                # Initialize screen context manager
                self._screen_manager = ScreenContextManager(
                    config=screen_config,
                    main_config=main_config
                )
                
                # Mark as initialized
                self._initialized = True
                
                init_time = time.perf_counter() - start_time
                self._stats['initialization_time'] = init_time
                
                logging.info(f"✅ Screen capture system loaded in {init_time:.2f}s")
                
            except Exception as e:
                logging.error(f"❌ Failed to load screen capture components: {e}")
                raise RuntimeError(f"Screen capture initialization failed: {e}")
    
    async def capture_and_analyse(self) -> ScreenCaptureResult:
        """
        Capture the current screen and extract text using OCR
        
        On first call, this loads the screen capture and OCR components.
        Subsequent calls use the cached components.
        
        Returns:
            ScreenCaptureResult with capture outcome and extracted text
        """
        start_time = time.perf_counter()
        
        try:
            # Update statistics (count the capture attempt)
            self._stats['total_captures'] += 1
            if self._stats['first_capture_at'] is None:
                self._stats['first_capture_at'] = datetime.now()
            self._stats['last_capture_at'] = datetime.now()
            
            # Ensure components are loaded (lazy loading)
            if not self._initialized:
                await self._load_components()
            
            # Capture screen through manager
            capture = await self._screen_manager.capture_screen()
            
            if capture is None:
                # Privacy mode or sensitive app detected
                self._stats['failed_captures'] += 1
                return ScreenCaptureResult(
                    success=False,
                    error="Capture blocked by privacy settings",
                    processing_time=time.perf_counter() - start_time
                )
            
            # Update success statistics
            self._stats['successful_captures'] += 1
            processing_time = time.perf_counter() - start_time
            
            # Update average capture time
            if self._stats['successful_captures'] > 0:
                total_time = self._stats['avg_capture_time'] * (self._stats['successful_captures'] - 1) + processing_time
                self._stats['avg_capture_time'] = total_time / self._stats['successful_captures']
            
            return ScreenCaptureResult(
                success=True,
                capture_id=capture.id,
                text_content=capture.text_content,
                bounding_boxes=capture.bounding_boxes,
                processing_time=processing_time,
                ocr_confidence=capture.ocr_confidence
            )
                
        except Exception as e:
            self._stats['failed_captures'] += 1
            processing_time = time.perf_counter() - start_time
            
            logging.error(f"Screen capture failed: {e}")
            
            return ScreenCaptureResult(
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    async def get_recent_captures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of recent screen captures
        
        Loads components if not already loaded.
        """
        if not self._initialized:
            await self._load_components()
        
        captures = self._screen_manager.get_recent_captures(limit=limit)
        
        # Convert to lightweight format
        results = []
        for capture in captures:
            results.append({
                'id': capture.id,
                'timestamp': capture.timestamp,
                'text_content': capture.text_content,
                'bounding_boxes': capture.bounding_boxes,
                'ocr_confidence': capture.ocr_confidence,
                'relevance_score': capture.relevance_score,
                'processing_time': capture.processing_time
            })
        
        return results
    
    async def search_by_text(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search screen captures by text content
        
        Loads components if not already loaded.
        """
        if not self._initialized:
            await self._load_components()
        
        captures = await self._screen_manager.search_captures_by_text(search_term, limit)
        
        # Convert to lightweight format
        results = []
        for capture in captures:
            results.append({
                'id': capture.id,
                'timestamp': capture.timestamp,
                'text_content': capture.text_content,
                'relevance_score': capture.relevance_score,
                'ocr_confidence': capture.ocr_confidence,
                'match_type': 'text_search'
            })
        
        return results
    
    async def start_continuous_capture(self) -> bool:
        """
        Start continuous screen capture mode
        
        Loads components if not already loaded.
        """
        if not self._initialized:
            await self._load_components()
        
        try:
            await self._screen_manager.start_capture()
            return True
        except Exception as e:
            logging.error(f"Failed to start continuous capture: {e}")
            return False
    
    async def stop_continuous_capture(self) -> bool:
        """
        Stop continuous screen capture mode
        """
        if not self._initialized:
            return True  # Nothing to stop
        
        try:
            await self._screen_manager.stop_capture()
            return True
        except Exception as e:
            logging.error(f"Failed to stop continuous capture: {e}")
            return False
    
    def is_capture_system_loaded(self) -> bool:
        """Check if the screen capture system has been loaded"""
        return self._initialized
    
    def get_capture_state(self) -> str:
        """Get current capture state"""
        if not self._initialized:
            return "uninitialized"
        
        return self._screen_manager.state.value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get screen context service statistics"""
        stats = self._stats.copy()
        
        # Add screen manager stats if available
        if self._screen_manager:
            manager_stats = self._screen_manager.get_performance_stats()
            stats.update(manager_stats)
        
        return stats
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get health check information"""
        return {
            'service_name': 'ScreenContextService',
            'status': 'ready' if self._initialized else 'uninitialized',
            'capture_system_loaded': self._initialized,
            'total_captures': self._stats['total_captures'],
            'capture_state': self.get_capture_state(),
            'components': {
                'screen_manager': self._screen_manager is not None,
                'ocr_engine': self._initialized,  # OCR is part of screen manager
                'privacy_manager': self._initialized  # Privacy is part of screen manager
            }
        }
    
    async def close(self):
        """Cleanup resources"""
        if self._screen_manager:
            try:
                await self._screen_manager.stop_capture()
                # Note: ScreenContextManager doesn't have a cleanup method
                logging.info("Screen capture system stopped")
            except Exception as e:
                logging.error(f"Error stopping screen capture: {e}")
        
        self._screen_manager = None
        self._initialized = False
        
        logging.info("ScreenContextService closed") 