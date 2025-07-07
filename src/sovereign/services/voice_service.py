"""
Voice Service - Lazy-loaded voice input/output system

Provides on-demand voice input/output with heavy voice libraries
loaded only on first listen() or speak() request. Subsequent calls reuse the initialized components.
No heavy imports at module level.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class VoiceState(Enum):
    """Voice interface states"""
    STOPPED = "stopped"
    LISTENING = "listening"
    WAKE_WORD_DETECTED = "wake_word_detected"
    VOICE_DETECTED = "voice_detected"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class VoiceResult:
    """Lightweight container for voice operation results"""
    success: bool
    operation: str  # "listen", "speak", "wake_word"
    text_content: str = ""
    confidence: float = 0.0
    error: Optional[str] = None
    processing_time: float = 0.0
    audio_duration: float = 0.0


@dataclass
class SpeechResult:
    """Result from speech recognition"""
    success: bool
    text: str = ""
    confidence: float = 0.0
    language: str = "en"
    processing_time: float = 0.0


class VoiceService:
    """
    On-demand voice service with lazy loading
    
    Voice I/O initialization only happens on first listen() or speak() call.
    Subsequent calls reuse the existing components for performance.
    Thread-safe with async locks to prevent double-loading during concurrent access.
    """
    
    def __init__(self):
        """Initialize with minimal overhead - no heavy imports"""
        self._voice_manager = None
        self._initialized = False
        self._loading_lock = asyncio.Lock()
        
        # Voice state
        self._state = VoiceState.STOPPED
        self._is_listening = False
        self._is_speaking = False
        
        # Statistics
        self._stats = {
            'total_listen_calls': 0,
            'total_speak_calls': 0,
            'successful_recognitions': 0,
            'successful_syntheses': 0,
            'failed_operations': 0,
            'initialization_time': 0.0,
            'avg_recognition_time': 0.0,
            'avg_synthesis_time': 0.0,
            'first_operation_at': None,
            'last_operation_at': None,
            'wake_word_detections': 0
        }
        
        # Callbacks
        self._callbacks = {
            'on_wake_word': [],
            'on_speech_start': [],
            'on_speech_end': [],
            'on_transcription': [],
            'on_synthesis_start': [],
            'on_synthesis_end': [],
            'on_error': [],
            'on_state_change': []
        }
        
        logging.info("VoiceService created (lazy loading enabled)")
    
    async def _load_components(self):
        """
        Load heavy voice processing components on first use
        
        This is where the actual voice manager initialization happens.
        Protected by async lock to prevent double-loading.
        """
        async with self._loading_lock:
            if self._initialized:
                return  # Already loaded by another thread
            
            logging.info("🔄 Loading voice processing components...")
            start_time = time.perf_counter()
            
            try:
                # Import heavy modules only when needed
                from ..voice_interface import VoiceInterfaceManager
                from ..config import Config
                
                # Create configuration
                config = Config()
                
                # Initialize voice interface manager
                self._voice_manager = VoiceInterfaceManager(config)
                
                # Initialize the voice manager
                success = await self._voice_manager.initialize()
                if not success:
                    raise RuntimeError("Voice interface initialization failed")
                
                # Mark as initialized
                self._initialized = True
                
                init_time = time.perf_counter() - start_time
                self._stats['initialization_time'] = init_time
                
                logging.info(f"✅ Voice processing system loaded in {init_time:.2f}s")
                
            except Exception as e:
                logging.error(f"❌ Failed to load voice processing components: {e}")
                raise RuntimeError(f"Voice processing initialization failed: {e}")
    
    async def listen(self, timeout: float = 5.0, wake_word_required: bool = False) -> VoiceResult:
        """
        Listen for voice input and return transcribed text
        
        On first call, this loads the voice processing components.
        Subsequent calls use the cached components.
        
        Args:
            timeout: Maximum time to listen in seconds
            wake_word_required: Whether to wait for wake word before listening
            
        Returns:
            VoiceResult with transcription outcome
        """
        start_time = time.perf_counter()
        
        try:
            # Update statistics
            self._stats['total_listen_calls'] += 1
            if self._stats['first_operation_at'] is None:
                self._stats['first_operation_at'] = datetime.now()
            self._stats['last_operation_at'] = datetime.now()
            
            # Ensure components are loaded (lazy loading)
            if not self._initialized:
                await self._load_components()
            
            # Start listening
            self._is_listening = True
            await self._set_state(VoiceState.LISTENING)
            
            if not await self._voice_manager.start_listening():
                self._stats['failed_operations'] += 1
                return VoiceResult(
                    success=False,
                    operation="listen",
                    error="Failed to start listening",
                    processing_time=time.perf_counter() - start_time
                )
            
            # Wait for speech or timeout
            # This is a simplified implementation - the actual voice manager
            # handles the complex audio processing and speech recognition
            await asyncio.sleep(timeout)  # Placeholder for actual listening logic
            
            # Stop listening
            await self._voice_manager.stop_listening()
            self._is_listening = False
            await self._set_state(VoiceState.STOPPED)
            
            # Get transcription results (placeholder - would come from voice manager)
            text_result = "Hello, this is a placeholder transcription"  # Placeholder
            confidence = 0.85  # Placeholder
            
            # Update success statistics
            self._stats['successful_recognitions'] += 1
            processing_time = time.perf_counter() - start_time
            
            # Update average recognition time
            if self._stats['successful_recognitions'] > 0:
                total_time = self._stats['avg_recognition_time'] * (self._stats['successful_recognitions'] - 1) + processing_time
                self._stats['avg_recognition_time'] = total_time / self._stats['successful_recognitions']
            
            return VoiceResult(
                success=True,
                operation="listen",
                text_content=text_result,
                confidence=confidence,
                processing_time=processing_time,
                audio_duration=timeout
            )
                
        except Exception as e:
            self._stats['failed_operations'] += 1
            processing_time = time.perf_counter() - start_time
            
            logging.error(f"Voice listening failed: {e}")
            
            return VoiceResult(
                success=False,
                operation="listen",
                error=str(e),
                processing_time=processing_time
            )
        finally:
            self._is_listening = False
    
    async def speak(self, text: str, voice_name: Optional[str] = None) -> VoiceResult:
        """
        Synthesize and speak the given text
        
        Loads components if not already loaded.
        
        Args:
            text: Text to speak
            voice_name: Optional voice name to use
            
        Returns:
            VoiceResult with synthesis outcome
        """
        start_time = time.perf_counter()
        
        try:
            # Update statistics
            self._stats['total_speak_calls'] += 1
            if self._stats['first_operation_at'] is None:
                self._stats['first_operation_at'] = datetime.now()
            self._stats['last_operation_at'] = datetime.now()
            
            # Ensure components are loaded (lazy loading)
            if not self._initialized:
                await self._load_components()
            
            # Start speaking
            self._is_speaking = True
            await self._set_state(VoiceState.SPEAKING)
            
            # Synthesize speech
            success = await self._voice_manager.synthesize_speech(text)
            
            if not success:
                self._stats['failed_operations'] += 1
                return VoiceResult(
                    success=False,
                    operation="speak",
                    text_content=text,
                    error="Speech synthesis failed",
                    processing_time=time.perf_counter() - start_time
                )
            
            # Update success statistics
            self._stats['successful_syntheses'] += 1
            processing_time = time.perf_counter() - start_time
            
            # Update average synthesis time
            if self._stats['successful_syntheses'] > 0:
                total_time = self._stats['avg_synthesis_time'] * (self._stats['successful_syntheses'] - 1) + processing_time
                self._stats['avg_synthesis_time'] = total_time / self._stats['successful_syntheses']
            
            return VoiceResult(
                success=True,
                operation="speak",
                text_content=text,
                processing_time=processing_time,
                audio_duration=len(text) * 0.1  # Rough estimate
            )
                
        except Exception as e:
            self._stats['failed_operations'] += 1
            processing_time = time.perf_counter() - start_time
            
            logging.error(f"Speech synthesis failed: {e}")
            
            return VoiceResult(
                success=False,
                operation="speak",
                text_content=text,
                error=str(e),
                processing_time=processing_time
            )
        finally:
            self._is_speaking = False
            await self._set_state(VoiceState.STOPPED)
    
    async def enable_wake_word(self, enabled: bool = True) -> bool:
        """
        Enable or disable wake word detection
        
        Loads components if not already loaded.
        """
        if not self._initialized:
            await self._load_components()
        
        try:
            success = await self._voice_manager.toggle_wake_word(enabled)
            if success and enabled:
                logging.info("🎤 Wake word detection enabled")
            elif success and not enabled:
                logging.info("🎤 Wake word detection disabled")
            return success
        except Exception as e:
            logging.error(f"Failed to toggle wake word: {e}")
            return False
    
    def add_voice_callback(self, event: str, callback: Callable):
        """Add a callback for voice events"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def remove_voice_callback(self, event: str, callback: Callable):
        """Remove a callback for voice events"""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
    
    async def _set_state(self, new_state: VoiceState):
        """Set voice service state and notify callbacks"""
        old_state = self._state
        self._state = new_state
        
        if old_state != new_state:
            logging.debug(f"Voice state: {old_state.value} -> {new_state.value}")
            
            # Notify callbacks
            for callback in self._callbacks['on_state_change']:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(old_state, new_state)
                    else:
                        callback(old_state, new_state)
                except Exception as e:
                    logging.error(f"Voice state callback error: {e}")
    
    def is_voice_system_loaded(self) -> bool:
        """Check if the voice processing system has been loaded"""
        return self._initialized
    
    def get_voice_state(self) -> str:
        """Get current voice state"""
        return self._state.value
    
    def is_listening(self) -> bool:
        """Check if currently listening for voice input"""
        return self._is_listening
    
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self._is_speaking
    
    def get_stats(self) -> Dict[str, Any]:
        """Get voice service statistics"""
        stats = self._stats.copy()
        
        # Add voice manager stats if available
        if self._voice_manager:
            try:
                voice_stats = asyncio.create_task(self._voice_manager.get_performance_stats())
                # Note: This would need to be handled differently in a real async context
                stats.update({"voice_manager_stats": "available"})
            except Exception:
                pass
        
        return stats
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get health check information"""
        return {
            'service_name': 'VoiceService',
            'status': 'ready' if self._initialized else 'uninitialized',
            'voice_system_loaded': self._initialized,
            'total_operations': self._stats['total_listen_calls'] + self._stats['total_speak_calls'],
            'voice_state': self.get_voice_state(),
            'is_listening': self.is_listening(),
            'is_speaking': self.is_speaking(),
            'components': {
                'voice_manager': self._voice_manager is not None,
                'speech_recognition': self._initialized,
                'text_to_speech': self._initialized,
                'wake_word_detection': self._initialized
            }
        }
    
    async def close(self):
        """Cleanup resources"""
        if self._voice_manager:
            try:
                await self._voice_manager.close()
                logging.info("Voice processing system stopped")
            except Exception as e:
                logging.error(f"Error stopping voice processing: {e}")
        
        self._voice_manager = None
        self._initialized = False
        self._is_listening = False
        self._is_speaking = False
        self._state = VoiceState.STOPPED
        
        logging.info("VoiceService closed") 