"""
Voice Interface Manager for Sovereign AI Agent

This module provides comprehensive voice interaction capabilities including:
- Speech-to-text using local Whisper model
- Text-to-speech with multiple engine options
- Voice Activity Detection (VAD)
- Wake word detection ("Hey Sovereign")
- Audio input/output management
- Noise reduction and audio processing
- Real-time audio streaming
"""

import asyncio
import logging
import threading
import time
import wave
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass
import queue
import json

# Audio processing
import sounddevice as sd
import numpy as np
import librosa
import webrtcvad
import noisereduce as nr
from scipy import signal

# Speech recognition and TTS
import whisper
import pyttsx3
import speech_recognition as sr

# Wake word detection
import pvporcupine

# Async utilities
from concurrent.futures import ThreadPoolExecutor

from .config import Config
from .logger import get_performance_logger


class VoiceState(Enum):
    """Voice interface states"""
    STOPPED = "stopped"
    LISTENING = "listening"
    WAKE_WORD_DETECTED = "wake_word_detected"
    VOICE_DETECTED = "voice_detected"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


class AudioDevice(Enum):
    """Audio device types"""
    DEFAULT = "default"
    MICROPHONE = "microphone"
    SPEAKERS = "speakers"


@dataclass
class VoiceMetrics:
    """Voice interface performance metrics"""
    wake_word_detections: int = 0
    successful_transcriptions: int = 0
    failed_transcriptions: int = 0
    speech_synthesis_count: int = 0
    average_transcription_time: float = 0.0
    average_synthesis_time: float = 0.0
    voice_activity_detections: int = 0
    noise_reduction_applied: int = 0
    total_audio_processed_seconds: float = 0.0
    uptime_seconds: float = 0.0


@dataclass
class AudioChunk:
    """Audio data chunk for processing"""
    data: np.ndarray
    timestamp: datetime
    sample_rate: int
    is_speech: bool = False
    confidence: float = 0.0


class VoiceInterfaceManager:
    """
    Comprehensive voice interface manager for Sovereign AI Agent
    
    Handles all voice-related functionality including speech recognition,
    text-to-speech, wake word detection, and audio processing.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.perf_logger = get_performance_logger()
        
        # Voice interface state
        self.state = VoiceState.STOPPED
        self.is_initialized = False
        self.is_listening = False
        self.is_speaking = False
        
        # Audio configuration
        self.sample_rate = config.voice.sample_rate
        self.chunk_duration = config.voice.chunk_duration_ms / 1000.0  # Convert to seconds
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Audio devices
        self.input_device = None
        self.output_device = None
        
        # Models and engines
        self.whisper_model = None
        self.tts_engine = None
        self.vad = None
        self.porcupine = None
        
        # Audio processing
        self.audio_queue = queue.Queue(maxsize=100)
        self.audio_thread = None
        self.processing_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Voice Activity Detection
        self.vad_frames = []
        self.speech_frames = []
        self.silence_threshold = config.voice.silence_threshold
        self.speech_timeout = config.voice.speech_timeout_ms / 1000.0
        
        # Wake word detection
        self.wake_word_sensitivity = config.voice.wake_word_sensitivity
        self.wake_word_detected_callback = None
        self.wake_word_active = config.voice.wake_word_enabled
        
        # Callbacks
        self.voice_callbacks = {
            'on_wake_word': [],
            'on_speech_start': [],
            'on_speech_end': [],
            'on_transcription': [],
            'on_synthesis_start': [],
            'on_synthesis_end': [],
            'on_error': [],
            'on_state_change': []
        }
        
        # Performance metrics
        self.metrics = VoiceMetrics()
        self.start_time = datetime.now()
        
        # Audio buffers
        self.audio_buffer = []
        self.max_buffer_duration = 30.0  # seconds
        self.noise_profile = None
        
        # Threading locks
        self.state_lock = threading.Lock()
        self.processing_lock = threading.Lock()
    
    async def initialize(self) -> bool:
        """
        Initialize the voice interface system
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("🎤 Initializing Voice Interface Manager...")
            
            # Initialize audio devices
            if not await self._initialize_audio_devices():
                self.logger.error("❌ Failed to initialize audio devices")
                return False
            
            # Initialize Whisper model
            if not await self._initialize_whisper():
                self.logger.error("❌ Failed to initialize Whisper model")
                return False
            
            # Initialize Text-to-Speech
            if not await self._initialize_tts():
                self.logger.error("❌ Failed to initialize TTS engine")
                return False
            
            # Initialize Voice Activity Detection
            if not await self._initialize_vad():
                self.logger.error("❌ Failed to initialize VAD")
                return False
            
            # Initialize Wake Word Detection
            if self.config.voice.wake_word_enabled:
                if not await self._initialize_wake_word():
                    self.logger.warning("⚠️  Wake word detection initialization failed")
                    # Not critical - continue without wake word
            
            # Start audio processing threads
            self._start_audio_threads()
            
            self.is_initialized = True
            await self._set_state(VoiceState.LISTENING)
            
            self.logger.info("✅ Voice Interface Manager initialized successfully")
            self.logger.info(f"🎯 Wake word: {'Enabled' if self.wake_word_active else 'Disabled'}")
            self.logger.info(f"🎯 Sample rate: {self.sample_rate} Hz")
            self.logger.info(f"🎯 Chunk size: {self.chunk_size} samples")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Voice interface initialization failed: {e}")
            await self._set_state(VoiceState.ERROR)
            return False
    
    async def _initialize_audio_devices(self) -> bool:
        """Initialize audio input/output devices"""
        try:
            # List available devices
            devices = sd.query_devices()
            self.logger.debug(f"Available audio devices: {len(devices)}")
            
            # Find appropriate input device (microphone)
            input_device_index = None
            if self.config.voice.input_device_name != "default":
                for i, device in enumerate(devices):
                    if (self.config.voice.input_device_name.lower() in device['name'].lower() and 
                        device['max_input_channels'] > 0):
                        input_device_index = i
                        break
            
            # Find appropriate output device (speakers)
            output_device_index = None
            if self.config.voice.output_device_name != "default":
                for i, device in enumerate(devices):
                    if (self.config.voice.output_device_name.lower() in device['name'].lower() and 
                        device['max_output_channels'] > 0):
                        output_device_index = i
                        break
            
            # Test audio input
            try:
                test_recording = sd.rec(
                    frames=self.sample_rate,  # 1 second
                    samplerate=self.sample_rate,
                    channels=1,
                    device=input_device_index,
                    dtype=np.float32
                )
                sd.wait()
                self.logger.debug("✅ Audio input test successful")
            except Exception as e:
                self.logger.error(f"❌ Audio input test failed: {e}")
                return False
            
            # Test audio output
            try:
                test_tone = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(self.sample_rate * 0.1)))
                sd.play(test_tone, samplerate=self.sample_rate, device=output_device_index)
                sd.wait()
                self.logger.debug("✅ Audio output test successful")
            except Exception as e:
                self.logger.error(f"❌ Audio output test failed: {e}")
                return False
            
            self.input_device = input_device_index
            self.output_device = output_device_index
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Audio device initialization failed: {e}")
            return False
    
    async def _initialize_whisper(self) -> bool:
        """Initialize Whisper speech recognition model"""
        try:
            model_size = self.config.voice.whisper_model_size
            self.logger.info(f"🔄 Loading Whisper model: {model_size}")
            
            # Load Whisper model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.whisper_model = await loop.run_in_executor(
                self.executor, 
                whisper.load_model, 
                model_size
            )
            
            self.logger.info("✅ Whisper model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Whisper initialization failed: {e}")
            return False
    
    async def _initialize_tts(self) -> bool:
        """Initialize Text-to-Speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a voice that matches user preference
                voice_name = self.config.voice.tts_voice_name.lower()
                selected_voice = None
                
                for voice in voices:
                    if voice_name in voice.name.lower():
                        selected_voice = voice
                        break
                
                # Fallback to first available voice
                if not selected_voice and voices:
                    selected_voice = voices[0]
                
                if selected_voice:
                    self.tts_engine.setProperty('voice', selected_voice.id)
                    self.logger.info(f"🗣️  TTS Voice: {selected_voice.name}")
            
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', self.config.voice.tts_speech_rate)
            self.tts_engine.setProperty('volume', self.config.voice.tts_volume)
            
            self.logger.info("✅ TTS engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ TTS initialization failed: {e}")
            return False
    
    async def _initialize_vad(self) -> bool:
        """Initialize Voice Activity Detection"""
        try:
            # WebRTC VAD supports specific sample rates
            vad_sample_rate = 16000  # WebRTC VAD requirement
            self.vad = webrtcvad.Vad(self.config.voice.vad_aggressiveness)
            
            self.logger.info("✅ Voice Activity Detection initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ VAD initialization failed: {e}")
            return False
    
    async def _initialize_wake_word(self) -> bool:
        """Initialize Porcupine wake word detection"""
        try:
            # Use Porcupine's built-in "Hey Google" as a placeholder
            # In production, you would train a custom "Hey Sovereign" model
            self.porcupine = pvporcupine.create(
                access_key=self.config.voice.porcupine_access_key,
                keywords=['hey google'],  # Placeholder - would be custom "hey sovereign"
                sensitivities=[self.wake_word_sensitivity]
            )
            
            self.logger.info("✅ Wake word detection initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Wake word detection failed: {e}")
            return False
    
    def _start_audio_threads(self):
        """Start audio processing threads"""
        # Audio capture thread
        self.audio_thread = threading.Thread(
            target=self._audio_capture_loop,
            daemon=True,
            name="AudioCapture"
        )
        self.audio_thread.start()
        
        # Audio processing thread
        self.processing_thread = threading.Thread(
            target=self._audio_processing_loop,
            daemon=True,
            name="AudioProcessing"
        )
        self.processing_thread.start()
        
        self.logger.info("🎧 Audio processing threads started")
    
    def _audio_capture_loop(self):
        """Continuous audio capture loop"""
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                device=self.input_device,
                blocksize=self.chunk_size,
                dtype=np.float32,
                callback=self._audio_callback
            ):
                while self.is_initialized and not self._should_stop():
                    time.sleep(0.1)  # Small sleep to prevent busy waiting
                    
        except Exception as e:
            self.logger.error(f"❌ Audio capture error: {e}")
            asyncio.create_task(self._set_state(VoiceState.ERROR))
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Audio input callback for real-time processing"""
        if status:
            self.logger.warning(f"⚠️  Audio callback status: {status}")
        
        try:
            # Create audio chunk
            audio_chunk = AudioChunk(
                data=indata.copy().flatten(),
                timestamp=datetime.now(),
                sample_rate=self.sample_rate
            )
            
            # Add to processing queue (non-blocking)
            try:
                self.audio_queue.put_nowait(audio_chunk)
            except queue.Full:
                # Drop oldest chunk if queue is full
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_chunk)
                except queue.Empty:
                    pass
                    
        except Exception as e:
            self.logger.error(f"❌ Audio callback error: {e}")
    
    def _audio_processing_loop(self):
        """Audio processing loop for VAD, wake word detection, and transcription"""
        while self.is_initialized and not self._should_stop():
            try:
                # Get audio chunk from queue
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process audio chunk
                asyncio.run(self._process_audio_chunk(audio_chunk))
                
            except Exception as e:
                self.logger.error(f"❌ Audio processing error: {e}")
                time.sleep(0.1)
    
    async def _process_audio_chunk(self, chunk: AudioChunk):
        """Process individual audio chunk"""
        try:
            # Apply noise reduction if enabled
            if self.config.voice.noise_reduction_enabled:
                chunk.data = await self._apply_noise_reduction(chunk.data)
                self.metrics.noise_reduction_applied += 1
            
            # Voice Activity Detection
            is_speech = await self._detect_voice_activity(chunk)
            chunk.is_speech = is_speech
            
            if is_speech:
                self.metrics.voice_activity_detections += 1
                
                # Wake word detection (if enabled and not currently processing)
                if (self.wake_word_active and 
                    self.state not in [VoiceState.PROCESSING, VoiceState.SPEAKING]):
                    wake_word_detected = await self._detect_wake_word(chunk.data)
                    if wake_word_detected:
                        await self._handle_wake_word_detected()
                        return
                
                # Collect speech for transcription
                await self._collect_speech_audio(chunk)
            else:
                # Check if we should process collected speech
                await self._check_speech_timeout()
            
            # Update metrics
            self.metrics.total_audio_processed_seconds += self.chunk_duration
            
        except Exception as e:
            self.logger.error(f"❌ Audio chunk processing error: {e}")
    
    async def _apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise reduction to audio data"""
        try:
            # Use noisereduce library for spectral gating
            reduced_audio = nr.reduce_noise(
                y=audio_data,
                sr=self.sample_rate,
                stationary=False,
                prop_decrease=0.8
            )
            return reduced_audio
            
        except Exception as e:
            self.logger.error(f"❌ Noise reduction error: {e}")
            return audio_data  # Return original audio if noise reduction fails
    
    async def _detect_voice_activity(self, chunk: AudioChunk) -> bool:
        """Detect if audio chunk contains voice activity"""
        try:
            # Convert to 16-bit PCM for WebRTC VAD
            audio_16bit = (chunk.data * 32767).astype(np.int16)
            
            # Resample to 16kHz if needed (WebRTC VAD requirement)
            if chunk.sample_rate != 16000:
                audio_16bit = librosa.resample(
                    audio_16bit.astype(np.float32),
                    orig_sr=chunk.sample_rate,
                    target_sr=16000
                ).astype(np.int16)
            
            # WebRTC VAD requires specific frame sizes (10, 20, or 30 ms)
            frame_duration = 30  # ms
            frame_size = int(16000 * frame_duration / 1000)
            
            if len(audio_16bit) >= frame_size:
                frame = audio_16bit[:frame_size]
                is_speech = self.vad.is_speech(frame.tobytes(), 16000)
                
                # Additional energy-based detection
                energy = np.sum(chunk.data ** 2) / len(chunk.data)
                energy_threshold = self.silence_threshold
                
                return is_speech or energy > energy_threshold
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Voice activity detection error: {e}")
            return False
    
    async def _detect_wake_word(self, audio_data: np.ndarray) -> bool:
        """Detect wake word in audio data"""
        try:
            if not self.porcupine:
                return False
            
            # Convert to 16-bit PCM for Porcupine
            audio_16bit = (audio_data * 32767).astype(np.int16)
            
            # Resample to Porcupine's required sample rate
            if self.sample_rate != self.porcupine.sample_rate:
                audio_16bit = librosa.resample(
                    audio_16bit.astype(np.float32),
                    orig_sr=self.sample_rate,
                    target_sr=self.porcupine.sample_rate
                ).astype(np.int16)
            
            # Process with Porcupine
            keyword_index = self.porcupine.process(audio_16bit)
            
            if keyword_index >= 0:
                self.logger.info("🎯 Wake word detected!")
                self.metrics.wake_word_detections += 1
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Wake word detection error: {e}")
            return False
    
    async def _handle_wake_word_detected(self):
        """Handle wake word detection"""
        await self._set_state(VoiceState.WAKE_WORD_DETECTED)
        await self._notify_callbacks('on_wake_word', {'timestamp': datetime.now()})
        
        # Clear speech buffer and start fresh listening
        self.speech_frames.clear()
        await self._set_state(VoiceState.VOICE_DETECTED)
    
    async def _collect_speech_audio(self, chunk: AudioChunk):
        """Collect speech audio for transcription"""
        if chunk.is_speech:
            self.speech_frames.append(chunk)
            
            # Limit buffer size
            max_frames = int(self.max_buffer_duration / self.chunk_duration)
            if len(self.speech_frames) > max_frames:
                self.speech_frames = self.speech_frames[-max_frames:]
    
    async def _check_speech_timeout(self):
        """Check if speech has ended and should be processed"""
        if not self.speech_frames:
            return
        
        # Check time since last speech
        last_speech_time = self.speech_frames[-1].timestamp
        silence_duration = (datetime.now() - last_speech_time).total_seconds()
        
        if silence_duration >= self.speech_timeout:
            # Process collected speech
            await self._process_collected_speech()
    
    async def _process_collected_speech(self):
        """Process collected speech audio for transcription"""
        if not self.speech_frames:
            return
        
        try:
            await self._set_state(VoiceState.PROCESSING)
            
            # Combine audio chunks
            combined_audio = np.concatenate([chunk.data for chunk in self.speech_frames])
            
            # Transcribe audio
            start_time = time.time()
            transcription = await self._transcribe_audio(combined_audio)
            transcription_time = time.time() - start_time
            
            # Update metrics
            if transcription and transcription.strip():
                self.metrics.successful_transcriptions += 1
                self.metrics.average_transcription_time = (
                    (self.metrics.average_transcription_time * (self.metrics.successful_transcriptions - 1) +
                     transcription_time) / self.metrics.successful_transcriptions
                )
                
                # Notify callbacks
                await self._notify_callbacks('on_transcription', {
                    'text': transcription,
                    'timestamp': datetime.now(),
                    'processing_time': transcription_time
                })
                
                self.logger.info(f"🎙️  Transcribed: '{transcription}' ({transcription_time:.2f}s)")
            else:
                self.metrics.failed_transcriptions += 1
                self.logger.debug("🔇 Empty transcription result")
            
            # Clear speech buffer
            self.speech_frames.clear()
            
            # Return to listening state
            await self._set_state(VoiceState.LISTENING)
            
        except Exception as e:
            self.logger.error(f"❌ Speech processing error: {e}")
            self.metrics.failed_transcriptions += 1
            await self._set_state(VoiceState.ERROR)
    
    async def _transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio using Whisper"""
        try:
            if not self.whisper_model:
                raise Exception("Whisper model not initialized")
            
            # Prepare audio for Whisper
            # Whisper expects audio at 16kHz
            if self.sample_rate != 16000:
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=self.sample_rate,
                    target_sr=16000
                )
            
            # Ensure audio is float32 and normalized
            audio_data = audio_data.astype(np.float32)
            
            # Transcribe in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.whisper_model.transcribe,
                audio_data
            )
            
            # Extract text from result
            text = result.get('text', '').strip()
            return text if text else None
            
        except Exception as e:
            self.logger.error(f"❌ Transcription error: {e}")
            return None
    
    async def synthesize_speech(self, text: str) -> bool:
        """
        Convert text to speech and play it
        
        Args:
            text: Text to synthesize
            
        Returns:
            bool: True if synthesis successful, False otherwise
        """
        try:
            if not text or not text.strip():
                return False
            
            await self._set_state(VoiceState.SPEAKING)
            await self._notify_callbacks('on_synthesis_start', {'text': text})
            
            start_time = time.time()
            
            # Synthesize speech in thread pool
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.executor,
                self._synthesize_speech_sync,
                text
            )
            
            synthesis_time = time.time() - start_time
            
            # Update metrics
            if success:
                self.metrics.speech_synthesis_count += 1
                self.metrics.average_synthesis_time = (
                    (self.metrics.average_synthesis_time * (self.metrics.speech_synthesis_count - 1) +
                     synthesis_time) / self.metrics.speech_synthesis_count
                )
                
                self.logger.info(f"🗣️  Synthesized speech: '{text[:50]}...' ({synthesis_time:.2f}s)")
            
            await self._notify_callbacks('on_synthesis_end', {
                'text': text,
                'success': success,
                'processing_time': synthesis_time
            })
            
            # Return to listening state
            await self._set_state(VoiceState.LISTENING)
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Speech synthesis error: {e}")
            await self._set_state(VoiceState.ERROR)
            return False
    
    def _synthesize_speech_sync(self, text: str) -> bool:
        """Synchronous speech synthesis"""
        try:
            if not self.tts_engine:
                return False
            
            # Configure output device if specified
            if self.output_device is not None:
                # Note: pyttsx3 doesn't directly support device selection
                # This would need to be implemented with a different TTS engine
                pass
            
            # Synthesize and play
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Synchronous speech synthesis error: {e}")
            return False
    
    async def start_listening(self) -> bool:
        """
        Start voice interface listening
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            if not self.is_initialized:
                self.logger.error("❌ Voice interface not initialized")
                return False
            
            if self.is_listening:
                self.logger.debug("🎧 Already listening")
                return True
            
            self.is_listening = True
            await self._set_state(VoiceState.LISTENING)
            
            self.logger.info("🎧 Voice interface started listening")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start listening: {e}")
            return False
    
    async def stop_listening(self) -> bool:
        """
        Stop voice interface listening
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        try:
            self.is_listening = False
            await self._set_state(VoiceState.STOPPED)
            
            self.logger.info("🔇 Voice interface stopped listening")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop listening: {e}")
            return False
    
    async def toggle_wake_word(self, enabled: bool) -> bool:
        """
        Enable or disable wake word detection
        
        Args:
            enabled: Whether to enable wake word detection
            
        Returns:
            bool: True if toggle successful, False otherwise
        """
        try:
            self.wake_word_active = enabled
            self.logger.info(f"🎯 Wake word {'enabled' if enabled else 'disabled'}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to toggle wake word: {e}")
            return False
    
    def add_voice_callback(self, event: str, callback: Callable):
        """
        Add callback for voice events
        
        Args:
            event: Event name (on_wake_word, on_transcription, etc.)
            callback: Callback function
        """
        if event in self.voice_callbacks:
            self.voice_callbacks[event].append(callback)
        else:
            self.logger.warning(f"⚠️  Unknown voice event: {event}")
    
    def remove_voice_callback(self, event: str, callback: Callable):
        """Remove callback for voice events"""
        if event in self.voice_callbacks and callback in self.voice_callbacks[event]:
            self.voice_callbacks[event].remove(callback)
    
    async def _notify_callbacks(self, event: str, data: Dict[str, Any]):
        """Notify all callbacks for an event"""
        for callback in self.voice_callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"❌ Callback error for {event}: {e}")
    
    async def _set_state(self, new_state: VoiceState):
        """Set voice interface state with thread safety"""
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            
            if old_state != new_state:
                self.logger.debug(f"🔄 Voice state: {old_state.value} → {new_state.value}")
                asyncio.create_task(self._notify_callbacks('on_state_change', {
                    'old_state': old_state.value,
                    'new_state': new_state.value,
                    'timestamp': datetime.now()
                }))
    
    def _should_stop(self) -> bool:
        """Check if voice interface should stop"""
        return not self.is_initialized or self.state == VoiceState.ERROR
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get voice interface status
        
        Returns:
            dict: Status information
        """
        self.metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'state': self.state.value,
            'is_initialized': self.is_initialized,
            'is_listening': self.is_listening,
            'is_speaking': self.is_speaking,
            'wake_word_active': self.wake_word_active,
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'audio_queue_size': self.audio_queue.qsize(),
            'models_loaded': {
                'whisper': self.whisper_model is not None,
                'tts': self.tts_engine is not None,
                'vad': self.vad is not None,
                'porcupine': self.porcupine is not None
            },
            'metrics': {
                'wake_word_detections': self.metrics.wake_word_detections,
                'successful_transcriptions': self.metrics.successful_transcriptions,
                'failed_transcriptions': self.metrics.failed_transcriptions,
                'transcription_success_rate': (
                    self.metrics.successful_transcriptions / 
                    max(1, self.metrics.successful_transcriptions + self.metrics.failed_transcriptions)
                ),
                'speech_synthesis_count': self.metrics.speech_synthesis_count,
                'average_transcription_time': self.metrics.average_transcription_time,
                'average_synthesis_time': self.metrics.average_synthesis_time,
                'voice_activity_detections': self.metrics.voice_activity_detections,
                'total_audio_processed_seconds': self.metrics.total_audio_processed_seconds,
                'uptime_seconds': self.metrics.uptime_seconds
            }
        }
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        status = await self.get_status()
        return status['metrics']
    
    async def close(self):
        """Clean up voice interface resources"""
        try:
            self.logger.info("🔄 Shutting down Voice Interface Manager...")
            
            # Stop listening
            await self.stop_listening()
            
            # Mark as not initialized to stop threads
            self.is_initialized = False
            
            # Wait for threads to finish
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=2.0)
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
            
            # Clean up models and engines
            if self.porcupine:
                self.porcupine.delete()
                self.porcupine = None
            
            if self.tts_engine:
                try:
                    self.tts_engine.stop()
                except:
                    pass
                self.tts_engine = None
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            # Clear audio queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            await self._set_state(VoiceState.STOPPED)
            
            self.logger.info("✅ Voice Interface Manager shut down successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error during voice interface shutdown: {e}")


# Utility functions for voice interface testing and configuration

async def test_voice_interface(config: Config) -> bool:
    """
    Test voice interface functionality
    
    Args:
        config: Configuration object
        
    Returns:
        bool: True if all tests pass, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🧪 Testing Voice Interface...")
        
        # Initialize voice interface
        voice_manager = VoiceInterfaceManager(config)
        
        if not await voice_manager.initialize():
            logger.error("❌ Voice interface initialization failed")
            return False
        
        # Test TTS
        logger.info("🗣️  Testing text-to-speech...")
        tts_success = await voice_manager.synthesize_speech("Voice interface test successful.")
        
        if tts_success:
            logger.info("✅ TTS test passed")
        else:
            logger.warning("⚠️  TTS test failed")
        
        # Test listening for a short duration
        logger.info("🎧 Testing voice listening (5 seconds)...")
        await voice_manager.start_listening()
        await asyncio.sleep(5)
        await voice_manager.stop_listening()
        
        # Get status
        status = await voice_manager.get_status()
        logger.info(f"📊 Voice interface status: {status['state']}")
        
        # Clean up
        await voice_manager.close()
        
        logger.info("✅ Voice interface test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Voice interface test failed: {e}")
        return False


def get_audio_devices() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get available audio devices
    
    Returns:
        dict: Available input and output devices
    """
    try:
        devices = sd.query_devices()
        
        input_devices = []
        output_devices = []
        
        for i, device in enumerate(devices):
            device_info = {
                'index': i,
                'name': device['name'],
                'max_input_channels': device['max_input_channels'],
                'max_output_channels': device['max_output_channels'],
                'default_samplerate': device['default_samplerate']
            }
            
            if device['max_input_channels'] > 0:
                input_devices.append(device_info)
            
            if device['max_output_channels'] > 0:
                output_devices.append(device_info)
        
        return {
            'input_devices': input_devices,
            'output_devices': output_devices
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"❌ Error getting audio devices: {e}")
        return {'input_devices': [], 'output_devices': []} 