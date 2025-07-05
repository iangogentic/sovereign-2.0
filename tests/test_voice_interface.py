"""
Test cases for Voice Interface Manager

This module contains comprehensive tests for the voice interface functionality
including speech recognition, text-to-speech, voice activity detection,
wake word detection, and audio processing.
"""

import pytest
import asyncio
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import threading
import time

# Import the classes to test
from src.sovereign.voice_interface import (
    VoiceInterfaceManager,
    VoiceState,
    VoiceMetrics,
    AudioChunk,
    test_voice_interface,
    get_audio_devices
)
from src.sovereign.config import Config


class TestVoiceInterfaceManager:
    """Test cases for VoiceInterfaceManager"""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        config = Config()
        # Configure for testing
        config.voice.enabled = True
        config.voice.wake_word_enabled = False  # Disable for testing
        config.voice.sample_rate = 16000  # Lower sample rate for testing
        config.voice.chunk_duration_ms = 50  # Smaller chunks for testing
        config.voice.whisper_model_size = "tiny"  # Small model for testing
        config.voice.noise_reduction_enabled = False  # Disable for testing
        config.voice.porcupine_access_key = ""  # No key for testing
        return config
    
    @pytest.fixture
    def voice_manager(self, config):
        """Create a VoiceInterfaceManager instance"""
        return VoiceInterfaceManager(config)
    
    def test_initialization(self, voice_manager, config):
        """Test VoiceInterfaceManager initialization"""
        assert voice_manager.config == config
        assert voice_manager.state == VoiceState.STOPPED
        assert not voice_manager.is_initialized
        assert not voice_manager.is_listening
        assert not voice_manager.is_speaking
        assert voice_manager.sample_rate == config.voice.sample_rate
        assert voice_manager.chunk_size == int(config.voice.sample_rate * config.voice.chunk_duration_ms / 1000)
    
    def test_voice_state_enum(self):
        """Test VoiceState enum values"""
        assert VoiceState.STOPPED.value == "stopped"
        assert VoiceState.LISTENING.value == "listening"
        assert VoiceState.WAKE_WORD_DETECTED.value == "wake_word_detected"
        assert VoiceState.VOICE_DETECTED.value == "voice_detected"
        assert VoiceState.PROCESSING.value == "processing"
        assert VoiceState.SPEAKING.value == "speaking"
        assert VoiceState.ERROR.value == "error"
    
    def test_voice_metrics_initialization(self):
        """Test VoiceMetrics dataclass initialization"""
        metrics = VoiceMetrics()
        assert metrics.wake_word_detections == 0
        assert metrics.successful_transcriptions == 0
        assert metrics.failed_transcriptions == 0
        assert metrics.speech_synthesis_count == 0
        assert metrics.average_transcription_time == 0.0
        assert metrics.average_synthesis_time == 0.0
        assert metrics.voice_activity_detections == 0
        assert metrics.noise_reduction_applied == 0
        assert metrics.total_audio_processed_seconds == 0.0
        assert metrics.uptime_seconds == 0.0
    
    def test_audio_chunk_creation(self):
        """Test AudioChunk dataclass creation"""
        data = np.random.random(1024).astype(np.float32)
        timestamp = datetime.now()
        sample_rate = 16000
        
        chunk = AudioChunk(
            data=data,
            timestamp=timestamp,
            sample_rate=sample_rate
        )
        
        assert np.array_equal(chunk.data, data)
        assert chunk.timestamp == timestamp
        assert chunk.sample_rate == sample_rate
        assert chunk.is_speech == False
        assert chunk.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_voice_callbacks(self, voice_manager):
        """Test voice callback system"""
        callback_called = False
        callback_data = None
        
        def test_callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data
        
        # Add callback
        voice_manager.add_voice_callback('on_wake_word', test_callback)
        assert test_callback in voice_manager.voice_callbacks['on_wake_word']
        
        # Test callback notification
        test_data = {'test': 'data'}
        await voice_manager._notify_callbacks('on_wake_word', test_data)
        
        assert callback_called
        assert callback_data == test_data
        
        # Remove callback
        voice_manager.remove_voice_callback('on_wake_word', test_callback)
        assert test_callback not in voice_manager.voice_callbacks['on_wake_word']
    
    @pytest.mark.asyncio
    async def test_state_management(self, voice_manager):
        """Test voice interface state management"""
        # Test initial state
        assert voice_manager.state == VoiceState.STOPPED
        
        # Test state change
        await voice_manager._set_state(VoiceState.LISTENING)
        assert voice_manager.state == VoiceState.LISTENING
        
        # Test state change with callback
        state_changed = False
        old_state = None
        new_state = None
        
        def state_change_callback(data):
            nonlocal state_changed, old_state, new_state
            state_changed = True
            old_state = data['old_state']
            new_state = data['new_state']
        
        voice_manager.add_voice_callback('on_state_change', state_change_callback)
        await voice_manager._set_state(VoiceState.PROCESSING)
        
        # Give callback time to execute
        await asyncio.sleep(0.1)
        
        assert state_changed
        assert old_state == "listening"
        assert new_state == "processing"
    
    @pytest.mark.asyncio
    @patch('src.sovereign.voice_interface.sd.query_devices')
    @patch('src.sovereign.voice_interface.sd.rec')
    @patch('src.sovereign.voice_interface.sd.play')
    @patch('src.sovereign.voice_interface.sd.wait')
    async def test_audio_device_initialization(self, mock_wait, mock_play, mock_rec, mock_query_devices, voice_manager):
        """Test audio device initialization"""
        # Mock device list
        mock_devices = [
            {'name': 'Default Input', 'max_input_channels': 2, 'max_output_channels': 0, 'default_samplerate': 44100},
            {'name': 'Default Output', 'max_input_channels': 0, 'max_output_channels': 2, 'default_samplerate': 44100},
        ]
        mock_query_devices.return_value = mock_devices
        
        # Mock successful audio tests
        mock_rec.return_value = np.zeros((16000, 1), dtype=np.float32)
        mock_wait.return_value = None
        mock_play.return_value = None
        
        # Test initialization
        success = await voice_manager._initialize_audio_devices()
        assert success
        
        # Verify calls
        mock_query_devices.assert_called_once()
        mock_rec.assert_called_once()
        mock_play.assert_called_once()
        assert mock_wait.call_count == 2  # Called for both rec and play
    
    @pytest.mark.asyncio
    @patch('src.sovereign.voice_interface.whisper.load_model')
    async def test_whisper_initialization(self, mock_load_model, voice_manager):
        """Test Whisper model initialization"""
        # Mock Whisper model
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        # Test initialization
        success = await voice_manager._initialize_whisper()
        assert success
        assert voice_manager.whisper_model == mock_model
        
        # Verify model loading
        mock_load_model.assert_called_once_with(voice_manager.config.voice.whisper_model_size)
    
    @pytest.mark.asyncio
    @patch('src.sovereign.voice_interface.pyttsx3.init')
    async def test_tts_initialization(self, mock_tts_init, voice_manager):
        """Test text-to-speech initialization"""
        # Mock TTS engine
        mock_engine = Mock()
        mock_voice = Mock()
        mock_voice.name = "Test Voice"
        mock_voice.id = "test_voice_id"
        mock_engine.getProperty.return_value = [mock_voice]
        mock_tts_init.return_value = mock_engine
        
        # Test initialization
        success = await voice_manager._initialize_tts()
        assert success
        assert voice_manager.tts_engine == mock_engine
        
        # Verify configuration
        mock_engine.setProperty.assert_called()
        mock_tts_init.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.sovereign.voice_interface.webrtcvad.Vad')
    async def test_vad_initialization(self, mock_vad_class, voice_manager):
        """Test Voice Activity Detection initialization"""
        # Mock VAD
        mock_vad = Mock()
        mock_vad_class.return_value = mock_vad
        
        # Test initialization
        success = await voice_manager._initialize_vad()
        assert success
        assert voice_manager.vad == mock_vad
        
        # Verify VAD creation
        mock_vad_class.assert_called_once_with(voice_manager.config.voice.vad_aggressiveness)
    
    @pytest.mark.asyncio
    @patch('src.sovereign.voice_interface.pvporcupine.create')
    async def test_wake_word_initialization(self, mock_porcupine_create, voice_manager):
        """Test wake word detection initialization"""
        # Enable wake word for this test
        voice_manager.config.voice.wake_word_enabled = True
        voice_manager.config.voice.porcupine_access_key = "test_key"
        
        # Mock Porcupine
        mock_porcupine = Mock()
        mock_porcupine_create.return_value = mock_porcupine
        
        # Test initialization
        success = await voice_manager._initialize_wake_word()
        assert success
        assert voice_manager.porcupine == mock_porcupine
        
        # Verify Porcupine creation
        mock_porcupine_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_voice_activity_detection(self, voice_manager):
        """Test voice activity detection"""
        # Mock VAD
        mock_vad = Mock()
        mock_vad.is_speech.return_value = True
        voice_manager.vad = mock_vad
        
        # Create test audio chunk
        audio_data = np.random.random(800).astype(np.float32) * 0.1  # High energy audio
        chunk = AudioChunk(
            data=audio_data,
            timestamp=datetime.now(),
            sample_rate=16000
        )
        
        # Test VAD
        is_speech = await voice_manager._detect_voice_activity(chunk)
        assert is_speech
        
        # Verify VAD was called
        mock_vad.is_speech.assert_called()
    
    @pytest.mark.asyncio
    async def test_noise_reduction(self, voice_manager):
        """Test noise reduction functionality"""
        # Create noisy audio data
        audio_data = np.random.random(1000).astype(np.float32)
        
        # Test noise reduction
        with patch('src.sovereign.voice_interface.nr.reduce_noise') as mock_reduce_noise:
            mock_reduce_noise.return_value = audio_data * 0.5  # Simulated noise reduction
            
            reduced_audio = await voice_manager._apply_noise_reduction(audio_data)
            
            # Verify noise reduction was applied
            mock_reduce_noise.assert_called_once()
            assert not np.array_equal(reduced_audio, audio_data)  # Should be different
    
    @pytest.mark.asyncio
    async def test_wake_word_detection(self, voice_manager):
        """Test wake word detection"""
        # Mock Porcupine
        mock_porcupine = Mock()
        mock_porcupine.sample_rate = 16000
        mock_porcupine.process.return_value = 0  # Wake word detected
        voice_manager.porcupine = mock_porcupine
        
        # Create test audio
        audio_data = np.random.random(512).astype(np.float32)
        
        # Test wake word detection
        detected = await voice_manager._detect_wake_word(audio_data)
        assert detected
        
        # Verify Porcupine was called
        mock_porcupine.process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_speech_transcription(self, voice_manager):
        """Test speech transcription with Whisper"""
        # Mock Whisper model
        mock_model = Mock()
        mock_model.transcribe.return_value = {'text': 'Hello, world!'}
        voice_manager.whisper_model = mock_model
        
        # Create test audio
        audio_data = np.random.random(16000).astype(np.float32)
        
        # Test transcription
        transcription = await voice_manager._transcribe_audio(audio_data)
        assert transcription == 'Hello, world!'
        
        # Verify Whisper was called
        mock_model.transcribe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_speech_synthesis(self, voice_manager):
        """Test text-to-speech synthesis"""
        # Mock TTS engine
        mock_engine = Mock()
        voice_manager.tts_engine = mock_engine
        
        # Test speech synthesis
        success = await voice_manager.synthesize_speech("Hello, world!")
        assert success
        
        # Verify TTS was called
        mock_engine.say.assert_called_once_with("Hello, world!")
        mock_engine.runAndWait.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_stop_listening(self, voice_manager):
        """Test start and stop listening functionality"""
        # Mark as initialized
        voice_manager.is_initialized = True
        
        # Test start listening
        success = await voice_manager.start_listening()
        assert success
        assert voice_manager.is_listening
        assert voice_manager.state == VoiceState.LISTENING
        
        # Test stop listening
        success = await voice_manager.stop_listening()
        assert success
        assert not voice_manager.is_listening
        assert voice_manager.state == VoiceState.STOPPED
    
    @pytest.mark.asyncio
    async def test_wake_word_toggle(self, voice_manager):
        """Test wake word enable/disable functionality"""
        # Test enable
        success = await voice_manager.toggle_wake_word(True)
        assert success
        assert voice_manager.wake_word_active
        
        # Test disable
        success = await voice_manager.toggle_wake_word(False)
        assert success
        assert not voice_manager.wake_word_active
    
    @pytest.mark.asyncio
    async def test_status_reporting(self, voice_manager):
        """Test status reporting functionality"""
        # Mark as initialized for testing
        voice_manager.is_initialized = True
        voice_manager.whisper_model = Mock()
        voice_manager.tts_engine = Mock()
        voice_manager.vad = Mock()
        voice_manager.porcupine = Mock()
        
        # Get status
        status = await voice_manager.get_status()
        
        # Verify status structure
        assert 'state' in status
        assert 'is_initialized' in status
        assert 'is_listening' in status
        assert 'is_speaking' in status
        assert 'wake_word_active' in status
        assert 'sample_rate' in status
        assert 'chunk_size' in status
        assert 'audio_queue_size' in status
        assert 'models_loaded' in status
        assert 'metrics' in status
        
        # Verify model status
        models = status['models_loaded']
        assert models['whisper'] == True
        assert models['tts'] == True
        assert models['vad'] == True
        assert models['porcupine'] == True
    
    @pytest.mark.asyncio
    async def test_performance_stats(self, voice_manager):
        """Test performance statistics"""
        # Update some metrics
        voice_manager.metrics.successful_transcriptions = 5
        voice_manager.metrics.speech_synthesis_count = 3
        voice_manager.metrics.wake_word_detections = 2
        
        # Get performance stats
        stats = await voice_manager.get_performance_stats()
        
        # Verify stats
        assert stats['successful_transcriptions'] == 5
        assert stats['speech_synthesis_count'] == 3
        assert stats['wake_word_detections'] == 2
    
    @pytest.mark.asyncio
    async def test_audio_chunk_processing(self, voice_manager):
        """Test audio chunk processing pipeline"""
        # Setup mocks
        voice_manager.config.voice.noise_reduction_enabled = False
        voice_manager.vad = Mock()
        voice_manager.vad.is_speech.return_value = True
        voice_manager.porcupine = None  # Disable wake word for this test
        
        # Create test chunk
        audio_data = np.random.random(800).astype(np.float32)
        chunk = AudioChunk(
            data=audio_data,
            timestamp=datetime.now(),
            sample_rate=16000,
            is_speech=False
        )
        
        # Process chunk
        await voice_manager._process_audio_chunk(chunk)
        
        # Verify chunk was processed
        assert voice_manager.metrics.total_audio_processed_seconds > 0
    
    @pytest.mark.asyncio
    async def test_speech_collection_and_timeout(self, voice_manager):
        """Test speech audio collection and timeout handling"""
        # Create speech chunks
        chunks = []
        for i in range(3):
            chunk = AudioChunk(
                data=np.random.random(800).astype(np.float32),
                timestamp=datetime.now() - timedelta(seconds=i),
                sample_rate=16000,
                is_speech=True
            )
            chunks.append(chunk)
        
        # Collect speech
        for chunk in chunks:
            await voice_manager._collect_speech_audio(chunk)
        
        # Verify collection
        assert len(voice_manager.speech_frames) == 3
        
        # Test timeout (simulate old speech)
        old_chunk = AudioChunk(
            data=np.random.random(800).astype(np.float32),
            timestamp=datetime.now() - timedelta(seconds=10),
            sample_rate=16000,
            is_speech=True
        )
        voice_manager.speech_frames = [old_chunk]
        
        # Mock transcription for timeout test
        with patch.object(voice_manager, '_transcribe_audio', return_value="test"):
            await voice_manager._check_speech_timeout()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, voice_manager):
        """Test error handling in various scenarios"""
        # Test transcription error
        voice_manager.whisper_model = None
        result = await voice_manager._transcribe_audio(np.random.random(1000))
        assert result is None
        
        # Test synthesis error
        voice_manager.tts_engine = None
        success = await voice_manager.synthesize_speech("test")
        assert not success
        
        # Test VAD error with invalid audio
        voice_manager.vad = Mock()
        voice_manager.vad.is_speech.side_effect = Exception("Test error")
        
        chunk = AudioChunk(
            data=np.array([]),  # Empty audio
            timestamp=datetime.now(),
            sample_rate=16000
        )
        
        is_speech = await voice_manager._detect_voice_activity(chunk)
        assert not is_speech  # Should return False on error
    
    @pytest.mark.asyncio
    async def test_cleanup(self, voice_manager):
        """Test resource cleanup"""
        # Setup some resources
        voice_manager.is_initialized = True
        voice_manager.porcupine = Mock()
        voice_manager.tts_engine = Mock()
        voice_manager.executor = Mock()
        voice_manager.executor.shutdown = Mock()
        
        # Mock threads
        voice_manager.audio_thread = Mock()
        voice_manager.audio_thread.is_alive.return_value = True
        voice_manager.audio_thread.join = Mock()
        
        voice_manager.processing_thread = Mock()
        voice_manager.processing_thread.is_alive.return_value = True
        voice_manager.processing_thread.join = Mock()
        
        # Test cleanup
        await voice_manager.close()
        
        # Verify cleanup
        assert not voice_manager.is_initialized
        assert voice_manager.state == VoiceState.STOPPED
        voice_manager.porcupine.delete.assert_called_once()
        voice_manager.executor.shutdown.assert_called_once()


class TestVoiceInterfaceUtilities:
    """Test utility functions for voice interface"""
    
    @pytest.mark.asyncio
    @patch('src.sovereign.voice_interface.VoiceInterfaceManager')
    async def test_voice_interface_testing(self, mock_voice_manager_class):
        """Test voice interface testing function"""
        # Mock VoiceInterfaceManager
        mock_manager = AsyncMock()
        mock_manager.initialize.return_value = True
        mock_manager.synthesize_speech.return_value = True
        mock_manager.start_listening.return_value = True
        mock_manager.stop_listening.return_value = True
        mock_manager.get_status.return_value = {'state': 'listening'}
        mock_manager.close.return_value = None
        mock_voice_manager_class.return_value = mock_manager
        
        # Create test config
        config = Config()
        
        # Test voice interface
        success = await test_voice_interface(config)
        assert success
        
        # Verify calls
        mock_manager.initialize.assert_called_once()
        mock_manager.synthesize_speech.assert_called_once()
        mock_manager.start_listening.assert_called_once()
        mock_manager.stop_listening.assert_called_once()
        mock_manager.close.assert_called_once()
    
    @patch('src.sovereign.voice_interface.sd.query_devices')
    def test_get_audio_devices(self, mock_query_devices):
        """Test get_audio_devices utility function"""
        # Mock device list
        mock_devices = [
            {
                'name': 'Microphone',
                'max_input_channels': 2,
                'max_output_channels': 0,
                'default_samplerate': 44100
            },
            {
                'name': 'Speakers',
                'max_input_channels': 0,
                'max_output_channels': 2,
                'default_samplerate': 44100
            },
            {
                'name': 'USB Headset',
                'max_input_channels': 1,
                'max_output_channels': 2,
                'default_samplerate': 48000
            }
        ]
        mock_query_devices.return_value = mock_devices
        
        # Get devices
        devices = get_audio_devices()
        
        # Verify structure
        assert 'input_devices' in devices
        assert 'output_devices' in devices
        
        # Verify filtering
        input_devices = devices['input_devices']
        output_devices = devices['output_devices']
        
        # Should have Microphone and USB Headset as input
        assert len(input_devices) == 2
        # Should have Speakers and USB Headset as output
        assert len(output_devices) == 2
        
        # Verify device info structure
        for device in input_devices:
            assert 'index' in device
            assert 'name' in device
            assert 'max_input_channels' in device
            assert 'max_output_channels' in device
            assert 'default_samplerate' in device
    
    @patch('src.sovereign.voice_interface.sd.query_devices')
    def test_get_audio_devices_error_handling(self, mock_query_devices):
        """Test get_audio_devices error handling"""
        # Mock error
        mock_query_devices.side_effect = Exception("Audio system error")
        
        # Get devices (should handle error gracefully)
        devices = get_audio_devices()
        
        # Should return empty lists
        assert devices == {'input_devices': [], 'output_devices': []}


class TestVoiceInterfaceIntegration:
    """Integration tests for voice interface"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_basic_voice_workflow(self):
        """Test basic voice interface workflow (requires actual audio hardware)"""
        # This test requires actual audio hardware and is marked as integration
        config = Config()
        config.voice.wake_word_enabled = False  # Disable wake word for testing
        config.voice.whisper_model_size = "tiny"  # Use smallest model
        
        voice_manager = VoiceInterfaceManager(config)
        
        try:
            # Initialize (may fail on systems without audio)
            success = await voice_manager.initialize()
            if not success:
                pytest.skip("Audio hardware not available")
            
            # Test basic functionality
            await voice_manager.start_listening()
            status = await voice_manager.get_status()
            assert status['state'] == 'listening'
            
            # Test TTS
            success = await voice_manager.synthesize_speech("Testing voice interface")
            # Note: TTS success depends on system audio configuration
            
            await voice_manager.stop_listening()
            
        finally:
            await voice_manager.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 