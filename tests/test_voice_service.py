"""
Tests for VoiceService

Tests are organized into two categories:
- not_heavy: Tests service creation and basic functionality without triggering heavy imports
- heavy: Integration tests that load voice processing components
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from src.sovereign.services.voice_service import VoiceService, VoiceResult, VoiceState


class TestVoiceServiceConstruction:
    """Test service creation and lightweight operations"""
    
    @pytest.mark.not_heavy
    def test_service_creation(self):
        """Test that VoiceService can be created without heavy imports"""
        service = VoiceService()
        
        # Basic service properties
        assert service is not None
        assert not service.is_voice_system_loaded()
        assert service.get_voice_state() == "stopped"
        assert not service.is_listening()
        assert not service.is_speaking()
        
        # Statistics are initialized
        stats = service.get_stats()
        assert stats['total_listen_calls'] == 0
        assert stats['total_speak_calls'] == 0
        assert stats['successful_recognitions'] == 0
        assert stats['successful_syntheses'] == 0
        assert stats['failed_operations'] == 0
        assert stats['initialization_time'] == 0.0
    
    @pytest.mark.not_heavy
    def test_health_check_uninitialized(self):
        """Test health check before system is loaded"""
        service = VoiceService()
        
        health = service.get_health_check()
        
        assert health['service_name'] == 'VoiceService'
        assert health['status'] == 'uninitialized'
        assert not health['voice_system_loaded']
        assert health['total_operations'] == 0
        assert health['voice_state'] == 'stopped'
        assert not health['is_listening']
        assert not health['is_speaking']
        assert not health['components']['voice_manager']
        assert not health['components']['speech_recognition']
        assert not health['components']['text_to_speech']
        assert not health['components']['wake_word_detection']
    
    @pytest.mark.not_heavy
    def test_voice_result_creation(self):
        """Test VoiceResult creation and properties"""
        # Success case
        result = VoiceResult(
            success=True,
            operation="listen",
            text_content="Hello world",
            confidence=0.95,
            processing_time=1.5,
            audio_duration=3.0
        )
        
        assert result.success
        assert result.operation == "listen"
        assert result.text_content == "Hello world"
        assert result.confidence == 0.95
        assert result.processing_time == 1.5
        assert result.audio_duration == 3.0
        assert result.error is None
    
    @pytest.mark.not_heavy
    def test_voice_result_error_case(self):
        """Test VoiceResult error case"""
        result = VoiceResult(
            success=False,
            operation="speak",
            error="Audio device not found",
            processing_time=0.1
        )
        
        assert not result.success
        assert result.operation == "speak"
        assert result.error == "Audio device not found"
        assert result.text_content == ""
        assert result.confidence == 0.0
        assert result.processing_time == 0.1
        assert result.audio_duration == 0.0
    
    @pytest.mark.not_heavy
    def test_voice_state_enum(self):
        """Test VoiceState enum values"""
        assert VoiceState.STOPPED.value == "stopped"
        assert VoiceState.LISTENING.value == "listening"
        assert VoiceState.SPEAKING.value == "speaking"
        assert VoiceState.PROCESSING.value == "processing"
        assert VoiceState.ERROR.value == "error"
    
    @pytest.mark.not_heavy
    def test_voice_callbacks(self):
        """Test voice callback management"""
        service = VoiceService()
        
        # Test callback addition
        def test_callback(old_state, new_state):
            pass
        
        service.add_voice_callback('on_state_change', test_callback)
        assert test_callback in service._callbacks['on_state_change']
        
        # Test callback removal
        service.remove_voice_callback('on_state_change', test_callback)
        assert test_callback not in service._callbacks['on_state_change']


class TestVoiceServiceIntegration:
    """Test service integration with heavy voice processing components"""
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_lazy_loading_on_first_listen(self):
        """Test that voice processing components load only on first listen"""
        service = VoiceService()
        
        # Mock voice interface manager and its dependencies
        with patch('src.sovereign.services.voice_service.VoiceInterfaceManager') as mock_manager_class, \
             patch('src.sovereign.services.voice_service.Config') as mock_config_class:
            
            # Setup mocks
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            
            mock_manager = Mock()
            mock_manager.initialize = AsyncMock(return_value=True)
            mock_manager.start_listening = AsyncMock(return_value=True)
            mock_manager.stop_listening = AsyncMock(return_value=True)
            mock_manager_class.return_value = mock_manager
            
            # Verify service is uninitialized
            assert not service.is_voice_system_loaded()
            
            # First listen should trigger loading
            result = await service.listen(timeout=0.1)
            
            # Verify components were loaded
            assert service.is_voice_system_loaded()
            
            # Verify listen result
            assert result.success
            assert result.operation == "listen"
            assert result.text_content == "Hello, this is a placeholder transcription"
            assert result.confidence == 0.85
            
            # Verify manager was called correctly
            mock_manager.initialize.assert_called_once()
            mock_manager.start_listening.assert_called_once()
            mock_manager.stop_listening.assert_called_once()
            
            # Check statistics were updated
            stats = service.get_stats()
            assert stats['total_listen_calls'] == 1
            assert stats['successful_recognitions'] == 1
            assert stats['failed_operations'] == 0
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_lazy_loading_on_first_speak(self):
        """Test that voice processing components load only on first speak"""
        service = VoiceService()
        
        with patch('src.sovereign.services.voice_service.VoiceInterfaceManager') as mock_manager_class, \
             patch('src.sovereign.services.voice_service.Config') as mock_config_class:
            
            # Setup mocks
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            
            mock_manager = Mock()
            mock_manager.initialize = AsyncMock(return_value=True)
            mock_manager.synthesize_speech = AsyncMock(return_value=True)
            mock_manager_class.return_value = mock_manager
            
            # Verify service is uninitialized
            assert not service.is_voice_system_loaded()
            
            # First speak should trigger loading
            result = await service.speak("Hello world")
            
            # Verify components were loaded
            assert service.is_voice_system_loaded()
            
            # Verify speak result
            assert result.success
            assert result.operation == "speak"
            assert result.text_content == "Hello world"
            
            # Verify manager was called correctly
            mock_manager.initialize.assert_called_once()
            mock_manager.synthesize_speech.assert_called_once_with("Hello world")
            
            # Check statistics were updated
            stats = service.get_stats()
            assert stats['total_speak_calls'] == 1
            assert stats['successful_syntheses'] == 1
            assert stats['failed_operations'] == 0
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_voice_manager_initialization_failure(self):
        """Test handling of voice manager initialization failure"""
        service = VoiceService()
        
        with patch('src.sovereign.services.voice_service.VoiceInterfaceManager') as mock_manager_class:
            mock_manager_class.side_effect = Exception("Failed to initialize voice interface")
            
            # Attempt to listen should fail gracefully
            result = await service.listen()
            
            assert not result.success
            assert "Voice processing initialization failed" in result.error
            assert result.operation == "listen"
            assert result.processing_time > 0
            
            # Service should not be marked as initialized
            assert not service.is_voice_system_loaded()
            
            # Statistics should reflect the failed attempt
            stats = service.get_stats()
            assert stats['total_listen_calls'] == 1
            assert stats['successful_recognitions'] == 0
            assert stats['failed_operations'] == 1
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_concurrent_loading_protection(self):
        """Test that concurrent access doesn't cause double loading"""
        service = VoiceService()
        
        # Track initialization calls
        init_call_count = 0
        
        def count_init(*args, **kwargs):
            nonlocal init_call_count
            init_call_count += 1
            mock = Mock()
            mock.initialize = AsyncMock(return_value=True)
            mock.start_listening = AsyncMock(return_value=True)
            mock.stop_listening = AsyncMock(return_value=True)
            mock.synthesize_speech = AsyncMock(return_value=True)
            return mock
        
        with patch('src.sovereign.services.voice_service.VoiceInterfaceManager', side_effect=count_init), \
             patch('src.sovereign.services.voice_service.Config'):
            
            # Start multiple concurrent operations
            tasks = [
                service.listen(timeout=0.1),
                service.speak("test"),
                service.listen(timeout=0.1)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert all(result.success for result in results)
            
            # Voice manager should only be initialized once despite concurrent access
            assert init_call_count == 1
            assert service.is_voice_system_loaded()
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_listen_operation_failure(self):
        """Test handling of listen operation failure"""
        service = VoiceService()
        
        with patch('src.sovereign.services.voice_service.VoiceInterfaceManager') as mock_manager_class, \
             patch('src.sovereign.services.voice_service.Config'):
            
            mock_manager = Mock()
            mock_manager.initialize = AsyncMock(return_value=True)
            mock_manager.start_listening = AsyncMock(return_value=False)  # Fail to start
            mock_manager_class.return_value = mock_manager
            
            # Listen should fail
            result = await service.listen()
            
            assert not result.success
            assert "Failed to start listening" in result.error
            assert result.operation == "listen"
            
            # Should be marked as failed operation
            stats = service.get_stats()
            assert stats['total_listen_calls'] == 1
            assert stats['successful_recognitions'] == 0
            assert stats['failed_operations'] == 1
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_speak_operation_failure(self):
        """Test handling of speak operation failure"""
        service = VoiceService()
        
        with patch('src.sovereign.services.voice_service.VoiceInterfaceManager') as mock_manager_class, \
             patch('src.sovereign.services.voice_service.Config'):
            
            mock_manager = Mock()
            mock_manager.initialize = AsyncMock(return_value=True)
            mock_manager.synthesize_speech = AsyncMock(return_value=False)  # Fail synthesis
            mock_manager_class.return_value = mock_manager
            
            # Speak should fail
            result = await service.speak("test")
            
            assert not result.success
            assert "Speech synthesis failed" in result.error
            assert result.operation == "speak"
            assert result.text_content == "test"
            
            # Should be marked as failed operation
            stats = service.get_stats()
            assert stats['total_speak_calls'] == 1
            assert stats['successful_syntheses'] == 0
            assert stats['failed_operations'] == 1
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_wake_word_functionality(self):
        """Test wake word enable/disable functionality"""
        service = VoiceService()
        
        with patch('src.sovereign.services.voice_service.VoiceInterfaceManager') as mock_manager_class, \
             patch('src.sovereign.services.voice_service.Config'):
            
            mock_manager = Mock()
            mock_manager.initialize = AsyncMock(return_value=True)
            mock_manager.toggle_wake_word = AsyncMock(return_value=True)
            mock_manager_class.return_value = mock_manager
            
            # Enable wake word should work
            success = await service.enable_wake_word(True)
            assert success
            mock_manager.toggle_wake_word.assert_called_with(True)
            
            # Disable wake word should work
            success = await service.enable_wake_word(False)
            assert success
            mock_manager.toggle_wake_word.assert_called_with(False)
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_service_cleanup(self):
        """Test proper cleanup of service resources"""
        service = VoiceService()
        
        with patch('src.sovereign.services.voice_service.VoiceInterfaceManager') as mock_manager_class, \
             patch('src.sovereign.services.voice_service.Config'):
            
            mock_manager = Mock()
            mock_manager.initialize = AsyncMock(return_value=True)
            mock_manager.synthesize_speech = AsyncMock(return_value=True)
            mock_manager.close = AsyncMock()
            mock_manager_class.return_value = mock_manager
            
            # Initialize by speaking
            await service.speak("test")
            
            # Verify service is initialized
            assert service.is_voice_system_loaded()
            
            # Cleanup the service
            await service.close()
            
            # Verify cleanup was called on manager
            mock_manager.close.assert_called_once()
            
            # Verify service state is reset
            assert not service.is_voice_system_loaded()
            assert service.get_voice_state() == "stopped"
            assert not service.is_listening()
            assert not service.is_speaking()
    
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_voice_state_transitions(self):
        """Test proper voice state transitions during operations"""
        service = VoiceService()
        
        # Track state changes
        state_changes = []
        
        def track_state_change(old_state, new_state):
            state_changes.append((old_state.value, new_state.value))
        
        service.add_voice_callback('on_state_change', track_state_change)
        
        with patch('src.sovereign.services.voice_service.VoiceInterfaceManager') as mock_manager_class, \
             patch('src.sovereign.services.voice_service.Config'):
            
            mock_manager = Mock()
            mock_manager.initialize = AsyncMock(return_value=True)
            mock_manager.start_listening = AsyncMock(return_value=True)
            mock_manager.stop_listening = AsyncMock(return_value=True)
            mock_manager.synthesize_speech = AsyncMock(return_value=True)
            mock_manager_class.return_value = mock_manager
            
            # Test listen state transitions
            await service.listen(timeout=0.1)
            
            # Should have transitions: stopped -> listening -> stopped
            listen_transitions = [change for change in state_changes if 'listening' in change]
            assert len(listen_transitions) >= 1
            
            # Test speak state transitions
            await service.speak("test")
            
            # Should have transitions including speaking
            speak_transitions = [change for change in state_changes if 'speaking' in change]
            assert len(speak_transitions) >= 1 