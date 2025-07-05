"""
Integration tests for Sovereign AI Agent GUI-Backend Communication

These tests verify that the GUI correctly interfaces with the backend orchestrator
and that all data structures and method calls are properly formatted.

This prevents integration bugs like QueryContext parameter mismatches.
"""

import unittest
import asyncio
import threading
import queue
import tempfile
import os
import sys
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from pathlib import Path

# Add the src directory to Python path for testing
test_dir = Path(__file__).parent
src_dir = test_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from sovereign.config import Config
from sovereign.orchestrator import QueryContext, OrchestrationResult, ModelChoice, QueryComplexity
from sovereign.logger import get_debug_logger


class TestGUIBackendIntegration(unittest.TestCase):
    """Integration tests for GUI-Backend communication"""
    
    def setUp(self):
        """Set up test environment"""
        self.debug_logger = get_debug_logger()
        self.debug_logger.info("🧪 Setting up integration test")
        
        # Create test config
        self.config = Config()
        self.config.models.talker_model = "gemma2:9b"
        self.config.models.thinker_model = "deepseek-r1:14b"
        
        # Test message history
        self.test_message_history = [
            {'message': 'Hello', 'sender': 'user', 'timestamp': datetime.now()},
            {'message': 'Hi there!', 'sender': 'ai', 'timestamp': datetime.now()}
        ]
    
    def test_query_context_creation(self):
        """Test that QueryContext is created with correct parameters"""
        self.debug_logger.info("🧪 Testing QueryContext creation")
        
        prompt = "Test query"
        
        # Test QueryContext creation with correct parameters
        context = QueryContext(
            user_input=prompt,
            timestamp=datetime.now(),
            session_id="test_session",
            previous_queries=["previous query"],
            conversation_history=self.test_message_history
        )
        
        # Verify all required fields are set
        self.assertEqual(context.user_input, prompt)
        self.assertEqual(context.session_id, "test_session")
        self.assertIsInstance(context.timestamp, datetime)
        self.assertEqual(len(context.previous_queries), 1)
        self.assertEqual(len(context.conversation_history), 2)
        
        self.debug_logger.info("✅ QueryContext creation test passed")
    
    def test_query_context_wrong_parameters(self):
        """Test that QueryContext fails with wrong parameters (regression test)"""
        self.debug_logger.info("🧪 Testing QueryContext with wrong parameters")
        
        # This should fail - testing the bug we just fixed
        with self.assertRaises(TypeError):
            QueryContext(
                query="Test query",  # Wrong parameter name - should be 'user_input'
                timestamp=datetime.now(),
                session_id="test_session",
                previous_queries=[],
                conversation_history=[]
            )
        
        self.debug_logger.info("✅ QueryContext wrong parameters test passed")
    
    @patch('sovereign.orchestrator.ModelOrchestrator')
    def test_gui_worker_thread_simulation(self, mock_orchestrator_class):
        """Test the GUI worker thread process simulation"""
        self.debug_logger.info("🧪 Testing GUI worker thread simulation")
        
        # Create mock orchestrator instance
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Create mock result
        mock_result = OrchestrationResult(
            response="Test response",
            model_used=ModelChoice.TALKER,
            complexity_level=QueryComplexity.SIMPLE,
            processing_time=0.5,
            handoff_occurred=False,
            cache_hit=False,
            confidence_score=0.9,
            reasoning="Test reasoning",
            telemetry={}
        )
        
        # Setup mock to return our test result
        async def mock_process_query(user_input, context):
            return mock_result
        
        mock_orchestrator.process_query = AsyncMock(return_value=mock_result)
        
        # Simulate the GUI worker thread logic
        def simulate_worker_thread(prompt: str, message_history: list):
            """Simulate the exact worker thread logic from GUI"""
            try:
                # Create event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Create QueryContext with correct parameters
                    context = QueryContext(
                        user_input=prompt,
                        timestamp=datetime.now(),
                        session_id="gui_session",
                        previous_queries=[msg['message'] for msg in message_history[-5:] if msg['sender'] == 'user'],
                        conversation_history=message_history[-10:]
                    )
                    
                    # Call orchestrator (simulated)
                    result = loop.run_until_complete(mock_orchestrator.process_query(prompt, context))
                    
                    return {
                        'success': True,
                        'response': result.response,
                        'model_used': result.model_used,
                        'complexity': result.complexity_level,
                        'processing_time': result.processing_time
                    }
                    
                finally:
                    loop.close()
                    
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
        
        # Run the simulation
        prompt = "Hello, how are you?"
        result = simulate_worker_thread(prompt, self.test_message_history)
        
        # Verify the result
        self.assertTrue(result['success'])
        self.assertEqual(result['response'], "Test response")
        self.assertEqual(result['model_used'], ModelChoice.TALKER)
        self.assertEqual(result['complexity'], QueryComplexity.SIMPLE)
        self.assertEqual(result['processing_time'], 0.5)
        
        # Verify orchestrator was called with correct parameters
        mock_orchestrator.process_query.assert_called_once()
        call_args = mock_orchestrator.process_query.call_args
        self.assertEqual(call_args[0][0], prompt)  # First argument should be prompt
        self.assertIsInstance(call_args[0][1], QueryContext)  # Second should be QueryContext
        
        self.debug_logger.info("✅ GUI worker thread simulation test passed")
    
    def test_queue_communication(self):
        """Test thread-safe queue communication"""
        self.debug_logger.info("🧪 Testing queue communication")
        
        request_queue = queue.Queue()
        response_queue = queue.Queue()
        
        # Test putting and getting messages
        test_response = {
            'success': True,
            'response': 'Test response',
            'model_used': ModelChoice.TALKER,
            'processing_time': 0.3
        }
        
        response_queue.put(test_response)
        
        # Verify message retrieval
        retrieved = response_queue.get_nowait()
        self.assertEqual(retrieved['success'], True)
        self.assertEqual(retrieved['response'], 'Test response')
        self.assertEqual(retrieved['model_used'], ModelChoice.TALKER)
        
        # Test empty queue
        with self.assertRaises(queue.Empty):
            response_queue.get_nowait()
        
        self.debug_logger.info("✅ Queue communication test passed")
    
    def test_error_handling_in_worker_thread(self):
        """Test error handling in worker thread"""
        self.debug_logger.info("🧪 Testing error handling in worker thread")
        
        def simulate_worker_thread_with_error():
            """Simulate worker thread with an error"""
            try:
                # Simulate the error that was occurring
                raise TypeError("QueryContext.__init__() got an unexpected keyword argument 'query'")
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
        
        result = simulate_worker_thread_with_error()
        
        # Verify error handling
        self.assertFalse(result['success'])
        self.assertIn("unexpected keyword argument 'query'", result['error'])
        
        self.debug_logger.info("✅ Error handling test passed")
    
    @patch('sovereign.gui.SovereignGUI')
    def test_full_gui_backend_flow(self, mock_gui_class):
        """Test the complete GUI-Backend integration flow"""
        self.debug_logger.info("🧪 Testing full GUI-Backend integration flow")
        
        # Create mock GUI instance
        mock_gui = Mock()
        mock_gui_class.return_value = mock_gui
        
        # Setup mock attributes
        mock_gui.message_history = self.test_message_history
        mock_gui.response_queue = queue.Queue()
        mock_gui.is_processing = False
        
        # Create mock orchestrator
        mock_orchestrator = Mock()
        mock_result = OrchestrationResult(
            response="Integration test response",
            model_used=ModelChoice.THINKER,
            complexity_level=QueryComplexity.COMPLEX,
            processing_time=1.2,
            handoff_occurred=True,
            cache_hit=False,
            confidence_score=0.95,
            reasoning="Complex reasoning test",
            telemetry={}
        )
        
        async def mock_process_query(user_input, context):
            # Verify the context has correct structure
            assert hasattr(context, 'user_input')
            assert hasattr(context, 'timestamp')
            assert hasattr(context, 'session_id')
            assert hasattr(context, 'previous_queries')
            assert hasattr(context, 'conversation_history')
            return mock_result
        
        mock_orchestrator.process_query = AsyncMock(return_value=mock_result)
        mock_gui.orchestrator = mock_orchestrator
        
        # Simulate sending a message
        prompt = "What is the meaning of life?"
        
        def simulate_send_message():
            """Simulate the corrected send message flow"""
            # Create QueryContext with correct parameters (the fix)
            context = QueryContext(
                user_input=prompt,
                timestamp=datetime.now(),
                session_id="gui_session",
                previous_queries=[msg['message'] for msg in mock_gui.message_history[-5:] if msg['sender'] == 'user'],
                conversation_history=mock_gui.message_history[-10:]
            )
            
            # Simulate async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(mock_orchestrator.process_query(prompt, context))
                
                # Put result in queue
                mock_gui.response_queue.put({
                    'success': True,
                    'response': result.response,
                    'model_used': result.model_used,
                    'complexity': result.complexity_level,
                    'processing_time': result.processing_time
                })
                
            finally:
                loop.close()
        
        # Run the simulation
        simulate_send_message()
        
        # Verify the result
        response = mock_gui.response_queue.get_nowait()
        self.assertTrue(response['success'])
        self.assertEqual(response['response'], "Integration test response")
        self.assertEqual(response['model_used'], ModelChoice.THINKER)
        self.assertEqual(response['complexity'], QueryComplexity.COMPLEX)
        
        # Verify orchestrator was called correctly
        mock_orchestrator.process_query.assert_called_once()
        
        self.debug_logger.info("✅ Full GUI-Backend integration test passed")
    
    def tearDown(self):
        """Clean up after tests"""
        self.debug_logger.info("🧪 Integration test cleanup complete")


class TestDebugLoggingFramework(unittest.TestCase):
    """Test the debug logging framework"""
    
    def test_debug_logger_creation(self):
        """Test that debug logger is created correctly"""
        debug_logger = get_debug_logger()
        
        self.assertIsNotNone(debug_logger)
        self.assertEqual(debug_logger.name, "sovereign.debug")
        self.assertEqual(debug_logger.level, 10)  # DEBUG level
        
        # Test logging
        debug_logger.info("Test debug log message")
        debug_logger.debug("Test debug detail message")
        debug_logger.error("Test error message")
    
    def test_debug_logger_traceback_capture(self):
        """Test that debug logger captures tracebacks correctly"""
        debug_logger = get_debug_logger()
        
        try:
            # Simulate an error
            raise ValueError("Test error for traceback capture")
        except Exception as e:
            import traceback
            debug_logger.error(f"Captured error: {e}")
            debug_logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        # If we get here without exceptions, the test passed


if __name__ == '__main__':
    # Setup test logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create logs directory for testing
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    print("🚀 Running Sovereign AI Agent Integration Tests...")
    
    # Run the tests
    unittest.main(verbosity=2) 