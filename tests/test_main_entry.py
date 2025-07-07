"""
Tests for main.py - Application Entry Point

Tests the main entry point for the Sovereign AI Agent including
startup performance and basic functionality.
"""

import pytest
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch, call
from io import StringIO

# Add src to path (same as main.py does)
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import main functionality
import main
from src.sovereign.core.app import CoreApp


class TestMainEntry:
    """Test the main.py entry point"""
    
    @patch('main.CoreApp')
    def test_main_function(self, mock_core_app_class):
        """Test the main function"""
        # Setup mock
        mock_app = Mock()
        mock_core_app_class.return_value = mock_app
        
        # Capture stdout
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            main.main()
        
        # Verify CoreApp was created and run
        mock_core_app_class.assert_called_once()
        mock_app.run.assert_called_once()
        
        # Check output
        output = captured_output.getvalue()
        assert "🚀 Sovereign AI Agent v2.0 - Starting..." in output
        assert "✅ Startup completed in" in output
    
    @patch('main.CoreApp')
    def test_main_with_exception(self, mock_core_app_class):
        """Test main function when CoreApp raises exception"""
        # Setup mock to raise exception
        mock_app = Mock()
        mock_app.run.side_effect = Exception("Test error")
        mock_core_app_class.return_value = mock_app
        
        # Should propagate the exception
        with pytest.raises(Exception, match="Test error"):
            main.main()
    
    def test_main_startup_timing(self):
        """Test that startup timing is measured and reported"""
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            # Mock CoreApp to control timing
            with patch('main.CoreApp') as mock_core_app_class:
                mock_app = Mock()
                mock_core_app_class.return_value = mock_app
                
                # Add a small delay to test timing
                def slow_run():
                    time.sleep(0.01)  # 10ms delay
                
                mock_app.run.side_effect = slow_run
                
                main.main()
        
        output = captured_output.getvalue()
        
        # Check that timing is reported
        assert "✅ Startup completed in" in output
        
        # Extract the timing value (should be > 0.01 seconds)
        lines = output.split('\n')
        timing_line = [line for line in lines if "Startup completed in" in line][0]
        
        # Parse timing value
        import re
        match = re.search(r'(\d+\.\d+)s', timing_line)
        assert match is not None
        
        timing = float(match.group(1))
        assert timing >= 0.01  # Should be at least our artificial delay


class TestMainIntegration:
    """Integration tests for the complete main.py flow"""
    
    def test_real_startup(self):
        """Test actual startup with real CoreApp (not mocked)"""
        captured_output = StringIO()
        
        start_time = time.time()
        
        with patch('sys.stdout', captured_output):
            main.main()
        
        end_time = time.time()
        actual_startup_time = end_time - start_time
        
        # Verify output messages
        output = captured_output.getvalue()
        assert "🚀 Sovereign AI Agent v2.0 - Starting..." in output
        assert "✅ Startup completed in" in output
        
        # Most importantly: verify sub-second startup
        assert actual_startup_time < 1.0, f"Startup took {actual_startup_time:.3f}s, exceeds 1s requirement!"
        
        # Extract reported timing
        import re
        match = re.search(r'(\d+\.\d+)s', output)
        assert match is not None
        
        reported_timing = float(match.group(1))
        
        # Reported timing should be close to actual timing
        # Allow some tolerance for measurement differences
        assert abs(reported_timing - actual_startup_time) < 0.1
    
    def test_path_setup(self):
        """Test that sys.path is correctly modified"""
        # This test verifies the path setup logic in main.py
        original_path = sys.path.copy()
        
        # Import main (which modifies sys.path)
        import importlib
        importlib.reload(main)
        
        # Check that src path was added
        expected_src_path = str(Path(__file__).parent.parent / "src")
        assert expected_src_path in sys.path
        
        # Should be early in the path for priority
        src_index = sys.path.index(expected_src_path)
        assert src_index <= 1  # Should be at index 0 or 1
    
    def test_can_import_core_components(self):
        """Test that all core components can be imported after path setup"""
        # After main.py sets up the path, we should be able to import core components
        try:
            from sovereign.core.app import CoreApp
            from sovereign.core.service_manager import ServiceManager, ServiceStatus
            from sovereign.core import CoreApp as CoreAppFromInit
            
            # Verify they're the same classes
            assert CoreApp is CoreAppFromInit
            
        except ImportError as e:
            pytest.fail(f"Failed to import core components: {e}")


class TestPerformanceRequirement:
    """Critical performance tests"""
    
    @pytest.mark.not_heavy
    def test_cold_start_under_one_second(self):
        """
        CRITICAL TEST: Verify cold start is under 1 second
        
        This is the key requirement from the PRD.
        """
        # Measure multiple runs to ensure consistency
        times = []
        
        for i in range(3):  # Test 3 times
            start_time = time.time()
            
            # Suppress output for clean timing
            with patch('sys.stdout', StringIO()):
                main.main()
            
            end_time = time.time()
            startup_time = end_time - start_time
            times.append(startup_time)
            
            # Each individual run must be under 1 second
            assert startup_time < 1.0, f"Run {i+1}: Startup took {startup_time:.3f}s, exceeds 1s requirement!"
        
        # Average should also be well under 1 second
        avg_time = sum(times) / len(times)
        assert avg_time < 0.5, f"Average startup time {avg_time:.3f}s is too slow for consistent sub-second performance"
        
        print(f"✅ Cold start performance verified: {avg_time:.3f}s average ({min(times):.3f}s - {max(times):.3f}s)")
    
    def test_no_heavy_imports_on_startup(self):
        """Test that no heavy libraries are imported during startup"""
        # These modules should NOT be imported during startup
        heavy_modules = [
            'torch', 'tensorflow', 'transformers', 'sentence_transformers',
            'faiss', 'opencv-cv2', 'cv2', 'PIL', 'pillow',
            'whisper', 'sounddevice', 'pyaudio', 'pydub'
        ]
        
        # Clear any previously imported modules
        for module in heavy_modules:
            if module in sys.modules:
                del sys.modules[module]
        
        # Run main
        with patch('sys.stdout', StringIO()):
            main.main()
        
        # Check that heavy modules weren't imported
        for module in heavy_modules:
            assert module not in sys.modules, f"Heavy module '{module}' was imported during startup!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 