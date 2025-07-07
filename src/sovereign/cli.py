"""
Command Line Interface for Sovereign AI Agent
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import config
from .logger import setup_logger
from .hardware import check_system_requirements, hardware_detector
from .talker_model import TalkerModel
from .thinker_model import ThinkerModel, TaskType
from .orchestrator import ModelOrchestrator, QueryContext
from .external_model_connector import ExternalModelConnector, ExternalRoutingCriteria
from .consent_manager import ConsentManager, ConsentMethod, cli_consent_callback

# Optional voice interface imports
try:
    from .voice_interface import VoiceInterfaceManager, test_voice_interface, get_audio_devices
    VOICE_INTERFACE_AVAILABLE = True
except ImportError:
    VoiceInterfaceManager = None
    test_voice_interface = None
    get_audio_devices = None
    VOICE_INTERFACE_AVAILABLE = False

# Optional GUI imports
try:
    from .gui import run_gui
    GUI_AVAILABLE = True
except ImportError:
    run_gui = None
    GUI_AVAILABLE = False


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser"""
    
    parser = argparse.ArgumentParser(
        prog="sovereign",
        description="Sovereign AI Agent - A private, powerful, locally-running AI assistant",
        epilog="For more information, visit: https://github.com/sovereign-ai/sovereign"
    )
    
    # Version
    parser.add_argument(
        "--version",
        action="version",
        version="Sovereign AI Agent v1.0.0"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: logs/sovereign_YYYYMMDD.log)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    
    # Configuration options
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.json",
        help="Configuration file path (default: config/config.json)"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU usage and run on CPU only"
    )
    
    # System options
    parser.add_argument(
        "--check-requirements",
        action="store_true",
        help="Check system requirements and exit"
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run initial setup and configuration"
    )
    
    parser.add_argument(
        "--test-talker",
        action="store_true",
        help="Test the Talker model functionality"
    )
    
    parser.add_argument(
        "--test-thinker",
        action="store_true",
        help="Test the Thinker model functionality"
    )
    
    parser.add_argument(
        "--test-integration",
        action="store_true",
        help="Test Talker-Thinker model integration and handoff"
    )
    
    parser.add_argument(
        "--test-orchestrator",
        action="store_true",
        help="Test the orchestration system functionality"
    )
    
    parser.add_argument(
        "--test-external",
        action="store_true",
        help="Test external model routing functionality"
    )
    
    parser.add_argument(
        "--test-consent",
        action="store_true",
        help="Test user consent mechanism for external routing"
    )
    
    parser.add_argument(
        "--test-voice",
        action="store_true",
        help="Test the voice interface functionality"
    )
    
    parser.add_argument(
        "--diagnose-gpu",
        action="store_true",
        help="Run comprehensive GPU/CUDA diagnostic and exit"
    )
    
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List available audio input/output devices"
    )
    
    parser.add_argument(
        "--voice-profile",
        action="store_true",
        help="Configure voice interface settings interactively"
    )
    
    # Interface options
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the graphical user interface"
    )
    
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable voice interface"
    )
    
    parser.add_argument(
        "--no-screen-capture",
        action="store_true",
        help="Disable screen capture functionality"
    )
    
    return parser


async def test_talker_model():
    """Test the Talker model functionality"""
    logger = logging.getLogger("sovereign")
    
    logger.info("🧪 Testing Talker Model...")
    
    # Initialize TalkerModel
    talker = TalkerModel()
    
    try:
        # Test initialization
        logger.info("🔧 Initializing Talker model...")
        if not await talker.initialize():
            logger.error("❌ Failed to initialize Talker model")
            return 1
        
        logger.info("✅ Talker model initialized successfully")
        
        # Test basic generation
        test_prompts = [
            "Hello, how are you?",
            "What's the weather like today?",
            "Tell me a short joke",
            "Write a complex Python function to sort a list",  # Should be detected as complex
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"🧪 Test {i}: {prompt}")
            
            # Check complexity detection
            is_complex = talker.detect_complex_query(prompt)
            logger.info(f"🔍 Complexity detected: {is_complex}")
            
            # Generate response
            try:
                response = await talker.generate_response(prompt)
                logger.info(f"🤖 Response: {response}")
                
                if is_complex:
                    logger.info("ℹ️  This query would normally be handed off to the Thinker model")
                
            except Exception as e:
                logger.error(f"❌ Error generating response: {e}")
        
        # Display performance stats
        stats = talker.get_performance_stats()
        logger.info("📊 Performance Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  - {key}: {value:.3f}")
            else:
                logger.info(f"  - {key}: {value}")
        
        logger.info("✅ Talker model test completed")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return 1
    finally:
        await talker.close()


async def test_thinker_model():
    """Test the Thinker model functionality"""
    logger = logging.getLogger("sovereign")
    
    logger.info("🧠 Testing Thinker Model...")
    
    # Initialize ThinkerModel
    thinker = ThinkerModel()
    
    try:
        # Test initialization
        logger.info("🔧 Initializing Thinker model...")
        if not await thinker.initialize():
            logger.error("❌ Failed to initialize Thinker model")
            logger.info("💡 Make sure Ollama is running and deepseek-r1:14b model is available")
            logger.info("   Run: ollama run deepseek-r1:14b")
            return 1
        
        logger.info("✅ Thinker model initialized successfully")
        
        # Test different task types
        test_cases = [
            {
                "task_type": "deep_reasoning",
                "prompt": "Analyze the philosophical implications of artificial intelligence becoming sentient",
                "method": thinker.deep_reasoning
            },
            {
                "task_type": "code_generation", 
                "prompt": "Write a Python function that implements a binary search algorithm with error handling",
                "method": thinker.code_generation
            },
            {
                "task_type": "tool_use_planning",
                "prompt": "Plan how to build a web scraper that respects robots.txt and rate limits",
                "method": thinker.tool_use_planning,
                "context": "Available tools: requests, BeautifulSoup, selenium, scrapy"
            },
            {
                "task_type": "auto_detection",
                "prompt": "Solve this complex mathematical proof: Prove that the square root of 2 is irrational",
                "method": thinker.auto_process
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"🧪 Test {i}: {test_case['task_type']}")
            logger.info(f"📝 Prompt: {test_case['prompt']}")
            
            try:
                # Call the appropriate method
                if 'context' in test_case:
                    response = await test_case['method'](test_case['prompt'], test_case['context'])
                else:
                    response = await test_case['method'](test_case['prompt'])
                
                logger.info(f"🤖 Response preview: {response[:200]}...")
                logger.info("✅ Test case completed successfully")
                
            except Exception as e:
                logger.error(f"❌ Error in test case: {e}")
        
        # Display performance stats
        stats = thinker.get_performance_stats()
        logger.info("📊 Thinker Model Performance Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  - {key}: {value:.3f}")
            else:
                logger.info(f"  - {key}: {value}")
        
        logger.info("✅ Thinker model test completed")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return 1
    finally:
        await thinker.close()


async def test_integration():
    """Test Talker-Thinker model integration and handoff"""
    logger = logging.getLogger("sovereign")
    
    logger.info("🔄 Testing Talker-Thinker Integration...")
    
    # Initialize both models
    talker = TalkerModel()
    thinker = ThinkerModel()
    
    try:
        # Initialize both models
        logger.info("🔧 Initializing both models...")
        
        talker_ok = await talker.initialize()
        thinker_ok = await thinker.initialize()
        
        if not talker_ok:
            logger.error("❌ Failed to initialize Talker model")
            return 1
        if not thinker_ok:
            logger.error("❌ Failed to initialize Thinker model")
            return 1
        
        logger.info("✅ Both models initialized successfully")
        
        # Test integration scenarios
        test_scenarios = [
            {
                "name": "Simple Query (Talker Only)",
                "prompt": "Hello, how are you today?",
                "expect_handoff": False
            },
            {
                "name": "Complex Reasoning (Should Hand Off)",
                "prompt": "Analyze the pros and cons of different machine learning algorithms for text classification and recommend the best approach",
                "expect_handoff": True
            },
            {
                "name": "Code Generation (Should Hand Off)",
                "prompt": "Implement a complete REST API in Python with authentication, error handling, and database integration",
                "expect_handoff": True
            },
            {
                "name": "Tool Planning (Should Hand Off)",
                "prompt": "Design a system to automatically backup and sync files across multiple cloud providers",
                "expect_handoff": True
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            logger.info(f"🧪 Integration Test {i}: {scenario['name']}")
            logger.info(f"📝 Prompt: {scenario['prompt']}")
            
            # Check if Talker detects complexity
            is_complex = talker.detect_complex_query(scenario['prompt'])
            logger.info(f"🔍 Complexity detected: {is_complex}")
            
            if is_complex != scenario['expect_handoff']:
                logger.warning(f"⚠️  Expected handoff: {scenario['expect_handoff']}, but detected: {is_complex}")
            
            try:
                if is_complex:
                    # Simulate handoff to Thinker
                    logger.info("🔄 Handing off to Thinker model...")
                    response = await thinker.auto_process(scenario['prompt'])
                    logger.info("🤖 Thinker response preview: " + response[:150] + "...")
                else:
                    # Process with Talker
                    logger.info("⚡ Processing with Talker model...")
                    response = await talker.generate_response(scenario['prompt'])
                    logger.info("🤖 Talker response: " + response[:150] + "...")
                
                logger.info("✅ Integration scenario completed")
                
            except Exception as e:
                logger.error(f"❌ Error in integration test: {e}")
        
        # Display combined performance stats
        talker_stats = talker.get_performance_stats()
        thinker_stats = thinker.get_performance_stats()
        
        logger.info("📊 Integration Performance Summary:")
        logger.info(f"  Talker - Queries: {talker_stats['query_count']}, Avg Time: {talker_stats.get('average_response_time', 0):.3f}s")
        logger.info(f"  Thinker - Tasks: {thinker_stats['task_count']}, Avg Time: {thinker_stats.get('average_processing_time', 0):.3f}s")
        
        logger.info("✅ Integration test completed")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return 1
    finally:
        await talker.close()
        await thinker.close()


async def test_orchestrator():
    """Test the orchestration system functionality"""
    logger = logging.getLogger("sovereign")
    
    logger.info("🎯 Testing Orchestration System...")
    
    # Initialize orchestrator
    orchestrator = ModelOrchestrator(config)
    
    try:
        # Test initialization
        logger.info("🔧 Initializing orchestrator...")
        await orchestrator.initialize()
        logger.info("✅ Orchestrator initialized successfully")
        
        # Test queries of different complexity levels
        test_queries = [
            ("Hello, how are you?", "Simple greeting"),
            ("What's the weather like?", "Simple question"),
            ("Explain the difference between machine learning and deep learning", "Complex explanation"),
            ("Write a Python function to implement a binary search algorithm with detailed comments", "Complex code generation"),
            ("Analyze the pros and cons of microservices architecture compared to monolithic architecture", "Complex analysis"),
            ("Create a step-by-step plan for deploying a web application to AWS", "Complex planning"),
        ]
        
        for i, (query, description) in enumerate(test_queries, 1):
            logger.info(f"\n🧪 Test {i}: {description}")
            logger.info(f"📝 Query: {query}")
            
            # Create query context
            context = QueryContext(
                user_input=query,
                timestamp=datetime.now(),
                session_id="test_session",
                previous_queries=[],
                conversation_history=[]
            )
            
            # Process query
            try:
                result = await orchestrator.process_query(query, context)
                
                # Display results
                logger.info(f"🤖 Response: {result.response[:200]}{'...' if len(result.response) > 200 else ''}")
                logger.info(f"🎯 Model Used: {result.model_used.value}")
                logger.info(f"📊 Complexity Level: {result.complexity_level.value}")
                logger.info(f"⏱️  Processing Time: {result.processing_time:.3f}s")
                logger.info(f"🔄 Handoff Occurred: {result.handoff_occurred}")
                logger.info(f"🗃️  Cache Hit: {result.cache_hit}")
                logger.info(f"📈 Confidence Score: {result.confidence_score:.3f}")
                logger.info(f"💡 Reasoning: {result.reasoning}")
                
            except Exception as e:
                logger.error(f"❌ Error processing query: {e}")
        
        # Display telemetry
        status = await orchestrator.get_status()
        logger.info("\n📊 Orchestrator Status:")
        logger.info(f"  - Talker Model Ready: {status['talker_model_ready']}")
        logger.info(f"  - Thinker Model Ready: {status['thinker_model_ready']}")
        logger.info(f"  - Cache Size: {status['cache_size']}")
        
        telemetry = status['telemetry']
        logger.info("\n📈 Telemetry:")
        logger.info(f"  - Total Queries: {telemetry['total_queries']}")
        logger.info(f"  - Talker Queries: {telemetry['talker_queries']}")
        logger.info(f"  - Thinker Queries: {telemetry['thinker_queries']}")
        logger.info(f"  - Handoff Queries: {telemetry['handoff_queries']}")
        logger.info(f"  - Cache Hit Rate: {telemetry['cache_hit_rate']:.3f}")
        logger.info(f"  - Average Response Time: {telemetry['avg_response_time']:.3f}s")
        logger.info(f"  - Error Count: {telemetry['error_count']}")
        
        logger.info("✅ Orchestrator test completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Orchestrator test failed: {e}")
        return 1
    finally:
        await orchestrator.close()


async def test_voice_interface_cli():
    """Test the voice interface functionality"""
    logger = logging.getLogger("sovereign")
    
    if not VOICE_INTERFACE_AVAILABLE:
        logger.error("❌ Voice interface not available - missing audio dependencies")
        logger.info("💡 Install audio dependencies: pip install sounddevice librosa speechrecognition pydub pyaudio pyttsx3 openai-whisper webrtcvad noisereduce scipy pvporcupine")
        return 1
    
    logger.info("🎤 Testing Voice Interface...")
    
    try:
        # Test voice interface
        success = await test_voice_interface(config)
        
        if success:
            logger.info("✅ Voice interface test completed successfully")
            return 0
        else:
            logger.error("❌ Voice interface test failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Voice interface test failed: {e}")
        return 1


async def list_audio_devices():
    """List available audio input/output devices"""
    logger = logging.getLogger("sovereign")
    
    if not VOICE_INTERFACE_AVAILABLE:
        logger.error("❌ Voice interface not available - missing audio dependencies")
        logger.info("💡 Install audio dependencies: pip install sounddevice librosa speechrecognition pydub pyaudio pyttsx3 openai-whisper webrtcvad noisereduce scipy pvporcupine")
        return 1
    
    logger.info("🎧 Available Audio Devices:")
    
    try:
        devices = get_audio_devices()
        
        print("\n🎙️  Input Devices (Microphones):")
        for device in devices['input_devices']:
            print(f"  [{device['index']}] {device['name']}")
            print(f"      Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']}")
        
        print("\n🔊 Output Devices (Speakers):")
        for device in devices['output_devices']:
            print(f"  [{device['index']}] {device['name']}")
            print(f"      Channels: {device['max_output_channels']}, Sample Rate: {device['default_samplerate']}")
        
        if not devices['input_devices'] and not devices['output_devices']:
            print("❌ No audio devices found!")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error listing audio devices: {e}")
        return 1


async def configure_voice_profile():
    """Configure voice interface settings interactively"""
    logger = logging.getLogger("sovereign")
    
    if not VOICE_INTERFACE_AVAILABLE:
        logger.error("❌ Voice interface not available - missing audio dependencies")
        logger.info("💡 Install audio dependencies: pip install sounddevice librosa speechrecognition pydub pyaudio pyttsx3 openai-whisper webrtcvad noisereduce scipy pvporcupine")
        return 1
    
    logger.info("🎛️  Voice Interface Configuration")
    
    try:
        print("\n🎤 Voice Interface Configuration")
        print("=" * 40)
        
        # Get available devices
        devices = get_audio_devices()
        
        # Configure input device
        print("\n🎙️  Select Input Device (Microphone):")
        for i, device in enumerate(devices['input_devices']):
            print(f"  [{i}] {device['name']}")
        
        while True:
            try:
                choice = input(f"\nEnter device number (0-{len(devices['input_devices'])-1}): ")
                device_idx = int(choice)
                if 0 <= device_idx < len(devices['input_devices']):
                    selected_input = devices['input_devices'][device_idx]
                    print(f"✅ Selected input: {selected_input['name']}")
                    break
                else:
                    print("❌ Invalid selection. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")
        
        # Configure output device
        print("\n🔊 Select Output Device (Speakers):")
        for i, device in enumerate(devices['output_devices']):
            print(f"  [{i}] {device['name']}")
        
        while True:
            try:
                choice = input(f"\nEnter device number (0-{len(devices['output_devices'])-1}): ")
                device_idx = int(choice)
                if 0 <= device_idx < len(devices['output_devices']):
                    selected_output = devices['output_devices'][device_idx]
                    print(f"✅ Selected output: {selected_output['name']}")
                    break
                else:
                    print("❌ Invalid selection. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")
        
        # Configure wake word
        print("\n🎯 Wake Word Configuration:")
        print(f"Current wake word: '{config.voice.wake_word}'")
        
        enable_wake_word = input("Enable wake word detection? (y/n): ").lower().startswith('y')
        
        if enable_wake_word:
            print("Note: Wake word detection requires a Porcupine access key.")
            porcupine_key = input("Enter Porcupine access key (or press Enter to skip): ").strip()
            if porcupine_key:
                config.voice.porcupine_access_key = porcupine_key
                print("✅ Porcupine access key configured")
            else:
                print("⚠️  Wake word detection will be disabled without access key")
                enable_wake_word = False
        
        # Configure Whisper model
        print("\n🗣️  Speech Recognition Configuration:")
        print("Available Whisper models: tiny, base, small, medium, large")
        print(f"Current model: {config.voice.whisper_model_size}")
        
        model_choice = input("Enter Whisper model size (or press Enter to keep current): ").strip()
        if model_choice in ['tiny', 'base', 'small', 'medium', 'large']:
            config.voice.whisper_model_size = model_choice
            print(f"✅ Whisper model set to: {model_choice}")
        
        # Configure TTS settings
        print("\n🔊 Text-to-Speech Configuration:")
        print(f"Current speech rate: {config.voice.tts_speech_rate} WPM")
        print(f"Current volume: {config.voice.tts_volume}")
        
        try:
            speech_rate = input("Enter speech rate (100-400 WPM, or press Enter to keep current): ").strip()
            if speech_rate:
                rate = int(speech_rate)
                if 100 <= rate <= 400:
                    config.voice.tts_speech_rate = rate
                    print(f"✅ Speech rate set to: {rate} WPM")
                else:
                    print("⚠️  Speech rate should be between 100-400 WPM")
        except ValueError:
            print("⚠️  Invalid speech rate entered")
        
        try:
            volume = input("Enter volume (0.0-1.0, or press Enter to keep current): ").strip()
            if volume:
                vol = float(volume)
                if 0.0 <= vol <= 1.0:
                    config.voice.tts_volume = vol
                    print(f"✅ Volume set to: {vol}")
                else:
                    print("⚠️  Volume should be between 0.0-1.0")
        except ValueError:
            print("⚠️  Invalid volume entered")
        
        # Update configuration
        config.voice.input_device_name = selected_input['name']
        config.voice.output_device_name = selected_output['name']
        config.voice.wake_word_enabled = enable_wake_word
        
        # Save configuration
        config.save_config()
        
        print("\n✅ Voice interface configuration saved!")
        print("\nConfiguration Summary:")
        print(f"  - Input Device: {config.voice.input_device_name}")
        print(f"  - Output Device: {config.voice.output_device_name}")
        print(f"  - Wake Word: {'Enabled' if config.voice.wake_word_enabled else 'Disabled'}")
        print(f"  - Whisper Model: {config.voice.whisper_model_size}")
        print(f"  - Speech Rate: {config.voice.tts_speech_rate} WPM")
        print(f"  - Volume: {config.voice.tts_volume}")
        
        logger.info("Voice profile configuration complete")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error configuring voice profile: {e}")
        return 1


async def test_external_routing():
    """Test external model routing functionality"""
    logger = logging.getLogger("sovereign")
    
    logger.info("🧪 Testing External Model Routing...")
    
    # Initialize external connector
    external_connector = ExternalModelConnector(config)
    
    try:
        # Test initialization
        logger.info("🔧 Initializing external model connector...")
        initialized = await external_connector.initialize()
        
        if not initialized:
            logger.warning("⚠️ External model connector not available (no API key or connection failed)")
            logger.info("💡 Set OPENROUTER_API_KEY environment variable to test external routing")
            return 0
        
        logger.info("✅ External model connector initialized successfully")
        
        # Test routing criteria detection
        test_queries = [
            "Hello, how are you?",  # Should not route
            "What are the latest news about AI?",  # Should route - specialized knowledge
            "Search online for Python tutorials",  # Should route - explicit request
            "I need the latest up-to-date information about Tesla stock",  # Should route - explicit request
            "What's the weather like today?",  # Should route - specialized knowledge
            "Explain how to create a simple Python function",  # Should not route
        ]
        
        logger.info("🔍 Testing routing criteria detection...")
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n🧪 Test {i}: {query}")
            
            # Check routing decision
            decision = external_connector.determine_external_need(query)
            logger.info(f"📊 Should route: {decision.should_route}")
            logger.info(f"📊 Confidence: {decision.confidence:.2f}")
            logger.info(f"📊 Criteria: {[c.value for c in decision.criteria]}")
            logger.info(f"📊 Reasoning: {decision.reasoning}")
        
        # Test performance stats
        stats = external_connector.get_performance_stats()
        logger.info("\n📈 Performance Stats:")
        for key, value in stats.items():
            logger.info(f"  - {key}: {value}")
        
        logger.info("\n✅ External routing tests completed")
        
    except Exception as e:
        logger.error(f"❌ Error testing external routing: {e}")
        return 1
    
    finally:
        await external_connector.close()
    
    return 0


async def test_consent_mechanism():
    """Test user consent mechanism"""
    logger = logging.getLogger("sovereign")
    
    logger.info("🧪 Testing User Consent Mechanism...")
    
    # Initialize external connector with consent callback
    external_connector = ExternalModelConnector(config)
    
    try:
        # Add CLI consent callback
        external_connector.add_consent_callback(cli_consent_callback)
        
        # Test consent with a sample query
        test_query = "What are the latest developments in AI technology?"
        
        logger.info(f"🔍 Testing consent for query: {test_query}")
        
        # Get routing decision
        decision = external_connector.determine_external_need(test_query)
        
        if decision.should_route:
            logger.info("📋 This query would trigger a consent request")
            logger.info("💡 In a real scenario, you would be prompted for consent")
            
            # Show what the consent request would look like
            logger.info("\n" + "="*60)
            logger.info("🔗 CONSENT REQUEST PREVIEW")
            logger.info("="*60)
            logger.info(f"Query: {test_query}")
            logger.info(f"Reasoning: {decision.reasoning}")
            logger.info(f"Confidence: {decision.confidence:.2f}")
            logger.info(f"Criteria: {[c.value for c in decision.criteria]}")
            logger.info("="*60)
            
            # For testing, we'll simulate a consent decision
            logger.info("💭 Simulating user consent (would be interactive in real use)")
            
        else:
            logger.info("📋 This query would not require external routing")
        
        logger.info("\n✅ Consent mechanism test completed")
        
    except Exception as e:
        logger.error(f"❌ Error testing consent mechanism: {e}")
        return 1
    
    finally:
        await external_connector.close()
    
    return 0


async def run_agent():
    """Main agent execution loop with integrated orchestration system"""
    logger = logging.getLogger("sovereign")
    
    print("DEBUG: Starting run_agent function...")
    logger.info("🚀 Starting Sovereign AI Agent...")
    
    print("DEBUG: Creating ModelOrchestrator instance...")
    # Initialize orchestrator
    orchestrator = ModelOrchestrator(config)
    print("DEBUG: ModelOrchestrator instance created.")
    
    print("DEBUG: Setting up session context...")
    # Session context for conversation history
    session_context = {
        'session_id': 'main_session',
        'conversation_history': [],
        'previous_queries': []
    }
    print("DEBUG: Session context set up.")
    
    print("DEBUG: Creating notification callback...")
    # Add notification callback
    async def notification_callback(message: str):
        print(f"💡 {message}")
    
    print("DEBUG: Adding notification callback to orchestrator...")
    orchestrator.add_notification_callback(notification_callback)
    print("DEBUG: Notification callback added.")
    
    try:
        print("DEBUG: About to initialize orchestration system...")
        # Initialize orchestrator
        logger.info("🔧 Initializing orchestration system...")
        await orchestrator.initialize()
        print("DEBUG: Orchestration system initialization completed.")
        logger.info("✅ Orchestration system initialized successfully")
        logger.info("💡 Agent ready for interaction")
        
        while True:
            # Get user input
            user_input = input("\n🤖 Sovereign > ")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                logger.info("👋 Shutting down Sovereign AI Agent")
                break
            elif user_input.lower() in ['help', 'h']:
                print("""
Available commands:
  help, h       - Show this help message
  status        - Show system status
  config        - Show current configuration
  stats         - Show orchestration statistics
  cache         - Show cache statistics
  exit, quit, q - Exit the application
  
Or just type any message to chat with Sovereign!
""")
            elif user_input.lower() == 'status':
                status = await orchestrator.get_status()
                print(f"""
🔧 System Status:
  - GPU Available: {hardware_detector.system_info.cuda_available}
  - GPU: {hardware_detector.gpu_info.name if hardware_detector.gpu_info else 'N/A'}
  - System Memory: {hardware_detector.system_info.memory_total:.1f} GB
  - CPU Cores: {hardware_detector.system_info.cpu_count}
  - Talker Model Ready: {status['talker_model_ready']}
  - Thinker Model Ready: {status['thinker_model_ready']}
  - Cache Size: {status['cache_size']}
""")
            elif user_input.lower() == 'config':
                print(f"""
⚙️ Configuration:
  - Talker Model: {config.models.talker_model}
  - Thinker Model: {config.models.thinker_model}
  - Ollama Endpoint: {config.models.ollama_endpoint}
  - GPU Enabled: {config.hardware.gpu_enabled}
  - Voice Enabled: {config.voice.enabled}
  - Screen Capture: {config.screen_capture.enabled}
  - Cache Enabled: {config.orchestrator.enable_caching}
  - Cache Max Size: {config.orchestrator.cache_max_size}
  - Cache TTL: {config.orchestrator.cache_ttl_hours}h
""")
            elif user_input.lower() == 'stats':
                status = await orchestrator.get_status()
                telemetry = status['telemetry']
                print("\n📊 Orchestration Statistics:")
                print(f"  - Total Queries: {telemetry['total_queries']}")
                print(f"  - Talker Queries: {telemetry['talker_queries']}")
                print(f"  - Thinker Queries: {telemetry['thinker_queries']}")
                print(f"  - Handoff Queries: {telemetry['handoff_queries']}")
                print(f"  - Cache Hit Rate: {telemetry['cache_hit_rate']:.3f}")
                print(f"  - Handoff Rate: {telemetry['handoff_rate']:.3f}")
                print(f"  - Average Response Time: {telemetry['avg_response_time']:.3f}s")
                print(f"  - Error Count: {telemetry['error_count']}")
                print(f"  - Uptime: {telemetry['uptime_seconds']:.1f}s")
                
                if 'complexity_distribution' in telemetry:
                    print("\n📈 Complexity Distribution:")
                    for complexity, count in telemetry['complexity_distribution'].items():
                        print(f"  - {complexity}: {count}")
                        
            elif user_input.lower() == 'cache':
                status = await orchestrator.get_status()
                print(f"\n🗃️ Cache Statistics:")
                print(f"  - Cache Size: {status['cache_size']}")
                print(f"  - Cache Hit Rate: {status['telemetry']['cache_hit_rate']:.3f}")
                print(f"  - Cache Hits: {status['telemetry']['cache_hits']}")
                print(f"  - Cache Misses: {status['telemetry']['cache_misses']}")
                
            else:
                # Generate AI response using orchestrator
                try:
                    # Create query context
                    context = QueryContext(
                        user_input=user_input,
                        timestamp=datetime.now(),
                        session_id=session_context['session_id'],
                        previous_queries=session_context['previous_queries'][-5:],  # Last 5 queries
                        conversation_history=session_context['conversation_history'][-10:]  # Last 10 exchanges
                    )
                    
                    # Process query through orchestrator
                    result = await orchestrator.process_query(user_input, context)
                    
                    # Display response
                    print(f"\n🤖 {result.response}")
                    
                    # Show debug info if in debug mode
                    if config.debug:
                        print(f"\n🔍 Debug Info:")
                        print(f"  - Model Used: {result.model_used.value}")
                        print(f"  - Complexity: {result.complexity_level.value}")
                        print(f"  - Processing Time: {result.processing_time:.3f}s")
                        print(f"  - Handoff Occurred: {result.handoff_occurred}")
                        print(f"  - Cache Hit: {result.cache_hit}")
                        print(f"  - Confidence: {result.confidence_score:.3f}")
                        print(f"  - Reasoning: {result.reasoning}")
                    
                    # Update conversation history
                    session_context['previous_queries'].append(user_input)
                    session_context['conversation_history'].append({
                        'user': user_input,
                        'assistant': result.response,
                        'timestamp': datetime.now().isoformat(),
                        'model_used': result.model_used.value
                    })
                    
                    # Keep history manageable
                    if len(session_context['previous_queries']) > 20:
                        session_context['previous_queries'] = session_context['previous_queries'][-10:]
                    if len(session_context['conversation_history']) > 20:
                        session_context['conversation_history'] = session_context['conversation_history'][-10:]
                    
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    print("❌ I encountered an error processing your request. Please try again.")
    
    except KeyboardInterrupt:
        logger.info("👋 Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1
    finally:
        await orchestrator.close()
    
    return 0


def main():
    """Main entry point for the Sovereign AI Agent"""
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger(
        log_level=args.log_level,
        log_file=args.log_file,
        debug=args.debug
    )
    
    # Update config with command line arguments
    if args.debug:
        config.debug = True
        config.log_level = "DEBUG"
    else:
        config.log_level = args.log_level
    
    if args.no_gpu:
        config.hardware.gpu_enabled = False
    
    if args.no_voice:
        config.voice.enabled = False
    
    if args.no_screen_capture:
        config.screen_capture.enabled = False
    
    # Load configuration
    config.load_config(args.config)
    
    # Handle special commands
    if args.check_requirements:
        logger.info("🔍 Checking system requirements...")
        meets_requirements = check_system_requirements()
        
        # Display optimization recommendations
        optimal_settings = hardware_detector.get_optimal_settings()
        logger.info("💡 Recommended settings for your system:")
        for key, value in optimal_settings.items():
            logger.info(f"  - {key}: {value}")
        
        sys.exit(0 if meets_requirements else 1)
    
    if args.setup:
        logger.info("🛠️  Running initial setup...")
        
        # Create necessary directories
        directories = [
            "config",
            "data",
            "logs",
            "models",
            "cache"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")
        
        # Save configuration
        config.save_config(args.config)
        logger.info(f"💾 Saved configuration to: {args.config}")
        
        logger.info("✅ Setup complete!")
        sys.exit(0)
    
    if args.test_talker:
        logger.info("🧪 Running Talker model test...")
        try:
            exit_code = asyncio.run(test_talker_model())
            sys.exit(exit_code)
        except KeyboardInterrupt:
            logger.info("👋 Test interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Test error: {e}")
            sys.exit(1)
    
    if args.test_thinker:
        logger.info("🧠 Running Thinker model test...")
        try:
            exit_code = asyncio.run(test_thinker_model())
            sys.exit(exit_code)
        except KeyboardInterrupt:
            logger.info("👋 Test interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Test error: {e}")
            sys.exit(1)
    
    if args.test_integration:
        logger.info("🔄 Running integration test...")
        try:
            exit_code = asyncio.run(test_integration())
            sys.exit(exit_code)
        except KeyboardInterrupt:
            logger.info("👋 Test interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Test error: {e}")
            sys.exit(1)
    
    if args.test_orchestrator:
        logger.info("🧠 Running orchestrator test...")
        try:
            exit_code = asyncio.run(test_orchestrator())
            sys.exit(exit_code)
        except KeyboardInterrupt:
            logger.info("👋 Test interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Test error: {e}")
            sys.exit(1)
    
    if args.test_external:
        logger.info("🔗 Running external routing test...")
        try:
            exit_code = asyncio.run(test_external_routing())
            sys.exit(exit_code)
        except KeyboardInterrupt:
            logger.info("👋 Test interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Test error: {e}")
            sys.exit(1)
    
    if args.test_consent:
        logger.info("🛡️ Running consent mechanism test...")
        try:
            exit_code = asyncio.run(test_consent_mechanism())
            sys.exit(exit_code)
        except KeyboardInterrupt:
            logger.info("👋 Test interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Test error: {e}")
            sys.exit(1)
    
    if args.test_voice:
        logger.info("🎤 Running voice interface test...")
        try:
            exit_code = asyncio.run(test_voice_interface_cli())
            sys.exit(exit_code)
        except KeyboardInterrupt:
            logger.info("👋 Test interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Test error: {e}")
            sys.exit(1)
    
    if args.list_audio_devices:
        logger.info("🎧 Listing audio devices...")
        try:
            exit_code = asyncio.run(list_audio_devices())
            sys.exit(exit_code)
        except KeyboardInterrupt:
            logger.info("👋 Interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            sys.exit(1)
    
    if args.voice_profile:
        logger.info("🎛️  Configuring voice profile...")
        try:
            exit_code = asyncio.run(configure_voice_profile())
            sys.exit(exit_code)
        except KeyboardInterrupt:
            logger.info("👋 Configuration interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Configuration error: {e}")
            sys.exit(1)
    
    if args.diagnose_gpu:
        logger.info("🔍 Running GPU/CUDA diagnostic...")
        try:
            from .hardware import diagnose_gpu_environment
            diagnose_gpu_environment()
            sys.exit(0)
        except KeyboardInterrupt:
            logger.info("👋 Diagnostic interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Diagnostic error: {e}")
            sys.exit(1)
    
    if args.gui:
        logger.info("🎨 Launching graphical user interface...")
        if not GUI_AVAILABLE:
            logger.error("❌ GUI not available. Please install GUI dependencies:")
            logger.error("   pip install customtkinter tkinter-tooltip pystray")
            sys.exit(1)
        
        try:
            run_gui()
            sys.exit(0)
        except KeyboardInterrupt:
            logger.info("👋 GUI interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ GUI error: {e}")
            sys.exit(1)
    
    print("DEBUG: About to check system requirements...")
    # Check requirements on startup
    logger.info("🔍 Verifying system requirements...")
    if not check_system_requirements():
        logger.warning("⚠️  System may not meet optimal requirements")
        response = input("Continue anyway? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            logger.info("👋 Exiting...")
            sys.exit(1)
    print("DEBUG: System requirements check completed.")
    
    print("DEBUG: About to apply hardware optimizations...")
    # Apply hardware optimizations
    logger.info("⚙️  Applying hardware optimizations...")
    hardware_detector.optimize_pytorch()
    print("DEBUG: Hardware optimizations applied.")
    
    print("DEBUG: About to run the main agent...")
    # Run the main agent
    try:
        exit_code = asyncio.run(run_agent())
        print("DEBUG: Main agent completed.")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 