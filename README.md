# Sovereign AI Agent

A private, powerful, locally-running AI assistant that puts you in complete control of your data and computational resources.

## 🎯 Project Vision

Sovereign AI Agent is built on the principle of **AI Sovereignty** - you own and control your AI completely. The system operates entirely on your local hardware for all core functionality, ensuring privacy and eliminating dependencies on third-party cloud services.

## ✨ Key Features

### 🧠 Dual-Model "Stacked" Architecture
- **Talker Model**: Fast conversational AI (Gemma2:9b) for instant responses (<2 seconds)
- **Thinker Model**: Powerful reasoning AI (DeepSeek-R1:14b) for complex tasks
- **Intelligent Orchestration**: Automatic handoff between models based on query complexity

### 👁️ Real-Time Screen Context
- Background screen capture with configurable intervals
- OCR text extraction for understanding screen content
- Privacy-first design with easy on/off toggle

### 🎤 Advanced Voice Interface
- High-quality speech-to-text transcription
- Natural text-to-speech output
- Voice Activity Detection (VAD) for smart activation
- Wake word support ("Hey Sovereign")

### 🔧 External Tool Integration
- Real-time web search capabilities
- Extensible function-calling framework
- Future-ready for additional tools and APIs

### 🧠 Long-Term Memory (RAG)
- Local conversation history storage
- Semantic search across past interactions
- Context-aware, personalized responses

## 🏗️ System Requirements

### Recommended Hardware
- **GPU**: NVIDIA RTX 5070 Ti 16GB or equivalent
- **System RAM**: 16GB+ 
- **Python**: 3.10 or higher
- **Storage**: 50GB+ free space for models and data

### Minimum Hardware
- **GPU**: Any CUDA-capable GPU with 8GB+ VRAM
- **System RAM**: 8GB (performance will be limited)
- **CPU**: Multi-core processor for CPU fallback mode

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/sovereign-ai/sovereign.git
cd sovereign
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -e .
```

4. **Run initial setup**
```bash
sovereign --setup
```

5. **Check system requirements**
```bash
sovereign --check-requirements
```

### First Run

Launch the Sovereign AI Agent with a single command:

```bash
sovereign
```

## 🛠️ Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

Key configuration options:
- `GPU_ENABLED`: Enable/disable GPU acceleration
- `VOICE_ENABLED`: Enable/disable voice interface
- `SCREEN_CAPTURE_ENABLED`: Enable/disable screen context (default: false for privacy)
- `LOG_LEVEL`: Set logging verbosity (DEBUG, INFO, WARNING, ERROR)

### Command Line Options

```bash
# Basic usage
sovereign

# Disable GPU (CPU-only mode)
sovereign --no-gpu

# Enable debug logging
sovereign --debug

# Disable voice interface
sovereign --no-voice

# Check system compatibility
sovereign --check-requirements

# Run setup wizard
sovereign --setup
```

## 🏛️ Architecture

The Sovereign AI Agent follows a modular architecture:

```
src/sovereign/
├── __init__.py          # Main package
├── cli.py              # Command-line interface
├── config.py           # Configuration management
├── logger.py           # Logging system
├── hardware.py         # Hardware detection & optimization
├── models/             # AI model implementations (Future)
├── voice/              # Voice interface (Future)
├── screen/             # Screen capture system (Future)
├── memory/             # RAG and memory systems (Future)
└── tools/              # External tool integrations (Future)
```

## 🔒 Privacy & Security

- **Local-First**: All core functionality runs on your hardware
- **No Cloud Dependencies**: Optional external services only for specialized tasks
- **Data Ownership**: All conversations and data stored locally
- **Privacy Controls**: Easy toggles for all data collection features
- **Transparent Operations**: Full visibility into what the AI is doing

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/sovereign-ai/sovereign/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sovereign-ai/sovereign/discussions)
- **Documentation**: [Project Wiki](https://github.com/sovereign-ai/sovereign/wiki)

## 🗺️ Roadmap

- [x] **Phase 1**: Project architecture and environment setup
- [ ] **Phase 2**: Talker model integration (Gemma2:9b)
- [ ] **Phase 3**: Thinker model integration (DeepSeek-R1:14b)
- [ ] **Phase 4**: Intelligent orchestration system
- [ ] **Phase 5**: Voice interface implementation
- [ ] **Phase 6**: Screen context system
- [ ] **Phase 7**: External tool framework
- [ ] **Phase 8**: Long-term memory and RAG
- [ ] **Phase 9**: User interface development
- [ ] **Phase 10**: Performance optimization and testing

---

**Built with ❤️ for AI sovereignty and privacy** 