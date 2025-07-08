# Contributing to Sovereign AI Agent

Thank you for your interest in contributing to the Sovereign AI Agent! This guide will help you set up your development environment for a smooth and consistent experience.

## Development Environment Setup

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11 or 3.12** (Python 3.13+ may have compatibility issues with some dependencies)
- **Git** for version control
- **NVIDIA GPU with CUDA 12.1 support** (optional, but recommended for optimal performance)
- **Windows 10/11** (primary development platform)

### Automated Setup (Recommended)

The fastest way to get started is using our automated setup script:

```bash
# Clone the repository
git clone <repository-url>
cd Soverign\ 2.0

# Run the automated setup script
setup_env.bat
```

This will automatically create the virtual environment and install all dependencies with the correct PyTorch version.

### Manual Setup (Step-by-Step)

If you prefer to set up the environment manually or need to troubleshoot:

#### Step 1: Create Virtual Environment

```bash
# Create a new virtual environment named .venv
python -m venv .venv
```

#### Step 2: Activate Virtual Environment

```bash
# On Windows
.venv\Scripts\activate

# You should see (.venv) in your command prompt
```

#### Step 3: Upgrade pip

```bash
# Ensure you have the latest pip
python -m pip install --upgrade pip
```

#### Step 4: Install PyTorch with CUDA 12.1 Support

**IMPORTANT**: Install PyTorch FIRST before other dependencies to ensure CUDA compatibility:

```bash
# Install stable PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Step 5: Install Remaining Dependencies

```bash
# Install all other project dependencies
pip install -r requirements.txt
```

#### Step 6: Verify Installation

```bash
# Test the installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
```

Expected output:
```
PyTorch: 2.5.1+cu121
CUDA Available: True
GPU Count: 1
```

### GPU Compatibility Notes

#### Supported GPUs
- **RTX 40-series and older**: Full CUDA 12.1 support
- **RTX 50-series (5070 Ti, etc.)**: Limited support - may fall back to CPU
- **RTX 30-series**: Excellent performance
- **RTX 20-series**: Good performance

#### RTX 50-Series Compatibility
If you have an RTX 5070 Ti or other RTX 50-series GPU, you may see warnings like:
```
NVIDIA GeForce RTX 5070 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

**This is expected behavior**. The system will automatically fall back to CPU processing, which is still functional but slower. We prioritize stability over experimental PyTorch nightly builds.

### Development Workflow

#### Running the Application

```bash
# Activate environment (if not already active)
.venv\Scripts\activate

# Run the main application
python main.py

# Or run the full CLI version
python -m sovereign.cli
```

#### Running Tests

```bash
# Run the test suite
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/sovereign
```

#### Code Quality

```bash
# Format code with black
black src/ tests/

# Check with flake8
flake8 src/ tests/
```

### Environment Variables

Create a `.env` file in the project root for configuration:

```env
# Optional: Set log level
TASKMASTER_LOG_LEVEL=INFO

# Optional: Ollama endpoint (if different from default)
OLLAMA_BASE_URL=http://localhost:11434

# Add any API keys for external services (if needed)
# ANTHROPIC_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
```

### Troubleshooting

#### Virtual Environment Issues

**Problem**: `python -m venv .venv` fails
**Solution**: Ensure you're using Python 3.11 or 3.12. Check with `python --version`

**Problem**: Package installation fails
**Solution**: Ensure virtual environment is activated and pip is upgraded

#### PyTorch/CUDA Issues

**Problem**: CUDA not available despite having NVIDIA GPU
**Solution**: 
1. Verify NVIDIA drivers are installed and up to date
2. Check CUDA version with `nvidia-smi`
3. Reinstall PyTorch with correct CUDA version

**Problem**: New GPU not supported (RTX 50-series)
**Solution**: This is expected. The application will use CPU processing which is still functional.

#### Ollama Issues

**Problem**: Cannot connect to Ollama
**Solution**: 
1. Install Ollama from https://ollama.ai
2. Start Ollama service
3. Pull required models: `ollama pull gemma2:9b` and `ollama pull deepseek-r1:14b`

### Contributing Guidelines

#### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and small (< 50 lines when possible)

#### Commit Messages
- Use conventional commit format: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Example: `feat(memory): add conversation export functionality`

#### Pull Requests
1. Create a feature branch from `master`
2. Make your changes with proper tests
3. Update documentation as needed
4. Submit pull request with clear description

#### Testing
- Add tests for new functionality
- Ensure all existing tests pass
- Aim for >80% code coverage
- Include both unit and integration tests

### Getting Help

- Check existing issues and documentation first
- Create detailed bug reports with environment information
- Include steps to reproduce any issues
- Provide system information: OS, Python version, GPU model

### Development Tools Integration

#### VS Code/Cursor
Recommended extensions:
- Python
- Pylance 
- Black Formatter
- autoDocstring

#### PyCharm
Configure Python interpreter to use `.venv/Scripts/python.exe`

#### Command Line
All development can be done from command line with the activated virtual environment.

---

Happy coding! 🚀 