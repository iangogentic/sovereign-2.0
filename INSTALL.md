# Installation Guide - Sovereign AI Agent

This guide provides detailed installation instructions for the Sovereign AI Agent.

## 📋 Prerequisites

### System Requirements

**Recommended Configuration:**
- NVIDIA RTX 5070 Ti 16GB or equivalent GPU
- 16GB+ system RAM
- Python 3.10 or higher
- 50GB+ free disk space
- Windows 10/11, Linux (Ubuntu 20.04+), or macOS 12+

**Minimum Configuration:**
- Any CUDA-capable GPU with 8GB+ VRAM
- 8GB system RAM (limited performance)
- Python 3.10+
- 20GB+ free disk space

### Software Dependencies

1. **Python 3.10+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure `pip` is included

2. **NVIDIA CUDA Toolkit** (for GPU acceleration)
   - Download CUDA 12.1+ from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
   - Verify installation: `nvcc --version`

3. **Git** (for cloning repository)
   - Download from [git-scm.com](https://git-scm.com/downloads)

## 🔧 Installation Steps

### Step 1: Clone Repository

```bash
git clone https://github.com/sovereign-ai/sovereign.git
cd sovereign
```

### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Upgrade pip and Install Build Tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Install PyTorch with CUDA Support

**For CUDA 12.1+:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU-only (not recommended):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Install Sovereign AI Agent

```bash
pip install -e .
```

### Step 7: Run Initial Setup

```bash
sovereign --setup
```

This will:
- Create necessary directories (`config/`, `data/`, `logs/`, `models/`, `cache/`)
- Generate default configuration files
- Verify system requirements

### Step 8: Verify Installation

```bash
sovereign --check-requirements
```

## 🐛 Troubleshooting

### Common Issues

#### 1. PyTorch CUDA Installation Issues

**Problem**: CUDA not detected after PyTorch installation

**Solution:**
```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify CUDA is working:**
```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should show your GPU name
```

#### 2. Audio Dependencies (Voice Interface)

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install espeak espeak-data
```

**macOS:**
```bash
brew install portaudio
```

**Windows:**
- PyAudio wheels should install automatically
- If issues occur, download from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

#### 3. OCR Dependencies (Screen Context)

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
- Download Tesseract from [GitHub Releases](https://github.com/UB-Mannheim/tesseract/wiki)
- Add to PATH or set `TESSERACT_CMD` environment variable

#### 4. Permission Issues (Linux/macOS)

If you encounter permission errors:

```bash
# Option 1: Use user installation
pip install --user -r requirements.txt

# Option 2: Fix pip permissions
sudo chown -R $(whoami) ~/.local/lib/python*/site-packages/
```

#### 5. Memory Issues During Installation

If installation fails due to memory constraints:

```bash
# Install with no cache
pip install --no-cache-dir -r requirements.txt

# Or install packages individually
pip install torch torchvision torchaudio
pip install transformers
# ... continue with other packages
```

### System-Specific Notes

#### Windows

1. **Long Path Support**: Enable long path support in Windows:
   ```cmd
   # Run as Administrator
   reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1
   ```

2. **Visual C++ Redistributable**: Some packages may require Visual C++ Redistributable
   - Download from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)

#### Linux

1. **NVIDIA Driver**: Ensure NVIDIA drivers are installed:
   ```bash
   # Ubuntu/Debian
   sudo apt install nvidia-driver-535
   
   # Check installation
   nvidia-smi
   ```

2. **X11 Dependencies** (for screen capture):
   ```bash
   sudo apt-get install python3-tk python3-dev
   ```

#### macOS

1. **Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

2. **Metal Performance Shaders**: May provide GPU acceleration on Apple Silicon

## ✅ Verification

After installation, verify everything works:

```bash
# Check system requirements
sovereign --check-requirements

# Test basic functionality
sovereign --help

# Test GPU detection
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test imports
python -c "from sovereign import Config; print('✅ Import successful')"
```

## 🚀 Next Steps

1. **Configure Environment**: Copy `.env.example` to `.env` and customize settings
2. **First Run**: Launch with `sovereign`
3. **Read Documentation**: Check `README.md` for usage instructions
4. **Optional Setup**: Configure voice interface and screen capture as needed

## 📞 Getting Help

If you encounter issues not covered here:

1. **Check Issues**: [GitHub Issues](https://github.com/sovereign-ai/sovereign/issues)
2. **System Info**: Run `sovereign --check-requirements` and include output
3. **Create Issue**: Provide system info, error messages, and steps to reproduce

## 🔄 Updating

To update to the latest version:

```bash
cd sovereign
git pull origin main
pip install -r requirements.txt
pip install -e .
``` 