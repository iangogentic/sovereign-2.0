@echo off
setlocal enabledelayedexpansion

:: Sovereign AI Agent - Development Environment Setup Script
:: This script automates the creation of a virtual environment and installation of dependencies
:: with stable PyTorch CUDA 12.1 support for maximum compatibility.

echo.
echo ==========================================
echo Sovereign AI Agent - Environment Setup
echo ==========================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 or 3.12 and try again
    echo Download from: https://python.org/downloads/
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python version: %PYTHON_VERSION%

:: Warn about Python 3.13+ compatibility
echo %PYTHON_VERSION% | findstr "3.13" >nul
if %errorlevel% equ 0 (
    echo.
    echo WARNING: Python 3.13+ may have compatibility issues
    echo Recommended: Python 3.11 or 3.12
    echo Continue anyway? (Y/N)
    set /p CONTINUE=
    if /i "!CONTINUE!" neq "Y" (
        echo Setup cancelled
        pause
        exit /b 1
    )
)

:: Check if virtual environment already exists
if exist ".venv" (
    echo.
    echo Virtual environment already exists in .venv
    echo Do you want to recreate it? This will delete the existing environment. (Y/N)
    set /p RECREATE=
    if /i "!RECREATE!" equ "Y" (
        echo Removing existing virtual environment...
        rmdir /s /q .venv
    ) else (
        echo Using existing virtual environment
        goto :activate_env
    )
)

:: Create virtual environment
echo.
echo Creating virtual environment...
python -m venv .venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    echo Make sure you have sufficient permissions and disk space
    pause
    exit /b 1
)
echo Virtual environment created successfully

:activate_env
:: Activate virtual environment
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)

:: Install PyTorch with CUDA 12.1 support FIRST
echo.
echo Installing PyTorch with CUDA 12.1 support...
echo This is the most important step for GPU compatibility
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    echo This is critical for the application to work properly
    pause
    exit /b 1
)
echo PyTorch installed successfully

:: Install remaining dependencies
echo.
echo Installing remaining dependencies from requirements.txt...
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    echo Make sure you're running this script from the project root directory
    pause
    exit /b 1
)

python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install some dependencies
    echo Check the output above for specific errors
    pause
    exit /b 1
)

:: Verify installation
echo.
echo Verifying installation...
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); import sys; print(f'Python: {sys.version}')"
if %errorlevel% neq 0 (
    echo WARNING: Verification failed, but dependencies may still be installed
)

:: Check for GPU compatibility issues
echo.
echo Checking GPU compatibility...
python -c "import torch; gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'; print(f'GPU: {gpu_name}'); import warnings; warnings.filterwarnings('error'); torch.cuda.is_available()" 2>gpu_check.tmp
if exist gpu_check.tmp (
    findstr "sm_120" gpu_check.tmp >nul
    if !errorlevel! equ 0 (
        echo.
        echo NOTE: RTX 50-series GPU detected (RTX 5070 Ti, etc.)
        echo Your GPU has limited PyTorch support and will fall back to CPU processing
        echo This is expected behavior - the application will still work, just slower
        echo We prioritize stability over experimental PyTorch versions
    )
    del gpu_check.tmp
)

:: Create .env file template if it doesn't exist
if not exist ".env" (
    echo.
    echo Creating .env configuration template...
    (
        echo # Sovereign AI Agent Configuration
        echo # Uncomment and set values as needed
        echo.
        echo # Logging level
        echo TASKMASTER_LOG_LEVEL=INFO
        echo.
        echo # Ollama endpoint (if different from default)
        echo # OLLAMA_BASE_URL=http://localhost:11434
        echo.
        echo # API Keys (if using external services)
        echo # ANTHROPIC_API_KEY=your_key_here
        echo # OPENAI_API_KEY=your_key_here
        echo # PERPLEXITY_API_KEY=your_key_here
    ) > .env
    echo .env template created
)

:: Final instructions
echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Your development environment is ready to use.
echo.
echo To get started:
echo   1. Activate the environment: .venv\Scripts\activate
echo   2. Run the application: python main.py
echo   3. Or run CLI version: python -m sovereign.cli
echo.
echo For testing:
echo   - Run tests: python -m pytest tests/
echo   - Run with coverage: python -m pytest tests/ --cov=src/sovereign
echo.
echo See CONTRIBUTING.md for detailed development guidelines.
echo.

:: Ask if user wants to test the installation
echo Would you like to test the installation now? (Y/N)
set /p TEST_NOW=
if /i "!TEST_NOW!" equ "Y" (
    echo.
    echo Testing installation...
    python main.py --help >nul 2>&1
    if !errorlevel! equ 0 (
        echo Installation test PASSED - main.py is working
    ) else (
        echo Installation test WARNING - there may be issues
        echo Check the requirements and dependencies
    )
)

echo.
echo Environment setup complete! Press any key to exit.
pause >nul
exit /b 0 