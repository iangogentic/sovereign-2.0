"""
Setup script for Sovereign AI Agent
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="sovereign-ai-agent",
    version="1.0.0",
    author="Sovereign AI Team",
    author_email="team@sovereign-ai.dev",
    description="A private, powerful, locally-running AI assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sovereign-ai/sovereign",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
        "full": [
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sovereign=sovereign.cli:main",
            "sovereign-ai=sovereign.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "sovereign": ["*.json", "*.yaml", "*.yml"],
    },
    zip_safe=False,
    keywords="ai artificial-intelligence assistant local-ai privacy gpu cuda",
    project_urls={
        "Bug Reports": "https://github.com/sovereign-ai/sovereign/issues",
        "Source": "https://github.com/sovereign-ai/sovereign",
        "Documentation": "https://github.com/sovereign-ai/sovereign/docs",
    },
) 