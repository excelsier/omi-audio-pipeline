#!/usr/bin/env python3
"""
OMI Audio Pipeline - Setup Script
--------------------------------
Setup script for the OMI Audio Pipeline package.
"""
from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read the requirements file
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f.readlines() if line.strip()]

setup(
    name="omi-audio-pipeline",
    version="0.1.0",
    author="OMI Team",
    author_email="info@omi.ai",
    description="Audio processing pipeline with speaker diarization and transcription",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/excelsier/omi-audio-pipeline",
    project_urls={
        "Bug Tracker": "https://github.com/excelsier/omi-audio-pipeline/issues",
        "Documentation": "https://github.com/excelsier/omi-audio-pipeline#readme",
        "Source Code": "https://github.com/excelsier/omi-audio-pipeline",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "isort>=5.9.1",
            "flake8>=3.9.2",
            "mypy>=0.812",
        ],
        "docs": [
            "sphinx>=4.0.2",
            "sphinx-rtd-theme>=0.5.2",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "omi-audio=omi_audio.cli.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
