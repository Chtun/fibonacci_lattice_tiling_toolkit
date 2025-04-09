"""Package configuration and setup for spherical tiling and heatmap visualization.

This module contains the package configuration for the spherical tiling and heatmap visualization
system, including dependencies, version information, and package metadata.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Package requirements
REQUIREMENTS = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "matplotlib>=3.4.0",
    "scipy>=1.7.0",
]

# Development requirements
DEV_REQUIREMENTS = [
    "pytest>=6.0.0",
    "pytest-cov>=2.0.0",
    "black>=21.0",
    "isort>=5.0.0",
    "flake8>=3.9.0",
    "mypy>=0.900",
]

# Documentation requirements
DOCS_REQUIREMENTS = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=0.5.0",
    "myst-parser>=0.15.0",
]

setup(
    name="fibonacci_lattice_tiling_toolkit",
    version="0.1.0",
    description="A system for analyzing tiling and heatmap visualizations on the sphere.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chitsein Htun",
    author_email="chtun@live.com",
    url="https://github.com/Chtun/fibonacci_lattice_tiling_toolkit",
    packages=find_packages(exclude=["tests*", "examples*"]),
    
    # Package dependencies
    install_requires=REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS,
        "docs": DOCS_REQUIREMENTS,
        "all": DEV_REQUIREMENTS + DOCS_REQUIREMENTS,
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    
    # Package data
    include_package_data=True,
    package_data={
        "fibonacci_lattice_tiling_toolkit": [
            "data/test_data/*.csv",
            "data/example_data/*.csv",
        ]
    },
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "fibonacci_lattice_tiling_toolkit=fibonacci_lattice_tiling_toolkit.cli:main",
        ]
    },
    
    # Keywords for PyPI
    keywords=[
        "fibonacci-lattice",
        "360-video",
        "spherical-tiling",
        "heatmap",
        "visualization",
    ],
    
    # Additional metadata
    platforms=["any"],
    license="MIT",
)
