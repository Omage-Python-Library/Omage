from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8") if (Path(__file__).parent / "README.md").exists() else ""

setup(
    name="omage",
    version="0.1.1",
    description="AI-First Python Library — Build AI the way you think",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Omage Team",
    author_email="",
    url="https://github.com/your-username/omage",
    packages=find_packages(exclude=["tests*", "examples*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "vision":  ["torchvision>=0.15.0"],
        "nlp":     ["transformers>=4.30.0"],
        "detect":  ["ultralytics>=8.0.0"],
        "full":    ["torchvision>=0.15.0", "transformers>=4.30.0", "ultralytics>=8.0.0"],
        "dev":     ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "omage=omage.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="ai machine-learning deep-learning pytorch neural-network",
)
