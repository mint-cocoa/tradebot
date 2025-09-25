"""
Setup script for Crypto-DLSA Bot
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="crypto-dlsa-bot",
    version="0.1.0",
    author="Crypto-DLSA Team",
    author_email="team@crypto-dlsa.com",
    description="Deep Learning Statistical Arbitrage Bot for Cryptocurrency Markets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/crypto-dlsa/crypto-dlsa-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "pre-commit>=2.20.0",
        ],
        "gpu": [
            "torch>=1.12.0+cu116",
            "torchvision>=0.13.0+cu116",
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto-dlsa=crypto_dlsa_bot.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "crypto_dlsa_bot": ["config/*.yaml", "config/*.json"],
    },
)