"""
Reuko-1B Setup Script
Kolay kurulum ve dağıtım için
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="reuko-1b",
    version="0.1.0",
    author="Rei Calasso",
    author_email="rei@example.com",
    description="Mini T5 Pipeline for QA and Summarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/reuko-1B",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
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
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "reuko-train=reuko_1b.cli.train:main",
            "reuko-test=reuko_1b.cli.test:main",
        ],
    },
    include_package_data=True,
    package_data={
        "reuko_1b": ["*.json", "*.yaml", "*.yml"],
    },
)
