# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Setup script for vLLM instrumentation package."""

from setuptools import setup, find_packages

# Read the README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vllm-instrumentation",
    version="0.1.0",
    description="OpenTelemetry instrumentation for vLLM components",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="vLLM Contributors",
    author_email="",
    url="https://github.com/vllm-project/vllm",
    packages=find_packages(),
    install_requires=[
        "opentelemetry-api>=1.20.0",
        "opentelemetry-sdk>=1.20.0",
        "opentelemetry-instrumentation>=0.41b0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "opentelemetry_instrumentor": [
            "io_processor = vllm.instrumentation.io_processor_instrumentor:IOProcessorInstrumentor",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
    ],
    keywords="vllm opentelemetry instrumentation tracing metrics observability",
    project_urls={
        "Bug Reports": "https://github.com/vllm-project/vllm/issues",
        "Source": "https://github.com/vllm-project/vllm",
        "Documentation": "https://docs.vllm.ai",
    },
)

# Made with Bob
