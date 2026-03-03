#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Build script for vllm-instrumentation package

set -e

echo "Building vllm-instrumentation package..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info

# Ensure build tools are installed
echo "Installing/upgrading build tools..."
#python -m uv pip install --upgrade pip build wheel setuptools
uv pip install --upgrade pip build wheel setuptools

# Build the package
echo "Building distribution packages..."
python -m build

echo ""
echo "Build complete! Distribution packages are in the dist/ directory."
echo ""
echo "Files created:"
ls -lh dist/
echo ""
echo "To install locally:"
echo "  pip install dist/vllm_instrumentation-*.whl"
echo ""
echo "To upload to PyPI:"
echo "  python -m pip install --upgrade twine"
echo "  python -m twine upload dist/*"

# Made with Bob
