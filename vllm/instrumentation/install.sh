#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Installation script for vllm-instrumentation package

set -e

echo "Installing vllm-instrumentation package..."

# Check if we're in the correct directory
if [ ! -f "setup.py" ]; then
    echo "Error: setup.py not found. Please run this script from the vllm/instrumentation directory."
    exit 1
fi

# Install in development mode
echo "Installing in development mode (editable install)..."
pip install -e .

echo ""
echo "Installation complete!"
echo ""
echo "To install with development dependencies:"
echo "  pip install -e .[dev]"
echo ""
echo "To verify installation:"
echo "  python -c 'from vllm.instrumentation import IOProcessorInstrumentor; print(\"Success!\")'"

# Made with Bob
