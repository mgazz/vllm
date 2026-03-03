# Build Instructions for vllm-instrumentation

This document provides instructions for building and distributing the vllm-instrumentation package.

## Prerequisites

- Python 3.9 or higher
- pip (latest version recommended)
- build tools: `pip install --upgrade build twine`

## Quick Start

### Option 1: Using the build script (Recommended)

```bash
cd vllm/instrumentation
./build.sh
```

This will:
1. Clean previous builds
2. Build both source distribution (.tar.gz) and wheel (.whl) packages
3. Place the built packages in the `dist/` directory

### Option 2: Manual build

```bash
cd vllm/instrumentation

# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Install build tools
python -m pip install --upgrade build

# Build the package
python -m build
```

## Installation

### Install from local build

After building, you can install the package locally:

```bash
# Install the wheel package
pip install dist/vllm_instrumentation-*.whl

# Or install in development/editable mode
pip install -e .

# Or use the install script
./install.sh
```

### Install with development dependencies

```bash
pip install -e .[dev]
```

## Verify Installation

```bash
python -c "from vllm.instrumentation import IOProcessorInstrumentor; print('Success!')"
```

## Publishing to PyPI

### Test PyPI (Recommended for testing)

```bash
# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# Install from Test PyPI to verify
pip install --index-url https://test.pypi.org/simple/ vllm-instrumentation
```

### Production PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*
```

You'll need PyPI credentials or an API token. Configure them in `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
username = __token__
password = pypi-your-test-api-token-here
```

## Package Structure

```
vllm/instrumentation/
├── __init__.py                    # Package initialization
├── io_processor_instrumentor.py  # Main instrumentor implementation
├── README.md                      # Package documentation
├── setup.py                       # Setup configuration (legacy)
├── pyproject.toml                 # Modern build configuration
├── MANIFEST.in                    # Package manifest
├── build.sh                       # Build script
├── install.sh                     # Installation script
├── .gitignore                     # Git ignore rules
└── BUILD_INSTRUCTIONS.md          # This file
```

## Build Outputs

After building, you'll find:

- `dist/vllm_instrumentation-0.1.0.tar.gz` - Source distribution
- `dist/vllm_instrumentation-0.1.0-py3-none-any.whl` - Wheel distribution
- `build/` - Temporary build directory (can be deleted)
- `vllm_instrumentation.egg-info/` - Package metadata (can be deleted)

## Troubleshooting

### Build fails with "No module named 'build'"

```bash
pip install --upgrade build
```

### Permission denied when running scripts

```bash
chmod +x build.sh install.sh
```

### Import errors after installation

Make sure you're not in the source directory when testing:

```bash
cd /tmp
python -c "from vllm.instrumentation import IOProcessorInstrumentor"
```

### Package not found after installation

Check if it's installed:

```bash
pip list | grep vllm-instrumentation
```

Reinstall if needed:

```bash
pip uninstall vllm-instrumentation
pip install dist/vllm_instrumentation-*.whl
```

## Development Workflow

1. Make changes to the code
2. Test locally: `pip install -e .`
3. Run tests (if available): `pytest tests/`
4. Build: `./build.sh`
5. Test the built package: `pip install dist/vllm_instrumentation-*.whl`
6. Publish to Test PyPI for validation
7. Publish to PyPI when ready

## Version Management

Update the version in both:
- `setup.py` (line 13)
- `pyproject.toml` (line 10)

Follow semantic versioning: MAJOR.MINOR.PATCH

## License

SPDX-License-Identifier: Apache-2.0