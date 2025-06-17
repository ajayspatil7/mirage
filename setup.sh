#!/bin/bash

# 🔥 Syntheta Mirage - ML Engine Setup Script
# Run from: Syntheta/mirage/
# Usage: chmod +x setup.sh && ./setup.sh

set -e  # Exit on any error

echo "🚀 Setting up Syntheta Mirage ML Engine..."
echo "================================================"

# Create main project structure
echo "📁 Creating directory structure..."

# Root level files
touch README.md
touch .env.example
touch .gitignore
touch run.py

# Main mirage package
mkdir -p mirage
touch mirage/__init__.py
touch mirage/main.py

# Core infrastructure
mkdir -p mirage/core
touch mirage/core/__init__.py
touch mirage/core/base.py
touch mirage/core/config.py
touch mirage/core/exceptions.py

# Model system
mkdir -p mirage/models/tabular
mkdir -p mirage/models/vision
touch mirage/models/__init__.py
touch mirage/models/registry.py
touch mirage/models/tabular/__init__.py
touch mirage/models/tabular/ctgan.py
touch mirage/models/tabular/tvae.py
touch mirage/models/tabular/copulagan.py
touch mirage/models/vision/__init__.py
touch mirage/models/vision/diffusion.py

# Data pipeline
mkdir -p mirage/data
touch mirage/data/__init__.py
touch mirage/data/loaders.py
touch mirage/data/preprocessors.py
touch mirage/data/validators.py

# Execution engine
mkdir -p mirage/engine
touch mirage/engine/__init__.py
touch mirage/engine/executor.py
touch mirage/engine/queue.py
touch mirage/engine/isolation.py

# API layer
mkdir -p mirage/api/v1
mkdir -p mirage/api/schemas
touch mirage/api/__init__.py
touch mirage/api/v1/__init__.py
touch mirage/api/v1/generate.py
touch mirage/api/v1/models.py
touch mirage/api/v1/jobs.py
touch mirage/api/schemas/__init__.py
touch mirage/api/schemas/requests.py
touch mirage/api/schemas/responses.py

# Metrics system
mkdir -p mirage/metrics
touch mirage/metrics/__init__.py
touch mirage/metrics/tabular.py
touch mirage/metrics/vision.py
touch mirage/metrics/common.py

# Utilities
mkdir -p mirage/utils
touch mirage/utils/__init__.py
touch mirage/utils/logger.py
touch mirage/utils/storage.py
touch mirage/utils/mps_utils.py
touch mirage/utils/health.py

# Storage directories
mkdir -p storage/models
mkdir -p storage/datasets
mkdir -p storage/outputs
mkdir -p storage/logs

# Testing structure
mkdir -p tests/unit
mkdir -p tests/integration
touch tests/__init__.py
touch tests/conftest.py
touch tests/unit/test_models.py
touch tests/unit/test_data.py
touch tests/unit/test_engine.py
touch tests/integration/test_api.py
touch tests/integration/test_e2e.py

# Scripts
mkdir -p scripts
touch scripts/setup.py
touch scripts/benchmark.py
touch scripts/demo.py

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[tool.poetry]
name = "mirage"
version = "0.1.0"
description = "Syntheta ML Engine - Production-grade synthetic data generation"
authors = ["Syntheta Team <team@syntheta.ai>"]
readme = "README.md"
packages = [{include = "mirage"}]

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.104.1"
uvicorn = {extras = ["standard"], version = "^0.24.0"}
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"
loguru = "^0.7.2"
httpx = "^0.25.2"
python-multipart = "^0.0.6"
# ML/DL Dependencies
torch = "^2.1.0"
torchvision = "^0.16.0"
numpy = "^1.24.3"
pandas = "^2.1.0"
scikit-learn = "^1.3.0"
# Synthetic Data
sdv = "^1.10.0"
# Metrics
scipy = "^1.11.0"
# Development
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
black = "^23.0.0"
isort = "^5.12.0"
mypy = "^1.6.0"

[tool.poetry.group.dev.dependencies]
pytest-cov = "^4.1.0"
pre-commit = "^3.5.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
asyncio_mode = "auto"
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
.venv/

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# Environment variables
.env
.env.local
.env.*.local

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# MacOS
.DS_Store
.AppleDouble
.LSOverride
Icon
._*

# Storage (local data)
storage/datasets/*
storage/outputs/*
storage/logs/*
!storage/datasets/.gitkeep
!storage/outputs/.gitkeep
!storage/logs/.gitkeep

# Model artifacts (large files)
storage/models/*
!storage/models/.gitkeep

# Temporary files
*.tmp
*.temp
.temp/
EOF

# Create .env.example
cat > .env.example << 'EOF'
# Mirage ML Engine Configuration

# Environment
ENVIRONMENT=development
DEBUG=true

# API Configuration
HOST=0.0.0.0
PORT=8001
WORKERS=1

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Storage
STORAGE_PATH=./storage
MAX_UPLOAD_SIZE_MB=100
CLEANUP_AFTER_HOURS=24

# Model Configuration
DEFAULT_MODEL=ctgan
MODEL_CACHE_SIZE=3
MAX_CONCURRENT_JOBS=5

# Hardware Optimization (M4 Pro)
USE_MPS=true
MPS_FALLBACK=true
TORCH_THREADS=8

# Development
RELOAD=true
API_PREFIX=/api/v1
EOF

# Create gitkeep files for storage directories
touch storage/models/.gitkeep
touch storage/datasets/.gitkeep
touch storage/outputs/.gitkeep
touch storage/logs/.gitkeep

# Create basic README
cat > README.md << 'EOF'
# 🔥 Mirage - Syntheta ML Engine

Production-grade synthetic data generation engine optimized for Apple Silicon M4 Pro.

## Quick Start

```bash
# Install dependencies
poetry install

# Set up environment
cp .env.example .env

# Run the service
poetry run python run.py
```

## Features

- 🚀 **Fast**: Optimized for Apple Silicon M4 Pro with MPS acceleration
- 🏥 **Healthcare-Ready**: HIPAA-compliant synthetic medical data generation
- 📊 **Multiple Models**: CTGAN, TVAE, CopulaGAN support
- 🔧 **Production-Grade**: Async job processing, metrics, monitoring
- 🎯 **API-First**: Complete REST API with OpenAPI documentation

## Architecture

```
mirage/
├── core/           # Base classes and configuration
├── models/         # ML model implementations
├── data/           # Data loading and preprocessing
├── engine/         # Job execution and queue management
├── api/            # FastAPI routes and schemas
├── metrics/        # Evaluation metrics
└── utils/          # Utilities and helpers
```

## Next Steps

1. Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
2. Run setup: `poetry install`
3. Start development: `poetry run python run.py`

For detailed documentation, see the `/docs` directory.
EOF

echo ""
echo "✅ Project structure created successfully!"
echo ""
echo "📁 Directory tree:"
find . -type d | head -20 | sed 's/^/  /'
echo "   ... and more"
echo ""
echo "🎯 Next steps:"
echo "  1. Install Poetry (if not installed): curl -sSL https://install.python-poetry.org | python3 -"
echo "  2. Install dependencies: poetry install"
echo "  3. Copy environment: cp .env.example .env"
echo "  4. Ready for Day 2: Core implementation!"
echo ""
echo "🚀 Mirage ML Engine scaffold ready!"