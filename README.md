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
