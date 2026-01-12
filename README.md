# Chronarch

Bitcoin price forecasting and trading intelligence platform.

## Overview

Chronarch ingests market data from multiple sources, generates technical and statistical features, trains ML models (PyTorch, LightGBM), and provides prediction APIs with quantified uncertainty.

## Tech Stack

- **Backend**: Python 3.11-3.12, FastAPI, Uvicorn
- **Database**: PostgreSQL, asyncpg, SQLModel, Alembic
- **ML**: PyTorch (CUDA 12.x), LightGBM, scikit-learn
- **Data**: Pandas, NumPy, httpx
- **Package Manager**: uv

## Requirements

- Python 3.11 - 3.12
- PostgreSQL
- CUDA 12.x (for GPU acceleration)

## Installation

```bash
uv sync
```

With dev dependencies (includes JupyterLab, testing, linting):

```bash
uv sync --all-extras
```

## Verify CUDA

```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key settings:
- `POSTGRES_*` - Database connection
- `BINANCE_*` - API credentials (optional for public endpoints)
- `APP_ENV` - development | staging | production

## Development

Run the API server:

```bash
uv run uvicorn chronarch.api.main:app --reload
```

Run JupyterLab:

```bash
uv run jupyter lab
```

Run tests:

```bash
uv run pytest
uv run pytest --cov=chronarch  # with coverage
```

Run linting:

```bash
uv run ruff check src/
uv run black --check src/
uv run mypy src/
```

Database migrations:

```bash
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "description"
```

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Generate prediction
- `GET /predictions` - Query historical predictions

## Project Structure

```
chronarch/
├── src/chronarch/
│   ├── api/            # FastAPI routes and schemas
│   ├── backtest/       # Backtesting and validation
│   ├── data/           # Data models, repository, ingestion
│   ├── db/             # Database connection
│   ├── features/       # Feature engineering pipeline
│   ├── ml/             # ML models, training, inference
│   └── config.py       # Configuration
├── notebooks/          # Jupyter notebooks for exploration
├── tests/              # Test suite
├── alembic/            # Database migrations
└── docker/             # Docker configuration
```

## License

Proprietary - All rights reserved.
