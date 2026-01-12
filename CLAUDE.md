# Chronarch - Bitcoin Intelligence Platform

## Project Overview

Chronarch is a Bitcoin price forecasting and trading intelligence platform built with Python 3.11+, FastAPI, PyTorch, and LightGBM. The system ingests market data, generates features, trains ML models, and provides prediction APIs.

## Tech Stack

- **Backend**: Python 3.11-3.12, FastAPI, Uvicorn
- **Database**: PostgreSQL with asyncpg, SQLModel ORM, Alembic migrations
- **ML**: PyTorch (CUDA 12.x), LightGBM, scikit-learn
- **Data**: Pandas, NumPy
- **HTTP Client**: httpx (async)
- **Config**: pydantic-settings
- **Scheduling**: APScheduler
- **Logging**: structlog
- **Package Manager**: uv

## Project Structure

```
src/chronarch/
├── api/           # FastAPI application and routes
│   ├── main.py    # FastAPI app entry point
│   ├── routes/    # API endpoints (health, predictions)
│   └── schemas.py # Pydantic request/response models
├── backtest/      # Backtesting and validation
│   ├── metrics.py # Performance metrics (Sharpe, drawdown, etc.)
│   └── validator.py
├── data/          # Data layer
│   ├── ingestion/ # Data fetchers (binance.py, scheduler.py)
│   ├── models.py  # SQLModel database models
│   └── repository.py
├── db/            # Database connection management
├── features/      # Feature engineering
│   ├── pipeline.py
│   ├── statistics.py
│   └── technical.py  # Technical indicators (RSI, MACD, etc.)
├── ml/            # Machine learning
│   ├── dataset.py # Data loading and preparation
│   ├── model.py   # Model definitions
│   ├── predictor.py
│   └── trainer.py
└── config.py      # Settings via pydantic-settings
```

## Commands

```bash
# Install dependencies
uv sync

# Install with dev dependencies (JupyterLab, testing, linting)
uv sync --all-extras

# Verify CUDA
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run API server
uv run uvicorn chronarch.api.main:app --reload

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=chronarch

# Linting and formatting
uv run black src tests
uv run ruff check src tests
uv run mypy src

# Run JupyterLab
uv run jupyter lab

# Database migrations
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "description"
```

## Code Standards

- **Style**: Black (88 char line length), Ruff for linting
- **Types**: Full type hints, mypy strict mode
- **Tests**: pytest with pytest-asyncio for async tests
- **Docstrings**: Google style

## Configuration

Settings are loaded from environment variables or `.env` file. See `.env.example` for available options.

Key settings:
- `POSTGRES_*`: Database connection
- `BINANCE_*`: API credentials (optional for public endpoints)
- `APP_ENV`: development | staging | production

## Architecture Principles

- Hexagonal architecture: domain logic isolated from infrastructure
- Repository pattern for data access
- Dependency injection for testability
- EAFP (Easier to Ask Forgiveness than Permission) for error handling
- Async-first design with asyncpg and httpx

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Generate prediction
- `GET /predictions` - Query historical predictions

## Design Constraints

### DO:
- Quantify uncertainty in all predictions
- Use proper statistical validation
- Design for horizontal scalability
- Implement circuit breakers and failsafes
- Log all predictions for audit
- Use descriptive variable/function names
- Use dependency injection for loose coupling

### DO NOT:
- Claim guaranteed returns or accuracy
- Overfit to historical data
- Use bare `except:` or suppress errors silently
- Use mutable default arguments
- Use global variables unless necessary
- Store secrets in code
- Skip backtesting validation
