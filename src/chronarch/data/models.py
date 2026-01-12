"""Database models for market data and predictions."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class CandleInterval(str, Enum):
    """Supported candle intervals."""

    M1 = "1m"
    M5 = "5m"
    H1 = "1h"
    D1 = "1d"


class Direction(str, Enum):
    """Price direction classification."""

    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class Candle(SQLModel, table=True):
    """OHLCV candle data.

    This table is designed to be a TimescaleDB hypertable partitioned by timestamp.
    """

    __tablename__ = "candles"

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    interval: CandleInterval = Field(index=True)
    timestamp: datetime = Field(index=True)
    open: Decimal = Field(decimal_places=8, max_digits=20)
    high: Decimal = Field(decimal_places=8, max_digits=20)
    low: Decimal = Field(decimal_places=8, max_digits=20)
    close: Decimal = Field(decimal_places=8, max_digits=20)
    volume: Decimal = Field(decimal_places=8, max_digits=30)
    quote_volume: Decimal = Field(decimal_places=8, max_digits=30)
    num_trades: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class Prediction(SQLModel, table=True):
    """Model predictions with confidence intervals."""

    __tablename__ = "predictions"

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    timestamp: datetime = Field(index=True)  # When prediction was made
    target_timestamp: datetime = Field(index=True)  # What time is being predicted
    horizon_hours: int  # Prediction horizon

    # Direction prediction
    direction: Direction
    direction_confidence: float = Field(ge=0.0, le=1.0)
    prob_up: float = Field(ge=0.0, le=1.0)
    prob_down: float = Field(ge=0.0, le=1.0)
    prob_flat: float = Field(ge=0.0, le=1.0)

    # Price prediction with confidence interval
    price_predicted: Decimal = Field(decimal_places=2, max_digits=20)
    price_lower_10: Decimal = Field(decimal_places=2, max_digits=20)  # 10th percentile
    price_upper_90: Decimal = Field(decimal_places=2, max_digits=20)  # 90th percentile

    # Current price at prediction time
    price_current: Decimal = Field(decimal_places=2, max_digits=20)

    # Model metadata
    model_version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Outcome tracking (filled in after target time passes)
    actual_price: Decimal | None = Field(
        default=None, decimal_places=2, max_digits=20
    )
    actual_direction: Direction | None = None
    was_direction_correct: bool | None = None
    was_in_confidence_interval: bool | None = None

    class Config:
        use_enum_values = True


class FeatureSnapshot(SQLModel, table=True):
    """Cached feature values for a given timestamp."""

    __tablename__ = "feature_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    timestamp: datetime = Field(index=True)
    interval: CandleInterval

    # Store features as JSON for flexibility
    features: dict = Field(default_factory=dict, sa_type=None)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True
