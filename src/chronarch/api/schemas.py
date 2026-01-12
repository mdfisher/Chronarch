"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    database: bool
    model_loaded: bool
    last_data_update: datetime | None = None


class PredictionRequest(BaseModel):
    """Request to generate a new prediction."""

    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    horizon_hours: int = Field(default=1, ge=1, le=24, description="Prediction horizon")


class DirectionPredictionResponse(BaseModel):
    """Direction prediction component of response."""

    direction: Literal["up", "down", "flat"]
    confidence: float = Field(ge=0, le=1)
    prob_up: float = Field(ge=0, le=1)
    prob_down: float = Field(ge=0, le=1)
    prob_flat: float = Field(ge=0, le=1)


class PricePredictionResponse(BaseModel):
    """Price prediction component of response."""

    predicted: float = Field(description="Predicted price (median)")
    lower_10: float = Field(description="10th percentile (lower bound)")
    upper_90: float = Field(description="90th percentile (upper bound)")
    current: float = Field(description="Current price at prediction time")


class PredictionResponse(BaseModel):
    """Complete prediction response."""

    id: int | None = Field(default=None, description="Database ID if saved")
    symbol: str
    timestamp: datetime = Field(description="When prediction was made")
    target_timestamp: datetime = Field(description="Time being predicted")
    horizon_hours: int

    direction: DirectionPredictionResponse
    price: PricePredictionResponse

    model_version: str
    created_at: datetime


class PredictionListResponse(BaseModel):
    """Response for listing multiple predictions."""

    predictions: list[PredictionResponse]
    total: int
    page: int
    page_size: int


class FeatureImportanceResponse(BaseModel):
    """Feature importance response."""

    features: dict[str, float] = Field(
        description="Map of feature names to importance scores"
    )
    model_version: str


class ModelConfidenceResponse(BaseModel):
    """Model confidence and calibration metrics."""

    model_version: str
    direction_accuracy_recent: float | None = Field(
        default=None, description="Direction accuracy over last N predictions"
    )
    ci_coverage_recent: float | None = Field(
        default=None, description="CI coverage over last N predictions"
    )
    predictions_evaluated: int = Field(
        description="Number of predictions used for metrics"
    )


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    detail: str | None = None
