"""Inference interface for generating predictions."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import structlog

from chronarch.data.models import Direction, Prediction
from chronarch.features.pipeline import FeaturePipeline
from chronarch.ml.model import DirectionClassifier, QuantileRegressor

logger = structlog.get_logger()


@dataclass
class PredictionResult:
    """Complete prediction result combining direction and price forecasts."""

    symbol: str
    timestamp: datetime
    target_timestamp: datetime
    horizon_hours: int

    # Direction prediction
    direction: Literal["up", "down", "flat"]
    direction_confidence: float
    prob_up: float
    prob_down: float
    prob_flat: float

    # Price prediction
    price_predicted: float
    price_lower_10: float
    price_upper_90: float

    # Current state
    price_current: float
    model_version: str


class Predictor:
    """Generates predictions using trained models."""

    def __init__(
        self,
        model_dir: Path,
        model_version: str | None = None,
        horizon_hours: int = 1,
        direction_threshold: float = 0.001,
    ) -> None:
        """Initialize predictor.

        Args:
            model_dir: Directory containing trained models
            model_version: Specific version to load (default: "latest")
            horizon_hours: Prediction horizon in hours
            direction_threshold: Threshold for direction classification
        """
        self._model_dir = Path(model_dir)
        self._horizon_hours = horizon_hours
        self._direction_threshold = direction_threshold

        self._classifier = DirectionClassifier()
        self._regressor = QuantileRegressor()
        self._model_version: str = ""
        self._feature_pipeline = FeaturePipeline(
            include_target=False,
            direction_threshold=direction_threshold,
        )

        self._load_models(model_version)

    def _load_models(self, version: str | None) -> None:
        """Load model files.

        Args:
            version: Model version or None for latest
        """
        if version is None:
            version_path = self._model_dir / "latest"
            if version_path.is_symlink():
                version = version_path.resolve().name
            else:
                raise FileNotFoundError(
                    f"No 'latest' model found in {self._model_dir}"
                )

        version_dir = self._model_dir / version

        self._classifier.load(version_dir / "direction.txt")
        self._regressor.load(version_dir / "price.txt")
        self._model_version = version

        logger.info("models_loaded", version=version)

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> PredictionResult:
        """Generate prediction from OHLCV data.

        Args:
            df: OHLCV DataFrame indexed by timestamp
            symbol: Trading pair symbol

        Returns:
            PredictionResult with direction and price forecasts
        """
        # Generate features
        features_df = self._feature_pipeline.transform(df)
        features_df = FeaturePipeline.drop_na_rows(features_df)

        if features_df.empty:
            raise ValueError("Insufficient data to generate features")

        # Get latest feature vector
        latest_features = features_df.iloc[-1:].values
        latest_timestamp = features_df.index[-1]

        # Get current price
        current_price = float(df["close"].iloc[-1])

        # Generate predictions
        dir_pred = self._classifier.predict(latest_features)[0]
        price_pred = self._regressor.predict(latest_features)[0]

        # Convert price predictions from returns to actual prices
        # (if model predicts returns, adjust here)
        predicted_price = price_pred.predicted
        lower_10 = price_pred.lower_10
        upper_90 = price_pred.upper_90

        # If predictions are returns, convert to prices
        if abs(predicted_price) < 0.5:  # Likely a return, not a price
            predicted_price = current_price * (1 + predicted_price)
            lower_10 = current_price * (1 + lower_10)
            upper_90 = current_price * (1 + upper_90)

        target_timestamp = latest_timestamp + timedelta(hours=self._horizon_hours)

        return PredictionResult(
            symbol=symbol,
            timestamp=latest_timestamp,
            target_timestamp=target_timestamp,
            horizon_hours=self._horizon_hours,
            direction=dir_pred.direction,
            direction_confidence=dir_pred.confidence,
            prob_up=dir_pred.prob_up,
            prob_down=dir_pred.prob_down,
            prob_flat=dir_pred.prob_flat,
            price_predicted=predicted_price,
            price_lower_10=lower_10,
            price_upper_90=upper_90,
            price_current=current_price,
            model_version=self._model_version,
        )

    def to_db_prediction(self, result: PredictionResult) -> Prediction:
        """Convert PredictionResult to database Prediction model.

        Args:
            result: PredictionResult to convert

        Returns:
            Prediction database model
        """
        direction_map = {
            "up": Direction.UP,
            "down": Direction.DOWN,
            "flat": Direction.FLAT,
        }

        return Prediction(
            symbol=result.symbol,
            timestamp=result.timestamp,
            target_timestamp=result.target_timestamp,
            horizon_hours=result.horizon_hours,
            direction=direction_map[result.direction],
            direction_confidence=result.direction_confidence,
            prob_up=result.prob_up,
            prob_down=result.prob_down,
            prob_flat=result.prob_flat,
            price_predicted=Decimal(str(round(result.price_predicted, 2))),
            price_lower_10=Decimal(str(round(result.price_lower_10, 2))),
            price_upper_90=Decimal(str(round(result.price_upper_90, 2))),
            price_current=Decimal(str(round(result.price_current, 2))),
            model_version=result.model_version,
        )

    @property
    def model_version(self) -> str:
        """Get current model version."""
        return self._model_version

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from direction classifier.

        Returns:
            Dict mapping feature names to importance scores
        """
        return self._classifier.get_feature_importance()
