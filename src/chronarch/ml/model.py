"""LightGBM model wrappers for classification and quantile regression."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np


@dataclass
class DirectionPrediction:
    """Prediction result for direction classification."""

    direction: Literal["up", "down", "flat"]
    confidence: float
    prob_up: float
    prob_down: float
    prob_flat: float


@dataclass
class PricePrediction:
    """Prediction result for price regression with confidence interval."""

    predicted: float
    lower_10: float  # 10th percentile
    upper_90: float  # 90th percentile


class DirectionClassifier:
    """LightGBM classifier for price direction prediction."""

    DEFAULT_PARAMS = {
        "objective": "multiclass",
        "num_class": 3,  # up, down, flat
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    # Map class indices to labels
    CLASS_LABELS = {0: "flat", 1: "up", -1: "down"}
    LABEL_TO_IDX = {"flat": 0, "up": 1, "down": 2}

    def __init__(self, params: dict | None = None) -> None:
        """Initialize classifier.

        Args:
            params: LightGBM parameters (merged with defaults)
        """
        self._params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._model: lgb.Booster | None = None
        self._feature_names: list[str] | None = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
    ) -> dict:
        """Train the classifier.

        Args:
            X_train: Training features
            y_train: Training labels (-1, 0, 1 for down/flat/up)
            X_val: Validation features
            y_val: Validation labels
            feature_names: Feature names for interpretability
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Rounds without improvement before stopping

        Returns:
            Training history dict
        """
        # Convert labels to indices (0, 1, 2)
        y_train_idx = self._labels_to_indices(y_train)

        train_data = lgb.Dataset(
            X_train,
            label=y_train_idx,
            feature_name=feature_names or "auto",
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            y_val_idx = self._labels_to_indices(y_val)
            val_data = lgb.Dataset(
                X_val,
                label=y_val_idx,
                reference=train_data,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")

        callbacks = [lgb.early_stopping(early_stopping_rounds)]
        history: dict = {}

        self._model = lgb.train(
            self._params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self._feature_names = feature_names

        return history

    def predict(self, X: np.ndarray) -> list[DirectionPrediction]:
        """Generate direction predictions.

        Args:
            X: Feature matrix

        Returns:
            List of DirectionPrediction objects
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Get probability predictions
        probs = self._model.predict(X)

        predictions = []
        for prob_row in probs:
            prob_flat, prob_up, prob_down = prob_row

            # Get predicted class
            pred_idx = np.argmax(prob_row)
            direction = ["flat", "up", "down"][pred_idx]
            confidence = prob_row[pred_idx]

            predictions.append(
                DirectionPrediction(
                    direction=direction,
                    confidence=float(confidence),
                    prob_up=float(prob_up),
                    prob_down=float(prob_down),
                    prob_flat=float(prob_flat),
                )
            )

        return predictions

    def _labels_to_indices(self, y: np.ndarray) -> np.ndarray:
        """Convert labels (-1, 0, 1) to indices (0, 1, 2)."""
        # -1 (down) -> 2, 0 (flat) -> 0, 1 (up) -> 1
        return np.where(y == -1, 2, y)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dict mapping feature names to importance scores
        """
        if self._model is None:
            raise RuntimeError("Model not trained.")

        importance = self._model.feature_importance(importance_type="gain")
        names = self._feature_names or [f"f{i}" for i in range(len(importance))]

        return dict(zip(names, importance.tolist()))

    def save(self, path: Path) -> None:
        """Save model to file.

        Args:
            path: Path to save model
        """
        if self._model is None:
            raise RuntimeError("Model not trained.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._model.save_model(str(path))

        # Save metadata
        metadata = {
            "params": self._params,
            "feature_names": self._feature_names,
        }
        with open(path.with_suffix(".meta.json"), "w") as f:
            json.dump(metadata, f)

    def load(self, path: Path) -> None:
        """Load model from file.

        Args:
            path: Path to model file
        """
        path = Path(path)

        self._model = lgb.Booster(model_file=str(path))

        # Load metadata
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            self._params = metadata.get("params", self.DEFAULT_PARAMS)
            self._feature_names = metadata.get("feature_names")


class QuantileRegressor:
    """LightGBM quantile regressor for price prediction with confidence interval."""

    DEFAULT_PARAMS = {
        "objective": "quantile",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    QUANTILES = [0.1, 0.5, 0.9]  # 10th, 50th (median), 90th percentiles

    def __init__(self, params: dict | None = None) -> None:
        """Initialize regressor.

        Args:
            params: LightGBM parameters (merged with defaults)
        """
        self._base_params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._models: dict[float, lgb.Booster] = {}
        self._feature_names: list[str] | None = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
    ) -> None:
        """Train quantile regressors for each quantile.

        Args:
            X_train: Training features
            y_train: Training target (future price or return)
            X_val: Validation features
            y_val: Validation target
            feature_names: Feature names for interpretability
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Rounds without improvement before stopping
        """
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=feature_names or "auto",
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_data,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")

        callbacks = [lgb.early_stopping(early_stopping_rounds)]

        for quantile in self.QUANTILES:
            params = {
                **self._base_params,
                "alpha": quantile,
            }

            self._models[quantile] = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )

        self._feature_names = feature_names

    def predict(self, X: np.ndarray) -> list[PricePrediction]:
        """Generate price predictions with confidence interval.

        Args:
            X: Feature matrix

        Returns:
            List of PricePrediction objects
        """
        if not self._models:
            raise RuntimeError("Model not trained. Call train() first.")

        # Get predictions for each quantile
        pred_10 = self._models[0.1].predict(X)
        pred_50 = self._models[0.5].predict(X)
        pred_90 = self._models[0.9].predict(X)

        predictions = []
        for p10, p50, p90 in zip(pred_10, pred_50, pred_90):
            predictions.append(
                PricePrediction(
                    predicted=float(p50),
                    lower_10=float(p10),
                    upper_90=float(p90),
                )
            )

        return predictions

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores (averaged across quantiles).

        Returns:
            Dict mapping feature names to importance scores
        """
        if not self._models:
            raise RuntimeError("Model not trained.")

        # Average importance across quantile models
        all_importance = []
        for model in self._models.values():
            all_importance.append(model.feature_importance(importance_type="gain"))

        avg_importance = np.mean(all_importance, axis=0)
        names = self._feature_names or [f"f{i}" for i in range(len(avg_importance))]

        return dict(zip(names, avg_importance.tolist()))

    def save(self, path: Path) -> None:
        """Save models to files.

        Args:
            path: Base path for model files
        """
        if not self._models:
            raise RuntimeError("Model not trained.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        for quantile, model in self._models.items():
            model_path = path.with_suffix(f".q{int(quantile * 100)}.txt")
            model.save_model(str(model_path))

        # Save metadata
        metadata = {
            "params": self._base_params,
            "feature_names": self._feature_names,
            "quantiles": self.QUANTILES,
        }
        with open(path.with_suffix(".meta.json"), "w") as f:
            json.dump(metadata, f)

    def load(self, path: Path) -> None:
        """Load models from files.

        Args:
            path: Base path for model files
        """
        path = Path(path)

        # Load metadata
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            self._base_params = metadata.get("params", self.DEFAULT_PARAMS)
            self._feature_names = metadata.get("feature_names")
            quantiles = metadata.get("quantiles", self.QUANTILES)
        else:
            quantiles = self.QUANTILES

        # Load quantile models
        for quantile in quantiles:
            model_path = path.with_suffix(f".q{int(quantile * 100)}.txt")
            self._models[quantile] = lgb.Booster(model_file=str(model_path))
