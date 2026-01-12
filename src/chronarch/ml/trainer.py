"""Model training orchestration."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import structlog

from chronarch.ml.dataset import TrainTestSplit, create_dataset
from chronarch.ml.model import DirectionClassifier, QuantileRegressor

logger = structlog.get_logger()


@dataclass
class TrainingResult:
    """Results from model training."""

    direction_accuracy: float
    direction_precision: dict[str, float]
    direction_recall: dict[str, float]
    price_mae: float
    price_coverage: float  # % of actuals within CI
    feature_importance: dict[str, float]
    model_version: str
    trained_at: datetime
    train_samples: int
    test_samples: int


class ModelTrainer:
    """Orchestrates training of direction and price models."""

    def __init__(
        self,
        model_dir: Path,
        classifier_params: dict | None = None,
        regressor_params: dict | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model_dir: Directory to save trained models
            classifier_params: LightGBM params for classifier
            regressor_params: LightGBM params for regressor
        """
        self._model_dir = Path(model_dir)
        self._classifier = DirectionClassifier(classifier_params)
        self._regressor = QuantileRegressor(regressor_params)

    def train(
        self,
        split: TrainTestSplit,
        target_prices: np.ndarray,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
    ) -> TrainingResult:
        """Train both direction and price models.

        Args:
            split: Train/test split for direction classification
            target_prices: Actual target prices for regression
            num_boost_round: Max boosting rounds
            early_stopping_rounds: Early stopping patience

        Returns:
            TrainingResult with metrics
        """
        logger.info(
            "starting_training",
            train_samples=len(split.X_train),
            test_samples=len(split.X_test),
        )

        # Train direction classifier
        logger.info("training_direction_classifier")
        self._classifier.train(
            X_train=split.X_train,
            y_train=split.y_train,
            X_val=split.X_test,
            y_val=split.y_test,
            feature_names=split.feature_names,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )

        # Train quantile regressor on price returns
        logger.info("training_quantile_regressor")
        train_end = len(split.y_train)
        y_train_price = target_prices[:train_end]
        y_test_price = target_prices[train_end:]

        self._regressor.train(
            X_train=split.X_train,
            y_train=y_train_price,
            X_val=split.X_test,
            y_val=y_test_price,
            feature_names=split.feature_names,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )

        # Evaluate models
        logger.info("evaluating_models")
        metrics = self._evaluate(split, y_test_price)

        # Generate version and save
        model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._save_models(model_version)

        return TrainingResult(
            direction_accuracy=metrics["direction_accuracy"],
            direction_precision=metrics["direction_precision"],
            direction_recall=metrics["direction_recall"],
            price_mae=metrics["price_mae"],
            price_coverage=metrics["price_coverage"],
            feature_importance=self._classifier.get_feature_importance(),
            model_version=model_version,
            trained_at=datetime.utcnow(),
            train_samples=len(split.X_train),
            test_samples=len(split.X_test),
        )

    def _evaluate(
        self,
        split: TrainTestSplit,
        y_test_price: np.ndarray,
    ) -> dict:
        """Evaluate model performance on test set.

        Args:
            split: Train/test split
            y_test_price: Actual target prices for test set

        Returns:
            Dict of metrics
        """
        # Direction predictions
        dir_preds = self._classifier.predict(split.X_test)
        pred_directions = np.array([
            1 if p.direction == "up" else (-1 if p.direction == "down" else 0)
            for p in dir_preds
        ])

        # Direction accuracy
        direction_accuracy = np.mean(pred_directions == split.y_test)

        # Per-class precision and recall
        direction_precision = {}
        direction_recall = {}

        for label, name in [(1, "up"), (-1, "down"), (0, "flat")]:
            pred_mask = pred_directions == label
            actual_mask = split.y_test == label

            # Precision: of predicted label, how many were correct
            if pred_mask.sum() > 0:
                direction_precision[name] = float(
                    (pred_mask & actual_mask).sum() / pred_mask.sum()
                )
            else:
                direction_precision[name] = 0.0

            # Recall: of actual label, how many were predicted
            if actual_mask.sum() > 0:
                direction_recall[name] = float(
                    (pred_mask & actual_mask).sum() / actual_mask.sum()
                )
            else:
                direction_recall[name] = 0.0

        # Price predictions
        price_preds = self._regressor.predict(split.X_test)

        # MAE
        pred_prices = np.array([p.predicted for p in price_preds])
        price_mae = float(np.mean(np.abs(pred_prices - y_test_price)))

        # CI coverage (% of actuals within 10-90 percentile)
        lower_10 = np.array([p.lower_10 for p in price_preds])
        upper_90 = np.array([p.upper_90 for p in price_preds])
        in_ci = (y_test_price >= lower_10) & (y_test_price <= upper_90)
        price_coverage = float(np.mean(in_ci))

        return {
            "direction_accuracy": float(direction_accuracy),
            "direction_precision": direction_precision,
            "direction_recall": direction_recall,
            "price_mae": price_mae,
            "price_coverage": price_coverage,
        }

    def _save_models(self, version: str) -> None:
        """Save trained models with version.

        Args:
            version: Model version string
        """
        version_dir = self._model_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        self._classifier.save(version_dir / "direction.txt")
        self._regressor.save(version_dir / "price.txt")

        # Update latest symlink
        latest_link = self._model_dir / "latest"
        latest_link.unlink(missing_ok=True)
        latest_link.symlink_to(version)

        logger.info(
            "models_saved",
            version=version,
            path=str(version_dir),
        )
