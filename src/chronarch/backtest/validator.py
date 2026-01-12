"""Walk-forward validation for model evaluation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from chronarch.backtest.metrics import PerformanceMetrics, compile_metrics
from chronarch.features.pipeline import FeaturePipeline
from chronarch.ml.dataset import create_walk_forward_splits
from chronarch.ml.model import DirectionClassifier, QuantileRegressor

logger = structlog.get_logger()


@dataclass
class ValidationResult:
    """Result from a single validation fold."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_samples: int
    test_samples: int
    metrics: PerformanceMetrics


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward validation."""

    folds: list[ValidationResult]
    mean_direction_accuracy: float
    std_direction_accuracy: float
    mean_sharpe_ratio: float
    std_sharpe_ratio: float
    mean_ci_coverage: float
    std_ci_coverage: float
    total_test_samples: int


class WalkForwardValidator:
    """Walk-forward validation for time series models."""

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: float = 0.3,
        classifier_params: dict | None = None,
        regressor_params: dict | None = None,
    ) -> None:
        """Initialize validator.

        Args:
            n_splits: Number of walk-forward folds
            min_train_size: Minimum fraction for first training set
            classifier_params: LightGBM params for classifier
            regressor_params: LightGBM params for regressor
        """
        self._n_splits = n_splits
        self._min_train_size = min_train_size
        self._classifier_params = classifier_params
        self._regressor_params = regressor_params

    def validate(
        self,
        df: pd.DataFrame,
        target_periods: int = 60,
        direction_threshold: float = 0.001,
        num_boost_round: int = 300,
        early_stopping_rounds: int = 30,
    ) -> WalkForwardResult:
        """Run walk-forward validation.

        Args:
            df: OHLCV DataFrame indexed by timestamp
            target_periods: Periods ahead for prediction
            direction_threshold: Threshold for direction classification
            num_boost_round: Max boosting rounds per fold
            early_stopping_rounds: Early stopping patience

        Returns:
            WalkForwardResult with all fold metrics
        """
        # Create feature pipeline with targets
        pipeline = FeaturePipeline(
            include_target=True,
            target_periods=target_periods,
            direction_threshold=direction_threshold,
        )

        # Get walk-forward splits for direction
        splits = create_walk_forward_splits(
            df=df,
            target_col="target_direction",
            feature_pipeline=pipeline,
            n_splits=self._n_splits,
            min_train_size=self._min_train_size,
        )

        # Get full features for price targets
        features_df = pipeline.transform(df)
        features_df = FeaturePipeline.drop_na_rows(features_df)
        features_df = features_df.dropna(subset=["target_price"])

        fold_results: list[ValidationResult] = []

        for i, split in enumerate(splits):
            logger.info(
                "running_fold",
                fold=i + 1,
                total_folds=len(splits),
                train_samples=len(split.X_train),
                test_samples=len(split.X_test),
            )

            # Train classifier
            classifier = DirectionClassifier(self._classifier_params)
            classifier.train(
                X_train=split.X_train,
                y_train=split.y_train,
                X_val=split.X_test,
                y_val=split.y_test,
                feature_names=split.feature_names,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )

            # Get target prices for this split
            train_end_idx = len(split.train_timestamps)
            test_end_idx = train_end_idx + len(split.test_timestamps)

            target_prices = features_df["target_price"].values
            y_train_price = target_prices[:train_end_idx]
            y_test_price = target_prices[train_end_idx:test_end_idx]

            # Train regressor
            regressor = QuantileRegressor(self._regressor_params)
            regressor.train(
                X_train=split.X_train,
                y_train=y_train_price,
                X_val=split.X_test,
                y_val=y_test_price,
                feature_names=split.feature_names,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )

            # Generate predictions
            dir_preds = classifier.predict(split.X_test)
            price_preds = regressor.predict(split.X_test)

            y_pred_direction = np.array([
                1 if p.direction == "up" else (-1 if p.direction == "down" else 0)
                for p in dir_preds
            ])

            y_pred_price = np.array([p.predicted for p in price_preds])
            y_lower = np.array([p.lower_10 for p in price_preds])
            y_upper = np.array([p.upper_90 for p in price_preds])

            # Get prices for return calculation
            close_prices = df["close"].loc[split.test_timestamps].values.astype(float)

            # Calculate metrics
            metrics = compile_metrics(
                y_true_direction=split.y_test,
                y_pred_direction=y_pred_direction,
                y_true_price=y_test_price,
                y_pred_price=y_pred_price,
                y_lower=y_lower,
                y_upper=y_upper,
                prices=close_prices,
            )

            fold_results.append(
                ValidationResult(
                    fold=i + 1,
                    train_start=split.train_timestamps[0],
                    train_end=split.train_timestamps[-1],
                    test_start=split.test_timestamps[0],
                    test_end=split.test_timestamps[-1],
                    train_samples=len(split.X_train),
                    test_samples=len(split.X_test),
                    metrics=metrics,
                )
            )

            logger.info(
                "fold_complete",
                fold=i + 1,
                accuracy=metrics.direction_accuracy,
                sharpe=metrics.sharpe_ratio,
                ci_coverage=metrics.ci_coverage_80,
            )

        # Aggregate results
        accuracies = [r.metrics.direction_accuracy for r in fold_results]
        sharpes = [r.metrics.sharpe_ratio for r in fold_results]
        coverages = [r.metrics.ci_coverage_80 for r in fold_results]

        return WalkForwardResult(
            folds=fold_results,
            mean_direction_accuracy=float(np.mean(accuracies)),
            std_direction_accuracy=float(np.std(accuracies)),
            mean_sharpe_ratio=float(np.mean(sharpes)),
            std_sharpe_ratio=float(np.std(sharpes)),
            mean_ci_coverage=float(np.mean(coverages)),
            std_ci_coverage=float(np.std(coverages)),
            total_test_samples=sum(r.test_samples for r in fold_results),
        )


def run_backtest(
    df: pd.DataFrame,
    n_splits: int = 5,
    target_periods: int = 60,
    direction_threshold: float = 0.001,
) -> WalkForwardResult:
    """Convenience function to run walk-forward backtest.

    Args:
        df: OHLCV DataFrame indexed by timestamp
        n_splits: Number of validation folds
        target_periods: Periods ahead for prediction
        direction_threshold: Threshold for direction classification

    Returns:
        WalkForwardResult with validation metrics
    """
    validator = WalkForwardValidator(n_splits=n_splits)
    return validator.validate(
        df=df,
        target_periods=target_periods,
        direction_threshold=direction_threshold,
    )
