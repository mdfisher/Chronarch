"""Dataset preparation for model training."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from chronarch.features.pipeline import FeaturePipeline


@dataclass
class TrainTestSplit:
    """Container for train/test split data."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    train_timestamps: pd.DatetimeIndex
    test_timestamps: pd.DatetimeIndex


def create_dataset(
    df: pd.DataFrame,
    target_col: str,
    feature_pipeline: FeaturePipeline,
    test_size: float = 0.2,
) -> TrainTestSplit:
    """Create train/test dataset from OHLCV DataFrame.

    Uses time-based split (no shuffling) to prevent look-ahead bias.

    Args:
        df: OHLCV DataFrame indexed by timestamp
        target_col: Name of target column ("target_direction" or "target_return")
        feature_pipeline: Configured FeaturePipeline instance
        test_size: Fraction of data to use for testing

    Returns:
        TrainTestSplit with train/test data
    """
    # Generate features with targets
    features_df = feature_pipeline.transform(df)

    # Drop NaN rows (from rolling windows)
    features_df = FeaturePipeline.drop_na_rows(features_df)

    # Drop rows with NaN targets (from future shift)
    features_df = features_df.dropna(subset=[target_col])

    # Separate features and target
    target_cols = ["target_return", "target_direction", "target_price"]
    feature_cols = [c for c in features_df.columns if c not in target_cols]

    X = features_df[feature_cols]
    y = features_df[target_col]
    timestamps = features_df.index

    # Time-based split (last test_size fraction is test)
    split_idx = int(len(X) * (1 - test_size))

    return TrainTestSplit(
        X_train=X.iloc[:split_idx].values,
        y_train=y.iloc[:split_idx].values,
        X_test=X.iloc[split_idx:].values,
        y_test=y.iloc[split_idx:].values,
        feature_names=feature_cols,
        train_timestamps=timestamps[:split_idx],
        test_timestamps=timestamps[split_idx:],
    )


def create_walk_forward_splits(
    df: pd.DataFrame,
    target_col: str,
    feature_pipeline: FeaturePipeline,
    n_splits: int = 5,
    min_train_size: float = 0.3,
) -> list[TrainTestSplit]:
    """Create multiple train/test splits for walk-forward validation.

    Each split uses all data up to a point for training and a fixed-size
    window for testing, progressively moving forward in time.

    Args:
        df: OHLCV DataFrame indexed by timestamp
        target_col: Name of target column
        feature_pipeline: Configured FeaturePipeline instance
        n_splits: Number of walk-forward splits
        min_train_size: Minimum fraction of data for first training set

    Returns:
        List of TrainTestSplit objects
    """
    # Generate features with targets
    features_df = feature_pipeline.transform(df)
    features_df = FeaturePipeline.drop_na_rows(features_df)
    features_df = features_df.dropna(subset=[target_col])

    target_cols = ["target_return", "target_direction", "target_price"]
    feature_cols = [c for c in features_df.columns if c not in target_cols]

    X = features_df[feature_cols]
    y = features_df[target_col]
    timestamps = features_df.index

    n_samples = len(X)
    test_size = int(n_samples * (1 - min_train_size) / n_splits)

    splits = []

    for i in range(n_splits):
        train_end = int(n_samples * min_train_size) + i * test_size
        test_end = train_end + test_size

        if test_end > n_samples:
            break

        splits.append(
            TrainTestSplit(
                X_train=X.iloc[:train_end].values,
                y_train=y.iloc[:train_end].values,
                X_test=X.iloc[train_end:test_end].values,
                y_test=y.iloc[train_end:test_end].values,
                feature_names=feature_cols,
                train_timestamps=timestamps[:train_end],
                test_timestamps=timestamps[train_end:test_end],
            )
        )

    return splits


def get_latest_features(
    df: pd.DataFrame,
    feature_pipeline: FeaturePipeline,
) -> tuple[np.ndarray, list[str], datetime]:
    """Get feature vector for the most recent data point.

    Args:
        df: OHLCV DataFrame indexed by timestamp
        feature_pipeline: Configured FeaturePipeline instance

    Returns:
        Tuple of (feature vector, feature names, timestamp)
    """
    # Generate features without targets
    pipeline = FeaturePipeline(include_target=False)
    features_df = pipeline.transform(df)

    # Get the last row with valid features
    last_valid = features_df.dropna().iloc[-1]

    return (
        last_valid.values.reshape(1, -1),
        list(last_valid.index),
        df.index[-1],
    )
