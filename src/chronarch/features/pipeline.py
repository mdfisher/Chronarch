"""Feature engineering pipeline combining all indicators."""

import pandas as pd

from chronarch.features import statistics as stats
from chronarch.features import technical as tech


class FeaturePipeline:
    """Pipeline for generating ML features from OHLCV data."""

    # Rolling window sizes for multi-scale features
    WINDOWS = [5, 10, 20, 50, 100]

    def __init__(
        self,
        include_target: bool = False,
        target_periods: int = 60,  # 1 hour for 1-minute candles
        direction_threshold: float = 0.001,
    ) -> None:
        """Initialize feature pipeline.

        Args:
            include_target: Whether to generate target columns
            target_periods: Periods ahead for target calculation
            direction_threshold: Threshold for direction classification
        """
        self._include_target = include_target
        self._target_periods = target_periods
        self._direction_threshold = direction_threshold

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform OHLCV DataFrame into feature matrix.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
                indexed by timestamp

        Returns:
            DataFrame with feature columns
        """
        features = pd.DataFrame(index=df.index)

        # Extract OHLCV columns
        open_price = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        # Returns
        features["log_return"] = stats.log_returns(close)
        features["simple_return"] = stats.simple_returns(close)

        # Technical indicators
        features["rsi_14"] = tech.rsi(close, period=14)

        macd_line, signal_line, histogram = tech.macd(close)
        features["macd"] = macd_line
        features["macd_signal"] = signal_line
        features["macd_histogram"] = histogram

        bb_upper, bb_middle, bb_lower = tech.bollinger_bands(close)
        features["bb_upper"] = bb_upper
        features["bb_middle"] = bb_middle
        features["bb_lower"] = bb_lower
        features["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)

        features["atr_14"] = tech.atr(high, low, close, period=14)
        features["obv"] = tech.obv(close, volume)

        stoch_k, stoch_d = tech.stochastic(high, low, close)
        features["stoch_k"] = stoch_k
        features["stoch_d"] = stoch_d

        features["williams_r"] = tech.williams_r(high, low, close)
        features["momentum_10"] = tech.momentum(close, period=10)
        features["roc_10"] = tech.rate_of_change(close, period=10)

        # Moving averages at different scales
        for window in self.WINDOWS:
            features[f"sma_{window}"] = tech.sma(close, window)
            features[f"ema_{window}"] = tech.ema(close, window)
            features[f"sma_ratio_{window}"] = close / tech.sma(close, window)

        # Rolling statistics at different scales
        returns = stats.log_returns(close)
        for window in self.WINDOWS:
            features[f"volatility_{window}"] = stats.rolling_volatility(
                returns, window
            )
            features[f"skewness_{window}"] = stats.rolling_skewness(returns, window)
            features[f"kurtosis_{window}"] = stats.rolling_kurtosis(returns, window)
            features[f"zscore_{window}"] = stats.rolling_zscore(close, window)

        # Price position features
        for window in self.WINDOWS:
            features[f"dist_high_{window}"] = stats.price_distance_from_high(
                close, high, window
            )
            features[f"dist_low_{window}"] = stats.price_distance_from_low(
                close, low, window
            )
            features[f"range_{window}"] = stats.rolling_range(high, low, window)

        # Volume features
        features["volume_sma_20"] = tech.sma(volume, 20)
        features["volume_ratio"] = volume / tech.sma(volume, 20)

        # Candle features
        features["body_size"] = (close - open_price).abs() / open_price
        features["upper_shadow"] = (high - close.where(close > open_price, open_price)) / close
        features["lower_shadow"] = (close.where(close < open_price, open_price) - low) / close
        features["is_bullish"] = (close > open_price).astype(int)

        # Time features (if index is datetime)
        if isinstance(df.index, pd.DatetimeIndex):
            features["hour"] = df.index.hour
            features["day_of_week"] = df.index.dayofweek
            features["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

        # Target variables (only for training data)
        if self._include_target:
            future_ret = stats.future_returns(close, self._target_periods)
            features["target_return"] = future_ret
            features["target_direction"] = stats.classify_direction(
                future_ret, self._direction_threshold
            )
            features["target_price"] = close.shift(-self._target_periods)

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature column names (excluding targets).

        Returns:
            List of feature names
        """
        # Create a small dummy DataFrame to get column names
        dummy_data = {
            "open": [1.0] * 200,
            "high": [1.1] * 200,
            "low": [0.9] * 200,
            "close": [1.0] * 200,
            "volume": [100.0] * 200,
        }
        dummy_df = pd.DataFrame(dummy_data)

        # Transform without targets
        pipeline = FeaturePipeline(include_target=False)
        features = pipeline.transform(dummy_df)

        return list(features.columns)

    @staticmethod
    def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with NaN values (from rolling window warmup).

        Args:
            df: Feature DataFrame

        Returns:
            DataFrame with NaN rows removed
        """
        return df.dropna()

    @staticmethod
    def normalize_features(
        df: pd.DataFrame,
        exclude_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
        """Normalize features to zero mean and unit variance.

        Args:
            df: Feature DataFrame
            exclude_cols: Columns to exclude from normalization

        Returns:
            Tuple of (normalized DataFrame, stats dict for inverse transform)
        """
        exclude_cols = exclude_cols or []
        stats_dict: dict[str, tuple[float, float]] = {}

        df_normalized = df.copy()

        for col in df.columns:
            if col in exclude_cols:
                continue

            mean_val = df[col].mean()
            std_val = df[col].std()

            if std_val > 0:
                df_normalized[col] = (df[col] - mean_val) / std_val
                stats_dict[col] = (mean_val, std_val)
            else:
                stats_dict[col] = (mean_val, 1.0)

        return df_normalized, stats_dict
