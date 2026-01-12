"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from chronarch.features import statistics as stats
from chronarch.features import technical as tech
from chronarch.features.pipeline import FeaturePipeline


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500

    # Generate random walk prices
    returns = np.random.randn(n) * 0.01
    close = 100 * np.exp(np.cumsum(returns))

    # Generate high/low around close
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_price = close.copy()
    open_price[1:] = close[:-1]

    volume = np.random.exponential(1000, n)

    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_volume": volume * close,
        "num_trades": np.random.randint(100, 1000, n),
    })

    df.index = pd.date_range("2024-01-01", periods=n, freq="h")
    return df


class TestTechnicalIndicators:
    """Tests for technical indicator calculations."""

    def test_sma(self, sample_ohlcv: pd.DataFrame):
        """Test Simple Moving Average calculation."""
        close = sample_ohlcv["close"]
        sma_20 = tech.sma(close, window=20)

        # First 19 values should be NaN
        assert sma_20.iloc[:19].isna().all()
        # 20th value should equal mean of first 20 prices
        expected = close.iloc[:20].mean()
        assert abs(sma_20.iloc[19] - expected) < 0.01

    def test_rsi_bounds(self, sample_ohlcv: pd.DataFrame):
        """Test RSI is bounded between 0 and 100."""
        close = sample_ohlcv["close"]
        rsi = tech.rsi(close, period=14)

        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_macd_components(self, sample_ohlcv: pd.DataFrame):
        """Test MACD line, signal, and histogram relationship."""
        close = sample_ohlcv["close"]
        macd_line, signal, histogram = tech.macd(close)

        # Histogram should equal MACD - Signal
        valid_mask = ~(macd_line.isna() | signal.isna())
        diff = np.abs(histogram[valid_mask] - (macd_line[valid_mask] - signal[valid_mask]))
        assert (diff < 0.0001).all()

    def test_bollinger_bands_order(self, sample_ohlcv: pd.DataFrame):
        """Test Bollinger Bands are in correct order (upper > middle > lower)."""
        close = sample_ohlcv["close"]
        upper, middle, lower = tech.bollinger_bands(close)

        valid_mask = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_mask] >= middle[valid_mask]).all()
        assert (middle[valid_mask] >= lower[valid_mask]).all()


class TestStatistics:
    """Tests for statistical calculations."""

    def test_log_returns(self, sample_ohlcv: pd.DataFrame):
        """Test log returns calculation."""
        close = sample_ohlcv["close"]
        returns = stats.log_returns(close)

        # First value should be NaN
        assert pd.isna(returns.iloc[0])

        # Manual calculation for second return
        expected = np.log(close.iloc[1] / close.iloc[0])
        assert abs(returns.iloc[1] - expected) < 0.0001

    def test_direction_classification(self):
        """Test direction classification with threshold."""
        returns = pd.Series([0.02, -0.02, 0.0005, -0.0003, 0.001])
        threshold = 0.001

        directions = stats.classify_direction(returns, threshold)

        assert directions.iloc[0] == 1   # up (> 0.001)
        assert directions.iloc[1] == -1  # down (< -0.001)
        assert directions.iloc[2] == 0   # flat (within threshold)
        assert directions.iloc[3] == 0   # flat (within threshold)
        assert directions.iloc[4] == 0   # flat (exactly at threshold)


class TestFeaturePipeline:
    """Tests for the feature engineering pipeline."""

    def test_transform_output_shape(self, sample_ohlcv: pd.DataFrame):
        """Test feature pipeline produces expected output shape."""
        pipeline = FeaturePipeline(include_target=False)
        features = pipeline.transform(sample_ohlcv)

        # Should have same number of rows as input
        assert len(features) == len(sample_ohlcv)

        # Should have many feature columns
        assert len(features.columns) > 50

    def test_transform_with_targets(self, sample_ohlcv: pd.DataFrame):
        """Test feature pipeline with target generation."""
        target_periods = 10
        pipeline = FeaturePipeline(
            include_target=True,
            target_periods=target_periods,
            direction_threshold=0.001,
        )
        features = pipeline.transform(sample_ohlcv)

        assert "target_return" in features.columns
        assert "target_direction" in features.columns
        assert "target_price" in features.columns

        # Target direction should be -1, 0, or 1
        valid_directions = features["target_direction"].dropna()
        assert set(valid_directions.unique()).issubset({-1, 0, 1})

    def test_drop_na_rows(self, sample_ohlcv: pd.DataFrame):
        """Test NaN row dropping."""
        pipeline = FeaturePipeline(include_target=False)
        features = pipeline.transform(sample_ohlcv)

        # Some rows should have NaN (from rolling windows)
        assert features.isna().any().any()

        # After dropping, no NaN should remain
        cleaned = FeaturePipeline.drop_na_rows(features)
        assert not cleaned.isna().any().any()

        # Should have fewer rows
        assert len(cleaned) < len(features)

    def test_feature_names(self):
        """Test feature name generation."""
        pipeline = FeaturePipeline(include_target=False)
        names = pipeline.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

        # Check some expected feature names
        assert "rsi_14" in names
        assert "macd" in names
        assert "sma_20" in names
