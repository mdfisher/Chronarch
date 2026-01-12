"""Rolling statistics and return calculations."""

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    """Calculate log returns.

    Args:
        close: Close price series

    Returns:
        Log return series
    """
    return np.log(close / close.shift(1))


def simple_returns(close: pd.Series) -> pd.Series:
    """Calculate simple (percentage) returns.

    Args:
        close: Close price series

    Returns:
        Simple return series
    """
    return close.pct_change()


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Calculate rolling volatility (standard deviation of returns).

    Args:
        returns: Return series
        window: Rolling window size

    Returns:
        Rolling volatility series
    """
    return returns.rolling(window=window).std()


def rolling_skewness(returns: pd.Series, window: int) -> pd.Series:
    """Calculate rolling skewness.

    Args:
        returns: Return series
        window: Rolling window size

    Returns:
        Rolling skewness series
    """
    return returns.rolling(window=window).skew()


def rolling_kurtosis(returns: pd.Series, window: int) -> pd.Series:
    """Calculate rolling kurtosis.

    Args:
        returns: Return series
        window: Rolling window size

    Returns:
        Rolling kurtosis series
    """
    return returns.rolling(window=window).kurt()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling z-score (how many std devs from rolling mean).

    Args:
        series: Input series
        window: Rolling window size

    Returns:
        Rolling z-score series
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return (series - rolling_mean) / rolling_std


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling percentile rank of current value.

    Args:
        series: Input series
        window: Rolling window size

    Returns:
        Rolling percentile rank (0-100)
    """
    def percentile_rank(x):
        return (x.argsort().argsort()[-1] + 1) / len(x) * 100

    return series.rolling(window=window).apply(percentile_rank, raw=False)


def price_distance_from_high(
    close: pd.Series,
    high: pd.Series,
    window: int,
) -> pd.Series:
    """Calculate percentage distance from rolling high.

    Args:
        close: Close price series
        high: High price series
        window: Rolling window size

    Returns:
        Distance from high as percentage
    """
    rolling_high = high.rolling(window=window).max()
    return (close - rolling_high) / rolling_high * 100


def price_distance_from_low(
    close: pd.Series,
    low: pd.Series,
    window: int,
) -> pd.Series:
    """Calculate percentage distance from rolling low.

    Args:
        close: Close price series
        low: Low price series
        window: Rolling window size

    Returns:
        Distance from low as percentage
    """
    rolling_low = low.rolling(window=window).min()
    return (close - rolling_low) / rolling_low * 100


def rolling_range(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Calculate rolling price range as percentage.

    Args:
        high: High price series
        low: Low price series
        window: Rolling window size

    Returns:
        Rolling range as percentage
    """
    rolling_high = high.rolling(window=window).max()
    rolling_low = low.rolling(window=window).min()
    return (rolling_high - rolling_low) / rolling_low * 100


def future_returns(close: pd.Series, periods: int) -> pd.Series:
    """Calculate future returns (target variable).

    Note: This creates look-ahead, only use for target creation in training.

    Args:
        close: Close price series
        periods: Number of periods ahead

    Returns:
        Future return series
    """
    return close.shift(-periods) / close - 1


def classify_direction(
    future_return: pd.Series,
    threshold: float = 0.001,
) -> pd.Series:
    """Classify future return into direction categories.

    Args:
        future_return: Future return series
        threshold: Threshold for flat classification (default 0.1%)

    Returns:
        Direction series (1=up, -1=down, 0=flat)
    """
    conditions = [
        future_return > threshold,
        future_return < -threshold,
    ]
    choices = [1, -1]
    return pd.Series(
        np.select(conditions, choices, default=0),
        index=future_return.index,
    )
