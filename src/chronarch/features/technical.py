"""Technical indicators for feature engineering."""

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average.

    Args:
        series: Price series
        window: Rolling window size

    Returns:
        SMA series
    """
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average.

    Args:
        series: Price series
        span: EMA span (decay factor)

    Returns:
        EMA series
    """
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index.

    Args:
        close: Close price series
        period: RSI period (default 14)

    Returns:
        RSI series (0-100)
    """
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence Divergence.

    Args:
        close: Close price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period

    Returns:
        Tuple of (MACD line, signal line, histogram)
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Args:
        close: Close price series
        window: Rolling window for SMA
        num_std: Number of standard deviations

    Returns:
        Tuple of (upper band, middle band, lower band)
    """
    middle = sma(close, window)
    rolling_std = close.rolling(window=window).std()

    upper = middle + (rolling_std * num_std)
    lower = middle - (rolling_std * num_std)

    return upper, middle, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period

    Returns:
        ATR series
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(span=period, adjust=False).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume.

    Args:
        close: Close price series
        volume: Volume series

    Returns:
        OBV series
    """
    direction = np.sign(close.diff())
    return (volume * direction).cumsum()


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: %K period
        d_period: %D smoothing period

    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()

    return k, d


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Williams %R.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Lookback period

    Returns:
        Williams %R series (-100 to 0)
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    return -100 * (highest_high - close) / (highest_high - lowest_low)


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """Price Momentum.

    Args:
        close: Close price series
        period: Lookback period

    Returns:
        Momentum series
    """
    return close - close.shift(period)


def rate_of_change(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change (ROC).

    Args:
        close: Close price series
        period: Lookback period

    Returns:
        ROC series (percentage)
    """
    return ((close - close.shift(period)) / close.shift(period)) * 100
