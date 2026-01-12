"""Performance metrics for backtesting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Collection of performance metrics."""

    # Direction metrics
    direction_accuracy: float
    direction_precision_up: float
    direction_precision_down: float
    direction_recall_up: float
    direction_recall_down: float

    # Price prediction metrics
    price_mae: float
    price_rmse: float
    price_mape: float
    ci_coverage_80: float  # % of actuals within 80% CI

    # Trading metrics (if direction signals used)
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float


def calculate_direction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate direction prediction metrics.

    Args:
        y_true: Actual directions (-1, 0, 1)
        y_pred: Predicted directions (-1, 0, 1)

    Returns:
        Dict of metric names to values
    """
    accuracy = np.mean(y_true == y_pred)

    metrics = {"accuracy": float(accuracy)}

    for label, name in [(1, "up"), (-1, "down")]:
        pred_mask = y_pred == label
        actual_mask = y_true == label

        # Precision
        if pred_mask.sum() > 0:
            precision = float((pred_mask & actual_mask).sum() / pred_mask.sum())
        else:
            precision = 0.0

        # Recall
        if actual_mask.sum() > 0:
            recall = float((pred_mask & actual_mask).sum() / actual_mask.sum())
        else:
            recall = 0.0

        metrics[f"precision_{name}"] = precision
        metrics[f"recall_{name}"] = recall

    return metrics


def calculate_price_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> dict[str, float]:
    """Calculate price prediction metrics.

    Args:
        y_true: Actual prices
        y_pred: Predicted prices (median)
        y_lower: Lower bound (10th percentile)
        y_upper: Upper bound (90th percentile)

    Returns:
        Dict of metric names to values
    """
    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # RMSE
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # MAPE (handle division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs((y_true - y_pred) / y_true)
        ape = np.where(np.isfinite(ape), ape, 0)
    mape = float(np.mean(ape) * 100)

    # CI coverage
    in_ci = (y_true >= y_lower) & (y_true <= y_upper)
    ci_coverage = float(np.mean(in_ci))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "ci_coverage_80": ci_coverage,
    }


def calculate_returns(
    prices: np.ndarray,
    signals: np.ndarray,
    transaction_cost: float = 0.001,
) -> np.ndarray:
    """Calculate strategy returns from price signals.

    Args:
        prices: Price series
        signals: Position signals (-1, 0, 1)
        transaction_cost: Cost per trade (default 0.1%)

    Returns:
        Array of strategy returns
    """
    # Calculate raw returns
    price_returns = np.diff(prices) / prices[:-1]

    # Strategy returns (signal from previous period)
    strategy_returns = signals[:-1] * price_returns

    # Apply transaction costs on position changes
    position_changes = np.abs(np.diff(signals))
    costs = np.concatenate([[0], position_changes * transaction_cost])
    strategy_returns = strategy_returns - costs[:-1]

    return strategy_returns


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365 * 24,  # Hourly data
) -> float:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    return float(
        np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    )


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365 * 24,
) -> float:
    """Calculate annualized Sortino ratio (downside deviation).

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return float("inf") if np.mean(excess_returns) > 0 else 0.0

    downside_std = np.std(downside_returns)
    return float(np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year))


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Cumulative returns or equity values

    Returns:
        Maximum drawdown as positive percentage
    """
    if len(equity_curve) == 0:
        return 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    return float(-np.min(drawdowns))


def calculate_win_rate(returns: np.ndarray) -> float:
    """Calculate percentage of winning trades.

    Args:
        returns: Array of trade returns

    Returns:
        Win rate as percentage
    """
    if len(returns) == 0:
        return 0.0

    return float(np.mean(returns > 0))


def calculate_profit_factor(returns: np.ndarray) -> float:
    """Calculate profit factor (gross profit / gross loss).

    Args:
        returns: Array of trade returns

    Returns:
        Profit factor (>1 is profitable)
    """
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()

    if losses == 0:
        return float("inf") if gains > 0 else 0.0

    return float(gains / losses)


def compile_metrics(
    y_true_direction: np.ndarray,
    y_pred_direction: np.ndarray,
    y_true_price: np.ndarray,
    y_pred_price: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    prices: np.ndarray,
    periods_per_year: int = 365 * 24,
) -> PerformanceMetrics:
    """Compile all performance metrics.

    Args:
        y_true_direction: Actual directions
        y_pred_direction: Predicted directions
        y_true_price: Actual prices
        y_pred_price: Predicted prices
        y_lower: Lower CI bound
        y_upper: Upper CI bound
        prices: Price series for return calculation
        periods_per_year: Periods per year for annualization

    Returns:
        PerformanceMetrics dataclass
    """
    dir_metrics = calculate_direction_metrics(y_true_direction, y_pred_direction)
    price_metrics = calculate_price_metrics(y_true_price, y_pred_price, y_lower, y_upper)

    # Calculate trading returns
    strategy_returns = calculate_returns(prices, y_pred_direction)

    # Cumulative returns
    cumulative = np.cumprod(1 + strategy_returns)
    total_return = float(cumulative[-1] - 1) if len(cumulative) > 0 else 0.0

    # Annualized return
    n_periods = len(strategy_returns)
    annualized_return = float(
        (1 + total_return) ** (periods_per_year / n_periods) - 1
    ) if n_periods > 0 else 0.0

    return PerformanceMetrics(
        direction_accuracy=dir_metrics["accuracy"],
        direction_precision_up=dir_metrics["precision_up"],
        direction_precision_down=dir_metrics["precision_down"],
        direction_recall_up=dir_metrics["recall_up"],
        direction_recall_down=dir_metrics["recall_down"],
        price_mae=price_metrics["mae"],
        price_rmse=price_metrics["rmse"],
        price_mape=price_metrics["mape"],
        ci_coverage_80=price_metrics["ci_coverage_80"],
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=calculate_sharpe_ratio(strategy_returns, periods_per_year=periods_per_year),
        sortino_ratio=calculate_sortino_ratio(strategy_returns, periods_per_year=periods_per_year),
        max_drawdown=calculate_max_drawdown(cumulative),
        win_rate=calculate_win_rate(strategy_returns),
        profit_factor=calculate_profit_factor(strategy_returns),
    )
