"""Data access layer for candles and predictions."""

from datetime import datetime

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from chronarch.data.models import Candle, CandleInterval, Prediction


class CandleRepository:
    """Repository for OHLCV candle data operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_candles(self, candles: list[Candle]) -> int:
        """Insert or update candles (upsert on conflict).

        Args:
            candles: List of Candle objects to upsert

        Returns:
            Number of rows affected
        """
        if not candles:
            return 0

        values = [
            {
                "symbol": c.symbol,
                "interval": c.interval.value if isinstance(c.interval, CandleInterval) else c.interval,
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "quote_volume": c.quote_volume,
                "num_trades": c.num_trades,
            }
            for c in candles
        ]

        stmt = insert(Candle).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "interval", "timestamp"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
                "quote_volume": stmt.excluded.quote_volume,
                "num_trades": stmt.excluded.num_trades,
            },
        )

        result = await self._session.execute(stmt)
        await self._session.commit()
        return result.rowcount

    async def get_candles(
        self,
        symbol: str,
        interval: CandleInterval,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> list[Candle]:
        """Fetch candles from database.

        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Optional maximum number of candles

        Returns:
            List of Candle objects ordered by timestamp
        """
        conditions = [
            Candle.symbol == symbol,
            Candle.interval == interval,
        ]

        if start_time is not None:
            conditions.append(Candle.timestamp >= start_time)
        if end_time is not None:
            conditions.append(Candle.timestamp <= end_time)

        stmt = (
            select(Candle)
            .where(and_(*conditions))
            .order_by(Candle.timestamp)
        )

        if limit is not None:
            stmt = stmt.limit(limit)

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_latest_candle(
        self,
        symbol: str,
        interval: CandleInterval,
    ) -> Candle | None:
        """Get the most recent candle for a symbol/interval.

        Args:
            symbol: Trading pair symbol
            interval: Candle interval

        Returns:
            Most recent Candle or None if not found
        """
        stmt = (
            select(Candle)
            .where(
                and_(
                    Candle.symbol == symbol,
                    Candle.interval == interval,
                )
            )
            .order_by(Candle.timestamp.desc())
            .limit(1)
        )

        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_candles_as_dataframe(
        self,
        symbol: str,
        interval: CandleInterval,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch candles and return as pandas DataFrame.

        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            DataFrame with OHLCV columns indexed by timestamp
        """
        candles = await self.get_candles(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
        )

        if not candles:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "quote_volume", "num_trades"]
            )

        data = [
            {
                "timestamp": c.timestamp,
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
                "quote_volume": float(c.quote_volume),
                "num_trades": c.num_trades,
            }
            for c in candles
        ]

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df


class PredictionRepository:
    """Repository for prediction data operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_prediction(self, prediction: Prediction) -> Prediction:
        """Save a new prediction.

        Args:
            prediction: Prediction object to save

        Returns:
            Saved Prediction with ID
        """
        self._session.add(prediction)
        await self._session.commit()
        await self._session.refresh(prediction)
        return prediction

    async def get_predictions(
        self,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> list[Prediction]:
        """Fetch predictions from database.

        Args:
            symbol: Trading pair symbol
            start_time: Optional start time filter (prediction timestamp)
            end_time: Optional end time filter
            limit: Optional maximum number of predictions

        Returns:
            List of Prediction objects ordered by timestamp desc
        """
        conditions = [Prediction.symbol == symbol]

        if start_time is not None:
            conditions.append(Prediction.timestamp >= start_time)
        if end_time is not None:
            conditions.append(Prediction.timestamp <= end_time)

        stmt = (
            select(Prediction)
            .where(and_(*conditions))
            .order_by(Prediction.timestamp.desc())
        )

        if limit is not None:
            stmt = stmt.limit(limit)

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_latest_prediction(self, symbol: str) -> Prediction | None:
        """Get the most recent prediction for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Most recent Prediction or None if not found
        """
        stmt = (
            select(Prediction)
            .where(Prediction.symbol == symbol)
            .order_by(Prediction.timestamp.desc())
            .limit(1)
        )

        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_pending_outcome_predictions(
        self,
        symbol: str,
        before: datetime,
    ) -> list[Prediction]:
        """Get predictions whose target time has passed but outcome not recorded.

        Args:
            symbol: Trading pair symbol
            before: Get predictions with target_timestamp before this time

        Returns:
            List of Prediction objects needing outcome update
        """
        stmt = (
            select(Prediction)
            .where(
                and_(
                    Prediction.symbol == symbol,
                    Prediction.target_timestamp <= before,
                    Prediction.actual_price.is_(None),
                )
            )
            .order_by(Prediction.target_timestamp)
        )

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_outcome(
        self,
        prediction_id: int,
        actual_price: float,
        actual_direction: str,
        was_direction_correct: bool,
        was_in_confidence_interval: bool,
    ) -> None:
        """Update a prediction with actual outcome.

        Args:
            prediction_id: ID of prediction to update
            actual_price: Actual price at target time
            actual_direction: Actual direction (up/down/flat)
            was_direction_correct: Whether direction prediction was correct
            was_in_confidence_interval: Whether actual price was in CI
        """
        stmt = (
            select(Prediction)
            .where(Prediction.id == prediction_id)
        )
        result = await self._session.execute(stmt)
        prediction = result.scalar_one()

        prediction.actual_price = actual_price
        prediction.actual_direction = actual_direction
        prediction.was_direction_correct = was_direction_correct
        prediction.was_in_confidence_interval = was_in_confidence_interval

        await self._session.commit()
