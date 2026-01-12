"""Binance API client for fetching OHLCV candle data."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TypedDict

import httpx
import structlog

from chronarch.config import get_settings
from chronarch.data.models import Candle, CandleInterval

logger = structlog.get_logger()


class RawCandle(TypedDict):
    """Raw candle data from Binance API."""

    open_time: int
    open: str
    high: str
    low: str
    close: str
    volume: str
    close_time: int
    quote_volume: str
    num_trades: int


class BinanceClient:
    """Async client for Binance REST API."""

    INTERVAL_MS = {
        CandleInterval.M1: 60_000,
        CandleInterval.M5: 300_000,
        CandleInterval.H1: 3_600_000,
        CandleInterval.D1: 86_400_000,
    }

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.binance_base_url
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=30.0,
                headers={"Accept": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_candles(
        self,
        symbol: str,
        interval: CandleInterval,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[Candle]:
        """Fetch OHLCV candles from Binance.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Candle interval
            start_time: Start time for historical data
            end_time: End time for historical data
            limit: Maximum number of candles to return (max 1000)

        Returns:
            List of Candle objects
        """
        client = await self._get_client()

        params: dict[str, str | int] = {
            "symbol": symbol,
            "interval": interval.value,
            "limit": min(limit, 1000),
        }

        if start_time is not None:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time is not None:
            params["endTime"] = int(end_time.timestamp() * 1000)

        logger.debug(
            "fetching_candles",
            symbol=symbol,
            interval=interval.value,
            start_time=start_time,
            end_time=end_time,
        )

        response = await client.get("/api/v3/klines", params=params)
        response.raise_for_status()

        raw_candles = response.json()
        return [
            self._parse_candle(raw, symbol, interval) for raw in raw_candles
        ]

    async def fetch_historical(
        self,
        symbol: str,
        interval: CandleInterval,
        days: int,
    ) -> list[Candle]:
        """Fetch historical candles going back N days.

        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            days: Number of days of history to fetch

        Returns:
            List of Candle objects sorted by timestamp
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        all_candles: list[Candle] = []
        current_start = start_time

        while current_start < end_time:
            candles = await self.fetch_candles(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=1000,
            )

            if not candles:
                break

            all_candles.extend(candles)

            # Move start time forward based on the last candle
            last_candle_time = candles[-1].timestamp
            current_start = last_candle_time + timedelta(
                milliseconds=self.INTERVAL_MS[interval]
            )

            logger.info(
                "fetched_candle_batch",
                symbol=symbol,
                interval=interval.value,
                batch_size=len(candles),
                total_candles=len(all_candles),
                progress_pct=round(
                    (current_start - start_time).total_seconds()
                    / (end_time - start_time).total_seconds()
                    * 100,
                    1,
                ),
            )

        return sorted(all_candles, key=lambda c: c.timestamp)

    async def fetch_latest(
        self,
        symbol: str,
        interval: CandleInterval,
        count: int = 100,
    ) -> list[Candle]:
        """Fetch the most recent candles.

        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            count: Number of recent candles to fetch

        Returns:
            List of Candle objects sorted by timestamp
        """
        return await self.fetch_candles(
            symbol=symbol,
            interval=interval,
            limit=count,
        )

    def _parse_candle(
        self,
        raw: list,
        symbol: str,
        interval: CandleInterval,
    ) -> Candle:
        """Parse raw Binance kline data into Candle object.

        Binance kline format:
        [
            0: open_time,
            1: open,
            2: high,
            3: low,
            4: close,
            5: volume,
            6: close_time,
            7: quote_volume,
            8: num_trades,
            9: taker_buy_base_volume,
            10: taker_buy_quote_volume,
            11: ignore
        ]
        """
        return Candle(
            symbol=symbol,
            interval=interval,
            timestamp=datetime.fromtimestamp(raw[0] / 1000, tz=timezone.utc),
            open=Decimal(raw[1]),
            high=Decimal(raw[2]),
            low=Decimal(raw[3]),
            close=Decimal(raw[4]),
            volume=Decimal(raw[5]),
            quote_volume=Decimal(raw[7]),
            num_trades=int(raw[8]),
        )
