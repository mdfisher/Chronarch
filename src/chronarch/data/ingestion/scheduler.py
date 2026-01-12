"""Data ingestion scheduler for periodic candle fetching."""

import asyncio
from datetime import datetime, timedelta, timezone

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from chronarch.config import get_settings
from chronarch.data.ingestion.binance import BinanceClient
from chronarch.data.models import CandleInterval
from chronarch.data.repository import CandleRepository
from chronarch.db.connection import Database

logger = structlog.get_logger()


class IngestionScheduler:
    """Schedules periodic data ingestion from Binance."""

    # Map intervals to fetch frequencies (seconds)
    FETCH_INTERVALS = {
        CandleInterval.M1: 60,
        CandleInterval.M5: 300,
        CandleInterval.H1: 3600,
        CandleInterval.D1: 86400,
    }

    def __init__(self) -> None:
        self._settings = get_settings()
        self._binance = BinanceClient()
        self._scheduler = AsyncIOScheduler()
        self._running = False

    async def start(self) -> None:
        """Start the ingestion scheduler."""
        logger.info("starting_ingestion_scheduler")

        # Fetch historical data on startup
        await self._fetch_historical()

        # Schedule periodic fetches for each interval
        for interval in self._settings.candle_intervals:
            candle_interval = CandleInterval(interval)
            seconds = self.FETCH_INTERVALS[candle_interval]

            self._scheduler.add_job(
                self._fetch_latest,
                trigger=IntervalTrigger(seconds=seconds),
                args=[candle_interval],
                id=f"fetch_{interval}",
                name=f"Fetch {interval} candles",
                replace_existing=True,
            )

            logger.info(
                "scheduled_fetch",
                interval=interval,
                frequency_seconds=seconds,
            )

        self._scheduler.start()
        self._running = True

    async def stop(self) -> None:
        """Stop the ingestion scheduler."""
        self._scheduler.shutdown(wait=False)
        await self._binance.close()
        self._running = False
        logger.info("ingestion_scheduler_stopped")

    async def _fetch_historical(self) -> None:
        """Fetch historical data for all intervals."""
        symbol = self._settings.default_symbol
        days = self._settings.historical_days

        for interval_str in self._settings.candle_intervals:
            interval = CandleInterval(interval_str)

            logger.info(
                "fetching_historical",
                symbol=symbol,
                interval=interval_str,
                days=days,
            )

            candles = await self._binance.fetch_historical(
                symbol=symbol,
                interval=interval,
                days=days,
            )

            async with Database.session() as session:
                repo = CandleRepository(session)
                count = await repo.upsert_candles(candles)

            logger.info(
                "historical_fetch_complete",
                symbol=symbol,
                interval=interval_str,
                candles_saved=count,
            )

    async def _fetch_latest(self, interval: CandleInterval) -> None:
        """Fetch latest candles for an interval."""
        symbol = self._settings.default_symbol

        # Fetch last 10 candles to handle any gaps
        candles = await self._binance.fetch_latest(
            symbol=symbol,
            interval=interval,
            count=10,
        )

        async with Database.session() as session:
            repo = CandleRepository(session)
            count = await repo.upsert_candles(candles)

        logger.debug(
            "latest_fetch_complete",
            symbol=symbol,
            interval=interval.value,
            candles_updated=count,
        )


async def main() -> None:
    """Main entry point for standalone ingestion worker."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )

    scheduler = IngestionScheduler()

    try:
        await scheduler.start()

        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("shutdown_requested")
    finally:
        await scheduler.stop()
        await Database.close()


if __name__ == "__main__":
    asyncio.run(main())
