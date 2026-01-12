"""Database connection management."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from chronarch.config import get_settings


class Database:
    """Async database connection manager."""

    _engine = None
    _session_factory = None

    @classmethod
    def get_engine(cls):
        """Get or create the async engine."""
        if cls._engine is None:
            settings = get_settings()
            cls._engine = create_async_engine(
                settings.database_url,
                echo=settings.debug,
                pool_size=5,
                max_overflow=10,
            )
        return cls._engine

    @classmethod
    def get_session_factory(cls) -> async_sessionmaker[AsyncSession]:
        """Get or create the session factory."""
        if cls._session_factory is None:
            cls._session_factory = async_sessionmaker(
                cls.get_engine(),
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return cls._session_factory

    @classmethod
    @asynccontextmanager
    async def session(cls) -> AsyncGenerator[AsyncSession, None]:
        """Context manager for database sessions."""
        session_factory = cls.get_session_factory()
        async with session_factory() as session:
            yield session

    @classmethod
    async def close(cls) -> None:
        """Close the database engine."""
        if cls._engine is not None:
            await cls._engine.dispose()
            cls._engine = None
            cls._session_factory = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI to get database sessions."""
    async with Database.session() as session:
        yield session
