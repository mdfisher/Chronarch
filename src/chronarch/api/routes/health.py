"""Health check endpoint."""

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from chronarch.api.schemas import HealthResponse
from chronarch.config import get_settings
from chronarch.db.connection import get_session

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    session: AsyncSession = Depends(get_session),
) -> HealthResponse:
    """Check service health status."""
    settings = get_settings()

    # Check database connectivity
    db_healthy = False
    try:
        await session.execute(text("SELECT 1"))
        db_healthy = True
    except Exception:
        pass

    # Check if model is loaded (placeholder - inject predictor dependency)
    model_loaded = False
    try:
        # This would check the predictor instance
        model_loaded = settings.model_dir.exists()
    except Exception:
        pass

    # Determine overall status
    if db_healthy and model_loaded:
        status = "healthy"
    elif db_healthy:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        version="0.1.0",
        database=db_healthy,
        model_loaded=model_loaded,
        last_data_update=None,  # Would query from DB
    )
