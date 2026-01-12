"""Prediction endpoints."""

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from chronarch.api.schemas import (
    DirectionPredictionResponse,
    FeatureImportanceResponse,
    ModelConfidenceResponse,
    PredictionListResponse,
    PredictionRequest,
    PredictionResponse,
    PricePredictionResponse,
)
from chronarch.config import get_settings
from chronarch.data.models import CandleInterval
from chronarch.data.repository import CandleRepository, PredictionRepository
from chronarch.db.connection import get_session
from chronarch.ml.predictor import Predictor

router = APIRouter(prefix="/predictions", tags=["predictions"])

# Predictor singleton (would use proper DI in production)
_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    """Get or create predictor instance."""
    global _predictor
    if _predictor is None:
        settings = get_settings()
        _predictor = Predictor(
            model_dir=settings.model_dir,
            horizon_hours=settings.prediction_horizon_hours,
            direction_threshold=settings.direction_threshold,
        )
    return _predictor


@router.post("", response_model=PredictionResponse)
async def create_prediction(
    request: PredictionRequest,
    session: AsyncSession = Depends(get_session),
    predictor: Predictor = Depends(get_predictor),
) -> PredictionResponse:
    """Generate a new prediction for the specified symbol."""
    settings = get_settings()

    # Get recent candle data
    candle_repo = CandleRepository(session)

    # Need enough data for feature calculation (at least 200 candles)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=200)  # 200 hourly candles

    df = await candle_repo.get_candles_as_dataframe(
        symbol=request.symbol,
        interval=CandleInterval.H1,
        start_time=start_time,
        end_time=end_time,
    )

    if df.empty or len(df) < 100:
        raise HTTPException(
            status_code=503,
            detail=f"Insufficient data for {request.symbol}. Need at least 100 candles.",
        )

    # Generate prediction
    result = predictor.predict(df=df, symbol=request.symbol)

    # Save to database
    pred_repo = PredictionRepository(session)
    db_prediction = predictor.to_db_prediction(result)
    db_prediction = await pred_repo.save_prediction(db_prediction)

    return PredictionResponse(
        id=db_prediction.id,
        symbol=result.symbol,
        timestamp=result.timestamp,
        target_timestamp=result.target_timestamp,
        horizon_hours=result.horizon_hours,
        direction=DirectionPredictionResponse(
            direction=result.direction,
            confidence=result.direction_confidence,
            prob_up=result.prob_up,
            prob_down=result.prob_down,
            prob_flat=result.prob_flat,
        ),
        price=PricePredictionResponse(
            predicted=result.price_predicted,
            lower_10=result.price_lower_10,
            upper_90=result.price_upper_90,
            current=result.price_current,
        ),
        model_version=result.model_version,
        created_at=datetime.now(timezone.utc),
    )


@router.get("/latest", response_model=PredictionResponse | None)
async def get_latest_prediction(
    symbol: str = Query(default="BTCUSDT"),
    session: AsyncSession = Depends(get_session),
) -> PredictionResponse | None:
    """Get the most recent prediction for a symbol."""
    pred_repo = PredictionRepository(session)
    prediction = await pred_repo.get_latest_prediction(symbol)

    if prediction is None:
        return None

    return PredictionResponse(
        id=prediction.id,
        symbol=prediction.symbol,
        timestamp=prediction.timestamp,
        target_timestamp=prediction.target_timestamp,
        horizon_hours=prediction.horizon_hours,
        direction=DirectionPredictionResponse(
            direction=prediction.direction.value,
            confidence=prediction.direction_confidence,
            prob_up=prediction.prob_up,
            prob_down=prediction.prob_down,
            prob_flat=prediction.prob_flat,
        ),
        price=PricePredictionResponse(
            predicted=float(prediction.price_predicted),
            lower_10=float(prediction.price_lower_10),
            upper_90=float(prediction.price_upper_90),
            current=float(prediction.price_current),
        ),
        model_version=prediction.model_version,
        created_at=prediction.created_at,
    )


@router.get("", response_model=PredictionListResponse)
async def list_predictions(
    symbol: str = Query(default="BTCUSDT"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    session: AsyncSession = Depends(get_session),
) -> PredictionListResponse:
    """List historical predictions with pagination."""
    pred_repo = PredictionRepository(session)

    # Get predictions
    predictions = await pred_repo.get_predictions(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        limit=page_size,
    )

    items = [
        PredictionResponse(
            id=p.id,
            symbol=p.symbol,
            timestamp=p.timestamp,
            target_timestamp=p.target_timestamp,
            horizon_hours=p.horizon_hours,
            direction=DirectionPredictionResponse(
                direction=p.direction.value,
                confidence=p.direction_confidence,
                prob_up=p.prob_up,
                prob_down=p.prob_down,
                prob_flat=p.prob_flat,
            ),
            price=PricePredictionResponse(
                predicted=float(p.price_predicted),
                lower_10=float(p.price_lower_10),
                upper_90=float(p.price_upper_90),
                current=float(p.price_current),
            ),
            model_version=p.model_version,
            created_at=p.created_at,
        )
        for p in predictions
    ]

    return PredictionListResponse(
        predictions=items,
        total=len(items),  # Would need COUNT query for true total
        page=page,
        page_size=page_size,
    )


@router.get("/features", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    predictor: Predictor = Depends(get_predictor),
) -> FeatureImportanceResponse:
    """Get feature importance from the current model."""
    importance = predictor.get_feature_importance()

    # Sort by importance descending
    sorted_importance = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)
    )

    return FeatureImportanceResponse(
        features=sorted_importance,
        model_version=predictor.model_version,
    )


@router.get("/confidence", response_model=ModelConfidenceResponse)
async def get_model_confidence(
    symbol: str = Query(default="BTCUSDT"),
    lookback: int = Query(default=100, ge=10, le=1000),
    session: AsyncSession = Depends(get_session),
    predictor: Predictor = Depends(get_predictor),
) -> ModelConfidenceResponse:
    """Get model confidence metrics based on recent prediction outcomes."""
    pred_repo = PredictionRepository(session)

    # Get predictions with outcomes
    predictions = await pred_repo.get_predictions(
        symbol=symbol,
        limit=lookback,
    )

    # Filter to those with outcomes
    evaluated = [p for p in predictions if p.actual_price is not None]

    if not evaluated:
        return ModelConfidenceResponse(
            model_version=predictor.model_version,
            direction_accuracy_recent=None,
            ci_coverage_recent=None,
            predictions_evaluated=0,
        )

    # Calculate metrics
    correct_directions = sum(1 for p in evaluated if p.was_direction_correct)
    in_ci = sum(1 for p in evaluated if p.was_in_confidence_interval)

    return ModelConfidenceResponse(
        model_version=predictor.model_version,
        direction_accuracy_recent=correct_directions / len(evaluated),
        ci_coverage_recent=in_ci / len(evaluated),
        predictions_evaluated=len(evaluated),
    )
