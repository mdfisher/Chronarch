"""Initial schema with candles, predictions, and feature snapshots.

Revision ID: 001
Revises:
Create Date: 2026-01-11

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create candles table
    op.create_table(
        "candles",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("interval", sa.String(10), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(20, 8), nullable=False),
        sa.Column("high", sa.Numeric(20, 8), nullable=False),
        sa.Column("low", sa.Numeric(20, 8), nullable=False),
        sa.Column("close", sa.Numeric(20, 8), nullable=False),
        sa.Column("volume", sa.Numeric(30, 8), nullable=False),
        sa.Column("quote_volume", sa.Numeric(30, 8), nullable=False),
        sa.Column("num_trades", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_candles_symbol", "candles", ["symbol"])
    op.create_index("ix_candles_interval", "candles", ["interval"])
    op.create_index("ix_candles_timestamp", "candles", ["timestamp"])
    op.create_index(
        "ix_candles_symbol_interval_timestamp",
        "candles",
        ["symbol", "interval", "timestamp"],
        unique=True,
    )

    # Create predictions table
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("target_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon_hours", sa.Integer(), nullable=False),
        sa.Column("direction", sa.String(10), nullable=False),
        sa.Column("direction_confidence", sa.Float(), nullable=False),
        sa.Column("prob_up", sa.Float(), nullable=False),
        sa.Column("prob_down", sa.Float(), nullable=False),
        sa.Column("prob_flat", sa.Float(), nullable=False),
        sa.Column("price_predicted", sa.Numeric(20, 2), nullable=False),
        sa.Column("price_lower_10", sa.Numeric(20, 2), nullable=False),
        sa.Column("price_upper_90", sa.Numeric(20, 2), nullable=False),
        sa.Column("price_current", sa.Numeric(20, 2), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("actual_price", sa.Numeric(20, 2), nullable=True),
        sa.Column("actual_direction", sa.String(10), nullable=True),
        sa.Column("was_direction_correct", sa.Boolean(), nullable=True),
        sa.Column("was_in_confidence_interval", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_predictions_symbol", "predictions", ["symbol"])
    op.create_index("ix_predictions_timestamp", "predictions", ["timestamp"])
    op.create_index(
        "ix_predictions_target_timestamp", "predictions", ["target_timestamp"]
    )

    # Create feature snapshots table
    op.create_table(
        "feature_snapshots",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("interval", sa.String(10), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_feature_snapshots_symbol", "feature_snapshots", ["symbol"])
    op.create_index(
        "ix_feature_snapshots_timestamp", "feature_snapshots", ["timestamp"]
    )

    # Convert candles to TimescaleDB hypertable (run this manually if using TimescaleDB)
    # op.execute("SELECT create_hypertable('candles', 'timestamp')")


def downgrade() -> None:
    op.drop_table("feature_snapshots")
    op.drop_table("predictions")
    op.drop_table("candles")
