from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260718_0005"
down_revision = "20260718_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "eval_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("eval_type", sa.String(length=50), nullable=False),
        sa.Column("model_key", sa.String(length=100), nullable=False),
        sa.Column("metric_name", sa.String(length=100), nullable=False),
        sa.Column("scope", sa.String(length=50), nullable=False, server_default="all"),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_eval_results_eval_type", "eval_results", ["eval_type"], unique=False)
    op.create_index("ix_eval_results_model_key", "eval_results", ["model_key"], unique=False)
    op.create_index("ix_eval_results_metric_name", "eval_results", ["metric_name"], unique=False)
    op.create_index("ix_eval_results_scope", "eval_results", ["scope"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_eval_results_scope", table_name="eval_results")
    op.drop_index("ix_eval_results_metric_name", table_name="eval_results")
    op.drop_index("ix_eval_results_model_key", table_name="eval_results")
    op.drop_index("ix_eval_results_eval_type", table_name="eval_results")
    op.drop_table("eval_results")
