"""llm call traces: prompt and response text for chosen purposes

Revision ID: a1d3e5f7b9c2
Revises: c73f27fb3f13
Create Date: 2026-11-24 07:10:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'a1d3e5f7b9c2'
down_revision: Union[str, None] = 'c73f27fb3f13'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('llm_calls', sa.Column('prompt', sa.Text(), nullable=True))
    op.add_column('llm_calls', sa.Column('response', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('llm_calls', 'response')
    op.drop_column('llm_calls', 'prompt')
