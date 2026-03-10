"""initial migration

Revision ID: 001
Revises: 
Create Date: 2025-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Создание таблицы verification_requests
    op.create_table(
        'verification_requests',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('result', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_verification_requests_id'), 'verification_requests', ['id'], unique=False)
    
    # Создание таблицы sources
    op.create_table(
        'sources',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('url', sa.String(), nullable=False),
        sa.Column('domain', sa.String(), nullable=True),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('date', sa.DateTime(), nullable=True),
        sa.Column('trust_level', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_sources_url'), 'sources', ['url'], unique=True)
    op.create_index(op.f('ix_sources_domain'), 'sources', ['domain'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_sources_domain'), table_name='sources')
    op.drop_index(op.f('ix_sources_url'), table_name='sources')
    op.drop_table('sources')
    op.drop_index(op.f('ix_verification_requests_id'), table_name='verification_requests')
    op.drop_table('verification_requests')

