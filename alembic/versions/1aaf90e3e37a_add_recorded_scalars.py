"""Add recorded scalars

Revision ID: 1aaf90e3e37a
Revises: 5b209f371e20
Create Date: 2019-05-03 13:26:32.256372

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1aaf90e3e37a'
down_revision = '5b209f371e20'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('recordedscalars',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.Integer(), nullable=True),
    sa.Column('episode_idx', sa.Integer(), nullable=True),
    sa.Column('label', sa.String(), nullable=True),
    sa.Column('value', sa.Float(), nullable=True),
    sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade():
    op.drop_table('recordedscalars')
