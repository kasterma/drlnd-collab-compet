"""Added episode_idx column to Model

Revision ID: 67428bb7f6db
Revises: 50494450f2df
Create Date: 2019-02-08 11:32:56.894752

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '67428bb7f6db'
down_revision = '50494450f2df'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('model', sa.Column('episode_idx', sa.Integer(), nullable=True))


def downgrade():
    op.drop_column('model', 'episode_idx')
