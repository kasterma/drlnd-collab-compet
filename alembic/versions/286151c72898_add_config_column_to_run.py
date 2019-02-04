"""Add config column to run

Revision ID: 286151c72898
Revises: 
Create Date: 2019-01-15 09:04:26.576280

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
from sqlalchemy import String, Column

revision = '286151c72898'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("runs",
                  Column("config", String()))


def downgrade():
    op.drop_column("runs", "config")
