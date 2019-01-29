"""ORM

Set up the object relational mapping (and general other database setup and interaction).
"""

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import yaml
import logging.config
from alembic.config import Config
from alembic import command

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("orm")


Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)  # has autoincrement semantics by default
    note = Column(String)
    config = Column(String)

    def __repr__(self):
        return f"Run(id={self.id}, note={self.note})"


class EpisodeScore(Base):
    __tablename__ = "scores"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"))
    episode_idx = Column(Integer)
    score = Column(Float)


engine = create_engine("sqlite:///data/rundb.sqlite", echo=True)
Base.metadata.create_all(engine)
alembic_cfg = Config("alembic.ini")
command.stamp(alembic_cfg, "head")
Session = sessionmaker(bind=engine)
session = Session()
run = None
config = None


def start_run(run_id=None, note=None):
    global run, config
    with open("config.yaml") as conf_file:
        conf_str = conf_file.read()
        config = yaml.load(conf_str)
    run = Run(id=run_id, note=note, config=conf_str)
    session.add(run)
    session.commit()
    log.info("Starting run %s", run)


def current_runid():
    """Given that the run was started with start_run with no explicit given run_id this will allow you to retrieve the
    assigned run_id.
    """
    global run
    return run.id


def save_score(episode_idx, score):
    score = EpisodeScore(run_id=run.id, episode_idx=episode_idx, score=score)
    session.add(score)
    session.commit()
