import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from collabcompet.config import config

import matplotlib.pyplot as plt

from collabcompet.orm import CriticValue, EpisodeScore, RecordedScalar

engine = create_engine(f"sqlite:///{config['database_file']}", connect_args={'check_same_thread': False})

Session = sessionmaker(bind=engine)
session = Session()


def plot_recorded_scalars(label, run_id):
    actor_scores = session.query(RecordedScalar).filter_by(label=label, run_id=run_id).all()
    scores = pd.DataFrame([{"idx": a.episode_idx, "val": a.value} for a in actor_scores])
    plt.scatter(scores.idx, scores.val, s=0.1)
    plt.title(label)
    plt.show()


plot_recorded_scalars("mean-scores", 44)
