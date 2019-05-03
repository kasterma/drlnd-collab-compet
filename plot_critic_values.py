import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from collabcompet.config import config

import matplotlib.pyplot as plt

from collabcompet.orm import CriticValue, EpisodeScore

engine = create_engine(f"sqlite:///{config['database_file']}", connect_args={'check_same_thread': False})

Session = sessionmaker(bind=engine)
session = Session()

critic_value = session.query(CriticValue).filter_by(input_id=200).all()
vals = [c.value1 for c in critic_value]
vals = pd.DataFrame({'value1':vals})

[c.episode_idx for c in critic_value]

vals.plot()
plt.show()

critic_values = session.query(CriticValue).filter_by(run_id=43).filter(CriticValue.input_id <= 400).all()
len(critic_values)
val_dat = pd.DataFrame([{'episode': c.episode_idx, 'input': c.input_id, 'value1': c.value1, 'value2': c.value2}
                        for c in critic_values])

for INPUT_ID in range(3, 399, 50):
    x = val_dat[val_dat.input == INPUT_ID]
    plt.scatter(x.episode, x.value1, alpha=0.4, s=0.1)
    plt.scatter(x.episode, x.value2, alpha=0.4, s=0.1)
plt.show()

for INPUT_ID in range(6, 399, 50):
    x = val_dat[val_dat.input == INPUT_ID]
    plt.scatter(x.episode, x.value1, alpha=0.4, s=0.1)
    plt.scatter(x.episode, x.value2, alpha=0.4, s=0.1)
plt.show()

for INPUT_ID in range(7, 399, 50):
    x = val_dat[val_dat.input == INPUT_ID]
    plt.scatter(x.episode, x.value1, alpha=0.4, s=0.1)
    plt.scatter(x.episode, x.value2, alpha=0.4, s=0.1)
plt.show()

for INPUT_ID in range(8, 399, 50):
    x = val_dat[val_dat.input == INPUT_ID]
    plt.scatter(x.episode, x.value1, alpha=0.4, s=0.1)
    plt.scatter(x.episode, x.value2, alpha=0.4, s=0.1)
plt.show()

scores = session.query(EpisodeScore).filter_by(run_id=43).all()
scored_df = pd.DataFrame([{'episode': s.episode_idx, 'score': s.score} for s in scores])
plt.plot(scored_df.episode, np.convolve(scored_df.score, np.ones(100)/100, mode='same'))
plt.show()
