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

critic_value = session.query(CriticValue).filter_by(input_id=200).all()
vals = [c.value1 for c in critic_value]
vals = pd.DataFrame({'value1': vals})

[c.episode_idx for c in critic_value]

vals.plot()
plt.show()

critic_values = session.query(CriticValue).filter_by(run_id=43)\
    .filter(CriticValue.input_id <= 1000)\
    .filter(CriticValue.input_id > 400).all()
critic_values = critic_values.all()
len(critic_values)
val_dat = pd.DataFrame([{'episode': c.episode_idx, 'input': c.input_id, 'value1': c.value1, 'value2': c.value2}
                        for c in critic_values])
val_dat.set_index('episode', inplace=True, drop=False)

for INPUT_ID in range(403, 1000, 100):
    x = val_dat[val_dat.input == INPUT_ID]
    plt.scatter(x.episode, x.value1, alpha=0.4, s=0.1)
    plt.scatter(x.episode, x.value2, alpha=0.4, s=0.1)
plt.ylim((-0.1, 0.5))
plt.show()

for INPUT_ID in range(406, 1000, 25):
    x = val_dat[val_dat.input == INPUT_ID]
    plt.scatter(x.episode, x.value1, alpha=0.4, s=0.1)
    plt.scatter(x.episode, x.value2, alpha=0.4, s=0.1)
plt.ylim((-0.1, 0.5))
plt.show()

# zoomed in on interesting looking synchronized bump in graphs
for INPUT_ID in range(406, 1000, 25):
    x = val_dat[val_dat.input == INPUT_ID].loc[5340:5420]
    plt.scatter(x.episode, x.value1, s=0.1)
    plt.scatter(x.episode, x.value2, s=0.1)
plt.ylim((-0.1, 0.5))
plt.show()

# zoomed in on interesting looking synchronized bump in graphs, line plot
for INPUT_ID in range(0, 1000, 1):
    x = val_dat[val_dat.input == INPUT_ID].loc[5340:5420]
    plt.plot(x.episode, x.value1, alpha=0.1)
    plt.plot(x.episode, x.value2, alpha=0.1)
plt.ylim((-0.1, 0.4))
plt.show()

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

scores = session.query(EpisodeScore).filter_by(run_id=43).all()
scored_df = pd.DataFrame([{'episode': s.episode_idx, 'score': s.score} for s in scores])
plt.plot(scored_df.episode, scored_df.score)
plt.show()

scores = session.query(EpisodeScore).filter_by(run_id=43).all()
scored_df = pd.DataFrame([{'episode': s.episode_idx, 'score': s.score} for s in scores])
scored_df.set_index('episode', inplace=True, drop=False)
dd = scored_df
plt.plot(dd.episode, dd.score)
plt.ylim((-0.1, 3))
plt.show()


x = 8040
scores = session.query(EpisodeScore).filter_by(run_id=43).all()
scored_df = pd.DataFrame([{'episode': s.episode_idx, 'score': s.score} for s in scores])
scored_df.set_index('episode', inplace=True, drop=False)
dd = scored_df.loc[x:x+100]
plt.plot(dd.episode, dd.score)
plt.ylim((-0.1, 3))
plt.show()
