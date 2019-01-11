# code to draw the episode score graph for in the report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from collabcompet.orm import EpisodeScore

engine = create_engine("sqlite:///data/rundb.sqlite", echo=True)
Session = sessionmaker(bind=engine)
session = Session()
scores = session.query(EpisodeScore).filter(EpisodeScore.run_id == 2)
score_dicts = [{'episode': s.episode_idx, 'score': s.score} for s in scores]
df = pd.DataFrame(score_dicts)
df_scores = df.reindex(df.episode)['score']
plt.scatter(df_scores.index, df_scores)
plt.show()


def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


dat = np.load("data/scores-28.npy")
#p lt.scatter(np.arange(len(dat)), dat, label="train")
plt.plot(dat, label="episodes")
ma = moving_average(dat)
plt.plot(np.concatenate([np.zeros(100), ma]), label="moving_average")
# plt.show()
plt.savefig("train-scores.png")

dat = np.load("data/evaluation-scores-28.npy")
plt.scatter(np.arange(len(dat)), dat, label="eval")
# plt.show()    # use this during development to show (not save) the graph
plt.savefig("evaluation-scores.png")
