import time

import matplotlib.pyplot as plt
from collabcompet import NNAnalysis
import numpy as np
import pandas as pd
import math

x = NNAnalysis(20)
dat = x.dats

# the weights for the two different episodes we want to look at
prop = (dat.label == "fc2.weight") & (dat.net == 'actor_1_local')
a = dat.loc[prop & (dat.episode_idx == 750)]
b = dat.loc[prop & (dat.episode_idx == 950)]

c = b.value - a.value
sortidx = np.argsort(c)

eps = np.unique(dat.episode_idx)
N = 51
ncols = 3
nrows = N // ncols
if ((N ) % 3) != 0:
    nrows += 1

fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
for idx in range(N):
    zz = dat.loc[prop & (dat.episode_idx == eps[idx])].value - dat.loc[prop & (dat.episode_idx == eps[idx + 1])].value
    xyz = zz.iloc[sortidx]
    axs[idx // ncols, idx % ncols, ].imshow(np.array(xyz.values).reshape(250, 150),
                    interpolation="nearest", aspect='auto', cmap='seismic', label=eps[idx],
                    vmin=-0.1, vmax=0.1)  # TODO: Wouter fix this
plt.show()

c.iloc[sortidx]

# quick plot of a specific weight over time
dat.loc[dat.net == 'actor_1_local']
dat.loc[dat.net == 'actor_1_local'].loc[37491]
dat.loc[dat.net == 'actor_1_local'].loc[37491].value

xx = dat.loc[dat.net == 'actor_1_local'].loc[37491]
plt.plot(xx.episode_idx, xx.value)
plt.show()

N = 5
BASE = 37300
fig, axs = plt.subplots(nrows=N, ncols=1)
for idx in range(N):
    xx = dat.loc[dat.net == 'actor_1_local'].loc[BASE + idx]
    axs[idx].plot(xx.episode_idx, xx.value)
plt.show()

datr = dat[prop]


def in_class_simple(t0, epsilon, theta):
    return in_class(t0//50, (t0 + 200)//50, (t0 + 400)//50, epsilon, theta)


def in_class(t0, t1, t2, epsilon, theta):
    def f(wt):
        diff_during = math.fabs(wt[t1] - wt[t0])
        diff_after = math.fabs(wt[t2] - wt[t1])
        return diff_during > theta and diff_after < epsilon

    return f


def in_class_simple_t(t0):
    return in_class_t(t0//50, (t0 + 200)//50, (t0 + 400)//50)


def in_class_t(t0, t1, t2):
    def f(wt):
        diff_during = math.fabs(wt[t1] - wt[t0])
        diff_after = math.fabs(wt[t2] - wt[t1])
        return (diff_during, diff_after)

    return f


p = in_class_simple(1700, 1, 4)
val_fn = in_class_t(1900//50, 2050//50, 2200//50)

class TimeIt:
    def __init__(self, id):
        self.id = id

    def __enter__(self):
        self.start = time.monotonic()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.monotonic()
        print(f"Took {self.end - self.start}")


with TimeIt("run1"):
    x = [i for i in range(37500) if p(datr.loc[i, 'value'].values)]


with TimeIt("run2"):
    x = [val_fn(datr.loc[i, 'value'].values) for i in range(37500)]

np.max(x, axis=0)
np.mean(x, axis=0)
np.min(x, axis=0)

[(a,b) for a,b in x if a > 0.05 and b < 0.01]
[i for i, x in enumerate(x) if x[0] == 0.0 and x[1] == 0.0]


with TimeIt("test1") as timer:
    [i for i in range(5000) if p(datr.loc[i, 'value'].values)]


def from40(wt):
    return math.fabs(wt[40]-wt[50])


with TimeIt("2"):
    w40_50 = [from40(datr.loc[i, 'value'].values) for i in range(37500)]


engine = create_engine(f"sqlite:///{config['database_file']}", connect_args={'check_same_thread': False})
Session = sessionmaker(bind=engine)
session = Session()

session.query(EpisodeScore).filter_by(run_id=20).all()
[s.score for s in session.query(EpisodeScore).filter_by(run_id=20).all()]
ss = [s.score for s in session.query(EpisodeScore).filter_by(run_id=20).all()]
plt.plot(range(len(ss)), ss)
plt.show()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
