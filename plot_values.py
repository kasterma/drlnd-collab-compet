import matplotlib.pyplot as plt
from collabcompet import NNAnalysis
import numpy as np
import pandas as pd

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
