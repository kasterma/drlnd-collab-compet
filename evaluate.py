# code to draw the episode score graph for in the report

import numpy as np
import matplotlib.pyplot as plt


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
# dat = np.load("evaluate-scores-15.npy")
# plt.plot(dat, label="eval")
# plt.legend()
# plt.show()    # use this during development to show (not save) the graph
# plt.savefig("both-scores.png")
