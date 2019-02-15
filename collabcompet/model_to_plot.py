import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from collections import defaultdict

from collabcompet.config import config
from collabcompet.agents import MADDPG
from collabcompet.orm import session, Model


class NNAnalysis:

    def __init__(self, run_id):

        self.run_id = run_id
        self.data = defaultdict(lambda: defaultdict(dict))
        self.episode_list = np.unique([m.episode_idx for m in session.query(Model).filter_by(run_id=self.run_id).all()])

        agent: MADDPG = MADDPG(replay_memory_size=config['replay_memory_size'],
                               state_size=config['state_size'],
                               action_size=config['action_size'],
                               actor_count=config['actor_count'],
                               run_id=self.run_id)

        print(self.episode_list)
        self.dats = pd.concat([agent.load(run_id, int(episode)).asDataFrame(int(episode))
                               for episode in self.episode_list])

    def sort_dat(self, d, nrow, ncol, dat_sort_idx):
        return d.flatten()[dat_sort_idx].reshape(nrow, ncol)

    def diff_episode_range_sort(self, epi_2, epi_1):
        actor = "actor_1"
        param = "fc2.weight"
        dat_sort_idx_diff = self.data[actor][param][epi_2] - self.data[actor][param][epi_1]
        return np.argsort(dat_sort_idx_diff.flatten())

    def plot(self):
        dat_sort_idx = self.diff_episode_range_sort(2350, 2400)
        ncols = 3
        nrows = len(self.episode_list) // ncols
        if ((len(self.episode_list) - 1) % 3) != 0:
            nrows += 1

        print(nrows)
        actor = "actor_1"

        for i, key in enumerate(self.data[actor]):
            print(key)
            if key == "fc2.weight":
                print(key)
                fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
                print(axs)
                for idx in range(len(self.episode_list) - 1):
                    diff = self.data[actor][key][self.episode_list[idx + 1]] - self.data[actor][key][
                        self.episode_list[idx]]
                    if len(diff.shape) == 1:
                        diff = np.expand_dims(diff, 0)
                    pos_c = idx % ncols
                    pos_r = idx // ncols
                    print(pos_c, pos_r)
                    axs[pos_r, pos_c].imshow(self.sort_dat(diff, diff.shape[0], diff.shape[1], dat_sort_idx),
                                             interpolation="nearest", aspect='auto', cmap='seismic', label=key,
                                             vmin=-0.1, vmax=0.1)
                plt.savefig(f"plots/plot-{key}.png")
                plt.show()
