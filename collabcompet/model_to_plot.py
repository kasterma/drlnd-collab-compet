import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
from collections import defaultdict

from collabcompet.config import config
from collabcompet.agents import MADDPG
from collabcompet.orm import session, Model, load_config_from_db

from pyecharts.charts import Scatter


class NNAnalysis:

    def __init__(self, run_id):

        self.run_id = run_id
        load_config_from_db(run_id)
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

    def diff_dats_range_sort(self, epi_2, epi_1):
        dat = self.dats
        actor_1_local_fc2wts = dat.loc[(dat.label == "fc2.weight") & (dat.net == "actor_1_local")]
        diff_series = actor_1_local_fc2wts[actor_1_local_fc2wts.loc[:, 'episode_idx'] == epi_2].loc[:, 'value'] - \
                      actor_1_local_fc2wts[actor_1_local_fc2wts.loc[:, 'episode_idx'] == epi_1].loc[:, 'value']
        return np.argsort(diff_series)

    def diff_episode_range_sort(self, epi_2, epi_1):
        actor = "actor_1"
        param = "fc2.weight"
        dat_sort_idx_diff = self.data[actor][param][epi_2] - self.data[actor][param][epi_1]
        return np.argsort(dat_sort_idx_diff.flatten())

    def plot(self):
        dat_sort_idx = self.diff_dats_range_sort(2050, 2400)
        ncols = 3
        nrows = len(self.episode_list) // ncols
        if ((len(self.episode_list) - 1) % 3) != 0:
            nrows += 1

        print(nrows)
        wts = self.dats.loc[(self.dats.label == "fc2.weight") & (self.dats.net == "actor_1_local")]

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        print(axs)
        for idx in range(len(self.episode_list) - 1):
            diff = wts.loc[wts.loc[:, 'episode_idx'] == self.episode_list[idx + 1]].loc[:, 'value'] - \
                  wts.loc[wts.loc[:, 'episode_idx'] == self.episode_list[idx]].loc[:, 'value']

            diff = np.array(diff).reshape((150, 250))

            if len(diff.shape) == 1:
                diff = np.expand_dims(diff, 0)
            pos_c = idx % ncols
            pos_r = idx // ncols
            print(pos_c, pos_r)
            axs[pos_r, pos_c].imshow(self.sort_dat(diff, diff.shape[0], diff.shape[1], dat_sort_idx),
                                     interpolation="nearest", aspect='auto', cmap='seismic', label="fc2.weights",
                                     vmin=-0.1, vmax=0.1)
        plt.savefig(f"plots/plot-fc2.weights.png")
        plt.show()

    def get_wts_over_episodes(self, label="fc2.weight", net="actor_1_local", idx=2):
        wts = self.dats.loc[(self.dats.label == label) & (self.dats.net == net)]
        return wts.loc[idx, ['episode_idx', 'value']].set_index('episode_idx')

    def plot_wt(self, ):
        for idx in range(1, 11):
            wts = self.get_wts_over_episodes(idx=idx)
            wts.plot(title=f"idx{idx}")
        plt.show()

    def compute_corss(self):
        cors = np.zeros((10, 10))
        for i in range(0, 10):
            for j in range(0, 10):
                cors[i, j] = self.get_wts_over_episodes(idx=i).loc[:, "value"].corr(
                    self.get_wts_over_episodes(idx=j).loc[:, "value"])
        return cors
