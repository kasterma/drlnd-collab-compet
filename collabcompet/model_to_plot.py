import glob
import re

import numpy as np

import torch
import matplotlib.pyplot as plt
from collections import defaultdict

from collabcompet.model import Actor, Critic


# # def import_data()
# files = glob.glob('runs/Jan12_20-11-33/checkpoint_actor_*')
# regex = re.compile(r'runs/Jan12_20-11-33/checkpoint_([a-z_]*)-([0-9]*)\.pth')

data = defaultdict(lambda: defaultdict(dict))

for filename in files:
    m = regex.search(filename)
    if m:
        agent = Critic(48, 4) if m.group(1) == "critic" else Actor(24, 2)
        agent.load_state_dict(torch.load(filename))
        for k, v in agent.named_parameters():
            data[m.group(1)][k][int(m.group(2))] = v.detach().numpy()


dat_sort_idx_diff = a1.fc2.weight.detach().numpy() - a2150.fc2.weight.detach().numpy()
dat_sort_idx = np.argsort(dat_sort_idx_diff.flatten())


def sort_dat(d, nrow, ncol, dat_sort_idx):
    return d.flatten()[dat_sort_idx].reshape(nrow, ncol)

def plot():

    episode_list = [a1, a50, a100, a150, a200, a250, a300, a350, a400, a450, a500, a550, a600,
                    a650, a700, a750, a800, a850, a900, a950, a1000, a1050, a1100, a1150, a1200, a1250,
                    a1300, a1350, a1400, a1450, a1500, a1550, a1600, a1650, a1700, a1750, a1800,
                    a1850, a1900, a1950, a2000, a2050, a2100, a2150]

    params = dict(episode_list[0].named_parameters()).keys()

    dat_dict_weight = defaultdict(list)
    for episode in episode_list:
        temp_dict = dict(episode.named_parameters())
        for param in params:
            dat_dict_weight[param].append(temp_dict[param].detach().numpy())

    ncols = 3
    nrows = len(episode_list) // ncols
    if ((len(episode_list) - 1) % 3) != 0:
        nrows += 1

    print(nrows)

    for i, key in enumerate(dat_dict_weight.keys()):
        print(key)
        if key == "fc2.weight":
            print(key)
            n_episode_list = len(dat_dict_weight[key])
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
            print(axs)
            for idx in range(n_episode_list - 1):
                diff = dat_dict_weight[key][idx] - dat_dict_weight[key][idx+1]
                # print(f"Episode index: {idx}, min: {np.min(diff)}, max: {np.max(diff)}")
                if len(diff.shape) == 1:
                    diff = np.expand_dims(diff, 0)
                pos_c = idx % ncols
                pos_r = idx // ncols
                print(pos_c, pos_r)
                axs[pos_r, pos_c].imshow(sort_dat(diff, diff.shape[0], diff.shape[1], dat_sort_idx), interpolation="nearest", aspect='auto', cmap='seismic', label=key, vmin=-0.1, vmax=0.1)
            plt.savefig(f"plots/plot-{key}.png")
            plt.show()
