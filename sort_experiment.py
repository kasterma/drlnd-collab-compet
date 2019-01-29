import numpy as np

NROW=3
NCOL=5

dat = np.random.uniform(-1, 1, NROW * NCOL).reshape(NROW, NCOL)
dat2 = np.random.uniform(-1, 1, NROW * NCOL).reshape(NROW, NCOL)

dat_flat = dat.flatten()
dat_sort_idx = np.argsort(dat_flat)


def sort_dat(d):
    return d.flatten()[dat_sort_idx].reshape(NROW, NCOL)
