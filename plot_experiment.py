import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

x1 = np.arange(100).reshape((1,100))
x2 = 20 * np.random.uniform(2, 5, 150*3).reshape((3, 150))+50

xs = {'x1': x1, 'x2': x2}

vmin = np.min([x.min() for x in xs.values()])
vmax = np.max([x.max() for x in xs.values()])
norm = colors.Normalize(vmin=vmin, vmax=vmax)

fig, axs = plt.subplots(1, len(xs))

images = []

for idx, kv in enumerate(xs.items()):
    print(kv[0])
    #axs[idx].title = kv[0]
    images.append(axs[idx].imshow(kv[1], aspect='auto', norm=norm))

fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=0.1)
plt.suptitle("Helleo")
plt.show()
