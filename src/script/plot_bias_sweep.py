import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib import cm # type: ignore
from mpl_toolkits.mplot3d import Axes3D # type: ignore

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", type=str, default="out/bias_sweep.json",
    help="JSON file containing measured data")

args = parser.parse_args()

fig = plt.figure()
ax = fig.gca(projection='3d')

with open(args.input_file) as json_data:
  obj = json.load(json_data)
  intra_biases = np.array(obj["intra_biases"])
  inter_biases = np.array(obj["inter_biases"])
  total_ents = np.array(obj["entropy_values"])

  x,y = np.meshgrid(intra_biases, inter_biases)
  ax.plot_surface(x, y, total_ents, cmap=cm.plasma)
  ax.set_xlabel("Intra-layer bias")
  ax.set_ylabel("Inter-layer bias")
  ax.set_zlabel("Test cross-entropy")
  nominal_ticks = [0.0, 0.05, 0.10, 0.15, 0.20]
  ax.set_xticks(nominal_ticks)
  ax.set_yticks(nominal_ticks)

ax.view_init(elev=30.0, azim=210)

plt.savefig('bias_plot.svg')
