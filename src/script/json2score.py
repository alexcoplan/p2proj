# turns generate JSON into score using music21

from score_utils import render_music21
import json
import sys
import argparse
import numpy as np # type: ignore

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", type=str, default="out/gend.json",
    help="JSON file containing score (and possibly entropy data)")
parser.add_argument("--plot-entropies", default=False, action="store_true",
    help="Plot entropy data using matplotlib")
parser.add_argument("--svg-path", type=str, default="out/entropy_plot.svg")
parser.add_argument("--ylim", type=float, default=None,
    help="Set ylim for plot for side-by-side consistency")

args = parser.parse_args()


with open(args.input_file) as json_data:
  obj = json.load(json_data)
  assert obj["notes"] is not None, \
    "JSON object must contain array with key 'notes'"
  render_music21(obj["notes"])

  if args.plot_entropies:
    import matplotlib.pyplot as plt # type: ignore

    entropies = obj["entropies"]
    xents = entropies["cross_entropies"]
    dents = entropies["dist_entropies"]

    print("Plotting entropies...")
    print("mean xent: {}".format(np.mean(np.array(xents))))
    
    h_xent, = plt.plot(xents, label="cross entropy")
    h_dent, = plt.plot(dents, label="distribution entropy")
    plt.legend(handles=[h_xent, h_dent])
    plt.xlabel("event number")
    plt.ylabel("entropy/bits")
    if args.ylim is not None:
      plt.ylim([0.0,args.ylim])
    plt.savefig(args.svg_path)

