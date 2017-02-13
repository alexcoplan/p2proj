# turns generate JSON into score using music21

from score_utils import render_music21
import json
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input-file", type=str, default="out/gend.json",
    help="JSON file containing score (and possibly entropy data)")
parser.add_argument("--plot-entropies", default=False, action="store_true",
    help="Plot entropy data using matplotlib")
parser.add_argument("--svg-path", type=str, default="out/entropy_plot.svg")

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
    
    plt.title("Entropy-time plot for RNN Sample (Random Walk)")
    h_xent, = plt.plot(xents, label="cross entropy")
    h_dent, = plt.plot(dents, label="distribution entropy")
    plt.legend(handles=[h_xent, h_dent])
    plt.xlabel("event number")
    plt.ylabel("entropy/bits")
    plt.savefig(args.svg_path)

