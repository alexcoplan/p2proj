# tpye hints
from typing import Dict,Any

# libs
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import argparse
import json

""" returns (arithmetic, geometric) combinations of input distribution list """
def combine_dists(dist_list, entropy_bias):
  dists = np.array(dist_list)
  sums = np.sum(dists, axis=1)
  assert np.all(np.ones_like(sums) == sums)
  dist_size = dists.shape[1]

  max_entropy = np.log2(dist_size)
  dist_entropies = -np.sum(dists * np.log2(dists), axis=1)
  weights = np.expand_dims(dist_entropies ** (-entropy_bias), axis=0)

  arith = np.squeeze(np.dot(np.transpose(dists), np.transpose(weights)) /
      np.sum(weights))

  geo_weighted = np.power(dists, np.transpose(weights))
  product = np.product(geo_weighted, axis=0)
  rooted = np.power(product, 1.0 / np.sum(weights))
  geom = rooted / np.sum(rooted)

  return list(arith), list(geom)

parser = argparse.ArgumentParser()
parser.add_argument("--gen-tests", default=False, action="store_true",
  help="if this option is set, then the script will generate test cases for\
      both arithmetic and geometric distribution combination. otherwise, we\
      will plot some example combinations")

args = parser.parse_args()

flat_dist = [0.25]*4
skew_dist = [0.5,0.25,1.0/8.0, 1.0/8.0]
peak_dist = [3.0/4.0, 1.0/8.0, 1.0/16.0, 1.0/16.0]

distributions = {
  "both_flat" : [flat_dist]*2,
  "both_skew" : [skew_dist,skew_dist],
  "flat+skew" : [flat_dist,skew_dist],
  "skew+flat" : [skew_dist,flat_dist],
  "peak+flat" : [peak_dist,flat_dist],
  "flat+peak" : [flat_dist,peak_dist],
  "opposite_skew" : [skew_dist, list(reversed(skew_dist))],
  "opposite_peak" : [list(reversed(peak_dist)), peak_dist], 
  "skew+flat+oppskew" : [skew_dist, flat_dist, list(reversed(skew_dist))]
}

if args.gen_tests:
  examples = []
  for name, dists in distributions.items():
    dist_matrix = np.array(dists)
    dist_totals = np.sum(dist_matrix, axis=1)
    assert np.all(dist_totals == np.ones_like(dist_totals)) # check everything adds to 1
    dlen = len(dists[0])
    for dist in dists:
      assert len(dist) == dlen # check distributions all over n events

    for eb in [0.0, 1.0, 2.0, 6.0]:
      arith, geom = combine_dists(dists, eb)
      example_obj = {
        "example_name" : name,
        "dists" : dists,
        "entropy_bias" : eb,
        "arithmetic_comb" : arith,
        "geometric_comb" : geom
      }
      examples.append(example_obj)

  root_obj = {
    "dist_comb_examples" : examples
  }
  with open("test/combination_examples.json", 'w') as outfile:
    outfile.write(json.dumps(root_obj, indent=2))
else: 
  # if we're not generating tests, then we'll just generate an example plot
  dist_x, dist_y = peak_dist, list(reversed(peak_dist))
  dist_size = len(dist_x)
  assert dist_size == len(dist_y)
  arith, geom = combine_dists([dist_x, dist_y], 6.0)
  bar_width = 0.2
  bar_idxs = np.arange(dist_size)

  fig, ax = plt.subplots()
  distx_bars = ax.bar(bar_idxs, dist_x, bar_width, color='b')
  disty_bars = ax.bar(bar_idxs + bar_width, dist_y, bar_width, color='g')
  arith_bars = ax.bar(bar_idxs + bar_width*2, arith, bar_width, color='c')
  geom_bars  = ax.bar(bar_idxs + bar_width*3, geom, bar_width, color='m')

  ax.set_ylabel('Probability')
  ax.set_title('Comparison of Geometric and Arithmetic Distribution Combination')
  ax.legend(
    (distx_bars[0], disty_bars[0], arith_bars[0], geom_bars[0]), 
    ('Distribution A', 'Distribution B', 'Arithmetic', 'Geometric')
  )

  plt.show()
