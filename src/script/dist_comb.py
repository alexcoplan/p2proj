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
vpeak_dist = [3.0/4.0, 3.0/16.0, 1.0/32.0, 1.0/32.0]

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
  dist_a, dist_b = peak_dist, list(reversed(peak_dist))
  dist_c, dist_d = flat_dist, vpeak_dist
  dist_size = len(dist_a)
  assert dist_size == len(dist_b) == len(dist_c) == len(dist_d)
  arith_ab, geom_ab = combine_dists([dist_a, dist_b], 2.0)
  arith_cd, geom_cd = combine_dists([dist_c, dist_d], 2.0)
  bar_width = 0.4
  bar_idxs = np.arange(dist_size)

  fig = plt.figure(figsize=(7,7))
  
  ax_a  = fig.add_subplot(321)
  ax_b  = fig.add_subplot(323)
  ax_ab = fig.add_subplot(325)

  ax_c  = fig.add_subplot(322)
  ax_d  = fig.add_subplot(324)
  ax_cd = fig.add_subplot(326)

  dista_bars = ax_a.bar(bar_idxs, dist_a, bar_width, color='r')
  distb_bars = ax_b.bar(bar_idxs, dist_b, bar_width, color='r')
  distc_bars = ax_c.bar(bar_idxs, dist_c, bar_width, color='r')
  distd_bars = ax_d.bar(bar_idxs, dist_d, bar_width, color='r')

  arith_ab_bars = ax_ab.bar(bar_idxs, arith_ab, bar_width, color='b')
  geom_ab_bars  = ax_ab.bar(bar_idxs + bar_width, geom_ab, bar_width, color='g')
  
  arith_cd_bars = ax_cd.bar(bar_idxs, arith_cd, bar_width, color='b')
  geom_cd_bars  = ax_cd.bar(bar_idxs + bar_width, geom_cd, bar_width, color='g')

  for axes in [ax_a, ax_b, ax_ab, ax_c, ax_d, ax_cd]:
    axes.set_ylabel('Probability')
    axes.set_ylim([0.0,1.0])

  ax_a.set_title('Distribution A')
  ax_b.set_title('Distribution B')
  ax_c.set_title('Distribution C')
  ax_d.set_title('Distribution D')
  ax_ab.set_title('Combination of A and B')
  ax_cd.set_title('Combination of C and D')

  ax_ab.legend(
    (arith_ab_bars[0], geom_ab_bars[0]), 
    ('Arithmetic', 'Geometric')
  )
  ax_cd.legend(
    (arith_cd_bars[0], geom_cd_bars[0]), 
    ('Arithmetic', 'Geometric')
  )

  plt.tight_layout()
  plt.subplots_adjust(bottom=0.1)

  plt.savefig('comb.svg')
