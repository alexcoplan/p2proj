import numpy as np # type: ignore
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", type=str,
  help="JSON file containing seqlevel cross-entropies")

args = parser.parse_args()

def standard_error(npa):
  return np.std(npa, ddof=1) / np.sqrt(np.size(npa))

def ninetyfive_confidence(npa):
  return standard_error(npa) * 1.96

with open(args.input_file, "r") as f:
  xent_j = json.load(f)

xents = np.array(xent_j["xents"])

print("mean:", np.mean(xents))
print("stderr:", standard_error(xents))
print("95conf:", ninetyfive_confidence(xents))
print("sstdev:", np.std(xents, ddof=1))
