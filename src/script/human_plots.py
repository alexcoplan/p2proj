import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import argparse
import json
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--data-file", type=str, default="out/survey_results.json",
  help="location of JSON file containing survey results")

args = parser.parse_args()

with open(args.data_file, "r") as f:
  survey_j = json.load(f)

model_j = survey_j["by_model"]
sample_j = survey_j["by_sample"]
xp_counts = survey_j["xp_counts"]

xp_idxs = np.arange(4)

def standard_error(npa):
  return np.std(npa, ddof=1) / np.sqrt(np.size(npa))

def ninetyfive_confidence(npa):
  return standard_error(npa) * 1.96

def coherency_stats(breakdown):
  points = [0.0] * breakdown[0] + [0.5] * breakdown[1] + [1.0] * breakdown[2]
  cdat = np.array(points)
  mean = np.mean(cdat)
  err = standard_error(cdat)
  return mean, err

def weighted_stats(human_confidences, comp_confidences):
  points = []
  for idx,amt in enumerate([0.0,0.25,0.5,0.75,1.0]):
    points += [amt]*human_confidences[idx]
    points += [-amt]*comp_confidences[idx]
  wdat = np.array(points)
  mean = np.mean(wdat)
  err = standard_error(wdat)
  return mean,err

def human_stats(comp_count, human_count):
  points = [-1.0] * comp_count + [1.0] * human_count
  hdat = np.array(points)
  mean = np.mean(hdat)
  err = standard_error(hdat)
  return mean, err

class BarSet():
  def __init__(self):
    self.means = []
    self.errors = []

  def add_point(self, mean, error):
    self.means.append(mean)
    self.errors.append(error)

model_proto = {
  "coherency" : BarSet(),
  "human" : BarSet(),
  "weighted" : BarSet()
}

per_model = {
  "MVS" : copy.deepcopy(model_proto),
  "RNN" : copy.deepcopy(model_proto),
  "Bach" : copy.deepcopy(model_proto)
}

model_names = ["MVS", "RNN", "Bach"]

num_xps = 4

for model in model_names:
  total_coherencies = np.array([0]*3)
  total_hvals = np.array([0,0])
  total_human_confs = np.array([0]*5)
  total_comp_confs = np.array([0]*5)

  model_dat = per_model[model]

  for xp in range(4):
    counts = model_j[model][xp]
    coherency_counts = counts["coherency_breakdown"]
    model_dat["coherency"].add_point(*coherency_stats(coherency_counts))

    hcounts = [counts["comp_count"], counts["human_count"]]
    model_dat["human"].add_point(*human_stats(*hcounts))

    confidences = [counts["human_confidences"], counts["comp_confidences"]]
    model_dat["weighted"].add_point(*weighted_stats(*confidences))

    total_coherencies += np.array(coherency_counts)
    total_hvals += np.array(hcounts)
    total_human_confs += np.array(confidences[0])
    total_comp_confs += np.array(confidences[1])

  model_dat["coherency"].add_point(*coherency_stats(total_coherencies))
  model_dat["human"].add_point(*human_stats(*total_hvals))
  model_dat["weighted"].add_point(*weighted_stats(total_human_confs,
    total_comp_confs))

idxs = np.arange(num_xps+1)
width = 0.2
the_capsize = 4

xp_labels = ["Novice", "Intermediate", "Advanced", "Expert", "Overall"]
model_colours = ['g','b','r']

# coherency plot
plt.figure(1)
plt.title("Mean coherency rating by model and experience")

model_bars = []
for idx,model in enumerate(model_names):
  cdat = per_model[model]["coherency"]
  bars = plt.bar(idxs + width*idx, cdat.means, width, color=model_colours[idx],
      yerr=cdat.errors, capsize=the_capsize)
  model_bars.append(bars[0])

plt.legend(model_bars, model_names)
plt.xlabel("Experience")
plt.ylabel("Coherency")
plt.xticks(idxs, xp_labels)
ax = plt.gca()
ax.set_ylim([0.0,1.0])

# turing-test plot
plt.figure(2)
plt.title("Mean human classification rate by model and experience")

model_bars = []
for idx,model in enumerate(model_names):
  hdat = per_model[model]["human"]
  bars = plt.bar(idxs + width*idx, hdat.means, width, color=model_colours[idx],
      yerr=hdat.errors, capsize=the_capsize)
  model_bars.append(bars[0])

plt.legend(model_bars, model_names)
plt.xlabel("Experience")
plt.ylabel("Human/computer classification (±1)")
plt.xticks(idxs, xp_labels)
plt.axhline(y=0, color='k') # show y=0

# weighted turing-test plot
plt.figure(3)
plt.title("Mean confidence-weighted classification by model and experience")

model_bars = []
for idx,model in enumerate(model_names):
  wdat = per_model[model]["weighted"]
  bars = plt.bar(idxs + width*idx, wdat.means, width, color=model_colours[idx],
      yerr=wdat.errors, capsize=the_capsize)
  model_bars.append(bars[0])

plt.legend(model_bars, model_names)
plt.xlabel("Experience")
plt.ylabel("Weighted human/computer classification (±1)")
plt.xticks(idxs, xp_labels)
plt.axhline(y=0, color='k')
ax = plt.gca()
ax.set_ylim([-0.45,0.7])

# show response/experiecne distribution
plt.figure(4)
plt.title("Distribution of survey responses by experience")
plt.xlabel("Experience")
plt.ylabel("Responses")
plt.bar(idxs[0:4], xp_counts, width*2, color='b')
plt.xticks(idxs[0:4], xp_labels[0:4])
plt.show()

