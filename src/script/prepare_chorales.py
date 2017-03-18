# this script reqiures:
# - Python 3
# - Latest music21 (in particular, 8b5d422 where I fix a bug in BWV 348)
# and it should be run from the main src directory

import music21 # type: ignore
import json
import argparse

from collections import Counter # for counting events in a given types domain
from typing import Dict,List # type annotations

from json_encoders import NoIndent, NoIndentEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--output-file", type=str, 
  default="corpus/chorale_dataset.json", help="path to output json to")
parser.add_argument("--ignore-modes", default=False, action="store_true",
  help="""
  this option does not include the mode of each choarle in the output, allowing
  those chorales for which the mode is not specified to be included in the
  corpus
  """)
parser.add_argument("--transpose-window", type=int, default=0,
  help="""
  transpose each chorale +/- n semitones so as to inflate the corpus and create
  a larger dataset. the code will ensure that transposing a chorale does not
  exceed the vocal range already present in the corpus
  """)
parser.add_argument("--rnn", default=False, action="store_true",
  help="""
  configure the preparation script with sensible defaults for generating a
  corpus for the RNN. 

  equivalent to --ignore-modes --transpose-window 3
  """) 
parser.add_argument("--validation-chorales", type=int, default=30,
  help="""
  number of chorales to use for validation
  """)

args = parser.parse_args() 

if args.rnn:
  args.transpose_window = 3 # type: ignore
  args.ignore_modes = True # type: ignore

################################################################################
# Begin main script
################################################################################

# these numbers are the Riemenschenider numbers of those chorale harmonisations
# in the corpus which have a unique tune w.r.t. other harmonisations
#
# these can be used for the validation set
individual_chorale_rs_nums = [
  177,186,39,48,153,128,159,180,208,5,124,304,1,10,230,245,210,197,56,200,196,
  228,311, 224,75,239,154,353,158,207,231,232,127,209,42,167,361,309,280,34,72,
  17,176,
]

validation_nums = individual_chorale_rs_nums[0:args.validation_chorales]

# event counters: these determine which values syntactic values of a given type
# are actually used in the corpus
counters : Dict[str, Counter] = {} 
types = ["pitch", "duration", "keysig", "timesig", "seqint", "intref", "rest"]
for t in types:
  counters[t] = Counter()

# this maps number of sharps in key -> pitch of referent
ref_map = {
  -4 : 8,
  -3 : 3,
  -2 : 10,
  -1 : 5,
   0 : 0,
   1 : 7,
   2 : 2,
   3 : 9,
   4 : 4
}

# quantize to semiquavers
def ql_quantize(ql):
  return round(ql/0.25)

def m21_to_internal(m21_notes, referent, anac_ql):
  c_notes = []

  prev_pitch = None
  prev_end_q = None # end = offset + duration

  counters["rest"].update([ql_quantize(anac_ql)])

  for n in m21_notes:
    counters["intref"].update([(n.pitch.midi - referent) % 12])

    if prev_pitch is not None:
      counters["seqint"].update([n.pitch.midi - prev_pitch])

    prev_pitch = n.pitch.midi

    duration_q = ql_quantize(n.duration.quarterLength)
    assert duration_q != 0
    offset_q = ql_quantize(n.offset)

    if prev_end_q is not None:
      counters["rest"].update([offset_q - prev_end_q])
    prev_end_q = offset_q + duration_q

    counters["pitch"].update([n.pitch.midi])
    counters["duration"].update([duration_q])
    c_notes.append([
      n.pitch.midi, 
      ql_quantize(n.offset + anac_ql), 
      duration_q
    ])

  return c_notes

print("Compiling corpus...")

# we will store the chorale json-like objects in here, and subsequently convert
# these objects to json
train_json = []
validate_json = []

# get bach chorales from the music21 core corpus, categorised according to
# Riemenschneider
bcl = music21.corpus.chorales.ChoraleListRKBWV()
num_chorales = max(bcl.byRiemenschneider.keys(), key=int)
base_chorales_added = 0 # type: int
train_chorales_added = 0 # type: int
validate_chorales_added = 0 # type: int

# transposition limits 
global_min_pitch = music21.pitch.Pitch('C4')
global_max_pitch = music21.pitch.Pitch('A5')
max_accidentals = 4 # cannot transpose into a key with more accidentals

for i in bcl.byRiemenschneider:
  info = bcl.byRiemenschneider[i]
  bwv = info["bwv"]
  title = info["title"]

  print("Processing %d of %d (BWV %s)" % (i, num_chorales, info["bwv"]))
  c = music21.corpus.parse('bach/bwv' + str(info["bwv"]))
  cStrip = c.stripTies(retainContainers=True)

  if len(c.parts) != 4:
    print(" * Skipping: BWV %s is not in four parts." % info["bwv"])
    continue

  if info["bwv"] == "36.4-2" or info["bwv"] == "432":
    print(" * WARNING: skipping due to inconvenient ornament.")
    continue
  elif info["bwv"] == "123.6":
    print(" * WARNING: skipping this chorale in double time.")
    continue

  # calculate anacrusis
  anac_bar_ql = c.measures(0,0).duration.quarterLength
  first_bar_ql = c.measures(1,1).duration.quarterLength

  anac_ql = first_bar_ql - anac_bar_ql # size of anacrusis
  time_sig_q = ql_quantize(first_bar_ql) # use first bar to determine time signature
  counters["timesig"].update([time_sig_q])

  # get time/key signature information
  ts = c.recurse().getElementsByClass('TimeSignature')[0]
  ks = c.recurse().getElementsByClass('KeySignature')[0]
  time_sig_str = ts.ratioString

  counters["keysig"].update([ks.sharps])
  key_sig_sharps = ks.sharps
  referent = ref_map[key_sig_sharps]
  
  if not args.ignore_modes:
    key_sig_major = True
    if isinstance(ks, music21.key.Key):
      key_sig_major = (ks.mode == "major")
    else:
      print(" * Skipping BWV %s has no mode specified." % info["bwv"])
      continue 
  
  sop = cStrip.parts[0].flat
  amb_analyser = music21.analysis.discrete.Ambitus()
  min_pitch, max_pitch = amb_analyser.getPitchSpan(sop)

  def transpose_and_add(amt):
    transd = sop.transpose(amt)
    ks_transd = ks.transpose(amt)

    direction = "up" if amt > 0 else "down"
    if amt == 0:
      title_ext = ""
    else:
      title_ext = " ({} {})".format(direction, amt) 

    internal_fmt = m21_to_internal(transd.notes, referent, anac_ql)

    obj = {
      "title" : title + title_ext,
      "bwv" : bwv,
      "time_sig_amt" : time_sig_q,
      "key_sig_sharps" : ks_transd.sharps,
      "notes" : NoIndent(internal_fmt)
    }

    global train_chorales_added, validate_chorales_added

    if not args.ignore_modes:
      obj["key_sig_major"] = key_sig_major

    if i in validation_nums:
      validate_chorales_added += 1
      validate_json.append(obj)
    else:
      train_chorales_added += 1
      train_json.append(obj)

  prev_added = train_chorales_added + validate_chorales_added

  # add the -ve transpositions
  for amt in range(-args.transpose_window, 0):
    if min_pitch.transpose(amt) < global_min_pitch:
      continue
    if abs(ks.transpose(amt).sharps) > max_accidentals:
      continue
    transpose_and_add(amt)

  # add the original and the +ve transpositions
  for amt in range(0, args.transpose_window+1):
    if max_pitch.transpose(amt) > global_max_pitch:
      continue
    if abs(ks.transpose(amt).sharps) > max_accidentals:
      continue
    transpose_and_add(amt)

  delta = (train_chorales_added + validate_chorales_added) - prev_added
  print("--> Added {} entries.".format(delta))

  base_chorales_added += 1

total_chorales_added = validate_chorales_added + train_chorales_added
validation_percentage = (validate_chorales_added / total_chorales_added) * 100

print("Done processing chorales.\n")
print("Total entries: {}.".format(total_chorales_added))
print("Train entries: {}.".format(train_chorales_added))
print("Validate enties: {}.".format(validate_chorales_added))
print("Validation set percentage: %.3f%%." % validation_percentage)
print("Transposition inflation: %.3f." %
    (total_chorales_added/base_chorales_added))

metadata_obj : Dict[str,NoIndent] = {}

print("\nSyntactic domains of musical types:")

# get syntactic domains for each type from counter objects
for t in types:
  label = t + "_domain"
  keys = list(counters[t].keys())
  keys.sort()
  print("{}: {}".format(label, keys))
  metadata_obj[label] = NoIndent(keys)

print("\nCompilation complete, writing JSON...")

outer_object = {
  "metadata" : metadata_obj,
  "corpus" : {
    "train" : train_json,
    "validate" : validate_json
  }
}

with open(args.output_file, 'w') as outfile:
  outfile.write(json.dumps(outer_object, indent=2, cls=NoIndentEncoder))

print("Done generating corpus!")

