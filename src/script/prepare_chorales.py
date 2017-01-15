# this script reqiures:
# - Python 3
# - Latest music21 (in particular, 8b5d422 where I fix a bug in BWV 348)
# and it should be run from the main src directory

from music21 import *
import json
import argparse

from json_encoders import NoIndent, NoIndentEncoder

# FIXME: de-duplicate since there are multiple harmonisation of the same melody!

parser = argparse.ArgumentParser()
parser.add_argument("--output-file", type=str, 
  default="corpus/chorale_dataset.json", help="path to output json to")
parser.add_argument("--ignore-modes", default=False, action="store_true",
  help="""
  this option does not include the mode of each choarle in the output, allowing
  those chorales for which the mode is not specified to be included in the
  corpus
  """)

args = parser.parse_args()

################################################################################
# Begin main script
################################################################################

# mask out which elements of each type are actually used
pitch_mask = [False] * 127
duration_mask = [False] * 64
sharps_mask = [False] * 15
timesig_mask = [False] * 32
seqint_mask = [False] * 42
rest_mask = [False] * 64

# quantize to semiquavers
def ql_quantize(ql):
  return round(ql/0.25)

# we will store the chorale json-like objects in here, and subsequently convert
# this object to json
json_objects = []

print("Compiling corpus...")

# get bach chorales from the music21 core corpus, categorised according to
# Riemenschneider
bcl = corpus.chorales.ChoraleListRKBWV()
num_chorales = max(bcl.byRiemenschneider.keys(), key=int)

for i in bcl.byRiemenschneider:
  info = bcl.byRiemenschneider[i]

  print("Processing %d of %d (BWV %s)" % (i, num_chorales, info["bwv"]))
  c = corpus.parse('bach/bwv' + str(info["bwv"]))
  cStrip = c.stripTies(retainContainers=True)

  if len(c.parts) != 4:
    print(" * Skipping: BWV %s is not in four parts." % info["bwv"])
    continue

  if info["bwv"] == "36.4-2" or info["bwv"] == "432":
    print(" * WARNING: skipping due to inconvenient ornament.")
    # note: 432 is a duplicate anyway
    continue

# calculate anacrusis
  anac_bar_ql = c.measures(0,0).duration.quarterLength
  first_bar_ql = c.measures(1,1).duration.quarterLength

  anacrusis_ql = first_bar_ql - anac_bar_ql # size of anacrusis
  time_sig_q = ql_quantize(first_bar_ql) # use first bar to determine time signature
  timesig_mask[time_sig_q - 1] = True

# get time/key signature information
  ts = c.recurse().getElementsByClass('TimeSignature')[0]
  ks = c.recurse().getElementsByClass('KeySignature')[0]
  time_sig_str = ts.ratioString

  sharps_mask[ks.sharps + 7] = True
  key_sig_sharps = ks.sharps
  
  if not args.ignore_modes:
    key_sig_major = True
    if isinstance(ks, key.Key):
      key_sig_major = (ks.mode == "major")
    else:
      print(" * Skipping BWV %s has no mode specified." % info["bwv"])
      continue 

  c_notes = []

  prev_pitch = None
  prev_end_q = None # end = offset + duration
  
  for n in cStrip.parts[0].flat.notes:
    if prev_pitch is not None:
      seqint_mask[n.pitch.midi - prev_pitch + 21] = True
    prev_pitch = n.pitch.midi

    duration_q = ql_quantize(n.duration.quarterLength)
    assert duration_q != 0
    offset_q = ql_quantize(n.offset)

    if prev_end_q is not None:
      rest_mask[offset_q - prev_end_q] = True
    prev_end_q = offset_q + duration_q

    pitch_mask[n.pitch.midi - 1]  = True
    duration_mask[duration_q - 1] = True
    c_notes.append([
      n.pitch.midi, 
      ql_quantize(n.offset + anacrusis_ql), 
      duration_q
    ])

  obj = {
      "title" : info["title"],
      "bwv" : info["bwv"],
      "time_sig_semis" : time_sig_q,
      "key_sig_sharps" : key_sig_sharps,
      "notes" : NoIndent(c_notes)
  } 

  if not args.ignore_modes:
    obj["key_sig_major"] = key_sig_major

  json_objects.append(obj)

print("Done processing chorales.\n")

# work out possible values of each type based on masks
pitch_domain = []
duration_domain = []
sharps_domain = []
time_sig_domain = []
seqint_domain = []
rest_domain = []

for idx,val in enumerate(pitch_mask):
  if val:
    pitch_domain.append(idx + 1)

print("Pitches used: ", end="")
print(pitch_domain)

for idx,val in enumerate(duration_mask):
  if val:
    duration_domain.append(idx + 1)

print("Durations used: ", end="")
print(duration_domain)

for idx,val in enumerate(sharps_mask):
  if val:
    sharps_domain.append(idx - 7)

print("Sharps used: ", end="")
print(sharps_domain)

for idx,val in enumerate(timesig_mask):
  if val:
    time_sig_domain.append(idx + 1)

print("Time signatures used: ", end="")
print(time_sig_domain)

for idx,val in enumerate(seqint_mask):
  if val:
    seqint_domain.append(idx - 21)

print("Intervals used (seqint): ", end="")
print(seqint_domain)

for idx,val in enumerate(rest_mask):
  if val:
    rest_domain.append(idx)

print("Rests used: ", end="")
print(rest_domain)

print("Compilation complete, writing JSON...")

outer_object = {
    "metadata" : {
      "pitch_domain" : NoIndent(pitch_domain),
      "duration_domain" : NoIndent(duration_domain),
      "key_sig_sharps_domain" : NoIndent(sharps_domain),
      "time_sig_domain" : NoIndent(time_sig_domain),
      "seqint_domain" : NoIndent(seqint_domain),
      "rest_domain" : NoIndent(rest_domain)
    },
    "corpus" : json_objects
}

with open(args.output_file, 'w') as outfile:
  outfile.write(json.dumps(outer_object, indent=2, cls=NoIndentEncoder))

print("Done generating corpus!")

