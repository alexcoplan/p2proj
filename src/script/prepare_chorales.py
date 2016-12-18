# this script reqiures:
# - Python 3
# - Latest music21 (in particular, 8b5d422 where I fix a bug in BWV 348)
# and it should be run from the main src directory

from music21 import *
import json

################################################################################
# Custom JSON encoder
#
# Taken from here: http://stackoverflow.com/a/25935321/840973
#
# This is to pretty-print the output, but avoid pretty-printing the inner data
# which we aren't particularly interested in when viewing the corpus.
################################################################################

import uuid

class NoIndent(object):
  def __init__(self, value):
    self.value = value


class NoIndentEncoder(json.JSONEncoder):
  def __init__(self, *args, **kwargs):
    super(NoIndentEncoder, self).__init__(*args, **kwargs)
    self.kwargs = dict(kwargs)
    del self.kwargs['indent']
    self._replacement_map = {}

  def default(self, o):
    if isinstance(o, NoIndent):
      key = uuid.uuid4().hex
      self._replacement_map[key] = json.dumps(o.value, **self.kwargs)
      return "@@%s@@" % (key,)
    else:
      return super(NoIndentEncoder, self).default(o)

  def encode(self, o):
    result = super(NoIndentEncoder, self).encode(o)
    for k, v in self._replacement_map.items():
      result = result.replace('"@@%s@@"' % (k,), v)
    return result


################################################################################
# Begin main script
################################################################################

# mask out which elements of each type are actually used
pitch_mask = [False] * 127
duration_mask = [False] * 64
sharps_mask = [False] * 15
timesig_mask = [False] * 32
seqint_mask = [False] * 42

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

  if len(c.parts) != 4:
    print(" * Skipping: BWV %s is not in four parts." % info["bwv"])
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
  key_sig_major = True
  if isinstance(ks, key.Key):
    key_sig_major = (ks.mode == "major")
  else:
    print(" * Skipping BWV %s has no mode specified." % info["bwv"])
    continue # TODO: in future check these and see if any definitely fit a mode

  c_notes = []

  prev_pitch = None

  for n in c.parts[0].flat.notes:
    if prev_pitch is not None:
      seqint_mask[n.pitch.midi - prev_pitch + 21] = True
    prev_pitch = n.pitch.midi

    duration_q = ql_quantize(n.duration.quarterLength)
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
      "key_sig_major" : key_sig_major,
      "notes" : NoIndent(c_notes)
  } 

  json_objects.append(obj)

print("Done processing chorales.\n")

# work out possible values of each type based on masks
pitch_domain = []
duration_domain = []
sharps_domain = []
time_sig_domain = []
seqint_domain = []

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

print("Compilation complete, writing JSON...")

outer_object = {
    "metadata" : {
      "pitch_domain" : NoIndent(pitch_domain),
      "duration_domain" : NoIndent(duration_domain),
      "key_sig_sharps_domain" : NoIndent(sharps_domain),
      "time_sig_domain" : NoIndent(time_sig_domain),
      "seqint_domain" : NoIndent(seqint_domain)
    },
    "corpus" : json_objects
}

with open('corpus/chorale_dataset.json', 'w') as outfile:
  outfile.write(json.dumps(outer_object, indent=2, cls=NoIndentEncoder))

print("Done generating corpus!")

