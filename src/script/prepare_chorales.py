from music21 import *
import json

# mask out which elements of each type are actually used
pitch_mask = [False] * 127
duration_mask = [False] * 64
sharps_mask = [False] * 14
# TODO: finish this

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
num_chorales = len(bcl.byRiemenschneider)

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

# get time/key signature information
  ts = c.recurse().getElementsByClass('TimeSignature')[0]
  ks = c.recurse().getElementsByClass('KeySignature')[0]
  time_sig_str = ts.ratioString

  key_sig_sharps = ks.sharps
  key_sig_major = True
  if isinstance(ks, key.Key):
    key_sig_major = (ks.mode == "major")
  else:
    print(" * Skipping BWV %s has no mode specified." % info["bwv"])
    continue # TODO: in future check these and see if any definitely fit a mode

  c_notes = []

  for n in c.parts[0].flat.notes:
    c_notes.append([
      n.pitch.midi, 
      ql_quantize(n.offset + anacrusis_ql), 
      ql_quantize(n.duration.quarterLength)
    ])

  obj = {
      "title" : info["title"],
      "bwv" : info["bwv"],
      "time_sig_semis" : time_sig_q,
      "key_sig_sharps" : key_sig_sharps,
      "key_sig_major" : key_sig_major,
      "notes" : c_notes
  } 

  json_objects.append(obj)

print("Done processing chorales. Writing JSON...")

f = open('chorale_dataset.json', 'w')
for chunk in json.JSONEncoder().iterencode(json_objects):
  f.write(chunk)

f.close()

print("Corpus compiled!")
