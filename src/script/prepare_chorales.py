from music21 import *
import json

# mask out which elements of each type are actually used
pitch_mask = [False] * 127
duration_mask = [False] * 64
sharps_mask = [False] * 15
timesig_mask = [False] * 32

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

  for n in c.parts[0].flat.notes:
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
      "notes" : c_notes
  } 

  json_objects.append(obj)

print("Done processing chorales.\n")

# work out possible values of each type based on masks
pitch_domain = []
duration_domain = []
sharps_domain = []
time_sig_domain = []

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

print("Compilation complete, writing JSON...")

outer_object = {
    "metadata" : {
      "pitch_domain" : pitch_domain,
      "duration_domain" : duration_domain,
      "key_sig_sharps_domain" : sharps_domain,
      "time_sig_domain" : time_sig_domain
    },
    "corpus" : json_objects
}

f = open('chorale_dataset.json', 'w')
for chunk in json.JSONEncoder().iterencode(outer_object):
  f.write(chunk)

f.close()

print("Done generating corpus!")

