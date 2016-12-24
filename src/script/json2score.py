# turns generate JSON into score using music21

from music21 import *
import json
import sys

if len(sys.argv) < 2:
  print("Please give me a filename!")
  sys.exit()

fname = sys.argv[1]

piece = stream.Stream()

with open(fname) as json_data:
  obj = json.load(json_data)
  if obj["notes"] is None:
    print("JSON object must contain array with key 'notes'")
    sys.exit()

  prev_end = 0.0

  for a in obj["notes"]:
    mid_pitch  = a[0]
    offset_q   = a[1]
    duration_q = a[2]

    duration_ql = duration_q * 0.25
    offset_ql   = offset_q * 0.25

    rest_amt = offset_ql - prev_end
    if rest_amt > 0.0:
      piece.insert(prev_end, note.Rest(quarterLength=rest_amt))

    n = note.Note(midi=mid_pitch, quarterLength=duration_ql)
    piece.insert(offset_ql, n)

    prev_end = offset_ql + duration_ql
    
piece.show()

