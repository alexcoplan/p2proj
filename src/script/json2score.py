# turns generate JSON into score using music21

from music21 import *
from scoreutils import render_music21
import json
import sys

if len(sys.argv) < 2:
  print("Please give me a filename!")
  sys.exit()

fname = sys.argv[1]

with open(fname) as json_data:
  obj = json.load(json_data)
  assert obj["notes"] is not None, \
    "JSON object must contain array with key 'notes'"
  render_music21(obj["notes"])

