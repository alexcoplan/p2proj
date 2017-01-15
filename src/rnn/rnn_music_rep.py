import json

"""
we make use of a string representation for musical events fed to the RNN.
this file provides encode/decode methods for that representation
"""

# used to separate pieces in the corpus
# we may not use this representaiton later, but this will do for now
divtoken = "|"

def encode_rest(duration):
  assert type(duration) is int
  return "r{}".format(duration)

def encode_note(pitch, duration):
  assert type(duration) is int
  assert type(pitch) is int
  return "n{}d{}".format(pitch, duration)

def encode_json_notes(notes):
  events = [divtoken]

  prev_end = 0

  for note_rep in notes:
    pitch    = note_rep[0]
    offset   = note_rep[1]
    duration = note_rep[2]
    delta = offset - prev_end
    if delta > 0:
      events.append(encode_rest(delta))
    events.append(encode_note(pitch, duration))
    prev_end = offset + duration

  return events

"""
takes a list of events in our RNN string representation format
and decodes them to the standard (pitch, offset, duration) format
we are using throughout the project
"""
def decode_events(events):
  offset = 0
  output_notes = []
  for e in events:
    if e == divtoken:
      continue
    if e[0] == "n": # it's a note
      _, note = e.split("n")
      pitch, duration = [int(s) for s in note.split("d")]
      output_notes.append([pitch, offset, duration])
      offset += duration
      continue
    if e[0] == "r": # it's a rest
      duration = int(e.split("r")[1])
      offset += duration
  return output_notes



