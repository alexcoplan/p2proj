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

""" 
takes a sequence of notes in our internal JSON encoding and turns it into a form
consumable by the RNN

returns (events, clock) pair

ts_semis: the number of semiquavers in a bar
 since the only possible time signatures for chorales are 3/4 or 4/4,
 this uniquely identifies a time signature.
"""
def encode_json_notes(notes, quantization, ts_semis):
  events = [divtoken]
  clock = [0]

  prev_end = 0

  for note_rep in notes:
    pitch    = note_rep[0]
    offset   = note_rep[1]
    duration = note_rep[2]
    delta = offset - prev_end
    if delta > 0:
      events.append(encode_rest(delta))
      clock.append(((prev_end % ts_semis) // quantization) + 1)
    events.append(encode_note(pitch, duration))
    clock.append(((offset % ts_semis) // quantization) + 1)
    prev_end = offset + duration

  return events, clock

pitch_table = [
  "C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"
]

duration_table = {
  1 : "Semiquaver",
  2 : "Quaver",
  3 : "Dotted quaver",
  4 : "Crotchet",
  6 : "Dotted crotchet",
  8 : "Minim",
  12 : "Dotted minim",
  14 : "Double dotted minim",
  16 : "Semibreve",
  20 : "Semibreve+Crotchet",
  24 : "Dotted semibreve",
  28 : "Doulbe dotted semibreve",
  32 : "Breve",
  56 : "Double dotted breve",
  64 : "Longa"
}

def readable_pitch(pitch):
  assert type(pitch) is int
  octave = (pitch // 12) - 1
  return pitch_table[pitch % 12] + str(octave)

def readable_duraiton(dur):
  return duration_table[dur]

def metadata_for_event(event):
  if event == divtoken:
    return "EOF", "0", "EOF"
  elif event[0] == "n":
    _, note = event.split("n")
    pitch, duration = [int(s) for s in note.split("d")]
    pitch_s = readable_pitch(pitch)
    human_readable = "{} {}".format(readable_duraiton(duration), pitch_s)
    return pitch_s, str(duration), human_readable
  elif event[0] == "r":
    dur = int(event.split("r")[1])
    human_readable = "{} rest".format(readable_duraiton(dur))
    return "Rest", str(dur), human_readable

"""
we generate a tsv with human-readable descriptions of each event type.
this is so we can visualise the embedding used in the RNN with TensorBoard
"""
def generate_metadata_tsv(events):
  tsv = "Pitch\tDuration\tDescription\n"
  for e in events:
    pitch_s, dur_s, desc_s = metadata_for_event(e)
    tsv += "{}\t{}\t{}\n".format(pitch_s, dur_s, desc_s)
  return tsv

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

def str_to_duration(e):
  if e[0] == "n":
    _, note = e.split("n")
    _, dur_s = note.split("d")
    return int(dur_s)
  elif e[0] == "r":
    _, dur_s = e.split("r")
    return int(dur_s)
  else:
    return None
