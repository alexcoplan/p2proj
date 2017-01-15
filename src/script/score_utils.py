from music21 import *

"""
takes notes in our internal format and renders them to a score using music21
"""
def render_music21(note_stream):
  piece = stream.Stream()
  prev_end = 0.0

  for mid_pitch, offset_q, duration_q in note_stream:
    duration_ql = duration_q * 0.25
    offset_ql   = offset_q * 0.25

    rest_amt = offset_ql - prev_end
    if rest_amt > 0.0:
      piece.insert(prev_end, note.Rest(quarterLength=rest_amt))

    n = note.Note(midi=mid_pitch, quarterLength=duration_ql)
    piece.insert(offset_ql, n)

    prev_end = offset_ql + duration_ql
    
  piece.show()
