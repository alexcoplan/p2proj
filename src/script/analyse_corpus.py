import music21 # type: ignore
from typing import List
import matplotlib.pyplot as plt # type: ignore

bcl = music21.corpus.chorales.ChoraleListRKBWV()
num_chorales = max(bcl.byRiemenschneider.keys(), key=int)

bwv_to_rs_num_map = {} # type: Dict[str,str]

rs_numbers = [] # type: List[int]
min_pitches = [] # type: List[int]
max_pitches = [] # type: List[int]
pitch_ranges = [] # type: List[int]
timesig_strs = [] # type: List[str]

# we just take pitches from some arbitrary chorale 
a_chorale_bwv = 'bwv26.6'
a_chorale = music21.corpus.parse(a_chorale_bwv)
some_pitches = a_chorale.parts[0].flat.pitches
min_min_pitch = some_pitches[0]
max_max_pitch = some_pitches[0]
min_min_bwv = a_chorale_bwv # type: str
max_max_bwv = a_chorale_bwv # type: str

# allow for +/- 7 sharps
num_sharps = [] # type: List[int]
min_sharps = 0
max_sharps = 0
min_sharps_bwv = a_chorale_bwv
max_sharps_bwv = a_chorale_bwv

for i in bcl.byRiemenschneider:
  info = bcl.byRiemenschneider[i]
  bwv = info["bwv"] # type: str
  bwv_to_rs_num_map[bwv] = i
  c = music21.corpus.parse('bach/bwv' + str(info["bwv"]))
  if len(c.parts) != 4:
    print("Skipping {} of {} (BWV {}): not in four parts."\
      .format(i,num_chorales,bwv))
    continue

  print("Processing {} of {} (BWV {})".format(i,num_chorales,bwv))
  rs_numbers.append(i)

  # process key signature
  ks = c.recurse().getElementsByClass('KeySignature')[0]
  num_sharps.append(ks.sharps)
  if ks.sharps < min_sharps:
    min_sharps = ks.sharps
    min_sharps_bwv = bwv
  if ks.sharps > max_sharps:
    max_sharps = ks.sharps
    max_sharps_bwv = bwv

  # process time signature
  ts = c.recurse().getElementsByClass('TimeSignature')[0]
  if ts.ratioString not in timesig_strs:
    timesig_strs.append(ts.ratioString)

  if ts.ratioString == "3/2":
    print("--> this one in 3/2!")

  # process min/max pitches
  melody_pitches = c.parts[0].flat.pitches
  min_pitch = melody_pitches[0]
  max_pitch = melody_pitches[0]
  for p in melody_pitches:
    if p < min_pitch:
      min_pitch = p
    if p > max_pitch:
      max_pitch = p
  
  if min_pitch < min_min_pitch:
    min_min_pitch = min_pitch
    min_min_bwv = bwv
  if max_pitch > max_max_pitch:
    max_max_pitch = max_pitch
    max_max_bwv = bwv

  min_pitches.append(min_pitch.midi)
  max_pitches.append(max_pitch.midi)
  pitch_ranges.append(max_pitch.midi - min_pitch.midi)
  
print("Done.")
print("=" * 50)

print("min pitch:", min_min_pitch, 
      "bwv:", min_min_bwv, "rs:", bwv_to_rs_num_map[min_min_bwv])
print("max pitch:", max_max_pitch, 
      "bwv:", max_max_bwv, "rs:", bwv_to_rs_num_map[max_max_bwv])

ultimin = music21.corpus.parse(min_min_bwv)
sop_pitches = ultimin.parts[0].flat.pitches
max_p = sop_pitches[0]
for p in sop_pitches:
  if p > max_p:
    max_p = p
print("max pitch of chorale with global pitch minimum", max_p)

ultimax = music21.corpus.parse(max_max_bwv)
sop_pitches = ultimax.parts[0].flat.pitches
min_p = sop_pitches[0]
for p in sop_pitches:
  if p < min_p:
    min_p = p
print("min pitch of chorale with global pitch maximum", min_p) # todo

print("min sharps:", min_sharps, 
      "bwv:", min_sharps_bwv, "rs:", bwv_to_rs_num_map[min_sharps_bwv])
print("max sharps:", max_sharps, 
      "bwv:", max_sharps_bwv, "rs:", bwv_to_rs_num_map[max_sharps_bwv])

print("time sig used:", timesig_strs)

plt.figure(1)
plt.title("Min/max pitches in chorale corpus")
plt.plot(rs_numbers, min_pitches)
plt.plot(rs_numbers, max_pitches)
plt.plot(rs_numbers, pitch_ranges)
plt.xlabel('Riemenschneider number')
plt.ylabel('Pitch')

plt.figure(2)
plt.title("Distribution of keys in chorales")
plt.hist(num_sharps, bins=range(min(num_sharps), max(num_sharps)+2))

plt.show()

# plt.figure(3)
# plt.title("Distribution of time signatures in chorales")
# plt.plot(timesig_denoms, timesig_nums)
# plt.xlabel('Division')
# plt.ylabel('Bar length')
# plt.show()


