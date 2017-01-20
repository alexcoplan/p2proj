"""
Since Bach harmonised many chorale melodies multiple times, if we take just the
melody line out of these harmonisations, there will naturally be some
duplication.

The purpose of this script is to assist in manually ascertaining the nature of
this duplication: are there exact copies, exact transpositions, or variations on
the same melody?
"""

import music21 # type: ignore
from tabulate import tabulate # type: ignore

# a list of duplicate groups in terms of their Riemenschneider number
dup_groups = [
  [40,279],
  [3,253,262],
  [156,217,308],
  [31,301],
  [125,249,313,326],
  [13,359],
  [270,286,340,367],
  [15,184,261,371],
  [66,119],
  [6,316],
  [81,113,198,307],
  [162,314],
  [53,178],
  [87,134],
  [100,126],
  [20,250,273],
  [9,102],
  [4,290,335],
  [260,362],
  [16,333,352],
  [29,64,67,76,282],
  [51,160,288],
  [235,319],
  [88,99,123],
  [101,303],
  [33,287],
  [73,266,294],
  [189,284],
  [317,318],
  [58,107,277],
  [59,78,105,111],
  [199,302],
  [155,368],
  [2,272,341],
  [50,140],
  [77,118],
  [37,269,297,369],
  [61,83,106],
  [96,138,263,283,324,356], # jesu meine freude
  [350,365],
  [11,252,327],
  [30,174],
  [175,338],
  [45,370],
  [131,328],
  [54,276,342],
  [44,310],
  [152,299,348],
  [130,358],
  [49,325],
  [36,84,97],
  [32,330],
  [28,170],
  [7,116,268,296],
  [63,103,117,289,355], # nur ruhen alle Wälder
  [26,274],
  [85,312,315,337],
  [74,80,89,98,345], # o haupt voll blut und wunden
  [201,306],
  [275,363,366],
  [213,219],
  [248,329,354],
  [24,108],
  [47,110,267],
  [91,215,259],
  [46,344],
  [114,191,332,364],
  [182,285],
  [94,145,300],
  [139,357],
  [255,291],
  [65,293,347],
  [41,115,120,265],
  [254,298],
  [52,322,351],
  [68,247],
  [95,121,233],
  [62,104,112,146,339],
  [86,195,278,305,323],
  [55,321,360],
  [25,281,331]
] # type: List[List[int]]

# check that no harmonisation is in multiple duplicate groups
duplicate_harmonisations = [] # type: List[int]
for group in dup_groups:
  for num in group:
    assert num not in duplicate_harmonisations,\
      "RS {} already in a dupe group".format(num)
    duplicate_harmonisations.append(num)

singles = [
  177,186,39,48,153,128,159,180,208,5,124,304,1,10,230,245,210,197,56,200,196,228,311,
  224,75,239,154,353,158,207,231,232,127,209,42,167,361,309,280,34,72,17,176,
  216,27,166,238,8,334,163,271,225,135,35,18,181,234,192,70,320,90,164,
  205,212,136,226,295,190,221,21,168,79,251,223,188,229,60,19,71,161,143,
  122,256,169,243,244,264,129,187,69,132,218,43,194,227,258,151,346,214,149,292,
  185,183,257,343,222,355,240,82,14,173,92,236,165,203,57,202,12,171,22,
  142,141,172,109,246,206,220,38,148,93,179,237,193,241,150,211,147,137,144,
  204,242,133,336,157,23
] # type: List[int]

# check that singles are unique
checked = [] # type: List[int]
for s in singles:
  assert s not in checked, "Duplicate single chorale: {}".format(s)
  checked.append(s)

for i in range(1,318):
  either = (i in singles and i not in duplicate_harmonisations)
  oar = (i not in singles and i in duplicate_harmonisations)
  assert either or oar, "Bad choarle: {}".format(i)



# start the review process
bcl = music21.corpus.chorales.ChoraleListRKBWV()
headers = ["RS","BWV","Parts","Key","Range","Title"]
for group_idx,group in enumerate(dup_groups):
  table = [] # type: List[List[str]]
  tunes = []

  skip = False
  for num in group:
    if num not in bcl.byRiemenschneider:
      print(" * Skipping RS {}, missing score.".format(num))
      continue

    info = bcl.byRiemenschneider[num]
    bwv = info["bwv"] 
    title = info["title"] 
    c = music21.corpus.parse('bach/bwv' + bwv)
    sop = c.parts[0]
    n_parts = len(c.parts)
    if n_parts != 4:
      print(" * Skipping RS {} in {} parts."\
        .format(num,n_parts))
      continue

    ks = c.recurse().getElementsByClass('KeySignature')[0]
    key_s = ks.name\
      if isinstance(ks, music21.key.Key)\
      else "{} sharps".format(ks.sharps)
    range_n = music21.analysis.discrete.analyzeStream(sop, 'range').name
    table.append([str(num),bwv,str(n_parts),key_s,range_n,title])
    tunes.append(sop)

  if len(table) <= 1:
    print(" * Skipping group {}/{}.".format(group_idx+1, len(dup_groups)))
    continue
  
  print("\nDuplicate group {}/{}:".format(group_idx+1, len(dup_groups)))
  print(tabulate(table, headers, tablefmt="orgtbl") + "\n")
  inp = input("blank: continue, q: quit, s: show: ")
  if (inp == "s"):
    for p in tunes:
      p.show()
    inp = input("blank: continue, q: quit: ")

  if (inp == "q"):
    break

  print()

# for group in dup_groups:
#   print("Duplicate group:", group)
#   for num in group:
#     print tabulate([[], ["RS","Key","Range","BWV","Title"])


