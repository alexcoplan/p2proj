# Introduction:
 - statement of project goals / work accomplished, ~~~ abstract
 - motivation: from perspective of end user of the system
 - background: define musical terms, give examiner necessary musical background,
   reference literature to explain why these things are difficult; 
   - metrical structure
   - justify musical scope (Western Classical)
 - related work: techniques that have been used in this area
 - context of work: comparison between MVS and RNN, why is this interesting?
 - briefly outline structure of dissertation

# Preparation:
 - outline core deliverables
 - justify all choices:
  - describe possible alternative choices in technical detail
  - describe MVS and RNN in full detail: *diagrams*
  - choice of tools, libraries
  - choice of software engineering strategy

# Implementation
 - remember to mention interesting challenges as well as successes
http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/

# Evaluation:
 - compare with numbers from C&W/Pearce/Whorley

Consider: why do we only just get as good numbers (1.87 bits/pitch) as C&W 1995?
 - PPM implementation: which escape method did they use?
 - We don't implement _fermata_ which the authors do. This is {y,h}uge.

## ideas
 - pathalogical examples!
 - cross entropy
 - human evaluation
 - subjective evaluation (annotated score + cross entropy/time)

