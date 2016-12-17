#include "chorale.hpp"
#include <array>
#include <cassert>

#define NAT(X,N) #X "$_{" #N "}$"
#define SHARP(X,N) #X "\\sharp{}$_{" #N "}$"
#define FLAT(X,N) "\\flatten{" #X "}_" #N

/***************************************************
 * ChoralePitch implementation
 ***************************************************/

// n.b. this is obviously not proper spelling of the scale but currently this is
// only for debugging purposes, so let's keep things simple
const std::array<const std::string, ChoralePitch::cardinality>
ChoralePitch::pitch_strings = {{
  NAT(C,4), SHARP(C,4), NAT(D,4), FLAT(E,4), NAT(E,4), NAT(F,4), SHARP(F,4),
    NAT(G,4), SHARP(G,4), NAT(A,4), FLAT(B,4), NAT(B,4), NAT(C,5), SHARP(C,5),
    NAT(D,5), FLAT(E,5), NAT(E,5), NAT(F,5), SHARP(F,5), NAT(G,5), SHARP(G,5),
    NAT(A,5)
}};

ChoralePitch::ChoralePitch(unsigned int c) : CodedEvent(c) {
  assert(c < cardinality);
}

ChoralePitch::ChoralePitch(const MidiPitch &mp) : CodedEvent(map_in(mp.pitch)) {
  auto pitch = mp.pitch;
  assert(pitch >= lowest_midi_pitch);
  assert(pitch < lowest_midi_pitch + cardinality);
}

/***************************************************
 * ChoraleDuration implementation
 ***************************************************/

const std::array<unsigned int, ChoraleDuration::cardinality> 
ChoraleDuration::duration_domain = {{1,2,3,4,6,8,12,16,24,64}};

const std::array<const std::string, ChoraleDuration::cardinality>
ChoraleDuration::pretty_durations = {{
  u8"\U0001D161",
  u8"\U0001D160",
  u8"\U0001D160.",
  u8"\U0001D15F",
  u8"\U0001D15F.",
  u8"\U0001D15E",
  u8"\U0001D15E.",
  u8"\U0001D15D",
  u8"\U0001D15D.",
  u8"\U0001D15C-\U0001D15C"
}};

unsigned int ChoraleDuration::map_in(unsigned int quantized_duration) {
  for (unsigned int i = 0; i < cardinality; i++) 
    if (duration_domain[i] == quantized_duration)
      return i;
  
  assert(! "Bad duration");
}

unsigned int ChoraleDuration::map_out(unsigned int code) {
  return duration_domain[code];
}

ChoraleDuration::ChoraleDuration(unsigned int c) : CodedEvent(c) {
  assert(c < cardinality);
}

ChoraleDuration::ChoraleDuration(const QuantizedDuration &qd) : 
  CodedEvent(map_in(qd.duration)) {}

/***************************************************
 * ChoraleKeySig implementation
 ***************************************************/

ChoraleKeySig::ChoraleKeySig(unsigned int c) : CodedEvent(c) {
  assert(c < cardinality);
}

ChoraleKeySig::ChoraleKeySig(const KeySig &ks) :
  CodedEvent(map_in(ks.num_sharps)) {
  auto sharps = ks.num_sharps;
  assert(sharps >= min_sharps);
  assert(sharps < min_sharps + cardinality);
}

/***************************************************
 * ChoraleTimeSig implementation
 ***************************************************/

const std::array<unsigned int, ChoraleTimeSig::cardinality> 
ChoraleTimeSig::time_sig_domain = {{12,16,24}};

unsigned int ChoraleTimeSig::map_in(unsigned int dur) {
  switch (dur) {
    case 12:
      return 0;
    case 16:
      return 1;
    case 24:
      return 2;
    default:
      assert(! "Bad time signature!");
  }
}

unsigned int ChoraleTimeSig::map_out(unsigned int c) {
  return time_sig_domain[c];
}

ChoraleTimeSig::ChoraleTimeSig(unsigned int c) : CodedEvent(c) {
  assert(c < cardinality);
}

ChoraleTimeSig::ChoraleTimeSig(const QuantizedDuration &qd) :
  CodedEvent(map_in(qd.duration)) {}

