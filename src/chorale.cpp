#include "chorale.hpp"
#include <array>
#include <cassert>

/***************************************************
 * ChoralePitch implementation
 ***************************************************/

ChoralePitch::ChoralePitch(unsigned int c) {
  assert(c < cardinality);
  code = c;
}

ChoralePitch::ChoralePitch(const MidiPitch &mp) {
  auto pitch = mp.pitch;
  assert(pitch >= lowest_midi_pitch);
  assert(pitch < lowest_midi_pitch + cardinality);
  code = map_in(pitch);
}

/***************************************************
 * ChoraleDuration implementation
 ***************************************************/

const std::array<unsigned int, ChoraleDuration::cardinality> 
ChoraleDuration::duration_domain = {{1,2,3,4,6,8,12,16,24,64}};

unsigned int ChoraleDuration::map_in(unsigned int quantized_duration) {
  for (unsigned int i = 0; i < cardinality; i++) 
    if (duration_domain[i] == quantized_duration)
      return i;
  
  assert(! "Bad duration");
}

unsigned int ChoraleDuration::map_out(unsigned int code) {
  return duration_domain[code];
}

ChoraleDuration::ChoraleDuration(unsigned int c) {
  assert(c < cardinality);
  code = c;
}

ChoraleDuration::ChoraleDuration(const QuantizedDuration &qd) {
  auto dur = qd.duration;
  code = map_in(dur);
}

/***************************************************
 * ChoraleKeySig implementation
 ***************************************************/

ChoraleKeySig::ChoraleKeySig(unsigned int c) {
  assert(c < cardinality);
  code = c;
}

ChoraleKeySig::ChoraleKeySig(const KeySig &ks) {
  auto sharps = ks.num_sharps;
  assert(sharps >= min_sharps);
  assert(sharps < min_sharps + cardinality);
  code = map_in(sharps);
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

ChoraleTimeSig::ChoraleTimeSig(unsigned int c) {
  assert(c < cardinality);
  code = c;
}

ChoraleTimeSig::ChoraleTimeSig(const QuantizedDuration &qd) {
  auto bar_len = qd.duration;
  code = map_in(bar_len); // validation done in map_in
}

