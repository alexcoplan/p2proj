#ifndef AJC_HGUARD_CHORALE
#define AJC_HGUARD_CHORALE

#include "event.hpp"
#include <cassert>
#include <string>
#include <array>

/* N.B. we define these little wrapper types such as KeySig and MidiPitch to
 * overload the constructors of the Chorale event types */
struct KeySig {
  int num_sharps;
  KeySig(int s) : num_sharps(s) { assert(-7 <= s && s <= 7); }
};

struct MidiPitch {
  unsigned int pitch;
  MidiPitch(unsigned int p) : pitch(p) { assert(p < 127); }
};

struct QuantizedDuration {
  unsigned int duration;
  QuantizedDuration(unsigned int d) : duration(d) {}
};

class CodedEvent : public SequenceEvent {
protected:
  const unsigned int code;
public:
  unsigned int encode() const override { return code; }
  std::string string_render() const override {
    return std::to_string(code);
  }
  CodedEvent(unsigned int c) : code(c) {}
};

/** Empirically (c.f. script/prepare_chorales.py) the domain for the pitch of
 * chorales in MIDI notation is the integers in [60,81]. */
class ChoralePitch : public CodedEvent {
public:
  constexpr static unsigned int cardinality = 22;

private:
  const static std::array<const std::string, cardinality> pitch_strings;
  constexpr static unsigned int lowest_midi_pitch = 60;
  constexpr static unsigned int map_in(unsigned int midi_pitch) {
    return midi_pitch - lowest_midi_pitch;
  }
  constexpr static unsigned int map_out(unsigned int some_code) {
    return some_code + lowest_midi_pitch;
  }

public:
  unsigned int encode() const override { return code; } 
  unsigned int raw_value() const { return map_out(code); } 
  std::string string_render() const override {
    return pitch_strings.at(code);
  }

  // need the "code" constructor for enumeration etc. to work 
  ChoralePitch(unsigned int code);
  ChoralePitch(const MidiPitch &pitch);
};

class ChoraleDuration : public CodedEvent {
public:
  constexpr static unsigned int cardinality = 10;

private:
  const static std::array<unsigned int, cardinality> duration_domain;
  static unsigned int map_in(unsigned int duration);
  static unsigned int map_out(unsigned int some_code);

public:
  unsigned int encode() const override { return code; } 
  unsigned int raw_value() const { return map_out(code); } 

  ChoraleDuration(unsigned int code);
  ChoraleDuration(const QuantizedDuration &qd);
};

class ChoraleKeySig : public CodedEvent {
private:
  constexpr static int min_sharps = -4;
  constexpr static unsigned int map_in(int num_sharps) {
    return num_sharps - min_sharps;
  }
  constexpr static int map_out(unsigned int some_code) {
    return some_code + min_sharps;
  }

public:
  constexpr static int cardinality = 9;

  unsigned int encode() const override { return code; } 
  unsigned int raw_value() const { return map_out(code); } 

  // need the "code" constructor for enumeration etc. to work 
  ChoraleKeySig(unsigned int code);
  ChoraleKeySig(const KeySig &ks);
};

class ChoraleTimeSig : public CodedEvent {
public:
  constexpr static unsigned int cardinality = 3;

private:
  static const std::array<unsigned int, cardinality> time_sig_domain;
  static unsigned int map_in(unsigned int bar_length);
  static unsigned int map_out(unsigned int code);

public:
  unsigned int encode() const override { return code; } 
  unsigned int raw_value() const { return map_out(code); } 

  // need the "code" constructor for enumeration etc. to work 
  ChoraleTimeSig(unsigned int code);
  ChoraleTimeSig(const QuantizedDuration &qd);
};

/* This type defines the event space we use to model chorales. ChoraleEvents
 * have as members all the different types that make up a chorale event (pitch,
 * duration, offset, etc.) */
class ChoraleEvent {
public:
  const ChoralePitch pitch;
  const ChoraleDuration duration;
  const ChoraleKeySig key_sig;
  const ChoraleTimeSig time_sig;
  const unsigned int offset;

  ChoraleEvent(const MidiPitch &mp, 
               const QuantizedDuration &dur, 
               const KeySig &ks, 
               const QuantizedDuration &bar_length,
               const unsigned int pos) :
    pitch(mp), duration(dur), key_sig(ks), time_sig(bar_length), offset(pos) {}
};

#endif
