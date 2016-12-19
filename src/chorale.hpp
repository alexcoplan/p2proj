#ifndef AJC_HGUARD_CHORALE
#define AJC_HGUARD_CHORALE

#include "event.hpp"
#include "viewpoint.hpp"
#include "sequence_model.hpp"
#include <cassert>
#include <string>
#include <array>
#include <map>
#include <cmath>

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

struct MidiInterval {
  int delta_pitch;
  MidiInterval(int dp) : delta_pitch(dp) { 
    assert(abs(dp) < 127);
  }
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
  bool operator==(const CodedEvent &other) {
    return this->code == other.code;
  }
  CodedEvent(unsigned int c) : code(c) {}
};

// forward declare to use in ChoralePitch
class ChoraleInterval;

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

  // check if transposition by the interval delta gives a valid chorale pitch
  bool is_valid_transposition(const ChoraleInterval &delta) const;
  MidiInterval operator-(const ChoralePitch &rh_pitch) const;
  ChoralePitch operator+(const ChoraleInterval &delta) const;

  // need the "code" constructor for enumeration etc. to work 
  ChoralePitch(unsigned int code);
  ChoralePitch(const MidiPitch &pitch);
};

class ChoraleDuration : public CodedEvent {
public:
  constexpr static unsigned int cardinality = 10;

private:
  const static std::array<unsigned int, cardinality> duration_domain;
  const static std::array<const std::string, cardinality> pretty_durations;
  static unsigned int map_in(unsigned int duration);
  static unsigned int map_out(unsigned int some_code);

public:
  unsigned int encode() const override { return code; } 
  unsigned int raw_value() const { return map_out(code); } 
  std::string string_render() const override {
    return pretty_durations.at(code);
  }

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

/**********************************************************
 * Derived types for the chorales
 **********************************************************/

class ChoraleInterval : public CodedEvent {
public:
  constexpr static unsigned int cardinality = 22;

private:
  // domain is {-12, -9, -8, ..., 8, 9, 12}
  constexpr static unsigned int map_in(int delta_p) {
    return (delta_p == -12) ? 0 :
           (delta_p == 12) ? 21 :
           delta_p + 10;
  }

  // inverse mapping
  constexpr static int map_out(unsigned int some_code) {
    return (some_code == 0) ? -12 :
           (some_code == 21) ? 12 :
           some_code - 10;
  }

  const static std::array<std::string, 13> interval_strings;

public:
  unsigned int encode() const override { return code; }
  int raw_value() const { return map_out(code); }

  std::string string_render() const override;

  ChoraleInterval(const MidiInterval &delta_pitch);
  ChoraleInterval(const ChoralePitch &from, const ChoralePitch &to);
  ChoraleInterval(unsigned int code);
};


/**********************************************************
 * Chorale Viewpoints
 **********************************************************/

class IntervalViewpoint : public Viewpoint<ChoraleInterval, ChoralePitch> {
private:
  using ParentVP = Viewpoint<ChoraleInterval, ChoralePitch>;

  std::unique_ptr<ChoraleInterval>
    project(const std::vector<ChoralePitch> &pitches, unsigned int upto) 
    const override;

  std::vector<ChoraleInterval>
    lift(const std::vector<ChoralePitch> &pitches) const override;

public:
  IntervalViewpoint(unsigned int h) : ParentVP(h) {}

  bool can_predict(const std::vector<ChoralePitch> &pitches) const override {
    return pitches.size() > 1;
  }

  EventDistribution<ChoralePitch>
    predict(const std::vector<ChoralePitch> &pitches) const override;

  IntervalViewpoint *clone() const override { 
    return new IntervalViewpoint(*this);
  }

  void debug();
};


/**********************************************************
 * Chorale Multiple Viewpoint System
 **********************************************************/

class ChoraleMVS {
private:
  double entropy_bias;
  std::vector<std::unique_ptr<Predictor<ChoralePitch>>> pitch_predictors;
  std::vector<std::unique_ptr<Predictor<ChoraleDuration>>> duration_predictors;

  template<typename T>
    const std::vector<std::unique_ptr<Predictor<T>>> &predictors() const;

public:
  template<typename T>
    EventDistribution<T> predict(const std::vector<T> &ctx) const;

  template<typename T>
    double avg_sequence_entropy(const std::vector<T> &seq) const;

  ChoraleMVS(double eb, 
   std::initializer_list<Predictor<ChoralePitch> *> pitch_vps,
   std::initializer_list<Predictor<ChoraleDuration> *> duration_vps);
};

// templated method implementations for ChoraleMVS

template<typename T>
EventDistribution<T>
ChoraleMVS::predict(const std::vector<T> &ctx) const {
  const auto &vps = predictors<T>();

  auto it = vps.begin();
  for (; it != vps.end(); ++it) {
    if ((*it)->can_predict(ctx))
      break;
  }

  if (it == vps.end())
    throw ViewpointPredictionException("No viewpoints can predict context");

  auto prediction = (*it)->predict(ctx);
  
  if (vps.size() == 1)
    return prediction;

  auto comb_strategy = WeightedEntropyCombination<T>(entropy_bias);

  for (; it != vps.end(); ++it) {
    if ((*it)->can_predict(ctx)) {
      auto new_prediction = (*it)->predict(ctx);
      prediction.combine_in_place(comb_strategy, new_prediction);
    }
  }

  return prediction;
}

template<typename T>
double ChoraleMVS::avg_sequence_entropy(const std::vector<T> &seq) const {
  std::vector<T> ngram_buf;

  double total_entropy = 0.0;
  auto dist = predict<T>({});

  for (auto &e : seq) {
    total_entropy -= std::log2(dist.probability_for(e));
    ngram_buf.push_back(e);
    dist = predict<T>(ngram_buf);
  }
  
  return total_entropy / seq.size();
}

#endif
