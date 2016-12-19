#include "chorale.hpp"
#include <array>
#include <cassert>

// convenience macros for LaTeX trie generation
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

MidiInterval ChoralePitch::operator-(const ChoralePitch &rhs) const {
  auto midi_pitch_from = static_cast<int>(rhs.raw_value());
  auto midi_pitch_to = static_cast<int>(this->raw_value());
  return MidiInterval(midi_pitch_to - midi_pitch_from);
}

ChoralePitch ChoralePitch::operator+(const ChoraleInterval &delta) const {
  return ChoralePitch(MidiPitch(this->raw_value() + delta.raw_value()));
}

bool ChoralePitch::is_valid_transposition(const ChoraleInterval &delta) const {
  auto new_val = this->raw_value() + delta.raw_value();
  return new_val >= lowest_midi_pitch && 
         new_val < lowest_midi_pitch + cardinality;
}

/***************************************************
 * ChoraleDuration implementation
 ***************************************************/

const std::array<unsigned int, ChoraleDuration::cardinality> 
ChoraleDuration::duration_domain = {{1,2,3,4,6,8,12,16,24,64}};

const std::array<const std::string, ChoraleDuration::cardinality>
ChoraleDuration::pretty_durations = {{
  "\\semiquaver",
  "\\quaver",
  "\\quaverDotted",
  "\\crotchet",
  "\\crotchetDotted",
  "\\minim",
  "\\minimDotted",
  "\\semibreve",
  "\\semibreveDotted",
  "$\\semibreve^4$"
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


/********************************************************************
 * Derived types below, starting with ChoraleInterval implementation
 ********************************************************************/

ChoraleInterval::ChoraleInterval(unsigned int c) :
  CodedEvent(c)  {
    assert(c < cardinality);
}

ChoraleInterval::ChoraleInterval(const MidiInterval &ival) :
  CodedEvent(map_in(ival.delta_pitch)) {
  assert(code < cardinality);
  auto dp = ival.delta_pitch;
  assert(dp != -11 && dp != -10 && dp != 11);
}

const std::array<std::string, 13> ChoraleInterval::interval_strings = {{
  "Z", "m2", "M2", "m3", "M3", "P4", "Tri", "P5", "m6", "M6", "m7", "M7", "8ve"
}};

std::string ChoraleInterval::string_render() const {
  auto ival = raw_value();
  auto base_str = interval_strings[std::abs(ival)];
  return base_str + 
    ((ival > 0) ? "$\\uparrow$" : (ival < 0) ? "$\\downarrow$" : "");
}

/********************************************************************
 * Viewpoint implementations below
 ********************************************************************/

std::unique_ptr<ChoraleInterval>
IntervalViewpoint::
project(const std::vector<ChoralePitch> &pitches, unsigned int upto) const {
  assert(upto <= pitches.size());

  if (upto <= 1)
    return std::unique_ptr<ChoraleInterval>();

  auto from = pitches[upto - 2];
  auto to = pitches[upto - 1];
  return std::unique_ptr<ChoraleInterval>(new ChoraleInterval(to - from));
}

std::vector<ChoraleInterval>
IntervalViewpoint::lift(const std::vector<ChoralePitch> &pitches) const {
  if (pitches.size() <= 1)
    return {};

  std::vector<ChoraleInterval> result;
  for (unsigned int i = 1; i < pitches.size(); i++) {
    auto from = pitches[i-1];
    auto to = pitches[i];
    result.push_back(to - from);
  }

  return result;
}

void IntervalViewpoint::debug() {
  for (ChoraleInterval iv : EventEnumerator<ChoraleInterval>()) {
    std::cout << iv.string_render() << " (" << iv.raw_value() << "): "
      << this->model.count_of({iv}) << std::endl;
  }
}

EventDistribution<ChoralePitch>
IntervalViewpoint::predict(const std::vector<ChoralePitch> &pitch_ctx) const {
  if (pitch_ctx.empty())
    throw ViewpointPredictionException("Viewpoint seqint needs at least one\
 pitch to be able to predict further pitches.");

  auto interval_ctx = lift(pitch_ctx);
  auto interval_dist = model.gen_successor_dist(interval_ctx);
  auto last_pitch = pitch_ctx.back();

  std::array<double, ChoralePitch::cardinality> new_values{{0.0}};

  unsigned int valid_predictions = 0;

  for (auto interval : EventEnumerator<ChoraleInterval>()) { 
    if (!last_pitch.is_valid_transposition(interval))
      continue;

    valid_predictions++;
    auto candidate_pitch = last_pitch + interval;
    new_values[candidate_pitch.encode()] = 
      interval_dist.probability_for(interval);
  }

  if (valid_predictions == ChoraleInterval::cardinality)
    return EventDistribution<ChoralePitch>(new_values);

  // need to correct for missing values due to the interval leading to an
  // invalid (out of range) pitch
  double correction = 
    (double)ChoraleInterval::cardinality / (double)valid_predictions;

  for (auto &v : new_values)
    v *= correction;

  return EventDistribution<ChoralePitch>(new_values);
}


/********************************************************************
 * ChoraleMVS implementations below
 ********************************************************************/

template<>
std::vector<std::unique_ptr<Predictor<ChoralePitch>>> &
ChoraleMVS::predictors<ChoralePitch>() {
  return pitch_predictors;
}

template<>
std::vector<std::unique_ptr<Predictor<ChoraleDuration>>> &
ChoraleMVS::predictors<ChoraleDuration>() {
  return duration_predictors;
}

template<typename T>
EventDistribution<T>
ChoraleMVS::predict(const std::vector<T> &ctx) const {
  auto vps = predictors<T>();

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

