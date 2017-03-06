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

ChoralePitch ChoralePitch::operator+(const MidiInterval &delta) const {
  return ChoralePitch(MidiPitch(this->raw_value() + delta.delta_pitch));
}

bool ChoralePitch::is_valid_transposition(const MidiInterval &delta) const {
  auto new_val = this->raw_value() + delta.delta_pitch;
  return new_val >= lowest_midi_pitch && 
         new_val < lowest_midi_pitch + cardinality;
}

/***************************************************
 * ChoraleDuration implementation
 ***************************************************/

const std::array<unsigned int, ChoraleDuration::cardinality> 
ChoraleDuration::duration_domain = {{1,2,3,4,6,8,12,14,16,20,24,28,32,56,64}};

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
}};

unsigned int ChoraleDuration::map_in(unsigned int quantized_duration) {
  for (unsigned int i = 0; i < cardinality; i++) 
    if (duration_domain[i] == quantized_duration)
      return i;
  
  std::cerr << "Unexpected duration: " << quantized_duration << std::endl;
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

const std::array<unsigned int, ChoraleKeySig::cardinality> 
ChoraleKeySig::referent_map = {{
  68,75,70,77,72,67,74,69,76
}};

/***************************************************
 * ChoraleTimeSig implementation
 ***************************************************/

const std::array<unsigned int, ChoraleTimeSig::cardinality> 
ChoraleTimeSig::time_sig_domain = {{12,16}};

unsigned int ChoraleTimeSig::map_in(unsigned int dur) {
  switch (dur) {
    case 12:
      return 0;
    case 16:
      return 1;
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

/***************************************************
 * ChoraleRest implementation
 ***************************************************/

const std::array<const ChoraleRest, ChoraleRest::cardinality> 
ChoraleRest::shared_instances = 
  {{ ChoraleRest(0), ChoraleRest(1), ChoraleRest(2) }};

const std::array<std::string, ChoraleRest::cardinality>
ChoraleRest::pretty_strs = {
  {"$\\rightarrow$ ", "\\crotchetRest{} ", "\\halfNoteRest{} "}
};

ChoraleRest::ChoraleRest(unsigned int c) : CodedEvent(c) {
  assert(c < cardinality);
}

ChoraleRest::ChoraleRest(const QuantizedDuration &qd) :
  CodedEvent(map_in(qd.duration)) {
  assert(code < cardinality);
}

/***************************************************
 * ChoraleEvent implementation
 ***************************************************/

template<>
ChoraleKeySig ChoraleEvent::project() const { return keysig; }

template<>
ChoralePitch ChoraleEvent::project() const { return pitch; }

template<>
ChoraleDuration ChoraleEvent::project() const { return duration; }

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

ChoraleIntref::ChoraleIntref(unsigned int c) :
  CodedEvent(c) {
  assert(c < cardinality);
}

ChoraleIntref::ChoraleIntref(const MidiInterval &ival) :
  CodedEvent(map_in(ival.delta_pitch)) {
  auto dp = ival.delta_pitch;
  assert(dp >= min_intref && dp <= max_intref);
}

std::string ChoraleIntref::string_render() const {
  auto delta_p = raw_value();
  auto num_str = std::to_string(delta_p);
  return (delta_p < 0) ? num_str : ("+" + num_str);
}

/********************************************************************
 * Viewpoint implementations below
 ********************************************************************/

std::vector<ChoraleInterval>
IntervalViewpoint::lift(const std::vector<ChoraleEvent> &events) const {
  auto pitches = ChoraleEvent::template lift<ChoralePitch>(events);

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

EventDistribution<ChoralePitch>
IntervalViewpoint::predict(const std::vector<ChoraleEvent> &ctx) const {
  if (ctx.empty())
    throw ViewpointPredictionException("Viewpoint seqint needs at least one\
 pitch to be able to predict further pitches.");

  auto interval_ctx = lift(ctx);
  auto interval_dist = model.gen_successor_dist(interval_ctx);
  auto last_pitch = ctx.back().project<ChoralePitch>();

  std::array<double, ChoralePitch::cardinality> new_values{{0.0}};

  double total_probability = 0.0;
  unsigned int valid_predictions = 0;

  for (auto interval : EventEnumerator<ChoraleInterval>()) { 
    auto midi_interval = interval.midi_interval();
    if (!last_pitch.is_valid_transposition(midi_interval))
      continue;

    auto candidate_pitch = last_pitch + midi_interval;
    auto prob = interval_dist.probability_for(interval);
    new_values[candidate_pitch.encode()] = prob;
    total_probability += prob;
    valid_predictions++;
  }

  if (valid_predictions == ChoraleInterval::cardinality)
    return EventDistribution<ChoralePitch>(new_values);

  for (auto &v : new_values)
    v /= total_probability;

  return EventDistribution<ChoralePitch>(new_values);
}

std::vector<ChoraleIntref>
IntrefViewpoint::lift(const std::vector<ChoraleEvent> &events) const {
  if (events.empty())
    return {};

  auto pitches = ChoraleEvent::template lift<ChoralePitch>(events);
  auto referent = events.front().project<ChoraleKeySig>().referent();
  
  std::vector<ChoraleIntref> result;
  for (const auto &p : pitches)
    result.push_back(p - referent);

  return result;
}

EventDistribution<ChoralePitch>
IntrefViewpoint::predict_given_key(
  const std::vector<ChoraleEvent> &ctx,
  const ChoraleKeySig &ks
) const {
  auto referent = ks.referent();
  auto intref_ctx = lift(ctx);
  auto intref_dist = model.gen_successor_dist(intref_ctx);

  std::array<double, ChoralePitch::cardinality> new_values{{0.0}};

  double total_probability = 0.0;
  unsigned int valid_predictions = 0;

  for (auto intref : EventEnumerator<ChoraleIntref>()) {
    auto midi_interval = intref.midi_interval();
    if (!referent.is_valid_transposition(midi_interval))
      continue;

    auto candidate_pitch = referent + midi_interval;
    auto prob = intref_dist.probability_for(intref);
    new_values[candidate_pitch.encode()] = prob;
    total_probability += prob;
    valid_predictions++;
  }

  if (valid_predictions == ChoraleIntref::cardinality)
    return EventDistribution<ChoralePitch>(new_values);

  for (auto &v : new_values)
    v /= total_probability;

  return EventDistribution<ChoralePitch>(new_values);
}


EventDistribution<ChoralePitch>
IntrefViewpoint::predict(const std::vector<ChoraleEvent> &ctx) const {
  if (ctx.empty()) {
    const std::string msg = "Need context to predict with intref viewpoint";
    throw ViewpointPredictionException(msg);
  }

  auto ks = ctx.front().project<ChoraleKeySig>();
  return predict_given_key(ctx, ks);
}

/***************************************************
 * ChoraleMVS implementation
 ***************************************************/

// note: the repeated for loops are a bit yucky here.
// we could probably do something with variadic templates to sort this out
// (and further generalise multiple viewpoint systems in C++ rather than the
// specific chorale case) but this is not a priority

void ChoraleVPLayer::learn(const std::vector<ChoraleEvent> &seq) {
  for (auto &vp_ptr : predictors<ChoralePitch>())
    vp_ptr->learn(seq);
  for (auto &vp_ptr : predictors<ChoraleDuration>())
    vp_ptr->learn(seq);
  for (auto &vp_ptr : predictors<ChoraleRest>())
    vp_ptr->learn(seq);
}

void ChoraleVPLayer::learn_from_tail(const std::vector<ChoraleEvent> &seq) {
  for (auto &vp_ptr : predictors<ChoralePitch>())
    vp_ptr->learn_from_tail(seq);
  for (auto &vp_ptr : predictors<ChoraleDuration>())
    vp_ptr->learn_from_tail(seq);
  for (auto &vp_ptr : predictors<ChoraleRest>())
    vp_ptr->learn_from_tail(seq);
}

void ChoraleVPLayer::reset_viewpoints() {
  for (auto &vp_ptr : predictors<ChoralePitch>())
    vp_ptr->reset();
  for (auto &vp_ptr : predictors<ChoraleDuration>())
    vp_ptr->reset();
  for (auto &vp_ptr : predictors<ChoraleRest>())
    vp_ptr->reset();
}

std::vector<ChoraleEvent> ChoraleMVS::random_walk(unsigned int len) {
  assert(len > 1);

  std::vector<ChoraleEvent> buffer;

  auto keysig = key_distribution.predict({}).sample();
  auto first_pitch = predict<ChoralePitch>(buffer).sample();
  auto first_dur   = predict<ChoraleDuration>(buffer).sample();

  ChoraleEvent first_event(keysig, first_pitch, first_dur, nullptr);
  buffer.push_back(first_event);
  short_term_layer.learn_from_tail(buffer);

  for (unsigned int i = 0; i < len - 1; i++) {
    auto pitch = predict<ChoralePitch>(buffer).sample();
    auto dur   = predict<ChoraleDuration>(buffer).sample();
    auto rest_ptr = predict<ChoraleRest>(buffer).sample().shared_instance();
    ChoraleEvent event(keysig, pitch, dur, rest_ptr);
    buffer.push_back(event);
    short_term_layer.learn_from_tail(buffer);
  }

  return buffer;
}

