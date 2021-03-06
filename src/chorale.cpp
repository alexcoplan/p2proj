#include "chorale.hpp"
#include <array>
#include <cassert>

// convenience macros for LaTeX trie generation
#define NAT(X,N) #X "$_{" #N "}$"
#define SHARP(X,N) #X "\\sharp{}$_{" #N "}$"
#define FLAT(X,N) "\\flatten{" #X "}_" #N

// list of all types currently implemented and their names within the multiple
// viewpoint formalism
// *** basic types
const std::string ChoralePitch::type_name    = "pitch";
const std::string ChoraleDuration::type_name = "duration";
const std::string ChoraleRest::type_name     = "rest";
const std::string ChoraleKeySig::type_name   = "keysig";
const std::string ChoraleTimeSig::type_name  = "timesig";

// *** derived types
const std::string ChoraleInterval::type_name = "seqint";
const std::string ChoraleIntref::type_name   = "intref";
const std::string ChoralePosinbar::type_name = "posinbar";
const std::string ChoraleFib::type_name      = "fib";
const std::string ChoraleFip::type_name      = "fip";
const std::string ChoraleIOI::type_name      = "ioi";

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
  8,3,10,5,0,7,2,9,4
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

const std::array<std::string, ChoraleRest::cardinality>
ChoraleRest::pretty_strs = {
  {"$\\rightarrow$ ", "\\crotchetRest{} ", "\\halfNoteRest{} ", 
    "\\halfNoteRest{}.", "\\wholeNoteRest{}" }
};

ChoraleRest::ChoraleRest(unsigned int c) : CodedEvent(c) {
  if (c >= cardinality) {
    std::string msg = "Bad direct initialisation of ChoraleRest (code = ";
    msg += std::to_string(c) + ")";
    throw ChoraleTypeError(msg);
  }
}

ChoraleRest::ChoraleRest(const QuantizedDuration &qd) :
  CodedEvent(map_in(qd.duration)) {
  if (code >= cardinality || (qd.duration % 4) != 0) {
    std::string msg = "Bad initialisation of ChoraleRest (duration = ";
    msg += std::to_string(qd.duration) + ")";
    throw ChoraleTypeError(msg);
  }
}

/***************************************************
 * ChoraleFib implementation
 ***************************************************/

ChoraleFib::ChoraleFib(unsigned int c) : CodedEvent(c) {
  if (c >= cardinality) {
    std::string msg = "Bad direct initialisation of ChoraleFib (code = ";
    msg += std::to_string(c) + ")";
    throw ChoraleTypeError(msg);
  }
}

/***************************************************
 * ChoraleFip implementation
 ***************************************************/

ChoraleFip::ChoraleFip(unsigned int c) : CodedEvent(c) {
  if (c >= cardinality) {
    std::string msg = "Bad direct initialisation of ChoraleFip (code = ";
    msg += std::to_string(c) + ")";
    throw ChoraleTypeError(msg);
  }
}

/********************************************************************
 * Implementation of stringification for chorale types
 ********************************************************************/

std::ostream& operator<<(std::ostream &os, const ChoralePosinbar &pos) {
  os << "pos(" << pos.encode() << ")";
  return os;
}

std::ostream& operator<<(std::ostream &os, const ChoraleTimeSig &ts) {
  os << (ts.raw_value() == 12 ? "3/4" : "4/4");
  return os;
}

/*************************************************************
 * ChoraleEvent implementation: reification for derived types
 *************************************************************/

EventDistribution<ChoralePitch>
ChoraleEvent::reify(
  const std::vector<ChoraleEvent> &ctx,
  const EventDistribution<ChoraleIntref> &intref_dist
) {
  if (ctx.empty()) {
    const std::string msg = "Need context to predict with intref viewpoint";
    throw ViewpointPredictionException(msg);
  }

  auto ks = ctx.front().project<ChoraleKeySig>();
  auto referent = ks.referent();

  std::array<double, ChoralePitch::cardinality> new_values{{0.0}};
  double total_probability = 0.0;

  std::array<MidiPitch, 3> base_pitches{{ 48,60,72 }};

  for (auto intref : EventEnumerator<ChoraleIntref>()) {
    for (auto base : base_pitches) {
      MidiPitch transposed(base.pitch + referent.pitch + intref.encode());

      if (!ChoralePitch::is_valid_pitch(transposed))
        continue;

      ChoralePitch candidate_pitch(transposed);

      auto prob = intref_dist.probability_for(intref);
      new_values[candidate_pitch.encode()] += prob;
      total_probability += prob;
    }
  }

  // normalise
  for (auto &v : new_values)
    v /= total_probability;

  return new_values;
}

EventDistribution<ChoralePitch>
ChoraleEvent::reify(
  const std::vector<ChoraleEvent> &ctx, 
  const EventDistribution<ChoraleInterval> &seqint_dist
) {
  if (ctx.empty()) {
    const std::string msg = 
      "Seqint needs at least one pitch to predict further pitches";
    throw ViewpointPredictionException(msg);
  }

  auto last_pitch = ctx.back().project<ChoralePitch>();
  std::array<double, ChoralePitch::cardinality> new_values{{0.0}};

  double total_probability = 0.0;
  unsigned int valid_predictions = 0;

  for (auto interval : EventEnumerator<ChoraleInterval>()) { 
    auto midi_interval = interval.midi_interval();
    if (!last_pitch.is_valid_transposition(midi_interval))
      continue;

    auto candidate_pitch = last_pitch + midi_interval;
    auto prob = seqint_dist.probability_for(interval);
    new_values[candidate_pitch.encode()] = prob;
    total_probability += prob;
    valid_predictions++;
  }

  if (valid_predictions == ChoraleInterval::cardinality)
    return new_values;

  for (auto &v : new_values)
    v /= total_probability;

  return new_values;
}

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

// intref implementation
ChoraleIntref::ChoraleIntref(unsigned int c) :
  CodedEvent(c) {
  assert(c < cardinality);
}

const std::array<unsigned int, ChoraleIOI::cardinality>
ChoraleIOI::ioi_domain = {
  { 1, 2, 3, 4, 6, 8, 12, 14, 16, 20, 24 }
};

unsigned int ChoraleIOI::map_in(unsigned int dur) {
  for (unsigned int i = 0; i < cardinality; i++)
    if (ioi_domain[i] == dur)
      return i;

  std::string msg = "Bad IOI: " + std::to_string(dur);
  throw ChoraleTypeError(msg);
}

unsigned int ChoraleIOI::map_out(unsigned int c) {
  return ioi_domain[c];
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
    return new_values;

  for (auto &v : new_values)
    v /= total_probability;

  return new_values;
}

std::vector<ChoraleIntref>
IntrefViewpoint::lift(const std::vector<ChoraleEvent> &events) const {
  if (events.empty())
    return {};

  auto pitches = ChoraleEvent::template lift<ChoralePitch>(events);
  auto referent = events.front().project<ChoraleKeySig>().referent();
  
  std::vector<ChoraleIntref> result;
  for (const auto &p : pitches)
    result.push_back((p.raw_value() - referent.pitch) % 12);

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

  std::array<MidiPitch, 3> base_pitches{{ 48,60,72 }};

  for (auto intref : EventEnumerator<ChoraleIntref>()) {
    for (auto base : base_pitches) {
      MidiPitch transposed(base.pitch + referent.pitch + intref.encode());

      if (!ChoralePitch::is_valid_pitch(transposed))
        continue;

      ChoralePitch candidate_pitch(transposed);

      auto prob = intref_dist.probability_for(intref);
      new_values[candidate_pitch.encode()] += prob;
      total_probability += prob;
    }
  }

  // normalise
  for (auto &v : new_values)
    v /= total_probability;

  EventDistribution<ChoralePitch> dist(new_values);

  return dist;
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

std::vector<ChoraleEvent> 
ChoraleMVS::random_walk(unsigned int len, const QuantizedDuration &timesig) {
  assert(len > 1);

  std::vector<ChoraleEvent> buffer;

  auto keysig = key_distribution.predict({}).sample();

  // for now just start on the tonic
  ChoralePitch first_pitch(MidiPitch(60 + keysig.referent().pitch));

  auto first_dur   = predict<ChoraleDuration>({}).sample();
  auto first_rest  = predict<ChoraleRest>({}).sample();

  ChoraleEvent first_event(keysig, timesig, first_pitch, first_dur, first_rest);
  buffer.push_back(first_event);
  short_term_layer.learn_from_tail(buffer);

  for (unsigned int i = 0; i < len - 1; i++) {
    auto pitch = predict<ChoralePitch>(buffer).sample();
    auto dur   = predict<ChoraleDuration>(buffer).sample();
    auto rest  = predict<ChoraleRest>(buffer).sample();
    ChoraleEvent event(keysig, timesig, pitch, dur, rest);
    buffer.push_back(event);
    short_term_layer.learn_from_tail(buffer);
  }

  return buffer;
}

