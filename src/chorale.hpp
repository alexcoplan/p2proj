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
#include <memory>

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
  bool operator==(const CodedEvent &other) const {
    return this->code == other.code;
  }

  CodedEvent(unsigned int c) : code(c) {}
};

// some forward declarations of types used in other types
class ChoraleInterval;
class ChoraleIntref;

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
  constexpr static bool is_valid_pitch(const MidiPitch &mp) {
    return mp.pitch >= lowest_midi_pitch && 
      mp.pitch < lowest_midi_pitch + cardinality;
  }
  unsigned int encode() const override { return code; } 
  unsigned int raw_value() const { return map_out(code); } 
  std::string string_render() const override {
    return pitch_strings.at(code);
  }

  // check if transposition by the interval delta gives a valid chorale pitch
  bool is_valid_transposition(const MidiInterval &delta) const;
  MidiInterval operator-(const ChoralePitch &rh_pitch) const;
  ChoralePitch operator+(const MidiInterval &delta) const;

  // need the "code" constructor for enumeration etc. to work 
  ChoralePitch(unsigned int code);
  ChoralePitch(const MidiPitch &pitch);
};

class ChoraleDuration : public CodedEvent {
public:
  constexpr static unsigned int cardinality = 15;

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
public:
  constexpr static int cardinality = 9;

private:
  constexpr static int min_sharps = -4;
  constexpr static unsigned int map_in(int num_sharps) {
    return num_sharps - min_sharps;
  }
  constexpr static int map_out(unsigned int some_code) {
    return some_code + min_sharps;
  }

  const static std::array<unsigned int, cardinality> referent_map;

public:
  unsigned int encode() const override { return code; } 
  unsigned int raw_value() const { return map_out(code); } 
  bool intref_gives_valid_pitch(const ChoraleIntref &intref) const;
  MidiPitch referent() const { 
    return MidiPitch(referent_map[code]);
  }

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

class ChoraleRest : public CodedEvent {
public:
  using singleton_ptr_t = const ChoraleRest * const;
  constexpr static unsigned int cardinality = 3;
  const static std::array<const ChoraleRest, cardinality> shared_instances;

private:
  constexpr static unsigned int map_in(unsigned int rest_len) {
    return rest_len / 4;
  }
  constexpr static unsigned int map_out(unsigned int code) {
    return code * 4;
  }
  const static std::array<std::string, cardinality> pretty_strs;


public:
  unsigned int encode() const override { return code; }
  unsigned int raw_value() const { return map_out(code); }

  singleton_ptr_t shared_instance() {
    return &shared_instances[code];
  }

  std::string string_render() const override {
   return pretty_strs[code];
  }

  constexpr static bool valid_singleton_ptr(singleton_ptr_t ptr) {
    return (ptr == nullptr
         || ptr == &shared_instances[0]
         || ptr == &shared_instances[1]
         || ptr == &shared_instances[2]);
  }

  ChoraleRest(unsigned int c);
  ChoraleRest(const QuantizedDuration &qd);
};


/* This type defines the event space we use to model chorales. ChoraleEvents
 * have as members all the different types that make up a chorale event (pitch,
 * duration, offset, etc.) */
struct ChoraleEvent {
  ChoraleKeySig keysig;
  ChoralePitch pitch;
  ChoraleDuration duration;
  ChoraleRest::singleton_ptr_t rest;

  template<typename T>
  T project() const;

  template<typename T>
  static std::vector<T>
  lift(const std::vector<ChoraleEvent> &es);

  template<typename P, typename Q>
  static std::vector<EventPair<P,Q>>
  lift(const std::vector<ChoraleEvent> &es);

  ChoraleEvent(const KeySig &ks,
               const MidiPitch &mp, 
               const QuantizedDuration &dur, 
               ChoraleRest::singleton_ptr_t rest_ptr) :
    keysig(ks), pitch(mp), duration(dur), rest(rest_ptr) {
    assert(ChoraleRest::valid_singleton_ptr(rest_ptr));
  }

  ChoraleEvent(const ChoraleKeySig &ks,
               const ChoralePitch &cp,
               const ChoraleDuration &cd,
               ChoraleRest::singleton_ptr_t rest_ptr) :
    keysig(ks), pitch(cp), duration(cd), rest(rest_ptr) {
    assert(ChoraleRest::valid_singleton_ptr(rest_ptr));
  }
};

template<>
std::vector<ChoraleRest>
inline ChoraleEvent::lift(const std::vector<ChoraleEvent> &es) {
  std::vector<ChoraleRest> result;
  for (const auto &e : es) {
    if (e.rest)
      result.push_back(*e.rest);
  }

  return result;
}

template<typename T>
std::vector<T>
ChoraleEvent::lift(const std::vector<ChoraleEvent> &es) {
  std::vector<T> result;
  std::transform(es.begin(), es.end(), std::back_inserter(result),
      [](const ChoraleEvent &e) { return e.project<T>(); });
  return result;
}

template<typename P, typename Q>
std::vector<EventPair<P,Q>>
ChoraleEvent::lift(const std::vector<ChoraleEvent> &es) {
  std::vector<EventPair<P,Q>> result;
  std::transform(es.begin(), es.end(), std::back_inserter(result),
      [](const ChoraleEvent &e) {
        return EventPair<P,Q>(e.project<P>(), e.project<Q>());
      });
  return result;
}

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
  MidiInterval midi_interval() const { return MidiInterval(raw_value()); }
  std::string string_render() const override;

  ChoraleInterval(const MidiInterval &delta_pitch);
  ChoraleInterval(const ChoralePitch &from, const ChoralePitch &to);
  ChoraleInterval(unsigned int code);
};

// the representation of this type is very simple:
// because intervals are taken mod 12 and the domain is dense,
// then the surface type is a compact code by default
class ChoraleIntref : public CodedEvent {
public:
  constexpr static unsigned int cardinality = 12;

  unsigned int encode() const override { return code; }
  std::string string_render() const override {
    return std::to_string(code);
  }

  ChoraleIntref(unsigned int code);
};

/**********************************************************
 * Chorale Viewpoints
 **********************************************************/

class IntervalViewpoint : 
  public Viewpoint<ChoraleEvent, ChoraleInterval, ChoralePitch> {
private:
  using ParentVP = Viewpoint<ChoraleEvent, ChoraleInterval, ChoralePitch>;

  std::vector<ChoraleInterval>
    lift(const std::vector<ChoraleEvent> &events) const override;

public:
  IntervalViewpoint(unsigned int h) : ParentVP(h) {}

  bool can_predict(const std::vector<ChoraleEvent> &ctx) const override {
    return ctx.size() > 1;
  }

  IntervalViewpoint *clone() const override { 
    return new IntervalViewpoint(*this);
  }

  EventDistribution<ChoralePitch>
    predict(const std::vector<ChoraleEvent> &pitches) const override;

};

class IntrefViewpoint :
  public Viewpoint<ChoraleEvent, ChoraleIntref, ChoralePitch> {
private:
  using ParentVP = Viewpoint<ChoraleEvent, ChoraleIntref, ChoralePitch>;

  std::vector<ChoraleIntref>
    lift(const std::vector<ChoraleEvent> &events) const override;

public:
  IntrefViewpoint(unsigned int h) : ParentVP(h) {}

  bool can_predict(const std::vector<ChoraleEvent> &ctx) const override {
    return !ctx.empty(); // don't need any context to predict
  }

  IntrefViewpoint *clone() const override {
    return new IntrefViewpoint(*this);
  }

  EventDistribution<ChoralePitch>
   predict(const std::vector<ChoraleEvent> &ctx) const override;

  EventDistribution<ChoralePitch> predict_given_key(
    const std::vector<ChoraleEvent> &ctx, 
    const ChoraleKeySig &ks
  ) const;
};

/**********************************************************
 * Chorale Multiple Viewpoint System
 **********************************************************/

class ChoraleVPLayer {
private:
  template<class T>
  using BasicVP =
    BasicViewpoint<ChoraleEvent, T>;

  template<class T>
  using Pred =
    Predictor<ChoraleEvent, T>;

  template<class T>
  using PredictorList = 
    std::vector<std::unique_ptr<Pred<T>>>;

  PredictorList<ChoralePitch> pitch_predictors;
  PredictorList<ChoraleDuration> duration_predictors;
  PredictorList<ChoraleRest> rest_predictors;

  template<typename T>
    PredictorList<T> &predictors();

  template<typename T> const PredictorList<T> &predictors() const {
      return const_cast<ChoraleVPLayer *>(this)->predictors<T>();
  }

public:
  double entropy_bias; // used for intra-layer combination of VPs
  unsigned int vp_history;

  std::string debug_summary() const;

  template<class T>
  void add_viewpoint(Pred<T> *p) {
    Pred<T> *cloned_vp = p->clone();
    cloned_vp->set_history(vp_history);
    predictors<T>().push_back(std::unique_ptr<Pred<T>>(cloned_vp));
  }

  template<class P, class Q>
  void add_viewpoint(Pred<P> *p) {
    predictors<P>().push_back(std::unique_ptr<Pred<P>>(p->clone()));
    Pred<Q> *q = static_cast<Pred<Q>>(p);
    predictors<Q>().psuh_back(std::unique_ptr<Pred<Q>>(q->clone()));
  }

  template<class T>
  EventDistribution<T> predict(const std::vector<ChoraleEvent> &ctx) const;

  void reset_viewpoints();
  void learn(const std::vector<ChoraleEvent> &seq);
  void learn_from_tail(const std::vector<ChoraleEvent> &seq);

  ChoraleVPLayer(double eb, unsigned int vp_hist) : 
    entropy_bias(eb), vp_history(vp_hist) {}
};

inline std::string ChoraleVPLayer::debug_summary() const {
  auto total_vps = pitch_predictors.size() 
    + duration_predictors.size() + rest_predictors.size();
  
  std::string histories;
  for (const auto &vp_ptr : pitch_predictors)
    histories += std::to_string(vp_ptr->get_history()) + " ";
  for (const auto &vp_ptr : duration_predictors)
    histories += std::to_string(vp_ptr->get_history()) + " ";
  for (const auto &vp_ptr : rest_predictors)
    histories += std::to_string(vp_ptr->get_history()) + " ";

  return "ChoraleVPLayer: " + std::to_string(total_vps) 
    + " viewpoint(s)" + " with histories " + histories;
}

template<class T>
EventDistribution<T>
ChoraleVPLayer::predict(const std::vector<ChoraleEvent> &ctx) const {
  const auto &vps = predictors<T>();

  auto it = vps.begin();
  for (; it != vps.end(); ++it) {
    if ((*it)->can_predict(ctx))
      break;
  }

  if (it == vps.end())
    throw ViewpointPredictionException("No viewpoints can predict context");

  if (vps.size() == 1)
    return (*it)->predict(ctx);

  LogGeoEntropyCombination<T> comb_strategy(entropy_bias);
  std::vector<EventDistribution<T>> predictions;

  for (; it != vps.end(); ++it) {
    if ((*it)->can_predict(ctx)) {
      auto new_prediction = (*it)->predict(ctx);
      predictions.push_back(new_prediction);
    }
  }

  EventDistribution<T> combined(comb_strategy, predictions);
  return combined;
}


// ChoraleVPLayer template implementations 
template<>
inline std::vector<std::unique_ptr<Predictor<ChoraleEvent,ChoralePitch>>> &
ChoraleVPLayer::predictors() {
  return pitch_predictors;
}

template<>
inline std::vector<std::unique_ptr<Predictor<ChoraleEvent,ChoraleDuration>>> &
ChoraleVPLayer::predictors<ChoraleDuration>() {
  return duration_predictors;
}

template<>
inline std::vector<std::unique_ptr<Predictor<ChoraleEvent,ChoraleRest>>> &
ChoraleVPLayer::predictors<ChoraleRest>() {
  return rest_predictors;
}

struct MVSConfig {
  static MVSConfig long_term_only(double entropy_bias) {
    MVSConfig config;
    config.enable_short_term = false;
    config.intra_layer_bias = entropy_bias;
    config.mvs_name = "long-term only MVS";
    return config;
  }

  bool enable_short_term;
  double intra_layer_bias;
  double inter_layer_bias;
  unsigned int st_history;
  unsigned int lt_history;
  std::string mvs_name;

  // default constructor so properties can be sent manually for clarity
  MVSConfig() :
    enable_short_term(true),
    intra_layer_bias(1.0),
    inter_layer_bias(1.0),
    st_history(2),
    lt_history(3),
    mvs_name("default MVS") {}
};

class ChoraleMVS {
public:
  template<class T>
  using BasicVP =
    BasicViewpoint<ChoraleEvent, T>;

  template<class T1, class T2>
  using BasicLinkedVP =
    BasicLinkedViewpoint<ChoraleEvent, T1, T2>;

private:
  template<class T>
  using Pred =
    Predictor<ChoraleEvent, T>;

  ChoraleVPLayer short_term_layer;
  ChoraleVPLayer long_term_layer;
  BasicVP<ChoraleKeySig> key_distribution;
  bool enable_short_term;

public:
  double entropy_bias;
  const std::string mvs_name;

  void debug_print() const {
    std::cerr << "MVS debug: " << mvs_name << std::endl
      << "--> Short-term layer: " 
      << short_term_layer.debug_summary() << std::endl
      << "--> Long-term layer: "
      << long_term_layer.debug_summary() << std::endl;
  }

  void set_intra_layer_bias(double value) {
    short_term_layer.entropy_bias = value;
    long_term_layer.entropy_bias = value;
  }

  template<typename T>
    EventDistribution<T> predict(const std::vector<ChoraleEvent> &ctx) const;

  template<typename T>
    double avg_sequence_entropy(const std::vector<ChoraleEvent> &seq);

  template<typename T>
    std::vector<double>
    cross_entropies(const std::vector<ChoraleEvent> &seq) const;

  template<typename T>
    std::vector<double>
    dist_entropies(const std::vector<ChoraleEvent> &seq) const;

  std::vector<ChoraleEvent> random_walk(unsigned int len);

  void learn(const std::vector<ChoraleEvent> &seq);

  template<class T>
  void add_viewpoint(Pred<T> *p) {
    long_term_layer.add_viewpoint(p);
    short_term_layer.add_viewpoint(p);
  }

  ChoraleMVS(const MVSConfig &config) : 
    short_term_layer(config.intra_layer_bias, config.st_history),
    long_term_layer(config.intra_layer_bias, config.lt_history),
    key_distribution(1), // order-1 viewpoint
    enable_short_term(config.enable_short_term),
    entropy_bias(config.inter_layer_bias), 
    mvs_name(config.mvs_name) {}
};

void
inline ChoraleMVS::learn(const std::vector<ChoraleEvent> &seq) {
  key_distribution.learn({seq[0]});
  long_term_layer.learn(seq);
}

template<typename T>
EventDistribution<T>
ChoraleMVS::predict(const std::vector<ChoraleEvent> &ctx) const {
  auto lt_prediction = long_term_layer.predict<T>(ctx);
  if (enable_short_term) {
    LogGeoEntropyCombination<T> comb_strategy(entropy_bias);
    auto st_prediction = short_term_layer.predict<T>(ctx);
    return EventDistribution<T>(comb_strategy, {st_prediction, lt_prediction});
  }
  return lt_prediction;
}

template<typename T>
double 
ChoraleMVS::avg_sequence_entropy(const std::vector<ChoraleEvent> &seq) {
  if (enable_short_term)
    short_term_layer.reset_viewpoints();
  std::vector<ChoraleEvent> ngram_buf;

  double total_entropy = 0.0;
  auto dist = predict<T>({});

  for (const auto &e : seq) {
    const auto v = e.project<T>();
    total_entropy -= std::log2(dist.probability_for(v));
    ngram_buf.push_back(e);
    if (enable_short_term)
      short_term_layer.learn_from_tail(ngram_buf);
    dist = predict<T>(ngram_buf);
  }

  double avg_entropy = total_entropy/seq.size();
  if (avg_entropy > 100.0) 
    std::cerr << "Warning: very high average entropy (" 
      << avg_entropy << ")" << std::endl;
  
  return avg_entropy;
}

template<typename T>
std::vector<double>
ChoraleMVS::cross_entropies(const std::vector<ChoraleEvent> &seq) const {
  std::vector<ChoraleEvent> ngram_buf;
  std::vector<double> entropies;

  auto dist = predict<T>({});

  for (const auto &e : seq) {
    const auto v = e.project<T>();
    entropies.push_back(-std::log2(dist.probability_for(v)));
    ngram_buf.push_back(e);
    dist = predict<T>(ngram_buf);
  }

  return entropies;
}

template<typename T>
std::vector<double>
ChoraleMVS::dist_entropies(const std::vector<ChoraleEvent> &seq) const {
  std::vector<ChoraleEvent> ngram_buf;
  std::vector<double> entropies;

  auto dist = predict<T>({});

  for (const auto &e : seq) {
    entropies.push_back(dist.entropy());
    ngram_buf.push_back(e);
    dist = predict<T>(ngram_buf);
  }

  return entropies;
}

#endif
