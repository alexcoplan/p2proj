#ifndef AJC_HGUARD_SEQMODEL
#define AJC_HGUARD_SEQMODEL

#include <cassert>
#include <numeric> // gives us e.g. std::accumulate
#include <array>
#include <cmath>

#include "event.hpp"
#include "context_model.hpp"

// accuracy to which distributions must sum to 1
#define DISTRIBUTION_EPS 1e-15 

/* SequenceModel provides an abstraction of the ContextModel class, such that
 * the model takes abstract events and encodes them for appropriately for the
 * underlying ContextModel */

/**************************************************
 * EventDistribution: declaration
 **************************************************/

template<class T> class EventDistribution {
private:
  std::array<double, T::cardinality> values;

public:
  EventDistribution(const std::array<double, T::cardinality> &vs);
  constexpr static double max_entropy();
  EventDistribution<T> weighted_combination (
      const std::vector<EventDistribution<T> &> &vector);
  double probability_for(const T &event);
  double entropy();
};

/**************************************************
 * EventDistribution: public methods
 **************************************************/

template<class T> 
EventDistribution<T>::EventDistribution(
    const std::array<double, T::cardinality> &vs) : values(vs) {
  // enforce T : SequenceEvent
  static_assert(std::is_base_of<SequenceEvent, T>::value, "EventDistribution\
 can only be specialized on SequenceEvents");
  static_assert(T::cardinality > 0, "Event type must have strictly positive\
 cardinality!");

  double total_probability = std::accumulate(values.begin(), values.end(), 0.0);
  assert( std::abs(total_probability - 1.0) < DISTRIBUTION_EPS );
}

template<class T>
double EventDistribution<T>::probability_for(const T& event) {
  return values[event.encode()];
}

template<class T>
constexpr static double max_entropy() {
  return std::log2(T::cardinality);
}

// might need to be careful about numerical stability here.
// what if v > 0.0 but v ~~ 0.0?
template<class T>
double EventDistribution<T>::entropy() {
  double sum = 0.0;
  for (double v : values) {
    if (v == 0.0)
      continue;

    sum -= v * std::log2(v);
  }

  return sum;
}

/**************************************************
 * SequenceModel: declaration
 **************************************************/

template<class T> class SequenceModel {
private:
  ContextModel<T::cardinality> model; // underlying context model

  std::vector<unsigned int> encode_sequence(const std::vector<T> &seq);

public:
  SequenceModel(unsigned int history);
  void learn_sequence(const std::vector<T> &seq);
  double probability_of(const std::vector<T> &seq);
  EventDistribution<T> gen_successor_dist(const std::vector<T> &ctx);
};

/**************************************************
 * SequenceModel: public methods
 **************************************************/

template<class T> 
SequenceModel<T>::SequenceModel(unsigned int h) : model(h) {
  // enforce T : SequenceEvent
  static_assert(std::is_base_of<SequenceEvent, T>::value, "SequenceModel can\
 only be specialized on SequenceEvents");
  static_assert(T::cardinality > 0, "Event type must have strictly positive\
 cardinality!");
};

// simple wrappers around the context model
template<class T> 
void SequenceModel<T>::learn_sequence(const std::vector<T> &seq) {
  model.learn_sequence(encode_sequence(seq));
}

template<class T>
double SequenceModel<T>::probability_of(const std::vector<T> &seq) {
  return model.probability_of(encode_sequence(seq));
}

template<class T> EventDistribution<T> 
SequenceModel<T>::gen_successor_dist(const std::vector<T> &context) {
  std::vector<T> tmp_ctx(context);
  std::array<double, T::cardinality> values;

  // obvious TODO: define an iterator to generate all events of a given type
  for (unsigned int i = 0; i < T::cardinality; i++) {
    T candidate_event = T(i);
    tmp_ctx.push_back(candidate_event);
    values[i] = probability_of(tmp_ctx);
    tmp_ctx.pop_back();
  }

  return EventDistribution<T>(values);
}

/**************************************************
 * SequenceModel: private methods
 **************************************************/

// Although this might look inefficient since we are returning a "big" object (a
// vector of Ts), C++11's move semantics should have our back here.
template<class T> std::vector<unsigned int> 
SequenceModel<T>::encode_sequence(const std::vector<T> &seq) {
  std::vector<unsigned int> result(seq.size());
  std::transform(seq.begin(), seq.end(), result.begin(),
      [](const T& event) { return event.encode(); });
  return result;
}

#endif // header guard
