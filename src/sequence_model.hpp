#ifndef AJC_HGUARD_SEQMODEL
#define AJC_HGUARD_SEQMODEL

#include <cassert>
#include <numeric> // gives us e.g. std::accumulate
#include <array>
#include <cmath>

#include "event.hpp"
#include "context_model.hpp"

// accuracy to which distributions must sum to 1
#define DISTRIBUTION_EPS 1e-10 

/* SequenceModel provides an abstraction of the ContextModel class, such that
 * the model takes abstract events and encodes them for appropriately for the
 * underlying ContextModel */

/**************************************************
 * EventDistribution: declaration
 **************************************************/

template<class T> struct EventDistribtuion {
  std::array<double, T::cardinality> values;

  EventDistribtuion(const std::array<double, T::cardinality> &vs);
  double probability_for(const T &event);
};

/**************************************************
 * EventDistribution: implementation
 **************************************************/

template<class T> 
EventDistribtuion<T>::EventDistribtuion(
    const std::array<double, T::cardinality> &vs) : values(vs) {
  // enforce T : SequenceEvent
  static_assert(std::is_base_of<SequenceEvent, T>::value, "EventDistribtuion\
 can only be specialized on SequenceEvents");
  static_assert(T::cardinality > 0, "Event type must have strictly positive\
 cardinality!");

  double total_probability = std::accumulate(values.begin(), values.end(), 0.0);
  assert( std::abs(total_probability - 1.0) < 1e-10 );
}

template<class T>
double EventDistribtuion<T>::probability_for(const T& event) {
  return values[event.encode()];
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
  EventDistribtuion<T> gen_successor_dist(const std::vector<T> &ctx);
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

template<class T> EventDistribtuion<T> 
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

  return EventDistribtuion<T>(values);
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
