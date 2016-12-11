#ifndef AJC_HGUARD_SEQMODEL
#define AJC_HGUARD_SEQMODEL

#include <cassert>
#include <numeric> // gives us e.g. std::accumulate
#include <array>
#include <cmath>

#include "event.hpp"
#include "event_enumerator.hpp"
#include "context_model.hpp"

// accuracy to which distributions must sum to 1
#define DISTRIBUTION_EPS 1e-15 

// forward declaration
template<class T> class EventDistribution;

/* SequenceModel provides an abstraction of the ContextModel class, such that
 * the model takes abstract events and encodes them for appropriately for the
 * underlying ContextModel */

/* DistCombStrategy is an abstract class which specifies an algorithm for
 * combining event distributions */

template<class T>
struct DistCombStrategy {
  virtual std::array<double, T::cardinality>
    combine(std::initializer_list<EventDistribution<T>> list) const = 0;
};

template<class T>
struct WeightedEntropyCombination : public DistCombStrategy<T> {
  const double re_exponent;

  using values_t = std::array<double, T::cardinality>;

  values_t 
  combine(std::initializer_list<EventDistribution<T>> list) const override {
    values_t result = {{0.0}};

    double sum_of_weights = 0.0;

    for (auto dist : list) {
      double weight = std::pow(dist.normalised_entropy(), -re_exponent);
      sum_of_weights += weight;

      for (auto event : EventEnumerator<T>()) 
        result[event.encode()] += dist.probability_for(event) * weight;
    }

    for (unsigned int i = 0; i < T::cardinality; i++) 
      result[i] /= sum_of_weights;

    return result;
  }

  WeightedEntropyCombination(double exponent) : re_exponent(exponent) {}
};

/**************************************************
 * EventDistribution: declaration
 **************************************************/

template<class T> class EventDistribution {
private:
  const std::array<double, T::cardinality> values;

public:
  EventDistribution(const std::array<double, T::cardinality> &vs);
  EventDistribution(const DistCombStrategy<T> &strategy, 
      std::initializer_list<EventDistribution>);
  constexpr static double max_entropy() { return std::log2(T::cardinality); }
  EventDistribution<T> weighted_combination (
      const std::vector<EventDistribution<T> &> &vector);
  double probability_for(const T &event) const;
  double entropy() const;
  double normalised_entropy() const;
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

  if ( std::abs(total_probability - 1.0) >= DISTRIBUTION_EPS ) {
    std::cerr << "Distirbution failed: total prob = " << total_probability << 
      std::endl;

    std::cerr << "Values: " << std::endl << std::endl;
    for (unsigned int i = 0; i < vs.size(); i++) 
      std::cerr << "P(" << i << ") = " << vs[i] << std::endl;

    assert(! "Distribution does not add to one, aborting...");
  }
}

/* Combination constructor
 *
 * Takes a distribution combination strategy and uses the given strategy to
 * combine the distributions */
template<class T>
EventDistribution<T>::EventDistribution(const DistCombStrategy<T> &strategy,
    std::initializer_list<EventDistribution<T>> distributions) :
  EventDistribution(strategy.combine(distributions)) {}

template<class T>
double EventDistribution<T>::probability_for(const T& event) const {
  return values[event.encode()];
}

// might need to be careful about numerical stability here.
// what if v > 0.0 but v ~~ 0.0?
template<class T>
double EventDistribution<T>::entropy() const {
  double sum = 0.0;
  for (double v : values) {
    if (v == 0.0)
      continue;

    sum -= v * std::log2(v);
  }

  return sum;
}

template<class T>
double EventDistribution<T>::normalised_entropy() const {
  if (max_entropy() > 0.0) 
    return entropy() / max_entropy();
  
  return 1.0;
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

  for (auto candidate_event : EventEnumerator<T>()) {
    tmp_ctx.push_back(candidate_event);
    values[candidate_event.encode()] = probability_of(tmp_ctx);
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
