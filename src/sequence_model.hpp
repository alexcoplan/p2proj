#ifndef AJC_HGUARD_SEQMODEL
#define AJC_HGUARD_SEQMODEL

#include <cassert>
#include <numeric> // gives us e.g. std::accumulate
#include <array>
#include <cmath>

#include "event.hpp"
#include "event_enumerator.hpp"
#include "context_model.hpp"
#include "random_source.hpp"

// accuracy to which distributions must sum to 1
#define DISTRIBUTION_EPS 1e-14 

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
    combine(const std::vector<EventDistribution<T>> &list) const = 0;
};

template<class T>
struct ArithmeticEntropyCombination : public DistCombStrategy<T> {
  const double re_exponent;

  using values_t = std::array<double, T::cardinality>;

  values_t 
  combine(const std::vector<EventDistribution<T>> &list) const override {
    values_t result = {{0.0}};

    double sum_of_weights = 0.0;

    for (auto dist : list) {
      double norm_entropy = dist.normalised_entropy();
      if (norm_entropy == 0.0) 
        assert(! "Distributions must be non-exclusive to use weighted entropy\
 combination");
      
      double weight = std::pow(norm_entropy, -re_exponent);
      sum_of_weights += weight;

      for (auto event : EventEnumerator<T>()) 
        result[event.encode()] += dist.probability_for(event) * weight;
    }

    for (unsigned int i = 0; i < T::cardinality; i++) 
      result[i] /= sum_of_weights;

    return result;
  }

  ArithmeticEntropyCombination(double exponent) : re_exponent(exponent) {}
};

template<class T>
struct GeometricEntropyCombination : public DistCombStrategy<T> {
  const double re_exponent;

  using values_t = std::array<double, T::cardinality>;

  values_t
  combine(const std::vector<EventDistribution<T>> &list) const override {
    values_t result;
    for (auto &v : result)
      v = 1.0;

    double sum_of_weights = 0.0;
    std::vector<double> dist_weights;

    for (auto dist : list) {
      double norm_entropy = dist.normalised_entropy();
      if (norm_entropy == 0.0) {
        std::cerr << "Exclusive distribution! Values:" << std::endl;
        std::cerr << dist.debug_summary() << std::endl;
        assert(! "Distribution must be non-exclusive.");
      }

      double weight = std::pow(norm_entropy, -re_exponent);
      sum_of_weights += weight;
      dist_weights.push_back(weight);
    }

    assert(dist_weights.size() == list.size());

    double total_probability = 0.0;
    for (const auto &e : EventEnumerator<T>()) {
      unsigned int j = e.encode();
      unsigned int i = 0;
      for (const auto &dist : list)
        result[j] *= std::pow(dist.probability_for(e), dist_weights[i++]);
      result[j] = std::pow(result[j], 1.0 / sum_of_weights);
      total_probability += result[j];
    }

    // normalise (n.b. normalisation constant cannot be computed in advance for
    // geometric combination)
    for (auto &v : result)
      v /= total_probability;

    return result;
  }

  GeometricEntropyCombination(double exponent) :
    re_exponent(exponent) {}
};

template<class T>
struct LogGeoEntropyCombination : public DistCombStrategy<T> {
  const double re_exponent;

  using values_t = std::array<double, T::cardinality>;

  values_t
  combine(const std::vector<EventDistribution<T>> &dists) const override {
    values_t result{{0.0}};

    double sum_of_weights = 0.0;

    for (const auto &dist : dists) {
      double norm_entropy = dist.normalised_entropy();
      if (norm_entropy == 0.0) {
        std::cerr << "Exclusive distribution! Values:" << std::endl;
        std::cerr << dist.debug_summary() << std::endl;
        assert(! "Distribution must be non-exclusive.");
      }

      double weight = std::pow(norm_entropy, -re_exponent);
      sum_of_weights += weight;
      for (auto e : EventEnumerator<T>())
        result[e.encode()] += weight * std::log2(dist.probability_for(e));
    }

    double total_probability = 0.0;
    for (auto &v : result) {
      v = std::pow(2.0, v / sum_of_weights);
      total_probability += v;
    }
    
    // normalise
    for (auto &v : result)
      v /= total_probability;

    return result;
  }

  LogGeoEntropyCombination(double entropy_bias) : re_exponent(entropy_bias) {}
};

/**************************************************
 * EventDistribution: declaration
 **************************************************/

template<class T> class EventDistribution {
private:
  std::array<double, T::cardinality> values;

public:
  EventDistribution(const std::array<double, T::cardinality> &vs);
  EventDistribution(const DistCombStrategy<T> &strategy, 
      const std::vector<EventDistribution> &dist);
  constexpr static double max_entropy() { return std::log2(T::cardinality); }
  EventDistribution<T> weighted_combination (
      const std::vector<EventDistribution<T> &> &vector);
  double probability_for(const T &event) const;
  double entropy() const;
  double normalised_entropy() const;
  T sample() const;
  T sample_with_source(RandomSource *) const;
  void combine_in_place(const DistCombStrategy<T> &strategy, 
      const EventDistribution<T> &dist) {
    EventDistribution combined(strategy, {dist,*this});
    values = combined.values;
  }
  std::string debug_summary() const;
};

/**************************************************
 * EventDistribution: public methods
 **************************************************/

template<class T>
std::string
EventDistribution<T>::debug_summary() const {
  std::string result = "";
  for (unsigned int i = 0; i < T::cardinality; i++) {
    result += "P(" + std::to_string(i) + ") = " + 
      std::to_string(values[i]) + "\n";
  }
  return result;
}

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
    std::cerr << "Distribution failed: total prob = " << total_probability << 
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
    const std::vector<EventDistribution<T>> &distributions) :
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

template<class T>
T EventDistribution<T>::sample_with_source(RandomSource *rs) const {
  // build a cumulative frequency distribution and sample from that
  std::array<double, T::cardinality> cfd{{0.0}};
  double total_probability = 0.0;
  for (unsigned int i = 0; i < T::cardinality; i++) {
    total_probability += values[i];
    cfd[i] = total_probability;
  }

  double target_probability = rs->sample();
    
  // binary search to find i s.t.
  // cfd[i-1] < p <= cfd[i]
  // or 0 <= p <= cfd[1] if p <= cfd[1]
  unsigned int ub = T::cardinality; // exclusive
  unsigned int lb = 0; // inclusive

  while (ub - lb > 1) {
    unsigned int midpoint = (ub + lb)/2;
    if (target_probability <= cfd[midpoint - 1])
      ub = midpoint;
    else 
      lb = midpoint;
  }

  return T(lb);
}

template<class T>
T EventDistribution<T>::sample() const {
  return sample_with_source(&DefaultRandomSource::shared_source);
}

/**************************************************
 * SequenceModel: declaration
 **************************************************/

template<class T> class SequenceModel {
private:
  ContextModel<T::cardinality> model; // underlying context model

  std::vector<unsigned int> encode_sequence(const std::vector<T> &seq) const;

public:
  SequenceModel(unsigned int history);
  void learn_sequence(const std::vector<T> &seq);
  void clear_model();
  void set_history(unsigned int h);
  unsigned int get_history() const;
  void update_from_tail(const std::vector<T> &seq);
  double probability_of(const std::vector<T> &seq) const;
  double avg_sequence_entropy(const std::vector<T> &seq) const;
  unsigned int count_of(const std::vector<T> &seq) const;
  EventDistribution<T> gen_successor_dist(const std::vector<T> &ctx) const;
  void write_latex(std::string filename) const;

  // we pass the location of this function to the underlying context model in
  // order to generate correctly-labelled graphviz output
  static std::string string_decoder (unsigned int);
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
}

// simple wrappers around the context model
template<class T>
void SequenceModel<T>::set_history(unsigned int h) {
  model.set_history(h);
}

template<class T>
unsigned int SequenceModel<T>::get_history() const {
  return model.get_history();
}

template<class T> 
void SequenceModel<T>::learn_sequence(const std::vector<T> &seq) {
  model.learn_sequence(encode_sequence(seq));
}

template<class T>
void SequenceModel<T>::clear_model() {
  model.clear_model();
}

template<class T>
void SequenceModel<T>::update_from_tail(const std::vector<T> &seq) {
  model.update_from_tail(encode_sequence(seq));
}

template<class T>
double SequenceModel<T>::probability_of(const std::vector<T> &seq) const {
  return model.probability_of(encode_sequence(seq));
}

template<class T>
double SequenceModel<T>::avg_sequence_entropy(const std::vector<T> &seq) const {
  return model.avg_sequence_entropy(encode_sequence(seq));
}

template<class T>
unsigned int SequenceModel<T>::count_of(const std::vector<T> &seq) const {
  return model.count_of(encode_sequence(seq));
}

template<class T> EventDistribution<T> 
SequenceModel<T>::gen_successor_dist(const std::vector<T> &context) const {
  std::vector<T> tmp_ctx(context);
  std::array<double, T::cardinality> values;

  for (auto candidate_event : EventEnumerator<T>()) {
    tmp_ctx.push_back(candidate_event);
    values[candidate_event.encode()] = probability_of(tmp_ctx);
    tmp_ctx.pop_back();
  }

  return EventDistribution<T>(values);
}

template<class T>
std::string SequenceModel<T>::string_decoder(unsigned int code) {
  return T(code).string_render();
}

template<class T>
void SequenceModel<T>::write_latex(std::string filename) const {
  model.write_latex(filename, &SequenceModel<T>::string_decoder);
}

/**************************************************
 * SequenceModel: private methods
 **************************************************/

// Although this might look inefficient since we are returning a "big" object (a
// vector of Ts), C++11's move semantics should have our back here.
template<class T> std::vector<unsigned int> 
SequenceModel<T>::encode_sequence(const std::vector<T> &seq) const {
  std::vector<unsigned int> result(seq.size());
  std::transform(seq.begin(), seq.end(), result.begin(),
      [](const T& event) { return event.encode(); });
  return result;
}

#endif // header guard
