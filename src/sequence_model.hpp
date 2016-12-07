#ifndef AJC_HGUARD_SEQMODEL
#define AJC_HGUARD_SEQMODEL

#include "event.hpp"
#include "context_model.hpp"

/* SequenceModel provides an abstraction of the ContextModel class, such that
 * the model takes abstract events and encodes them for appropriately for the
 * underlying ContextModel */

template<class T> class SequenceModel {
private:
  ContextModel<T::cardinality> model; // underlying context model

  std::vector<unsigned int> encode_sequence(const std::vector<T> &seq);

public:
  SequenceModel(unsigned int history);
  void learn_sequence(const std::vector<T> &seq);
  double probability_of(const std::vector<T> &seq);
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
