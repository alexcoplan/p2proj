#ifndef AJC_HGUARD_VIEWPOINT
#define AJC_HGUARD_VIEWPOINT

#include "sequence_model.hpp"

template<class EventStructure, class T_predict>
class Predictor {
public:
  virtual EventDistribution<T_predict> 
    predict(const std::vector<EventStructure> &es) const = 0;
  
  virtual void
    learn(const std::vector<EventStructure> &es) = 0;

  virtual void
    learn_from_tail(const std::vector<EventStructure> &es) = 0;

  virtual void
    set_history(unsigned int h) = 0;

  virtual unsigned int
    get_history() const = 0;

  virtual void
    reset() = 0; // undoes any training (useful for short-term models)

  virtual bool 
    can_predict(const std::vector<EventStructure> &es) const = 0;

  virtual Predictor *clone() const = 0;
};

/* GeneralViewpoint
 *
 * This class should subsume all other viewpoint-related classes. It should work
 * for both derived and basic types. The idea is as follows:
 * 
 * In order to implement a viewpoint, there are two main bits of functionality
 * needed. We need to know how to *lift* a sequence of events of type
 * T_viewpoint form a stream of events of type EventStructure, i.e.
 *  - lift : vec<EventStrcutrue> --> vec<T_viewpoint>
 *
 * We can then model sequences of type vec<T_viewpoint> using a sequence model.
 * Also, we need to be able to predict a distribution over T_surface given a
 * context (a sequence of EventStructure events) and a distribution over
 * T_viewpoint. We call this process "reifying" a distribution over the hidden
 * (viewpoint type). In the MVS formalism this is (roughly) the inverse of the
 * phi function. In other words:
 *  - reify : vec<EventStructure> x dist<T_viewpoint> --> dist<T_surface>
 *
 * With GeneralViewpoints, both `lift` and `reify` are implemented as part of
 * the EventStructure itself, allowing us to create viewpoints on arbitrary
 * combinations of basic or derived types.
 */
template<class EventStructure, class T_viewpoint, class T_surface>
class GeneralViewpoint : public Predictor<EventStructure, T_surface> {
protected:
  using PredBase = Predictor<EventStructure, T_surface>;
  SequenceModel<T_viewpoint> model;

public:
  void reset() override { model.clear_model(); }
  void set_history(unsigned int h) override { model.set_history(h); }
  unsigned int get_history() const override { return model.get_history(); }
  void write_latex(std::string filename) const { model.write_latex(filename); }

  void learn(const std::vector<EventStructure> &events) override {
    model.learn_sequence(EventStructure::template lift<T_viewpoint>(events));
  }

  void learn_from_tail(const std::vector<EventStructure> &events) override {
    auto lifted = EventStructure::template lift<T_viewpoint>(events);
    if (lifted.size() > 0)
      model.update_from_tail(lifted);
  }

  EventDistribution<T_surface> 
  predict(const std::vector<EventStructure> &ctx) const override {
    auto lifted = EventStructure::template lift<T_viewpoint>(ctx);
    auto hidden_dist = model.gen_successor_dist(lifted);
    return EventStructure::reify(ctx, hidden_dist);
  }

  bool can_predict(const std::vector<EventStructure> &) const override {
    // TODO: once all VPs have been replaced with GeneralViewpoints, this method
    // can go and we will switch to an exception-based approach to this
    // 
    // this is just so that we are compatible with the existing Predictor<>
    // interface for now
    return true; 
  }

  PredBase *clone() const override { return new GeneralViewpoint(*this); }
  GeneralViewpoint(unsigned int hist) : model(hist) {}
};

/* T_viewpoint is the type internal to the viewpoint (such as interval).
 * T_surface is the basic musical type that T_viewpoint is derived from and thus
 * that this viewpoint is capable of predicting. */
template<class EventStructure, class T_viewpoint, class T_surface> 
class Viewpoint : public Predictor<EventStructure, T_surface> {
protected:
  SequenceModel<T_viewpoint> model;

  virtual std::vector<T_viewpoint> 
    lift(const std::vector<EventStructure> &events) const = 0; 

public:
  void reset() override { model.clear_model(); }
  void set_history(unsigned int h) override { model.set_history(h); }
  unsigned int get_history() const override { return model.get_history(); }
  void write_latex(std::string filename) const { model.write_latex(filename); }

  void learn(const std::vector<EventStructure> &events) override {
    model.learn_sequence(lift(events));
  }

  void learn_from_tail(const std::vector<EventStructure> &events) override {
    auto lifted = lift(events);
    if (lifted.size() > 0)
      model.update_from_tail(lifted);
  }

  Viewpoint(unsigned int history) : model(history) {}
};


template<class EventStructure, class T_hidden, class T_predict>
class BasicLinkedViewpoint : 
  public Viewpoint<EventStructure, EventPair<T_hidden, T_predict>, T_predict> 
{
  using BaseVP = 
    Viewpoint<EventStructure, EventPair<T_hidden, T_predict>, T_predict>;

  std::vector<EventPair<T_hidden, T_predict>> 
    lift(const std::vector<EventStructure> &events) const override {
    return EventStructure::template lift<T_hidden, T_predict>(events);
  }

public:

  EventDistribution<T_predict> 
  predict(const std::vector<EventStructure> &ctx) const override {
    // generate the joint distribution
    auto dist = this->model.gen_successor_dist(lift(ctx));
    
    // then marginalise (sum over the hidden type)
    std::array<double, T_predict::cardinality> values{{0.0}};
    for (auto e_predict : EventEnumerator<T_predict>())
      for (auto e_hidden : EventEnumerator<T_hidden>()) {
        EventPair<T_hidden, T_predict> pair(e_hidden, e_predict);
        values[e_predict.encode()] += dist.probability_for(pair);
      }

    return EventDistribution<T_predict>(values);
  }

  bool can_predict(const std::vector<EventStructure> &) const override {
    return true; // basic VPs can always predict
  }

  BasicLinkedViewpoint *clone() const override {
    return new BasicLinkedViewpoint(*this);
  }

  BasicLinkedViewpoint(unsigned int history) : BaseVP(history) {}
};

template<class EventStructure, class T_basic>
class BasicViewpoint : public Viewpoint<EventStructure, T_basic, T_basic> {
  using VPBase = Viewpoint<EventStructure, T_basic, T_basic>;

  std::vector<T_basic> 
    lift(const std::vector<EventStructure> &events) const override;

public:
  EventDistribution<T_basic> 
    predict(const std::vector<EventStructure> &context) const override;

  bool can_predict(const std::vector<EventStructure> &) const override {
    // basic viewpoints are essentially just wrapped-up context models, so they
    // can always predict
    return true;
  }

  VPBase *clone() const override { return new BasicViewpoint(*this); }

  BasicViewpoint(int history) : VPBase(history) {}
};

/* In basic viewpoints, we override `lift` for efficiency, since there is no
 * need to use the projection function and allocate a load of events on the
 * heap, we can just return the original sequence! */
template<class ES, class T> 
std::vector<T> 
BasicViewpoint<ES,T>::lift(const std::vector<ES> &events) const {
  return ES::template lift<T>(events);
}

template<class EventStructure, class T>
EventDistribution<T> 
BasicViewpoint<EventStructure,T>::
predict(const std::vector<EventStructure> &context) const {
  return 
    this->model.gen_successor_dist(EventStructure::template lift<T>(context));
}

struct ViewpointPredictionException : public std::runtime_error {
  ViewpointPredictionException(std::string msg) : 
    std::runtime_error(msg) {}
};

#endif
