#ifndef AJC_HGUARD_VIEWPOINT
#define AJC_HGUARD_VIEWPOINT

#include "sequence_model.hpp"
#include <type_traits>

/* Predictor
 *
 * The fully abstract interface implemented by all viewpoints */
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

namespace template_magic {
  /* IsDerived<T>
   *
   * This is a bit of template metaprogramming magic to check if a type has a
   * static member called `derived_from`.
   *
   * The motivation for this is that we want to handle derived types and basic
   * types differently inside viewpoints.
   *
   * Based on this: 
   * https://en.wikibooks.org/wiki/More_C++_Idioms/Member_Detector */
  template<typename T>
  class IsDerived
  {
      struct Fallback { int derived_from; }; 
      struct Derived : T, Fallback { };

      template<typename U, U> struct Check;

      typedef char ArrayOfOne[1];  // typedef for an array of size one.
      typedef char ArrayOfTwo[2];  // typedef for an array of size two.

      template<typename U> 
      static ArrayOfOne & func(Check<int Fallback::*, &U::derived_from> *);
      
      template<typename U> 
      static ArrayOfTwo & func(...);

    public:
      typedef IsDerived type;
      enum { value = sizeof(func<Derived>(0)) == 2 };
  };

  /* Beware: a bit of template wizardry to lazily-evaluate the surface type of a
   * type (we want lazy evaluation in case it doesn't have one). */
  template<typename T>
  struct just_T { using type = T; };

  template<typename T>
  struct surface_of_T { using type = typename T::derived_from; };

  /* SurfaceType<T>
   *
   * If T is a derived type (see above) then SurfaceType<T> is the type that T
   * is derived from. Otherwise, T is its own surface type. */
  template<typename T>
  using SurfaceType = 
    typename std::conditional<IsDerived<T>::value, 
                              surface_of_T<T>, 
                              just_T<T>>::type::type;
}

template<typename T>
using SurfaceType = template_magic::SurfaceType<T>;

/* Viewpoint
 *
 * Abstract base class for any Viewpoint that internally uses a ContextModel
 * (currently, all of them). Unlike Preditor which is fully abstract, this class
 * contains the ContextModel and implements some of the methods from Predictor.
 */
template<class EventStructure, class T_viewpoint, class T_predict> 
class Viewpoint : public Predictor<EventStructure, T_predict> {
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

/* GeneralViewpoint
 *
 * Concrete but generic implementation of viewpoints that uses type-specific
 * functionality implemented in EventStructure.
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
template<class EventStructure, class T_vp>
using GenVPBase = Viewpoint<EventStructure, T_vp, SurfaceType<T_vp>>;

template<class EventStructure, class T_viewpoint>
class GeneralViewpoint : 
  public GenVPBase<EventStructure, T_viewpoint> {
protected:
  using T_surface = SurfaceType<T_viewpoint>;
  using Base = GenVPBase<EventStructure, T_viewpoint>;
  using PredBase = Predictor<EventStructure, T_surface>;

public:
  std::vector<T_viewpoint> 
  lift(const std::vector<EventStructure> &events) const override {
    return EventStructure::template lift<T_viewpoint>(events);
  }

  EventDistribution<T_surface> 
  predict(const std::vector<EventStructure> &ctx) const override {
    auto lifted = EventStructure::template lift<T_viewpoint>(ctx);
    auto hidden_dist = this->model.gen_successor_dist(lifted);
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
  GeneralViewpoint(unsigned int hist) : Base(hist) {}
};

template<class EventStructure, class T_h, class T_p>
using GenLinkedBase = 
  Viewpoint<EventStructure, EventPair<T_h, T_p>, SurfaceType<T_p>>;

template<class EventStructure, class T_hidden, class T_predict>
class GeneralLinkedVP :
  public GenLinkedBase<EventStructure, T_hidden, T_predict> {
protected:
  using T_pair = EventPair<T_hidden, T_predict>;
  using T_surface = SurfaceType<T_predict>;
  using Base = GenLinkedBase<EventStructure, T_hidden, T_predict>;
  using PredBase = Predictor<EventStructure, T_surface>;

public:
  std::vector<T_pair>
  lift(const std::vector<EventStructure> &events) const override {
    // TODO: in the future this should be done with streams/iterators for
    // efficiency, but vectors will do for now.
    auto left = EventStructure::template lift<T_hidden>(events);
    auto right = EventStructure::template lift<T_predict>(events);
    return T_pair::zip_tail(left, right);
  }

  EventDistribution<T_surface>
  predict(const std::vector<EventStructure> &ctx) const override {
    auto lifted = lift(ctx);
    auto pair_dist = this->model.gen_successor_dist(lifted);
    std::array<double, T_predict::cardinality> predict_values{{0.0}};
    for (auto e_predict : EventEnumerator<T_predict>())
      for (auto e_hidden : EventEnumerator<T_hidden>()) {
        T_pair pair(e_hidden, e_predict);
        predict_values[e_predict.encode()] += pair_dist.probability_for(pair);
      }

    auto derived_dist = EventDistribution<T_predict>(predict_values);
    return EventStructure::reify(ctx, derived_dist);
  }

  bool can_predict(const std::vector<EventStructure> &) const override {
    // TODO: eventually remove this from the Predictor<> interface once all VPs
    // have been properly subsumed by these generalised VPs
    return true; 
  }

  PredBase *clone() const override { return new GeneralLinkedVP(*this); }
  GeneralLinkedVP(unsigned int hist) : Base(hist) {}
};

// template to model a triply-linked type. we use this template rather than
// writing out the full expression partly for brevity, but also because it
// enforces the same nesting convention everywhere.
template<class T_a, class T_b, class T_c>
using TripleLink = EventPair<EventPair<T_a, T_b>, T_c>;

template<class EventStructure, class T_l, class T_r, class T_p>
using TripleLinkedBase = 
  Viewpoint<EventStructure, TripleLink<T_l, T_r, T_p>, SurfaceType<T_p>>;

template<class EventStructure, class T_hleft, class T_hright, class T_predict>
class TripleLinkedVP :
  public TripleLinkedBase<EventStructure, T_hleft, T_hright, T_predict> {
protected:
  // set up the relevant types / template aliases
  using T_hidden = EventPair<T_hleft, T_hright>;
  using T_model = TripleLink<T_hleft, T_hright, T_predict>;
  using T_surface = SurfaceType<T_predict>;
  using Base = TripleLinkedBase<EventStructure, T_hleft, T_hright, T_predict>;
  using PredBase = Predictor<EventStructure, T_surface>;

public:
  std::vector<T_model>
  lift(const std::vector<EventStructure> &events) const override {
    auto h_left = EventStructure::template lift<T_hleft>(events);
    auto h_right = EventStructure::template lift<T_hright>(events);
    auto main_es = EventStructure::template lift<T_predict>(events);
    auto hidden_es = T_hidden::zip_tail(h_left, h_right);
    return T_model::zip_tail(hidden_es, main_es);
  }

  EventDistribution<T_surface>
  predict(const std::vector<EventStructure> &ctx) const override {
    auto lifted = lift(ctx);
    auto triple_dist = this->model.gen_successor_dist(lifted);
    std::array<double, T_predict::cardinality> summed_out{{0.0}};
    for (auto e_predict : EventEnumerator<T_predict>())
      for(auto e_hidden : EventEnumerator<T_hidden>()) {
        T_model pair(e_hidden, e_predict);
        summed_out[e_predict.encode()] += triple_dist.probability_for(pair);
      }

    auto derived_dist = EventDistribution<T_predict>(summed_out);
    return EventStructure::reify(ctx, derived_dist);
  }

  bool can_predict(const std::vector<EventStructure> &) const override {
    return true; // TODO: see other implementations in this file
  }

  PredBase *clone() const override { return new TripleLinkedVP(*this); }
  TripleLinkedVP(unsigned int hist) : Base(hist) {}
};

/************************************************************
 * Legacy code below here
 *
 * Use GeneralViewpoints as these are the future.
 ************************************************************/

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
