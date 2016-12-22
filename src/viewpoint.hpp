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

  virtual bool 
    can_predict(const std::vector<EventStructure> &es) const = 0;

  virtual Predictor *clone() const = 0;
};

/* T_viewpoint is the type internal to the viewpoint (such as interval).
 * T_surface is the basic musical type that T_viewpoint is derived from and thus
 * that this viewpoint is capable of predicting. */
template<class EventStructure, class T_viewpoint, class T_surface> 
class Viewpoint : public Predictor<EventStructure, T_surface> {
protected:
  SequenceModel<T_viewpoint> model;

  virtual std::unique_ptr<T_viewpoint> 
    project(const std::vector<T_surface> &events, unsigned int up_to) const = 0;

  virtual std::vector<T_viewpoint> 
    lift(const std::vector<T_surface> &events) const; 

public:
  void learn(const std::vector<EventStructure> &events) override {
    auto surface_seq = EventStructure::template lift<T_surface>(events);
    model.learn_sequence(lift(surface_seq));
  }

  Viewpoint(int history) : model(history) {}

  void write_latex(std::string filename) const {
    model.write_latex(filename);
  }
};

template<class EventStructure, class T_viewpoint, class T_surface> 
std::vector<T_viewpoint>
Viewpoint<EventStructure, T_viewpoint, T_surface>
::lift(const std::vector<T_surface> &events) const {
  std::vector<T_viewpoint> result;
  for (unsigned int i = 1; i <= events.size(); i++) {
    auto projection = project(events, i);
    if (projection)
      result.push_back(T_viewpoint(*projection));
  }

  return result;
}

template<class EventStructure, class T_basic>
class BasicViewpoint : public Viewpoint<EventStructure, T_basic, T_basic> {
  using VPBase = Viewpoint<EventStructure, T_basic, T_basic>;

  std::vector<T_basic> lift(const std::vector<T_basic> &events) const override;

  std::unique_ptr<T_basic> 
    project(const std::vector<T_basic> &es, unsigned int up_to) const override;

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

template<class EventStructure, class T> std::unique_ptr<T>
BasicViewpoint<EventStructure,T>
::project(const std::vector<T> &events, unsigned int up_to) const {
  if (up_to == 0)
    return std::unique_ptr<T>();
  return std::unique_ptr<T>(new T(events[up_to - 1]));
}

/* In basic viewpoints, we override `lift` for efficiency, since there is no
 * need to use the projection function and allocate a load of events on the
 * heap, we can just return the original sequence! */
template<class ES, class T> 
std::vector<T> 
BasicViewpoint<ES,T>::lift(const std::vector<T> &events) const {
  return events;
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
