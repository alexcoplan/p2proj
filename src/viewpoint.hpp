#include "sequence_model.hpp"
#include "event.hpp"
#include "chorale.hpp"

/* T_viewpoint is the type internal to the viewpoint (such as interval).
 * T_surface is the basic musical type that T_viewpoint is derived from and thus
 * that this viewpoint is capable of predicting. */
template<class T_viewpoint, class T_surface> 
class Viewpoint {
protected:
  SequenceModel<T_viewpoint> model;

  virtual std::unique_ptr<T_viewpoint> 
    project(const std::vector<T_surface> &events, unsigned int up_to) const = 0;

  virtual std::vector<T_viewpoint> 
    lift(const std::vector<T_surface> &events) const; 

public:
  virtual void learn(const std::vector<T_surface> &events) = 0;

  virtual EventDistribution<T_surface> 
    predict(const std::vector<T_surface> &context) const = 0;

  Viewpoint(int history) : model(history) {}

  void write_latex(std::string filename) const {
    model.write_latex(filename);
  }
};

template<class T_viewpoint, class T_surface> 
std::vector<T_viewpoint>
Viewpoint<T_viewpoint, T_surface>
::lift(const std::vector<T_surface> &events) const {
  std::vector<T_viewpoint> result;
  for (unsigned int i = 1; i <= events.size(); i++) {
    auto projection = project(events, i);
    if (projection)
      result.push_back(T_viewpoint(*projection));
  }

  return result;
}

template<class T_basic>
class BasicViewpoint : public Viewpoint<T_basic, T_basic> {
  using VPBase = Viewpoint<T_basic, T_basic>;

  std::vector<T_basic> lift(const std::vector<T_basic> &events) const override;

  std::unique_ptr<T_basic> 
    project(const std::vector<T_basic> &es, unsigned int up_to) const override;

public:
  EventDistribution<T_basic> 
    predict(const std::vector<T_basic> &context) const override;


  void learn(const std::vector<T_basic> &seq) override;

  BasicViewpoint(int history) : VPBase(history) {}
};

template<class T> std::unique_ptr<T>
BasicViewpoint<T>
::project(const std::vector<T> &events, unsigned int up_to) const {
  if (up_to == 0)
    return std::unique_ptr<T>();
  return std::unique_ptr<T>(new T(events[up_to - 1]));
}

/* In basic viewpoints, we override `lift` for efficiency, since there is no
 * need to use the projection function and allocate a load of events on the
 * heap, we can just return the original sequence! */
template<class T> 
std::vector<T> 
BasicViewpoint<T>::lift(const std::vector<T> &events) const {
  return events;
}

template<class T>
EventDistribution<T>
BasicViewpoint<T>::predict(const std::vector<T> &context) const {
  return this->model.gen_successor_dist(context);
}

template<class T> 
void BasicViewpoint<T>::learn(const std::vector<T> &seq) {
  this->model.learn_sequence(seq);
}
