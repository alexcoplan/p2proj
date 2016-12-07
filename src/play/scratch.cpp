#include "context_model.hpp"
#include <iostream>
// #include <type_traits>

class Event {
public:
  const static int bf = 1;
};

class PitchEvent : Event {
public:
  const static int bf = 12;
};

class DurEvent : Event {
public:
  const static int bf = 96;
};

template<class T> class Wrapper {
public:
  ContextModel<T::bf> model;

  Wrapper(unsigned int hist) : model(hist) {
    static_assert(std::is_base_of<Event, T>::value, "T not derived from Event");
  }

  void thang() {
    std::cout << "bf: " << T::bf << std::endl;
  }
};

int main(void)
{
  Wrapper<PitchEvent> w1(3);
  Wrapper<DurEvent> w2(3);

  w1.thang();
  w2.thang();
}
