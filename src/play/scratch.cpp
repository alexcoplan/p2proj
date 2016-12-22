#include <string>
#include <array>
#include <vector>
#include <iostream>

struct EventStructure {
  std::string a_string;
  double a_double;
  char a_char;

  template<typename T>
  T psi() const;

public:
  template<typename T>
  static std::vector<T> 
  phi(const std::vector<EventStructure> &es) {
    std::vector<T> result;
    std::transform(es.begin(), es.end(), std::back_inserter(result),
        [](const EventStructure &e) { return e.psi<T>(); });
    return result;
  }

  template<typename T1, typename T2>
  static std::vector<std::pair<T1,T2>>
  phi(const std::vector<EventStructure> &es) {
    std::vector<std::pair<T1,T2>> result;
    std::transform(es.begin(), es.end(), std::back_inserter(result),
      [](const EventStructure &e) { 
        return std::make_pair<T1,T2>(e.psi<T1>(), e.psi<T2>()); 
      } );
    return result;
  }

  EventStructure(std::string s, double d, char c) :
    a_string(s), a_double(d), a_char(c) {}
};

template<>
std::string EventStructure::psi() const { return a_string; }

template<>
double EventStructure::psi() const { return a_double; }

template<>
char EventStructure::psi() const { return a_char; }

int main(void) {
  std::vector<EventStructure> ss {
    EventStructure("pi", 3.1415, 'p'),
    EventStructure("e", 2.18, 'e')
  };

  for (auto x : EventStructure::phi<std::string>(ss)) 
    std::cout << x << std::endl;

  for (auto p : EventStructure::phi<char, double>(ss)) 
    std::cout << p.first << ": " << p.second << std::endl;

  return 0;
}
