#include <string>
#include <array>
#include <vector>
#include <iostream>
#include <chrono>

#include "random_source.hpp"
#include "sequence_model.hpp"
#include "chorale.hpp"

template<typename T>
class DetectX
{
    struct Fallback { int X; }; // add member name "X"
    struct Derived : T, Fallback { };

    template<typename U, U> struct Check;

    typedef char ArrayOfOne[1];  // typedef for an array of size one.
    typedef char ArrayOfTwo[2];  // typedef for an array of size two.

    template<typename U> 
    static ArrayOfOne & func(Check<int Fallback::*, &U::X> *);
    
    template<typename U> 
    static ArrayOfTwo & func(...);

  public:
    typedef DetectX type;
    enum { value = sizeof(func<Derived>(0)) == 2 };
};

template<typename T>
struct JustData {
  T x;
};

struct A {
};

struct B { 
  using X = std::string;
};

int main(void) {
  std::cout 
    << std::boolalpha 
    << "Does B have X? " 
    << DetectX<B>::value 
    << std::endl;

  std::cout 
    << std::boolalpha 
    << "Does A have X? " 
    << DetectX<A>::value 
    << std::endl;


  return 0;
}
