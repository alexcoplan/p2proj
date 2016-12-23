#include <iostream>
#include <random>
#include <array>

#include "random_source.hpp"
#include "xoroshiro.hpp"

int main(void) {
  /*
  DefaultRandomSource rs;

  for (int i = 0; i < 10; i++)
    std::cout << rs.sample() << std::endl;
  */

  xoroshiro128plus_engine eng;

  std::random_device rdev{};
  eng.seed([&rdev]() { return rdev(); });

  std::uniform_int_distribution<> dist{0,512};

  for (int i = 0; i < 100; i++)
    std::cout << dist(eng) << std::endl;
}


