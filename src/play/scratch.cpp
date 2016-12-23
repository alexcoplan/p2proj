#include <string>
#include <array>
#include <vector>
#include <iostream>
#include <chrono>

#include "random_source.hpp"
#include "sequence_model.hpp"

int main(void) {
  /*
  DefaultRandomSource rs1;
  DefaultRandomSource rs2;

  std::cout << "rs1: " << rs1.sample() << std::endl;
  std::cout << "rs2: " << rs2.sample() << std::endl;
  */

  /*
  for (int i = 0; i < 10; i++) {
    std::cout 
      << DefaultRandomSource::shared_source.sample()
      << std::endl;
  }
  */

  std::array<double, DummyEvent::cardinality>
    vals{{ 0.1, 0.3, 0.2, 0.4 }};
  EventDistribution<DummyEvent> dist{vals};

  ConstantSource zero_point_1001(0.1001);
  std::cout << 
    dist.sample_with_source(&zero_point_1001).encode() << std::endl;

  return 0;
}
