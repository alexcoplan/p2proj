#include <string>
#include <array>
#include <vector>
#include <iostream>
#include <chrono>

#include "random_source.hpp"

int main(void) {
  /*
  DefaultRandomSource rs1;
  DefaultRandomSource rs2;

  std::cout << "rs1: " << rs1.sample() << std::endl;
  std::cout << "rs2: " << rs2.sample() << std::endl;
  */

  for (int i = 0; i < 10; i++) {
    uint64_t ns =
      std::chrono::high_resolution_clock::now().time_since_epoch() /
      std::chrono::nanoseconds(1);
    
    std::cout << ns << std::endl;
  }

  return 0;
}
