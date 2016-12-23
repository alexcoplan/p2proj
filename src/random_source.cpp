#include "random_source.hpp"

DefaultRandomSource::DefaultRandomSource() : distribution{0.0, 1.0} {
  try {
    std::random_device rdev{};
    engine.seed([&rdev]() { return rdev(); });
  }
  catch (std::exception e) {
    std::cerr 
      << "Warning: failed to generate true random seed, using time seed." 
      << std::endl
      << e.what()
      << std::endl;

    engine.seed_long([]() -> uint64_t {
      return std::chrono::high_resolution_clock::now().time_since_epoch() /
             std::chrono::nanoseconds(1);
    } );
  }
}
