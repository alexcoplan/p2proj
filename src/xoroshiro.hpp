#ifndef AJC_HGUARD_XOROSHIRO128PLUS
#define AJC_HGUARD_XOROSHIRO128PLUS

#include <random>
#include <array>
#include <functional>

/* C++11 UniformRandomBitGenerator wrapper around xoroshiro128+.
 * Original C code from: http://xoroshiro.di.unimi.it/xoroshiro128plus.c */

struct xoroshiro128plus_engine {
private:
  uint64_t state[2];

  static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }

  void warm_up();

public:
  using result_type = uint64_t;
  constexpr static result_type min() { return 0; }
  constexpr static result_type max() { return -1; }

  result_type operator()();
  void seed(std::function<uint32_t(void)>);
  void seed_long(std::function<uint64_t(void)>);
  void seed(const std::array<uint32_t, 4> &);
};

#endif
