#include "catch.hpp"

#include "xoroshiro.hpp"

TEST_CASE("xoroshiro seeding gives same results via array and lambda") {
  xoroshiro128plus_engine eng_1;
  xoroshiro128plus_engine eng_2;

  std::array<uint32_t, 4> vals {{ 314, 42, 2718, 99 }};
  unsigned int i = 0;

  eng_1.seed([&i,&vals]() { return vals[i++]; });
  eng_2.seed(vals);

  for (int i = 0; i < 10; i++)
    REQUIRE( eng_1() == eng_2() );
}
