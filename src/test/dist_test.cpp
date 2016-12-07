#include <cassert>
#include <string>
#include <iostream>
#include <string>

#include "catch.hpp"
#include "event.hpp"

/* Distributions and events
 *
 * This test suite checks distributions and the event abstractions made around
 * them */

TEST_CASE("Dummy event encoding works correctly") {
  std::vector<char> chars{ 'G','A','B','D' };
  for (auto c : chars) {
    REQUIRE( DummyEvent(DummyEvent(c).encode()).raw_value() == c );
  }
}
