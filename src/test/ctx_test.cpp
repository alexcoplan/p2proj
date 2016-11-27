#include <cassert>
#include <string>
#include <iostream>
#include <string>

// TODO: function to check context model satisfies trie sum property

// include unit test library
// set this file to have a main
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "context_model.hpp"

#define HISTORY 3
#define NUM_NOTES 4

unsigned int encode(const char c) {
  switch(c) {
    case 'G' :
      return 0;
    case 'A' :
      return 1;
    case 'B' : 
      return 2;
    case 'D' : 
      return 3;
    default :
      assert(! "Bad character");
      return 4;
  }
}

char decode(const unsigned int x) {
  assert(x < 4);
  char a[4] = { 'G', 'A', 'B', 'D' };
  return a[x];
}

std::vector<unsigned int> encode_string(const std::string &str) {
  const std::vector<char> chars(str.begin(), str.end());
  std::vector<unsigned int> encoded(chars.size());
  std::transform(chars.begin(), chars.end(), encoded.begin(),
      [](char c) { return encode(c); });
  return encoded;
}

TEST_CASE("Test data encoding works correctly", "[selftest]") {
  REQUIRE( encode('G') == 0 );
  REQUIRE( encode('A') == 1 );
  REQUIRE (encode('B') == 2 );
  REQUIRE (encode('D') == 3 );
}

TEST_CASE("Test data decoding works correctly", "[selftest]") {
  REQUIRE( decode(0) == 'G' );
  REQUIRE( decode(1) == 'A' );
  REQUIRE( decode(2) == 'B' );
  REQUIRE( decode(3) == 'D' );
}

TEST_CASE("Context model training works correctly", "[ctxmodel]") {

  SECTION("Replicate example given in Conklin & Witten's 1995 MVS Paper") {
    ContextModel<NUM_NOTES> model(HISTORY);

    model.learnSequence(encode_string("GGDBAGGABA"));
    REQUIRE( model.count_of(std::vector<unsigned int>()) == 10 );
    // TODO: complete unit tests here
  }

}
