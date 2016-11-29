#include <cassert>
#include <string>
#include <iostream>
#include <string>

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
    // data here is taken directly from the paper (Table 1)
    // note that there is an error in the table, but it is clearly a duplication
    // typo
    ContextModel<NUM_NOTES> model(HISTORY);
    model.learnSequence(encode_string("GGDBAGGABA"));

    // total count (zero-grams)
    REQUIRE( model.count_of(std::vector<unsigned int>()) == 10 );

    // unigrams
    REQUIRE( model.count_of(encode_string("A")) == 3 );
    REQUIRE( model.count_of(encode_string("G")) == 4 );
    REQUIRE( model.count_of(encode_string("D")) == 1 );
    REQUIRE( model.count_of(encode_string("B")) == 2 );

    // bigrams
    REQUIRE( model.count_of(encode_string("AB")) == 1 );
    REQUIRE( model.count_of(encode_string("AG")) == 1 );
    REQUIRE( model.count_of(encode_string("GA")) == 1 );
    REQUIRE( model.count_of(encode_string("GG")) == 2 );
    REQUIRE( model.count_of(encode_string("GD")) == 1 );
    REQUIRE( model.count_of(encode_string("DB")) == 1 );
    REQUIRE( model.count_of(encode_string("BA")) == 2 );
    
    // trigrams
    REQUIRE( model.count_of(encode_string("ABA")) == 1 );
    REQUIRE( model.count_of(encode_string("AGG")) == 1 );
    REQUIRE( model.count_of(encode_string("GAB")) == 1 );
    REQUIRE( model.count_of(encode_string("GGA")) == 1 );
    REQUIRE( model.count_of(encode_string("GDB")) == 1 );
    REQUIRE( model.count_of(encode_string("DBA")) == 1 );
    REQUIRE( model.count_of(encode_string("BAG")) == 1 );
    REQUIRE( model.count_of(encode_string("GGD")) == 1 );
  }
}

TEST_CASE("Context model correctly calculates probabilities using PPM", 
    "[ctxmodel]") {
  SECTION("Trivial cases: untrained model gives equal probabilities") {
    ContextModel<2> model(1);
    REQUIRE( model.count_of(std::vector<unsigned int>()) == 0 ); // check init'd
    REQUIRE( model.probability_of(encode_string("G")) == 0.5 );
    REQUIRE( model.probability_of(encode_string("A")) == 0.5 );

    ContextModel<4> model_b(2);
    REQUIRE( model_b.probability_of(encode_string("G")) == 0.25 );
    REQUIRE( model_b.probability_of(encode_string("A")) == 0.25 );
    REQUIRE( model_b.probability_of(encode_string("B")) == 0.25 );
    REQUIRE( model_b.probability_of(encode_string("D")) == 0.25 );
  }

  // TODO: write many many more unit tests here
}

