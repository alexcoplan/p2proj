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

std::string decode_to_str(const unsigned int x) {
  return std::string(1, decode(x));
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
    model.learn_sequence(encode_string("GGDBAGGABA"));

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

TEST_CASE("Context model calculates correct probabilities using PPM A", 
    "[ctxmodel][ppm-a]") {
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

  SECTION("Simple case: h = 1, calculate probability for unseen event") {
    ContextModel<4> model(1);
    model.learn_sequence( encode_string("GGGGABB") );
    
    // first check that the counts are as expected
    // although this is not what we actually want to test here
    REQUIRE( model.count_of(std::vector<unsigned int>()) == 7 );
    REQUIRE( model.count_of(encode_string("G")) == 4 );
    REQUIRE( model.count_of(encode_string("A")) == 1 );
    REQUIRE( model.count_of(encode_string("B")) == 2 );
    REQUIRE( model.count_of(encode_string("D")) == 0 );

    REQUIRE( model.probability_of(encode_string("G")) == 1.0/2.0 );
    REQUIRE( model.probability_of(encode_string("A")) == 1.0/8.0 );
    REQUIRE( model.probability_of(encode_string("B")) == 1.0/4.0 );
    
    // unseen, calculate using PPM:
    REQUIRE( model.probability_of(encode_string("D")) == 1.0/8.0 ); 
  }

  SECTION("Nontrivial case: h = 3, probabilities match hand calculations for\
 C&W example") {
    ContextModel<4> model(3);
    model.learn_sequence(encode_string("GGDBAGGABA"));

    // sample bigrams
    REQUIRE( model.probability_of(encode_string("GG")) == 2.0/5.0 );
    REQUIRE( model.probability_of(encode_string("GA")) == 1.0/5.0 );
    REQUIRE( model.probability_of(encode_string("GB")) == 1.0/5.0 ); // unseen
    REQUIRE( model.probability_of(encode_string("GD")) == 1.0/5.0 );

    REQUIRE( model.probability_of(encode_string("AG")) == 1.0/3.0 );
    REQUIRE( model.probability_of(encode_string("AA")) == 1.0/4.0 );
    REQUIRE( model.probability_of(encode_string("AB")) == 1.0/3.0 );
    REQUIRE( model.probability_of(encode_string("AD")) == 1.0/12.0 );

    // sample trigrams
    REQUIRE( model.probability_of(encode_string("GGA")) == 1.0/3.0 );
    REQUIRE( model.probability_of(encode_string("GGD")) == 1.0/3.0 );
    REQUIRE( model.probability_of(encode_string("GGB")) == 1.0/9.0 );
    REQUIRE( model.probability_of(encode_string("GGG")) == 2.0/9.0 );

    REQUIRE( model.probability_of(encode_string("GAB")) == 1.0/2.0 );
    REQUIRE( model.probability_of(encode_string("GAA")) == 3.0/16.0 );
    REQUIRE( model.probability_of(encode_string("GAG")) == 1.0/4.0 );
    REQUIRE( model.probability_of(encode_string("GAD")) == 1.0/16.0 );
  }
}
