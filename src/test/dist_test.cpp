#include <cassert>
#include <string>
#include <iostream>
#include <string>

#include "catch.hpp"
#include "event.hpp"
#include "sequence_model.hpp"


/******************************************************************************
 * Distributions and events
 *
 * This test suite checks distributions and the event abstractions made around
 * them 
 ******************************************************************************/

// little helper so we can just write strings which get split up into vectors of
// DummyEvents purely for testing purposes
std::vector<DummyEvent> str_to_events(const std::string &str) {
  std::vector<DummyEvent> result;
  std::transform(str.begin(), str.end(), std::back_inserter(result), 
    [](const char c) { return DummyEvent(c); });
  return result;
}

TEST_CASE("Dummy event encoding works correctly", "[event]") {
  std::vector<char> chars{ 'G','A','B','D' };
  for (auto c : chars) {
    REQUIRE( DummyEvent(DummyEvent(c).encode()).raw_value() == c );
  }
}

TEST_CASE("SequenceModel correctly abstracts around ContextModel", 
    "[seqmodel]") { 
  SequenceModel<DummyEvent> seq_model(3);

  // we're going to check if the abstract sequence model exactly replicates the
  // output when fed the same data as the underlying context model in the
  // context model unit tests.
  //
  // TODO: abstract out this test data into JSON or YAML or something and load
  // the same data in both places... maybe?
  //
  // currently the asseritons below are just copy-pasted from the context model
  // test suite (!)
  seq_model.learn_sequence(str_to_events("GGDBAGGABA"));

  // sample bigrams
  REQUIRE( seq_model.probability_of(str_to_events("GG")) == 2.0/5.0 );
  REQUIRE( seq_model.probability_of(str_to_events("GA")) == 1.0/5.0 );
  REQUIRE( seq_model.probability_of(str_to_events("GB")) == 1.0/5.0 ); // unseen
  REQUIRE( seq_model.probability_of(str_to_events("GD")) == 1.0/5.0 );

  REQUIRE( seq_model.probability_of(str_to_events("AG")) == 1.0/3.0 );
  REQUIRE( seq_model.probability_of(str_to_events("AA")) == 1.0/4.0 );
  REQUIRE( seq_model.probability_of(str_to_events("AB")) == 1.0/3.0 );
  REQUIRE( seq_model.probability_of(str_to_events("AD")) == 1.0/12.0 );

  // sample trigrams
  REQUIRE( seq_model.probability_of(str_to_events("GGA")) == 1.0/3.0 );
  REQUIRE( seq_model.probability_of(str_to_events("GGD")) == 1.0/3.0 );
  REQUIRE( seq_model.probability_of(str_to_events("GGB")) == 1.0/9.0 );
  REQUIRE( seq_model.probability_of(str_to_events("GGG")) == 2.0/9.0 );

  REQUIRE( seq_model.probability_of(str_to_events("GAB")) == 1.0/2.0 );
  REQUIRE( seq_model.probability_of(str_to_events("GAA")) == 3.0/16.0 );
  REQUIRE( seq_model.probability_of(str_to_events("GAG")) == 1.0/4.0 );
  REQUIRE( seq_model.probability_of(str_to_events("GAD")) == 1.0/16.0 );
}

TEST_CASE("SequenceModel distribtuion construction works correctly", 
    "[seqmodel][distribution]") {
  SequenceModel<DummyEvent> seq_model(3);
  seq_model.learn_sequence(str_to_events("GGDBAGGABA"));

  // in this case we test on the same data (C&W) as the previous test case: we
  // are checking that the distribtuions are well-formed and contain the same
  // probabilities

  SECTION("Check P(e' | ()) i.e. given no context") {
    std::vector<DummyEvent> empty_context;
    auto distrib = seq_model.gen_successor_dist(empty_context);
    REQUIRE( distrib.probability_for(DummyEvent('G')) == 4.0/10.0 );
    REQUIRE( distrib.probability_for(DummyEvent('A')) == 3.0/10.0 );
    REQUIRE( distrib.probability_for(DummyEvent('B')) == 2.0/10.0 );
    REQUIRE( distrib.probability_for(DummyEvent('D')) == 1.0/10.0 );
  }

  SECTION("Check P(e' | G) gives the expected distribution") {
    auto distrib = seq_model.gen_successor_dist(str_to_events("G"));
    REQUIRE( distrib.probability_for(DummyEvent('G')) == 2.0/5.0 );
    REQUIRE( distrib.probability_for(DummyEvent('A')) == 1.0/5.0 );
    REQUIRE( distrib.probability_for(DummyEvent('B')) == 1.0/5.0 );
    REQUIRE( distrib.probability_for(DummyEvent('D')) == 1.0/5.0 );
  }

  SECTION("Check P(e' | A) gives the expected distribution") {
    auto distrib = seq_model.gen_successor_dist(str_to_events("A"));
    REQUIRE( distrib.probability_for(DummyEvent('G')) == 1.0/3.0 );
    REQUIRE( distrib.probability_for(DummyEvent('A')) == 1.0/4.0 );
    REQUIRE( distrib.probability_for(DummyEvent('B')) == 1.0/3.0 );
    REQUIRE( distrib.probability_for(DummyEvent('D')) == 1.0/12.0 );
  }

  SECTION("Check P(e' | B) gives the expected distribution") {
    auto distrib = seq_model.gen_successor_dist(str_to_events("B"));
    REQUIRE( distrib.probability_for(DummyEvent('G')) == 4.0/21.0 );
    REQUIRE( distrib.probability_for(DummyEvent('A')) == 2.0/3.0 );
    REQUIRE( distrib.probability_for(DummyEvent('B')) == 2.0/21.0 );
    REQUIRE( distrib.probability_for(DummyEvent('D')) == 1.0/21.0 );
  }

  SECTION("Check P(e' | D) gives the expected distribtuion") {
    auto distrib = seq_model.gen_successor_dist(str_to_events("D"));
    REQUIRE( distrib.probability_for(DummyEvent('G')) == 1.0/4.0 );
    REQUIRE( distrib.probability_for(DummyEvent('A')) == 3.0/16.0 );
    REQUIRE( distrib.probability_for(DummyEvent('B')) == 1.0/2.0 );
    REQUIRE( distrib.probability_for(DummyEvent('D')) == 1.0/16.0 );
  }

  // we have exhaustively checked first-order distributions, probably only need
  // to check one of the second-order distribtuions (given that we have also
  // checked these numbers in ctx_test)
  SECTION("Check P(e' | GG) gives the expected distribution") {
    auto distrib = seq_model.gen_successor_dist(str_to_events("GG"));
    REQUIRE( distrib.probability_for(DummyEvent('G')) == 2.0/9.0 );
    REQUIRE( distrib.probability_for(DummyEvent('A')) == 1.0/3.0 );
    REQUIRE( distrib.probability_for(DummyEvent('B')) == 1.0/9.0 );
    REQUIRE( distrib.probability_for(DummyEvent('D')) == 1.0/3.0 );
  }
}
