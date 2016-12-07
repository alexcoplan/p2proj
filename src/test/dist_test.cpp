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
