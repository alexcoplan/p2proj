#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <array>

#include "catch.hpp"
#include "event.hpp"
#include "sequence_model.hpp"
#include "json.hpp"

using json = nlohmann::json; // to load pre-gen'd test cases

/******************************************************************************
 * Distributions and events
 *
 * This test suite checks distributions and the event abstractions made around
 * them. 
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

TEST_CASE("Pair encoding works correctly", "[event]") {
  std::vector<char> chars{ 'G', 'A', 'B', 'D' };
  for (auto x : chars) {
    for (auto y : chars) {
      DummyEvent ex(x);
      DummyEvent ey(y);
      EventPair<DummyEvent, DummyEvent> pair(ex,ey);
      REQUIRE( pair.left().raw_value() == x );
      REQUIRE( pair.right().raw_value() == y );
    }
  }
}

TEST_CASE("zip_tail for event pairs works correctly", "[event]") {
  std::vector<unsigned int> left{0,1,2};
  std::vector<unsigned int> right{3,2};

  auto lambda = [](unsigned e) { return DummyEvent(e); };

  std::vector<DummyEvent> left_es;
  std::transform(left.begin(), left.end(), std::back_inserter(left_es), lambda);
  std::vector<DummyEvent> right_es;
  std::transform(right.begin(), 
                 right.end(), std::back_inserter(right_es), lambda);

  using T_pair = EventPair<DummyEvent, DummyEvent>;

  auto zipped = T_pair::zip_tail(left_es, right_es);
  std::vector<T_pair> expected {
    T_pair(left[1], right[0]),
    T_pair(left[2], right[1])
  };

  REQUIRE( expected == zipped );

  auto zipped_rl = T_pair::zip_tail(right_es, left_es);
  std::vector<T_pair> expected_rl {
   T_pair(right[0], left[1]),
   T_pair(right[1], left[2])
  };

  REQUIRE( expected_rl == zipped_rl );
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

TEST_CASE("Check entropy calculations", "[seqmodel]") {
  std::array<double, 4> values{{0.5, 0.25, 0.125, 0.125}};
  EventDistribution<DummyEvent> dist(values);

  std::array<double, 4> flat_vs{{0.25, 0.25, 0.25, 0.25}};
  EventDistribution<DummyEvent> flat(flat_vs);

  std::array<double, 4> delta_vs{{1.0, 0.0, 0.0, 0.0}};
  EventDistribution<DummyEvent> delta(delta_vs);

  SECTION("Shannon entropy") {
    REQUIRE( dist.entropy() == 1.75 );
    REQUIRE( flat.entropy() == 2.0 );
    REQUIRE( delta.entropy() == 0.0 );
  }

  SECTION("Normalised entropy") {
    REQUIRE( dist.normalised_entropy() == 0.875 );
    REQUIRE( flat.normalised_entropy() == 1.0 );
    REQUIRE( delta.normalised_entropy() == 0.0 );
  }
}

TEST_CASE("Weighted entropy combination works as expected", "[seqmodel]") {
  using array_t = std::array<double, 4>;

  array_t values{{0.5, 0.25, 0.125, 0.125}};
  EventDistribution<DummyEvent> dist(values);

  array_t flat_vs{{0.25, 0.25, 0.25, 0.25}};
  EventDistribution<DummyEvent> flat(flat_vs);

  // b = 1
  ArithmeticEntropyCombination<DummyEvent> strategy_1(1.0);
  array_t expected_1{{23.0/60.0, 15.0/60.0, 11.0/60.0, 11.0/60.0}};

  // b = 2 (squared bias to entropy weights)
  ArithmeticEntropyCombination<DummyEvent> strategy_2(2.0);
  array_t expected_2{
    {177.0/452.0, 113.0/452.0, 81.0/452.0, 81.0/452.0}
  };

  SECTION("Check copy combination") {
    EventDistribution<DummyEvent> combined(strategy_1, {dist, flat});
    for (auto e : EventEnumerator<DummyEvent>()) 
      REQUIRE( combined.probability_for(e) == Approx(expected_1[e.encode()]) );

    EventDistribution<DummyEvent> combined_2(strategy_2, {dist, flat});
    for (auto e : EventEnumerator<DummyEvent>())
      REQUIRE(combined_2.probability_for(e) == Approx(expected_2[e.encode()]));
  }

  SECTION("Check in-place combination") {
    EventDistribution<DummyEvent> dist_copy(values);
    dist_copy.combine_in_place(strategy_1, flat);
    for (auto e : EventEnumerator<DummyEvent>()) 
      REQUIRE( dist_copy.probability_for(e) == Approx(expected_1[e.encode()]) );

    EventDistribution<DummyEvent> dist_copy2(values);
    dist_copy2.combine_in_place(strategy_2, flat);
    for (auto e : EventEnumerator<DummyEvent>())
      REQUIRE(dist_copy2.probability_for(e) == Approx(expected_2[e.encode()]));
  }
}

TEST_CASE("Weighted entropy combination matches numpy-generated examples", 
    "[seqmodel]") {
  json j;
  std::ifstream examples_file("test/combination_examples.json");
  examples_file >> j;

  const auto &examples = j["dist_comb_examples"];
  for (const auto &eg : examples) {
    double entropy_bias = eg["entropy_bias"];
    std::vector<EventDistribution<DummyEvent>> source_dists;
    for (const auto &values : eg["dists"]) {
      assert(values.size() == DummyEvent::cardinality);
      std::array<double, DummyEvent::cardinality> vals;
      unsigned int i = 0;
      for (auto &v : values)
        vals[i++] = v;
      source_dists.push_back(vals);
    }

    ArithmeticEntropyCombination<DummyEvent> arith_strategy(entropy_bias);
    EventDistribution<DummyEvent> arith_comb(arith_strategy, source_dists);
    std::vector<double> expected_arith = eg["arithmetic_comb"];
    for (auto e : EventEnumerator<DummyEvent>())
      REQUIRE( 
        arith_comb.probability_for(e) == Approx(expected_arith[e.encode()])
      );

    GeometricEntropyCombination<DummyEvent> geom_strategy(entropy_bias);
    LogGeoEntropyCombination<DummyEvent> log_geo_strat(entropy_bias);
    EventDistribution<DummyEvent> geom_comb(geom_strategy, source_dists);
    EventDistribution<DummyEvent> log_geo_comb(log_geo_strat, source_dists);
    std::vector<double> expected_geom = eg["geometric_comb"];
    for (auto e : EventEnumerator<DummyEvent>()) {
      REQUIRE(
        geom_comb.probability_for(e) == Approx(expected_geom[e.encode()])
      );
      REQUIRE(
        log_geo_comb.probability_for(e) == Approx(expected_geom[e.encode()])
      );
    }
        
  }
}



