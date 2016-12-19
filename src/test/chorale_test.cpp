#include "catch.hpp"
#include "chorale.hpp"
#include "viewpoint.hpp"
#include <array>

TEST_CASE("Check Chorale event encodings", "[chorale][events]") {
  SECTION("Check interval encoding") {
    std::array<int, ChoraleInterval::cardinality> interval_domain =
    {{-12, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
       12}};

    for (unsigned int i = 0; i < interval_domain.size(); i++) {
      REQUIRE( ChoraleInterval(i).raw_value() == interval_domain[i] );
      MidiInterval midi_ival(interval_domain[i]);
      REQUIRE( ChoraleInterval(midi_ival).encode() == i );
    }
  }
}

TEST_CASE("Check Chorale event operaitons", "[chorale][events]") {
  SECTION("Check interval operations") {
    ChoralePitch p1(MidiPitch(60));
    ChoralePitch p2(MidiPitch(62));
    REQUIRE((p2 - p1).delta_pitch == 2);
    REQUIRE((p1 - p2).delta_pitch == -2);

    ChoraleInterval ival(p2 - p1);
    REQUIRE((p1 + ival) == p2);
  }
}

TEST_CASE("Check predictions/entropy calculations in ChoraleMVS") {
  const unsigned int order = 3;
  SequenceModel<ChoralePitch> model(order);
  BasicViewpoint<ChoralePitch> pitch_vp(order);

  // C D C E C F C G C A C B C C^
  auto eg = {60,62,60,64,60,65,60,67,60,69,60,71,60,72};
  std::vector<ChoralePitch> pitches;
  std::transform(eg.begin(), eg.end(), std::back_inserter(pitches),
      [](unsigned int p) { return ChoralePitch(MidiPitch(p)); });

  // train both model and trivial viewpoint on same data
  model.learn_sequence(pitches);
  pitch_vp.learn(pitches);

  double entropy_bias = 2.0; // arbitrary, this is a single viewpoint system
  ChoraleMVS mvs(entropy_bias, {&pitch_vp}, {});

  auto test_1 = {60,61,62,63,64,65,66,67,68,69,70};
  auto test_2 = {60};
  auto test_3 = {62,81,62,60};

  for (const auto &vs : {eg, test_1, test_2, test_3}) {
    std::vector<ChoralePitch> test_pitches;
    std::transform(vs.begin(), vs.end(), std::back_inserter(test_pitches),
        [](unsigned int p) { return ChoralePitch(MidiPitch(p)); });
    auto mvs_entropy = mvs.avg_sequence_entropy(test_pitches);
    auto model_entropy = model.avg_sequence_entropy(test_pitches);
    REQUIRE( mvs_entropy == model_entropy );
  }
}


