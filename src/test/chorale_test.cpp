#include "catch.hpp"
#include "chorale.hpp"
#include "viewpoint.hpp"
#include <array>

struct ChoraleMocker {
  static const MidiPitch default_pitch;
  static const QuantizedDuration default_dur; 

  static ChoraleEvent mock(const ChoralePitch &p) {
    return ChoraleEvent(MidiPitch(p.raw_value()), default_dur, nullptr);
  }

  static ChoraleEvent mock(const ChoraleDuration &d) {
    return 
      ChoraleEvent(default_pitch, QuantizedDuration(d.raw_value()), nullptr);
  }

  static ChoraleEvent mock(ChoraleRest::singleton_ptr_t ptr) {
    return ChoraleEvent(default_pitch, default_dur, ptr);
  }
  
  template<typename T>
  static std::vector<ChoraleEvent>
  mock_sequence(std::vector<T> vals) {
    std::vector<ChoraleEvent> result;
    std::transform(vals.begin(), vals.end(), std::back_inserter(result),
      [](const T &v) { return ChoraleMocker::mock(v); } );
    return result;
  }
};

const MidiPitch
ChoraleMocker::default_pitch = MidiPitch(60);

const QuantizedDuration 
ChoraleMocker::default_dur = QuantizedDuration(4);



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

  double entropy_bias = 2.0;
  ChoraleMVS mvs(entropy_bias);

  ChoraleMVS::BasicVP<ChoralePitch> pitch_vp(order);

  // C D C E C F C G C A C B C C^
  auto eg = {60,62,60,64,60,65,60,67,60,69,60,71,60,72};
  std::vector<ChoralePitch> pitches;
  std::transform(eg.begin(), eg.end(), std::back_inserter(pitches),
      [](unsigned int p) { return ChoralePitch(MidiPitch(p)); });

  // train both model and trivial viewpoint on same data
  model.learn_sequence(pitches);
  pitch_vp.learn(ChoraleMocker::mock_sequence(pitches));

  mvs.add_viewpoint(&pitch_vp);

  auto test_1 = {60,61,62,63,64,65,66,67,68,69,70};
  auto test_2 = {60};
  auto test_3 = {62,81,62,60};

  for (const auto &vs : {eg, test_1, test_2, test_3}) {
    std::vector<ChoralePitch> test_pitches;
    std::transform(vs.begin(), vs.end(), std::back_inserter(test_pitches),
        [](unsigned int p) { return ChoralePitch(MidiPitch(p)); });
    auto mvs_entropy = mvs.avg_sequence_entropy<ChoralePitch>(
        ChoraleMocker::mock_sequence(test_pitches));
    auto model_entropy = model.avg_sequence_entropy(test_pitches);
    REQUIRE( mvs_entropy == model_entropy );
  }
}

TEST_CASE("Check ChoraleEvent template magic") {
  std::vector<ChoraleEvent> test_events {
    ChoraleEvent(
        MidiPitch(60), QuantizedDuration(4), nullptr
    ),
    ChoraleEvent(
      MidiPitch(62), QuantizedDuration(6), ChoraleRest(2).shared_instance()
    ),
    ChoraleEvent(
      MidiPitch(64), QuantizedDuration(4), ChoraleRest(0).shared_instance()
    )
  };

  auto pitches = ChoraleEvent::lift<ChoralePitch>(test_events);
  auto durations = ChoraleEvent::lift<ChoraleDuration>(test_events);
  auto rests = ChoraleEvent::lift<ChoraleRest>(test_events);

  std::vector<unsigned int> raw_pitches = { 60,62,64 };
  std::vector<ChoralePitch> expected_pitches;
  std::transform(raw_pitches.begin(), raw_pitches.end(),
    std::back_inserter(expected_pitches),
    [](unsigned int x) { return ChoralePitch(MidiPitch(x)); });

  std::vector<unsigned int> raw_durs = { 4,6,4 };
  std::vector<ChoraleDuration> expected_durs;
  std::transform(raw_durs.begin(), raw_durs.end(), 
    std::back_inserter(expected_durs),
    [](unsigned int x) { return ChoraleDuration(QuantizedDuration(x)); });

  std::vector<ChoraleRest> expected_rests{ ChoraleRest(2), ChoraleRest(0) };
  
  REQUIRE( pitches == expected_pitches );
  REQUIRE( durations == expected_durs );
  REQUIRE( rests == expected_rests );

  std::vector<std::pair<ChoralePitch, ChoraleDuration>> expected_pairs {
    std::make_pair(expected_pitches[0], expected_durs[0]),
    std::make_pair(expected_pitches[1], expected_durs[1]),
    std::make_pair(expected_pitches[2], expected_durs[2])
  };

  auto pairs = ChoraleEvent::lift<ChoralePitch, ChoraleDuration>(test_events);
  REQUIRE( pairs == expected_pairs );
}


