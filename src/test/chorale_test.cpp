#include "catch.hpp"
#include "chorale.hpp"
#include "viewpoint.hpp"
#include <array>

struct ChoraleMocker {
  static const MidiPitch default_pitch;
  static const QuantizedDuration default_dur; 
  static const QuantizedDuration default_rest_dur;
  static const KeySig default_key;

  static ChoraleEvent mock(const ChoralePitch &p) {
    return ChoraleEvent(
      default_key, 
      MidiPitch(p.raw_value()), 
      default_dur, 
      default_rest_dur
    );
  }

  static ChoraleEvent mock(const ChoraleDuration &d) {
    return ChoraleEvent(
      default_key, 
      default_pitch, 
      d,
      default_rest_dur
    );
  }

  static ChoraleEvent mock(const ChoraleRest &r) {
    return ChoraleEvent(
      default_key,
      default_pitch, 
      default_dur, 
      r
    );
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

const KeySig
ChoraleMocker::default_key = KeySig(0);

const QuantizedDuration
ChoraleMocker::default_rest_dur = QuantizedDuration(0);


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

    MidiInterval ival(p2 - p1);
    REQUIRE((p1 + ival) == p2);
  }
}

TEST_CASE("Check predictions/entropy calculations in MVS") {
  const unsigned int lt_hist = 3;
  const unsigned int st_hist = 2;

  SequenceModel<ChoralePitch> model(lt_hist);

  double intra_bias = 1.0;
  double inter_bias = 2.0;
  auto lt_config = MVSConfig::long_term_only(intra_bias);
  MVSConfig full_config;
  full_config.enable_short_term = true;
  full_config.intra_layer_bias = intra_bias;
  full_config.inter_layer_bias = inter_bias;
  full_config.lt_history = lt_hist;
  full_config.st_history = st_hist;
  full_config.mvs_name = "test MVS (full)";

  ChoraleMVS lt_mvs(lt_config);
  ChoraleMVS full_mvs(full_config);

  ChoraleMVS::BasicVP<ChoralePitch> long_term_vp(lt_hist);
  ChoraleMVS::BasicVP<ChoralePitch> short_term_vp(st_hist);

  // C D C E C F C G C A C B C C^
  auto eg = {60,62,60,64,60,65,60,67,60,69,60,71,60,72};
  std::vector<ChoralePitch> pitches;
  std::transform(eg.begin(), eg.end(), std::back_inserter(pitches),
      [](unsigned int p) { return ChoralePitch(MidiPitch(p)); });

  // train both model and trivial viewpoint on same data
  model.learn_sequence(pitches);
  long_term_vp.learn(ChoraleMocker::mock_sequence(pitches));

  lt_mvs.add_viewpoint(&long_term_vp);
  full_mvs.add_viewpoint(&long_term_vp);

  auto test_1 = {60,61,62,63,64,65,66,67,68,69,70};
  auto test_2 = {60};
  auto test_3 = {62,81,62,60};

  for (const auto &vs : {eg, test_1, test_2, test_3}) {
    // pitch_vp + short-term calculations
    short_term_vp.reset();
    std::vector<ChoralePitch> test_pitches;
    std::transform(vs.begin(), vs.end(), std::back_inserter(test_pitches),
        [](unsigned int p) { return ChoralePitch(MidiPitch(p)); });

    std::vector<ChoraleEvent> st_buff;
    double total_entropy = 0.0;
    auto comb_strategy = LogGeoEntropyCombination<ChoralePitch>(inter_bias);
    auto prediction = EventDistribution<ChoralePitch>(comb_strategy, 
        {short_term_vp.predict({}), long_term_vp.predict({})});
    for (const auto &pitch : test_pitches) {
      total_entropy -= std::log2(prediction.probability_for(pitch));
      st_buff.push_back(ChoraleMocker::mock(pitch));
      short_term_vp.learn_from_tail(st_buff);
      prediction = EventDistribution<ChoralePitch>(comb_strategy,
        {short_term_vp.predict(st_buff), long_term_vp.predict(st_buff)});
    }

    auto expected_full_entropy = total_entropy / test_pitches.size();
    auto actual_full_entropy = full_mvs.avg_sequence_entropy<ChoralePitch>(
        ChoraleMocker::mock_sequence(test_pitches));
    
    REQUIRE( expected_full_entropy == actual_full_entropy );

    auto mvs_entropy = lt_mvs.avg_sequence_entropy<ChoralePitch>(
        ChoraleMocker::mock_sequence(test_pitches));
    auto model_entropy = model.avg_sequence_entropy(test_pitches);
    REQUIRE( mvs_entropy == model_entropy );
  }
}

TEST_CASE("Check ChoraleEvent template magic") {
  std::vector<ChoraleEvent> test_events {
    ChoraleEvent(
      ChoraleMocker::default_key, MidiPitch(60), 
      QuantizedDuration(4), QuantizedDuration(0)
    ),
    ChoraleEvent(
      ChoraleMocker::default_key,
      MidiPitch(62), QuantizedDuration(6), ChoraleRest(2)
    ),
    ChoraleEvent(
      ChoraleMocker::default_key,
      MidiPitch(64), QuantizedDuration(4), ChoraleRest(0)
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

  std::vector<ChoraleRest> expected_rests{ 
    ChoraleRest(0),
    ChoraleRest(2), 
    ChoraleRest(0) 
  };
  
  REQUIRE( pitches == expected_pitches );
  REQUIRE( durations == expected_durs );
  REQUIRE( rests == expected_rests );

  std::vector<EventPair<ChoralePitch, ChoraleDuration>> expected_pairs {
    {expected_pitches[0], expected_durs[0]},
    {expected_pitches[1], expected_durs[1]},
    {expected_pitches[2], expected_durs[2]}
  };

  auto pairs = ChoraleEvent::lift<ChoralePitch, ChoraleDuration>(test_events);
  REQUIRE( pairs == expected_pairs );
}

TEST_CASE("Check GeneralViewpoint works in place of BasicViewpoint") {
  const unsigned int hist = 3;

  BasicViewpoint<ChoraleEvent, ChoralePitch> basic_vp(hist);
  GeneralViewpoint<ChoraleEvent, ChoralePitch, ChoralePitch> gen_vp(hist);

  std::vector<MidiPitch> pitch_values {
    60, 61, 60, 62, 63, 62, 61, 60, 70, 65, 67, 69, 60
  };

  std::vector<ChoralePitch> pitches;
  std::transform(pitch_values.begin(), pitch_values.end(), 
      std::back_inserter(pitches),
      [](const MidiPitch &mp) { return mp; });

  auto mocked = ChoraleMocker::mock_sequence(pitches);

  basic_vp.learn(mocked);
  gen_vp.learn(mocked);

  std::vector<ChoralePitch> eg_1{ MidiPitch(60), MidiPitch(61), MidiPitch(62) };
  std::vector<ChoralePitch> eg_2{ MidiPitch(70), MidiPitch(69), MidiPitch(68) };
  std::vector<ChoralePitch> eg_3{ MidiPitch(65), MidiPitch(67), MidiPitch(62) };

  for (auto eg : {eg_1, eg_2, eg_3}) { 
    auto eg_mocked = ChoraleMocker::mock_sequence(pitches);
    auto basic_dist = basic_vp.predict(eg_mocked);
    auto gen_dist = gen_vp.predict(eg_mocked);
    for (auto e : EventEnumerator<ChoralePitch>()) 
      REQUIRE( basic_dist.probability_for(e) == gen_dist.probability_for(e) );
  }
}

TEST_CASE("Check GeneralViewpoint works in place of seqint & intref VPs") {
  const unsigned int hist = 3;

  GeneralViewpoint<ChoraleEvent, ChoraleInterval, ChoralePitch> gen_ival(hist);
  IntervalViewpoint old_ival(hist);
  
  GeneralViewpoint<ChoraleEvent, ChoraleIntref, ChoralePitch> gen_intref(hist);
  IntrefViewpoint old_intref(hist);

  std::vector<MidiPitch> pitch_values {
    60, 61, 60, 62, 63, 62, 61, 60, 70, 65, 67, 69, 60
  };

  std::vector<ChoralePitch> pitches;
  std::transform(pitch_values.begin(), pitch_values.end(), 
      std::back_inserter(pitches),
      [](const MidiPitch &mp) { return mp; });

  auto mocked = ChoraleMocker::mock_sequence(pitches);

  gen_ival.learn(mocked);
  old_ival.learn(mocked);
  gen_intref.learn(mocked);
  old_intref.learn(mocked);

  std::vector<ChoralePitch> eg_1{ MidiPitch(60), MidiPitch(61), MidiPitch(62) };
  std::vector<ChoralePitch> eg_2{ MidiPitch(70), MidiPitch(69), MidiPitch(68) };
  std::vector<ChoralePitch> eg_3{ MidiPitch(65), MidiPitch(67), MidiPitch(62) };
  std::vector<ChoralePitch> eg_4{ MidiPitch(70) };
  std::vector<ChoralePitch> eg_5{ MidiPitch(65), MidiPitch(67) };
  std::vector<ChoralePitch> eg_6{ 
    MidiPitch(60), MidiPitch(62), MidiPitch(61), MidiPitch(65)
  };

  for (auto eg : {eg_1, eg_2, eg_3, eg_4, eg_5, eg_6}) { 
    auto eg_mocked = ChoraleMocker::mock_sequence(pitches);
    auto gen_ival_dist = gen_ival.predict(eg_mocked);
    auto old_ival_dist = old_ival.predict(eg_mocked);
    auto gen_intref_dist = gen_intref.predict(eg_mocked);
    auto old_intref_dist = old_intref.predict(eg_mocked);

    for (auto e : EventEnumerator<ChoralePitch>()) {
      REQUIRE( 
        gen_ival_dist.probability_for(e) == old_ival_dist.probability_for(e) 
      );
      REQUIRE(
        gen_intref_dist.probability_for(e) == old_intref_dist.probability_for(e)
      );
    }
  }
}




