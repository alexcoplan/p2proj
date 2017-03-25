#include "catch.hpp"
#include "chorale.hpp"
#include "viewpoint.hpp"
#include <array>

struct ChoraleMocker {
  static const MidiPitch default_pitch;
  static const QuantizedDuration default_dur; 
  static const QuantizedDuration default_rest_dur;
  static const QuantizedDuration default_timesig;
  static const KeySig default_key;

  static std::vector<ChoralePitch> 
  box_pitches(const std::vector<unsigned int> &pitches) {
    std::vector<ChoralePitch> result;
    for (auto p : pitches)
      result.push_back(MidiPitch {p});
    return result;
  }

  // can box either ChoraleDuration or ChoraleRest
  template<typename T>
  static std::vector<T>
  box_durations(const std::vector<unsigned int> &durs) {
    std::vector<T> result;
    for (auto d : durs)
      result.push_back(QuantizedDuration {d});
    return result;
  }

  static ChoraleEvent mock(const ChoralePitch &p, const ChoraleDuration &d) {
    return ChoraleEvent(
      default_key,
      default_timesig,
      p,
      d,
      default_rest_dur
    );
  }

  static ChoraleEvent mock(const ChoralePitch &p) {
    return ChoraleEvent(
      default_key, 
      default_timesig,
      p,
      default_dur, 
      default_rest_dur
    );
  }

  static ChoraleEvent mock(const ChoraleDuration &d) {
    return ChoraleEvent(
      default_key, 
      default_timesig,
      default_pitch, 
      d,
      default_rest_dur
    );
  }

  static ChoraleEvent mock(const ChoraleDuration &d, const ChoraleTimeSig &ts) {
    return ChoraleEvent(
      default_key, ts,
      default_pitch,
      d,
      default_rest_dur
    );
  }

  static ChoraleEvent 
  mock(const ChoraleDuration &d, const ChoraleRest &r, const ChoraleTimeSig &ts) {
    return ChoraleEvent(
      default_key, ts,
      default_pitch,
      d, r);
  }

  static ChoraleEvent mock(const ChoraleRest &r) {
    return ChoraleEvent(
      default_key,
      default_timesig,
      default_pitch, 
      default_dur, 
      r
    );
  }


  template<typename T>
  static std::vector<ChoraleEvent>
  mock_sequence(const std::vector<T> &vals, const ChoraleTimeSig &ts) {
    std::vector<ChoraleEvent> result;
    std::transform(vals.begin(), vals.end(), std::back_inserter(result),
      [ts](const T &v) { return ChoraleMocker::mock(v, ts); });
    return result;
  }

  template<typename P, typename Q>
  static std::vector<ChoraleEvent>
  mock_sequence(const std::vector<P> &left, 
                const std::vector<Q> &right, 
                const ChoraleTimeSig &ts) { 
    assert(left.size() == right.size());
    std::vector<ChoraleEvent> result;
    for (unsigned int i = 0; i < left.size(); i++)
      result.push_back(ChoraleMocker::mock(left[i], right[i], ts));
    return result;
  }
  
  template<typename T>
  static std::vector<ChoraleEvent>
  mock_sequence(const std::vector<T> &vals) {
    std::vector<ChoraleEvent> result;
    std::transform(vals.begin(), vals.end(), std::back_inserter(result),
      [](const T &v) { return ChoraleMocker::mock(v); } );
    return result;
  }

  template<typename P, typename Q>
  static std::vector<ChoraleEvent>
  mock_sequence(const std::vector<P> &left, const std::vector<Q> &right) {
    assert(left.size() == right.size());
    std::vector<ChoraleEvent> result;
    for (unsigned int i = 0; i < left.size(); i++)
      result.push_back(ChoraleMocker::mock(left[i], right[i]));
    return result;
  }
};

const KeySig
ChoraleMocker::default_key = KeySig(0);

const QuantizedDuration
ChoraleMocker::default_timesig = QuantizedDuration(16);

const MidiPitch
ChoraleMocker::default_pitch = MidiPitch(60);

const QuantizedDuration 
ChoraleMocker::default_dur = QuantizedDuration(4);

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
  std::vector<unsigned int> eg_pitches = 
    {60,62,60,64,60,65,60,67,60,69,60,71,60,72};
  
  auto pitches = ChoraleMocker::box_pitches(eg_pitches);

  // train both model and trivial viewpoint on same data
  model.learn_sequence(pitches);
  long_term_vp.learn(ChoraleMocker::mock_sequence(pitches));

  lt_mvs.add_viewpoint(&long_term_vp);
  full_mvs.add_viewpoint(&long_term_vp);

  std::vector<unsigned int> test_1 = {60,61,62,63,64,65,66,67,68,69,70};
  std::vector<unsigned int> test_2 = {60};
  std::vector<unsigned int> test_3 = {62,81,62,60};

  for (const auto &vs : {eg_pitches, test_1, test_2, test_3}) {
    // pitch_vp + short-term calculations
    short_term_vp.reset();
    auto test_pitches = ChoraleMocker::box_pitches(vs);

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
      ChoraleMocker::default_key, ChoraleMocker::default_timesig,
      MidiPitch(60), QuantizedDuration(4), QuantizedDuration(0)
    ),
    ChoraleEvent(
      ChoraleMocker::default_key, ChoraleMocker::default_timesig,
      MidiPitch(62), QuantizedDuration(6), ChoraleRest(2)
    ),
    ChoraleEvent(
      ChoraleMocker::default_key, ChoraleMocker::default_timesig,
      MidiPitch(64), QuantizedDuration(4), ChoraleRest(0)
    )
  };

  auto pitches = ChoraleEvent::lift<ChoralePitch>(test_events);
  auto durations = ChoraleEvent::lift<ChoraleDuration>(test_events);
  auto rests = ChoraleEvent::lift<ChoraleRest>(test_events);

  std::vector<unsigned int> raw_pitches { 60,62,64 };
  auto expected_pitches = ChoraleMocker::box_pitches(raw_pitches);

  std::vector<unsigned int> raw_durs { 4,6,4 };
  auto expected_durs = ChoraleMocker::box_durations<ChoraleDuration>(raw_durs);

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
  GeneralViewpoint<ChoraleEvent, ChoralePitch> gen_vp(hist);

  std::vector<unsigned int> pitch_values = {
    60, 61, 60, 62, 63, 62, 61, 60, 70, 65, 67, 69, 60
  };

  auto pitches = ChoraleMocker::box_pitches(pitch_values);
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

  GeneralViewpoint<ChoraleEvent, ChoraleInterval> gen_ival(hist);
  IntervalViewpoint old_ival(hist);
  
  GeneralViewpoint<ChoraleEvent, ChoraleIntref> gen_intref(hist);
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

TEST_CASE("Check GeneralLinkedVP works in place of BasicLinkedVP") {
  ChoraleMVS::BasicLinkedVP<ChoralePitch, ChoraleDuration> basic_vp(3);
  ChoraleMVS::GenLinkedVP<ChoralePitch, ChoraleDuration> gen_vp(3);

  std::vector<MidiPitch> pitches { 
    60, 62, 64, 60, 68, 67, 66, 65, 64, 63, 62, 61, 60, 70, 78
  };

  std::vector<QuantizedDuration> durations {
    4,  4,  8,  4,  2,  2,  4,  1,  1,  1,  1,  4,  4,  8,  16
  };

  REQUIRE( pitches.size() == durations.size() );

  std::vector<ChoralePitch> c_pitches;
  std::vector<ChoraleDuration> c_durations;
  for (auto p : pitches)
    c_pitches.push_back(p);

  for (auto d : durations)
    c_durations.push_back(d);

  auto events = ChoraleMocker::mock_sequence(pitches, durations);

  std::vector<std::pair<std::vector<unsigned int>, std::vector<unsigned int>>> 
    egs {
    { {}, {} }, 
    { {60, 62}, { 4, 4 } },
    { {62, 61}, { 1 ,4 } },
    { {65, 64, 63}, { 4, 8, 16 } },
    { {60, 62, 64, 60}, { 4, 4, 8, 4 } },
    { {72, 74, 72} , { 1, 1, 1 } }
  };

  for (auto eg : egs) {
    auto pitches = ChoraleMocker::box_pitches(eg.first);
    auto durs    = ChoraleMocker::box_durations<ChoraleDuration>(eg.second);
    auto events  = ChoraleMocker::mock_sequence(pitches, durs);

    auto dist_old = basic_vp.predict(events);
    auto dist_gen = gen_vp.predict(events);

    for (auto e : EventEnumerator<ChoraleDuration>())
      REQUIRE( dist_old.probability_for(e) == dist_gen.probability_for(e) );
  }
}

TEST_CASE("Check posinbar lifting is correct") {
  const ChoraleTimeSig three_four(QuantizedDuration(12));
  const ChoraleTimeSig  four_four(QuantizedDuration(16));

  std::vector<unsigned> duration_eg { 4, 8, 4, 2, 1, 1, 2, 2, 4, 4 };
  std::vector<ChoralePosinbar> dur_expect_44 { 0, 4, 12, 0, 2, 3, 4, 6, 8, 12 };
  std::vector<ChoralePosinbar> dur_expect_34 { 0, 4, 0, 4, 6, 7, 8, 10, 0, 4 };

  auto boxed_durs = ChoraleMocker::box_durations<ChoraleDuration>(duration_eg);
  auto mocked_34 = ChoraleMocker::mock_sequence(boxed_durs, three_four);
  auto mocked_44 = ChoraleMocker::mock_sequence(boxed_durs, four_four);

  auto lifted_34 = ChoraleEvent::lift<ChoralePosinbar>(mocked_34);
  auto lifted_44 = ChoraleEvent::lift<ChoralePosinbar>(mocked_44);

  REQUIRE( lifted_34 == dur_expect_34 );
  REQUIRE( lifted_44 == dur_expect_44 );

  std::pair<std::vector<unsigned>, std::vector<unsigned>> dur_rest_eg {
    { 4, 2, 2, 4, 4, 8, 8 },
    { 8, 0, 0, 4, 0, 0, 0 }
  };

  std::vector<ChoralePosinbar> dr_expected_44 { 8, 12, 14, 4, 8, 12, 4 };
  std::vector<ChoralePosinbar> dr_expected_34 { 8, 0, 2, 8, 0, 4, 0 };

  auto dr_durs  = ChoraleMocker::box_durations<ChoraleDuration>(dur_rest_eg.first);
  auto dr_rests = ChoraleMocker::box_durations<ChoraleRest>(dur_rest_eg.second);
 
  auto dr_mocked_44 = ChoraleMocker::mock_sequence(dr_durs, dr_rests, four_four);
  auto dr_mocked_34 = ChoraleMocker::mock_sequence(dr_durs, dr_rests, three_four);

  auto dr_lifted_44 = ChoraleEvent::lift<ChoralePosinbar>(dr_mocked_44);
  auto dr_lifted_34 = ChoraleEvent::lift<ChoralePosinbar>(dr_mocked_34);

  REQUIRE( dr_lifted_44 == dr_expected_44 );
  REQUIRE( dr_lifted_34 == dr_expected_34 );
}



