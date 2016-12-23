#include "catch.hpp"
#include "xoroshiro.hpp"
#include "random_source.hpp"
#include "sequence_model.hpp"

TEST_CASE("xoroshiro seeding gives same results via array and lambda") {
  xoroshiro128plus_engine eng_1;
  xoroshiro128plus_engine eng_2;

  std::array<uint32_t, 4> vals {{ 314, 42, 2718, 99 }};
  unsigned int i = 0;

  eng_1.seed([&i,&vals]() { return vals[i++]; });
  eng_2.seed(vals);

  for (int i = 0; i < 10; i++)
    REQUIRE( eng_1() == eng_2() );
}

TEST_CASE("sanity check constant random source") {
  REQUIRE( ConstantSource{0.0}.sample() == 0.0 );
  REQUIRE( ConstantSource{0.5}.sample() == 0.5 );
  REQUIRE( ConstantSource{1.0}.sample() == 1.0 );
}

// quick mock event with cardinality five
class BrubeckEvent : public SequenceEvent {
private:
  const unsigned int code;
public:
  constexpr static int cardinality = 5;
  unsigned int encode() const { return code; }
  std::string string_render() const { return std::to_string(code); }
  BrubeckEvent(unsigned int c) : code(c) {}
  bool operator==(const BrubeckEvent &other) {
    return encode() == other.encode();
  }
};

TEST_CASE("Distribution sampling works correctly") {
  SECTION("Event space with even cardinality") {
    std::array<double, DummyEvent::cardinality>
      dist_vals{{0.1, 0.3, 0.2, 0.4}};
    EventDistribution<DummyEvent> dist{dist_vals};

    ConstantSource zero(0.0);
    ConstantSource zero_point_05(0.05);
    ConstantSource zero_point_1(0.1);
    ConstantSource zero_point_1001(0.1001);
    ConstantSource zero_point_2(0.2);
    ConstantSource zero_point_3(0.3);
    ConstantSource zero_point_399(0.399);
    ConstantSource zero_point_4(0.4);
    ConstantSource zero_point_4001(0.4001);
    ConstantSource zero_point_5(0.5);
    ConstantSource zero_point_5999(0.5999);
    ConstantSource zero_point_6(0.6);
    ConstantSource zero_point_6001(0.6001);
    ConstantSource zero_point_85(0.85);
    ConstantSource zero_point_999(0.999);
    ConstantSource one(1.0);
    
    // cfd is [0.1, 0.4, 0.6, 1.0]

    auto nth_event = [](unsigned int c) { return DummyEvent(c); };

    REQUIRE( dist.sample_with_source(&zero)            == nth_event(0) );
    REQUIRE( dist.sample_with_source(&zero_point_05)   == nth_event(0) );
    REQUIRE( dist.sample_with_source(&zero_point_1)    == nth_event(0) );
    REQUIRE( dist.sample_with_source(&zero_point_1001) == nth_event(1) );
    REQUIRE( dist.sample_with_source(&zero_point_2)    == nth_event(1) );
    REQUIRE( dist.sample_with_source(&zero_point_3)    == nth_event(1) );
    REQUIRE( dist.sample_with_source(&zero_point_399)  == nth_event(1) );
    REQUIRE( dist.sample_with_source(&zero_point_4)    == nth_event(1) );
    REQUIRE( dist.sample_with_source(&zero_point_4001) == nth_event(2) );
    REQUIRE( dist.sample_with_source(&zero_point_5)    == nth_event(2) );
    REQUIRE( dist.sample_with_source(&zero_point_5999) == nth_event(2) );
    REQUIRE( dist.sample_with_source(&zero_point_6)    == nth_event(2) );
    REQUIRE( dist.sample_with_source(&zero_point_6001) == nth_event(3) );
    REQUIRE( dist.sample_with_source(&zero_point_85)   == nth_event(3) );
    REQUIRE( dist.sample_with_source(&zero_point_999)  == nth_event(3) );
    REQUIRE( dist.sample_with_source(&one)             == nth_event(3) );
  }
  
  SECTION("Event space with odd cardinaltiy") {
    std::array<double, BrubeckEvent::cardinality>
      dist_vals{{1.0/6.0, 1.0/3.0, 0.1, 0.3, 0.1}};
    EventDistribution<BrubeckEvent> dist{dist_vals};

    // cfd: [1/6, 0.5, 0.6, 0.9, 1.0]

    ConstantSource zero(0.0);
    ConstantSource one_sixth(1.0/6.0);
    ConstantSource above_1_6th(1.0/6.0 + 0.001);
    ConstantSource zero_point_55(0.55);
    ConstantSource zero_point_6(0.6);
    ConstantSource zero_point_85(0.85);
    ConstantSource one(1.0);
    
    REQUIRE( dist.sample_with_source(&zero) == BrubeckEvent(0) );
    REQUIRE( dist.sample_with_source(&one_sixth) == BrubeckEvent(0) );
    REQUIRE( dist.sample_with_source(&above_1_6th) == BrubeckEvent(1) );
    REQUIRE( dist.sample_with_source(&zero_point_55) == BrubeckEvent(2) );
    REQUIRE( dist.sample_with_source(&zero_point_6) == BrubeckEvent(2) );
    REQUIRE( dist.sample_with_source(&zero_point_85) == BrubeckEvent(3) );
    REQUIRE( dist.sample_with_source(&one) == BrubeckEvent(4) );
  }
}


