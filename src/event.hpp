#ifndef AJC_HGUARD_EVENT
#define AJC_HGUARD_EVENT

#include <vector>
#include <array>
#include <string>
#include <cassert>

class SequenceEvent {
public:
  constexpr static unsigned int cardinality = 0;
  virtual unsigned int encode() const = 0;
  virtual std::string string_render() const = 0;
};

// dummy events in set {G,A,B,D}
class DummyEvent : public SequenceEvent {
private:
  const unsigned int coded;
  const char raw_char;

public:
  constexpr static int cardinality = 4;
  static const std::array<char, cardinality> shared_encoding;
  static unsigned int code_for(char c);
  unsigned int encode() const override;
  std::string string_render() const override;
  char raw_value();
  DummyEvent(unsigned int code);
  DummyEvent(char c);

  bool operator==(const DummyEvent &other) const {
    return encode() == other.encode();
  }
};

template<class T1, class T2>
class EventPair : public SequenceEvent {
private:
  const unsigned int coded;

public:
  constexpr static unsigned int cardinality =
    T1::cardinality * T2::cardinality;

  constexpr T1 left() const { return T1(coded % T1::cardinality); }
  constexpr T2 right() const { return T2(coded / T1::cardinality); }

  static std::vector<EventPair>
  zip(const std::vector<T1> &left, const std::vector<T2> &right) {
    assert(left.size() == right.size());
    std::vector<EventPair> result{};
    for (unsigned int i = 0; i < left.size(); i++)
      result.push_back({ left[i], right[i] });

    return result;
  }

  unsigned int encode() const override {
    return coded;
  }

  std::string string_render() const override {
    return "(" + left().string_render() + "," + right().string_render() + ")";
  }

  EventPair(unsigned int c) : coded(c) { assert(c < cardinality); }

  EventPair(const T1 &x, const T2 &y) :
    coded(x.encode() + T1::cardinality * y.encode()) {}
};

template<class P, class Q>
bool operator==(const EventPair<P,Q> &l, const EventPair<P,Q> &r)
{ return l.encode() == r.encode(); }

#endif // header guard
