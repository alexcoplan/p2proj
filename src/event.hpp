#include <vector>

class SequenceEvent {
public:
  constexpr static int cardinality = 0;
  virtual unsigned int encode() = 0;
};

// dummy events in set {G,A,B,D}
class DummyEvent : SequenceEvent {
private:
  unsigned int coded;
  char raw_char;

public:
  static const std::vector<char> shared_encoding();
  virtual unsigned int encode();
  char raw_value();
  DummyEvent(unsigned int code);
  DummyEvent(char c);
};
