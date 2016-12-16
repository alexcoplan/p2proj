#ifndef AJC_HGUARD_EVENT
#define AJC_HGUARD_EVENT

#include <vector>
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
};

#endif // header guard
