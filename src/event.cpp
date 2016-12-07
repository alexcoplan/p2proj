#include <cassert>
#include <vector>
#include "event.hpp"

DummyEvent::DummyEvent(unsigned int code) {
  // decode in constructor 
  coded = code;
  raw_char = shared_encoding()[code];
}

DummyEvent::DummyEvent(char c) : raw_char(c) {
  auto encoding = shared_encoding();
  for (unsigned int i = 0; i < encoding.size(); i++) {
    if (encoding[i] == c) {
      coded = i;
      return;
    }
  }

  assert(! "Bad character for DummyEvent");
}

const std::vector<char> DummyEvent::shared_encoding() {
  return std::vector<char>{'G','A','B','D'};
}

unsigned int DummyEvent::encode() const {
  return coded;
}

char DummyEvent::raw_value() { return raw_char; }
