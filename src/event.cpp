#include <cassert>
#include <vector>
#include "event.hpp"
#include <array>
#include <cassert>
#include <string>

const std::array<char, DummyEvent::cardinality> 
DummyEvent::shared_encoding = {{'G','A','B','D'}};

unsigned int DummyEvent::code_for(char c) {
  for (unsigned int i = 0; i < cardinality; i++)
    if (shared_encoding[i] == c)
      return i;

  assert(! "Bad character for DummyEvent");
}

DummyEvent::DummyEvent(char c) : coded(code_for(c)), raw_char(c) {}
DummyEvent::DummyEvent(unsigned int code) : 
  coded(code), raw_char(shared_encoding[code]) {}

unsigned int DummyEvent::encode() const {
  return coded;
}

char DummyEvent::raw_value() { return raw_char; }

std::string DummyEvent::string_render() const {
  return std::to_string(shared_encoding[coded]);
}
