#include <iostream>

#include "sequence_model.hpp"

std::vector<DummyEvent> str_to_events(const std::string &str) {
  std::vector<DummyEvent> result;
  std::transform(str.begin(), str.end(), std::back_inserter(result), 
    [](const char c) { return DummyEvent(c); });
  return result;
}

int main(void)
{
  SequenceModel<DummyEvent> seq_model(3);
  seq_model.learn_sequence(str_to_events("GGDBAGGABA"));

  std::vector<DummyEvent> a_ctx{ DummyEvent('A') };
  auto distrib = seq_model.gen_successor_dist(a_ctx);
  std::vector<char> letters{'G','A','B','D'};
  for (char c : letters) 
    std::cout << "P(" << c << "|A) = " << 
      distrib.probability_for(DummyEvent(c)) << std::endl;
}
