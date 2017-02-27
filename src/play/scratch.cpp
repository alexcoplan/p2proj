#include <string>
#include <array>
#include <vector>
#include <iostream>
#include <chrono>

#include "random_source.hpp"
#include "sequence_model.hpp"
#include "chorale.hpp"

int main(void) {
  SequenceModel<DummyEvent> seq_orig(3);
  std::vector<DummyEvent> es;
  for (auto e : EventEnumerator<DummyEvent>())
    es.push_back(e);
  
  seq_orig.learn_sequence({es[0], es[1], es[2], es[0]});

  SequenceModel<DummyEvent> cloned(seq_orig);
  cloned.clear_model();

  auto nxt = seq_orig.gen_successor_dist({es[0]});
  std::cerr << nxt.debug_summary() << std::endl;


  return 0;
}
