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
  
  auto test_seq = { es[0], es[1], es[2], es[0] };
  seq_orig.learn_sequence(test_seq);

  SequenceModel<DummyEvent> cloned(seq_orig);
  cloned.clear_model();

  std::vector<DummyEvent> buff;
  for (auto e : test_seq) {
    buff.push_back(e);
    cloned.update_from_tail(buff);
  }

  auto nxt = seq_orig.gen_successor_dist({es[0]});
  std::cerr << nxt.debug_summary() << std::endl;


  return 0;
}
