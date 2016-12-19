#include "chorale.hpp"

int main(void) {
  auto eg = {60,62,60,64,60,65,60,67,60,69,60,71,60,72};
  std::vector<ChoralePitch> pitches;
  std::transform(eg.begin(), eg.end(), std::back_inserter(pitches),
      [](unsigned int p) { return ChoralePitch(MidiPitch(p)); });

  SequenceModel<ChoralePitch> model(3);
  model.learn_sequence(pitches);

  auto test = {60, 61, 62};
  std::vector<ChoralePitch> test_pitches;
  std::transform(test.begin(), test.end(), std::back_inserter(test_pitches),
      [](unsigned int p) { return ChoralePitch(MidiPitch(p)); });

  // model.write_latex("out/tex/debug.tex");
  std::cout << model.avg_sequence_entropy(test_pitches);
}
