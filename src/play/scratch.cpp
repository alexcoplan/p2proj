#include "chorale.hpp"

int main(void) {
  auto eg = {60,62,64,65,67,65,64,62,60};
  std::vector<ChoralePitch> pitches;
  std::transform(eg.begin(), eg.end(), std::back_inserter(pitches),
      [](unsigned int p) { return ChoralePitch(MidiPitch(p)); });

  BasicViewpoint<ChoralePitch> pitch_vp(3);
  IntervalViewpoint interval_vp(3);

  pitch_vp.learn(pitches);
  interval_vp.learn(pitches);

  ChoraleMVS mvs_1(1.0, {&pitch_vp, &interval_vp}, {});
  ChoraleMVS mvs_2(2.0, {&pitch_vp, &interval_vp}, {});

  auto test = {60, 61, 62};
  std::vector<ChoralePitch> test_pitches;
  std::transform(test.begin(), test.end(), std::back_inserter(test_pitches),
      [](unsigned int p) { return ChoralePitch(MidiPitch(p)); });

  // model.write_latex("out/tex/debug.tex");
  std::cout << mvs_1.avg_sequence_entropy(test_pitches) << std::endl;
  std::cout << mvs_2.avg_sequence_entropy(test_pitches) << std::endl;

  mvs_1.entropy_bias = 2.0;
  std::cout << mvs_1.avg_sequence_entropy(test_pitches) << std::endl;
}
