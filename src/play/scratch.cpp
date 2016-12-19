#include <iostream>
#include <fstream>
#include "json.hpp"
#include "event.hpp"
#include "chorale.hpp"
#include "viewpoint.hpp"

using json = nlohmann::json;

int main(void) {
  std::ifstream corpus_file("corpus/chorale_dataset.json");
  json j;
  corpus_file >> j;

  BasicViewpoint<ChoralePitch> pitch_vp(3);
  BasicViewpoint<ChoraleDuration> duration_vp(3);
  IntervalViewpoint interval_vp(3);

  const auto num_chorales = j["corpus"].size();
  for (unsigned int i = 0; i < num_chorales; i++) {
    const auto &chorale_j = j["corpus"][i];

    std::cout << "Training (" << (i+1) << "/" << num_chorales << "): "
     <<  chorale_j["title"] << std::endl;

    std::vector<ChoralePitch> pitches;
    std::vector<ChoraleDuration> durations;

    for (const auto &note_j : chorale_j["notes"]) {
      MidiPitch pitch(static_cast<unsigned int>(note_j[0]));
      // auto offset = static_cast<unsigned int>(note_j[1]);
      QuantizedDuration dur(static_cast<unsigned int>(note_j[2]));

      // ChoraleEvent e(pitch, dur, ks, ts, offset);
      pitches.push_back(pitch);
      durations.push_back(dur);
    }

    pitch_vp.learn(pitches);
    duration_vp.learn(durations);
    interval_vp.learn(pitches);
  }

  pitch_vp.write_latex("out/tex/complete_pitch.tex");
  duration_vp.write_latex("out/tex/complete_dur.tex");
  interval_vp.write_latex("out/tex/complete_ival.tex");

  std::vector<ChoralePitch> am_pitches;
  std::vector<ChoraleDuration> am_durations;

  const auto &aus_meins_j = j["corpus"][0];
  for (const auto &note_j : aus_meins_j["notes"]) {
    MidiPitch pitch(static_cast<unsigned int>(note_j[0]));
    QuantizedDuration dur(static_cast<unsigned int>(note_j[2]));

    am_pitches.push_back(pitch);
    am_durations.push_back(dur);
  }

  double entropy_bias = 2.0; // for now
  ChoraleMVS single_vp(entropy_bias, {&pitch_vp}, {&duration_vp});
  ChoraleMVS multi_vp(entropy_bias, {&pitch_vp, &interval_vp}, {&duration_vp});

  std::cout << std::endl << std::endl;
  std::cout 
    << "Average entropy of pitch in first chorale using long-term model:" 
    << std::endl;
 
  std::cout 
    << "--> Single VP (pitch): " 
    << single_vp.avg_sequence_entropy(am_pitches) << std::endl;

  std::cout 
    << "--> Multi VP (pitch,interval): " 
    << multi_vp.avg_sequence_entropy(am_durations) << std::endl;
}


