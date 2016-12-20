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

  using entry_t = 
    std::pair<std::vector<ChoralePitch>, std::vector<ChoraleDuration>>;

  std::vector<entry_t> corpus;

  const auto num_chorales = j["corpus"].size();
  for (unsigned int i = 0; i < num_chorales; i++) {
    const auto &chorale_j = j["corpus"][i];

    std::cout << "Training (" << (i+1) << "/" << num_chorales << "): "
     <<  chorale_j["title"] << std::endl;

    std::vector<ChoralePitch> ps;
    std::vector<ChoraleDuration> ds;

    for (const auto &note_j : chorale_j["notes"]) {
      MidiPitch pitch(static_cast<unsigned int>(note_j[0]));
      // auto offset = static_cast<unsigned int>(note_j[1]);
      QuantizedDuration dur(static_cast<unsigned int>(note_j[2]));

      // ChoraleEvent e(pitch, dur, ks, ts, offset);
      ps.push_back(pitch);
      ds.push_back(dur);
    }

    pitch_vp.learn(ps);
    duration_vp.learn(ds);
    interval_vp.learn(ps);

    corpus.push_back(std::make_pair(ps, ds));
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

  std::cout 
    << std::endl
    << "Training complete! Evaluating models..." 
    << std::endl << std::endl;

  double svs_entropy = 0.0;
  double mvs_entropy = 0.0;

  unsigned int i = 1;

  for (const auto &c : corpus) {
    std::cout << std::to_string(i++) << "," << std::flush;
    const auto &pitches = c.first;
    svs_entropy += single_vp.avg_sequence_entropy(pitches);
    mvs_entropy += multi_vp.avg_sequence_entropy(pitches);
  }

  std::cout << "done!" << std::endl << std::endl;

  svs_entropy /= corpus.size();
  mvs_entropy /= corpus.size();


  std::cout 
    << "Average entropy of pitch using long-term model:" 
    << std::endl;
 
  std::cout 
    << "--> Single VP (pitch): " 
    << single_vp.avg_sequence_entropy(am_pitches) 
    << " bits" << std::endl;

  std::cout 
    << "--> Multi VP (pitch,interval): " 
    << multi_vp.avg_sequence_entropy(am_durations) 
    << " bits" << std::endl;
}


