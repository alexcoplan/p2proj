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

  json chorale_j = j["corpus"][0];
  std::cout << "Chorale title: " << chorale_j["title"] << std::endl;

  KeySig ks(static_cast<int>(chorale_j["key_sig_sharps"]));
  QuantizedDuration ts(static_cast<unsigned int>(chorale_j["time_sig_semis"]));

  std::vector<ChoraleEvent> events;

  for (const auto &note_j : chorale_j["notes"]) {
    MidiPitch pitch(static_cast<unsigned int>(note_j[0]));
    auto offset = static_cast<unsigned int>(note_j[1]);
    QuantizedDuration dur(static_cast<unsigned int>(note_j[2]));

    ChoraleEvent e(pitch, dur, ks, ts, offset);
    events.push_back(e);
  }

  // let's try out some basic viewpoints

  BasicViewpoint<ChoralePitch> pitch_vp(3);
  std::vector<ChoralePitch> pitches;
  std::transform(events.begin(), events.end(), std::back_inserter(pitches),
      [](ChoraleEvent e) { return e.pitch; }); 

  pitch_vp.learn(pitches);
  pitch_vp.write_latex("out/tex/pitch_vp.tex");

  BasicViewpoint<ChoraleDuration> duration_vp(3);
  std::vector<ChoraleDuration> durations;
  std::transform(events.begin(), events.end(), std::back_inserter(durations),
      [](ChoraleEvent e) { return e.duration; });

  duration_vp.learn(durations);
  duration_vp.write_latex("out/tex/duration_vp.tex");

  IntervalViewpoint interval_vp(3);
  interval_vp.learn(pitches);
  interval_vp.write_latex("out/tex/ival_vp.tex");

}


