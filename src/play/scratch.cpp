#include <iostream>
#include <fstream>
#include "json.hpp"
#include "event.hpp"
#include "chorale.hpp"
#include "viewpoint.hpp"

using json = nlohmann::json;

class ChoraleInterval : public CodedEvent {
public:
  constexpr static unsigned int cardinality = 25;
  constexpr static int min_interval = -12;
  ChoraleInterval(int interval) : CodedEvent(interval + min_interval) {
    assert(code < cardinality);
  }
};

/*
OptionalEvent<ChoraleInterval> 
project_interval_test(const std::vector<ChoralePitch> &pitches) {
  if (pitches.size() < 2) return NoEvent<ChoraleInterval>();

  auto pitchA = pitches[pitches.size() - 2];
  auto pitchB = pitches[pitches.size() - 1];
  int interval = (int)pitchB.raw_value() - (int)pitchA.raw_value();
  return SomeEvent<ChoraleInterval>( ChoraleInterval(interval) );
}
*/

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

  BasicViewpoint<ChoralePitch> pitch_vp(3);

  std::vector<ChoralePitch> pitches;
  std::transform(events.begin(), events.end(), std::back_inserter(pitches),
      [](ChoraleEvent e) { return e.pitch; }); 

  pitch_vp.learn(pitches);
  pitch_vp.write_graphviz("out/pitch_vp.gv");
}
