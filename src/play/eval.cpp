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
  BasicViewpoint<ChoraleRest> rest_vp(3);
  IntervalViewpoint interval_vp(3);

  std::vector<std::vector<ChoraleEvent>> corpus;

  const auto num_chorales = j["corpus"].size();
  for (unsigned int i = 0; i < num_chorales; i++) {
    const auto &chorale_j = j["corpus"][i];
    
    std::vector<ChoraleEvent> chorale_events;

    std::cout << "Training (" << (i+1) << "/" << num_chorales << "): "
     <<  chorale_j["title"] << std::endl;


    const auto &notes_j = chorale_j["note"];
    assert(notes_j.size() > 1);

    const auto &first_note_j = notes_j[0];
    unsigned int first_pitch   = first_note_j[0];
    unsigned int prev_offset   = first_note_j[1];
    unsigned int prev_duration = first_note_j[2];

    chorale_events.push_back(ChoraleEvent(
      MidiPitch(first_pitch), QuantizedDuration(prev_duration), nullptr
    ));

    for (unsigned int j = 1; i < notes_j.size(); j++) {
      const auto &note_j = notes_j[j];

      unsigned int pitch    = note_j[0];
      unsigned int offset   = note_j[1];
      unsigned int duration = note_j[2];

      assert(offset >= prev_offset + prev_duration);
      ChoraleRest rest(offset - prev_duration - prev_offset);

      chorale_events.push_back(ChoraleEvent(
        MidiPitch(pitch), QuantizedDuration(duration), rest.shared_instance()
      ));

      prev_offset = offset;
      prev_duration = duration;
    }

    const auto pitches = 
      ChoraleEvent::lift<ChoralePitch>(chorale_events);

    const auto rests =
      ChoraleEvent::lift<ChoraleRest>(chorale_events);

    pitch_vp.learn(pitches);
    interval_vp.learn(pitches);
    rest_vp.learn(ChoraleEvent::lift<ChoraleRest>(chorale_events));
    duration_vp.learn(ChoraleEvent::lift<ChoraleDuration>(chorale_events));
  }

  rest_vp.write_latex("out/tex/complete_rest.tex");

  /*
  pitch_vp.write_latex("out/tex/complete_pitch.tex");
  duration_vp.write_latex("out/tex/complete_dur.tex");
  interval_vp.write_latex("out/tex/complete_ival.tex");
  */

  /*
  ChoraleMVS single_vp(1.0, {&pitch_vp}, {&duration_vp}, {&rest_vp});
  ChoraleMVS multi_vp(1.0, {&pitch_vp, &interval_vp}, {&duration_vp});

  std::cout 
    << std::endl
    << "Training complete! Evaluating models..." 
    << std::endl << std::endl;

  for (double eb = 0.0; eb < 12.0; eb += 1.0) {
    single_vp.entropy_bias = eb;
    multi_vp.entropy_bias = eb;

    double svs_entropy = 0.0;
    double mvs_entropy = 0.0;

    double svs_entropy_2 = 0.0;
    double mvs_entropy_2 = 0.0;

    unsigned int i = 1;

    std::cout << "entropy_bias = " << eb << std::endl;

    for (const auto &c : corpus) {
      if (++i % 10 == 0)
        std::cout << "=" << std::flush;

      const auto &pitches = c.first;

      double h_s = single_vp.avg_sequence_entropy(pitches);
      svs_entropy += h_s;
      svs_entropy_2 += h_s * h_s;

      double h_m = multi_vp.avg_sequence_entropy(pitches);
      mvs_entropy += h_m;
      mvs_entropy_2 += h_m * h_m;
    }

    svs_entropy /= corpus.size();
    mvs_entropy /= corpus.size();

    auto mu_svs_2 = svs_entropy * svs_entropy;
    auto mu_mvs_2 = mvs_entropy * mvs_entropy;
    auto svs_variance = (svs_entropy_2 / corpus.size()) - mu_svs_2;
    auto mvs_variance = (mvs_entropy_2 / corpus.size()) - mu_mvs_2;

    std::cout 
      << std::endl
      << "Average entropy of pitch using long-term model:" 
      << std::endl;
   
    std::cout 
      << "--> Single VP (pitch): " 
      << svs_entropy << " bits" 
      << " (stdev = " << std::sqrt(svs_variance) << ")" << std::endl;

    std::cout 
      << "--> Multi VP (pitch,interval): " 
      << mvs_entropy << " bits" 
      << " (stdev = " << std::sqrt(mvs_variance) << ")"
      << std::endl << std::endl;
  }

  */
}


