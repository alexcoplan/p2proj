#include <iostream>
#include <fstream>
#include "json.hpp"
#include "event.hpp"
#include "chorale.hpp"
#include "viewpoint.hpp"

using json = nlohmann::json;

/**************************************************
 * TODO: Fix corpus prep script to collapse ties
 **************************************************/

int main(void) {
  std::ifstream corpus_file("corpus/chorale_dataset.json");
  json j;
  corpus_file >> j;

  ChoraleMVS::BasicVP<ChoralePitch> pitch_vp(3);
  ChoraleMVS::BasicVP<ChoraleDuration> duration_vp(3);
  ChoraleMVS::BasicVP<ChoraleRest> rest_vp(3);
  IntervalViewpoint interval_vp(3);

  ChoraleMVS svs(2.0);
  svs.add_viewpoint(&pitch_vp);
  svs.add_viewpoint(&duration_vp);
  svs.add_viewpoint(&rest_vp);

  ChoraleMVS mvs(2.0);
  mvs.add_viewpoint(&pitch_vp);
  mvs.add_viewpoint(&interval_vp);
  mvs.add_viewpoint(&duration_vp);
  mvs.add_viewpoint(&rest_vp);

  std::vector<std::vector<ChoraleEvent>> corpus;

  const auto num_chorales = j["corpus"].size();
  for (unsigned int i = 0; i < num_chorales; i++) {
    const auto &chorale_j = j["corpus"][i];
    
    std::vector<ChoraleEvent> chorale_events;

    std::cout << "Training (" << (i+1) << "/" << num_chorales << "): "
     <<  chorale_j["title"] << std::endl;


    const auto &notes_j = chorale_j["notes"];
    assert(notes_j.size() > 1);

    const auto &first_note_j = notes_j[0];
    unsigned int first_pitch   = first_note_j[0];
    unsigned int prev_offset   = first_note_j[1];
    unsigned int prev_duration = first_note_j[2];

    chorale_events.push_back(ChoraleEvent(
      MidiPitch(first_pitch), QuantizedDuration(prev_duration), nullptr
    ));

    for (unsigned int j = 1; j < notes_j.size(); j++) {
      const auto &note_j = notes_j[j];

      unsigned int pitch    = note_j[0];
      unsigned int offset   = note_j[1];
      unsigned int duration = note_j[2];

      assert(offset >= prev_offset + prev_duration);
      auto rest_amt = offset - prev_offset - prev_duration;
      ChoraleRest rest{QuantizedDuration(rest_amt)};

      chorale_events.push_back(ChoraleEvent(
        MidiPitch(pitch), QuantizedDuration(duration), rest.shared_instance()
      ));

      prev_offset = offset;
      prev_duration = duration;
    }

    svs.learn(chorale_events);
    mvs.learn(chorale_events);
    corpus.push_back(chorale_events);
  }

  /*
  pitch_vp.write_latex("out/tex/complete_pitch.tex");
  duration_vp.write_latex("out/tex/complete_dur.tex");
  interval_vp.write_latex("out/tex/complete_ival.tex");
  */

  std::cout 
    << std::endl
    << "Training complete! Evaluating models..." 
    << std::endl << std::endl;

  for (double eb = 6.0; eb < 7.0; eb += 1.0) {
    svs.entropy_bias = eb;
    mvs.entropy_bias = eb;

    double svs_entropy = 0.0;
    double mvs_entropy = 0.0;

    double svs_entropy_2 = 0.0;
    double mvs_entropy_2 = 0.0;

    double dur_entropy = 0.0;

    unsigned int i = 1;

    std::cout << "entropy_bias = " << eb << std::endl;

    for (const auto &c : corpus) {
      if (++i % 10 == 0)
        std::cout << "=" << std::flush;

      double h_s = svs.avg_sequence_entropy<ChoralePitch>(c);
      svs_entropy += h_s;
      svs_entropy_2 += h_s * h_s;

      double h_m = mvs.avg_sequence_entropy<ChoralePitch>(c);
      mvs_entropy += h_m;
      mvs_entropy_2 += h_m * h_m;

      double h_dur = svs.avg_sequence_entropy<ChoraleDuration>(c);
      dur_entropy += h_dur;
    }

    svs_entropy /= corpus.size();
    mvs_entropy /= corpus.size();
    dur_entropy /= corpus.size();

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
      << "--> Single VP (duration): "
      << dur_entropy << " bits" << std::endl;

    std::cout 
      << "--> Multi VP (pitch,interval): " 
      << mvs_entropy << " bits" 
      << " (stdev = " << std::sqrt(mvs_variance) << ")"
      << std::endl << std::endl;
  }
}


