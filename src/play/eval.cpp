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
      ps.push_back(ChoralePitch(pitch));
      ds.push_back(ChoraleDuration(dur));
    }

    pitch_vp.learn(ps);
    duration_vp.learn(ds);
    interval_vp.learn(ps);

    corpus.push_back(std::make_pair(ps, ds));
  }

  /*
  pitch_vp.write_latex("out/tex/complete_pitch.tex");
  duration_vp.write_latex("out/tex/complete_dur.tex");
  interval_vp.write_latex("out/tex/complete_ival.tex");
  */

  ChoraleMVS single_vp(1.0, {&pitch_vp}, {&duration_vp});
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
}


