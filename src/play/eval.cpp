#include <iostream>
#include <fstream>
#include "json.hpp"
#include "event.hpp"
#include "chorale.hpp"
#include "viewpoint.hpp"

using json = nlohmann::json;
using corpus_t = std::vector<std::vector<ChoraleEvent>>;

void parse(const std::string corpus_path, corpus_t &corpus) {
  std::ifstream corpus_file(corpus_path);
  json j;
  corpus_file >> j;

  std::cout << "Parsing corpus... " << std::flush;

  const auto num_chorales = j["corpus"].size();
  for (unsigned int i = 0; i < num_chorales; i++) {
    const auto &chorale_j = j["corpus"][i];
    
    std::vector<ChoraleEvent> chorale_events;

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

    corpus.push_back(chorale_events);
  }

  std::cout << "done." << std::endl;
}

void train(const corpus_t &corpus, std::initializer_list<ChoraleMVS *> mvss) {
  for (const auto &piece : corpus) {
    for (auto mvs_ptr : mvss)
      mvs_ptr->learn(piece);
  }
}

void render(const std::vector<ChoraleEvent> &piece, 
            const std::string &json_fname) {
  json notes_j;

  unsigned int offset = 0;
  for (const auto &event : piece) {
    unsigned int pitch = event.pitch.raw_value();
    unsigned int duration = event.duration.raw_value();

    if (event.rest)
      offset += event.rest->raw_value();

    notes_j.push_back({ pitch, offset, duration });
    offset += duration;
  }

  json result_j;
  result_j["notes"] = notes_j;

  std::ofstream o(json_fname);
  o << std::setw(2) << result_j << std::endl;
}

void evaluate(const corpus_t &corpus, double eb_min, double eb_max,
    std::initializer_list<ChoraleMVS *> mvss) {

  std::cout 
    << std::endl
    << "Evaluating models..." 
    << std::endl << std::endl;

  for (double eb = eb_min; eb < eb_max + 1.0; eb += 1.0) {
    for (auto mvs_ptr : mvss)
      mvs_ptr->entropy_bias = eb;

    std::vector<double> pitch_entropies(mvss.size());
    for (auto &h : pitch_entropies)
      h = 0.0;

    std::vector<double> dur_entropies(mvss.size());
    for (auto &h : dur_entropies)
      h = 0.0;

    unsigned int i = 1;

    std::cout << "entropy_bias = " << eb << std::endl;

    for (const auto &c : corpus) {
      if (++i % 10 == 0)
        std::cout << "=" << std::flush;

      unsigned int j = 0;
      for (auto mvs_ptr : mvss) {
        double h_pitch = mvs_ptr->avg_sequence_entropy<ChoralePitch>(c);
        double h_dur   = mvs_ptr->avg_sequence_entropy<ChoraleDuration>(c);
        pitch_entropies[j] += h_pitch;
        dur_entropies[j] += h_dur;
        j++;
      }
    }

    for (auto &h : pitch_entropies)
      h /= corpus.size(); 
    for (auto &h : dur_entropies)
      h /= corpus.size();

    std::cout 
      << std::endl
      << "Average entropies using long-term model:" 
      << std::endl;

    unsigned int j = 0;
    for (auto mvs_ptr : mvss) {
      std::cout << "    MVS " << mvs_ptr->name << ":" << std::endl;

      std::cout
        << "--> pitch entropy: "
        << pitch_entropies[j]
        << std::endl;

      std::cout
        << "--> duration entropy: "
        << dur_entropies[j]
        << std::endl;

      j++;
    }
    std::cout << std::endl;
  }
}

void generate(const ChoraleMVS &mvs, 
              const unsigned int len, 
              const std::string &json_fname) {
  std::cout << "Generating piece of length " << len << "... " << std::flush;
  auto piece = mvs.generate(len);
  std::cout << "done." << std::endl;

  std::cout << "Entropy of generated piece: " << std::endl << std::flush;
  auto pitch_entropy = mvs.avg_sequence_entropy<ChoralePitch>(piece);
  auto dur_entropy = mvs.avg_sequence_entropy<ChoraleDuration>(piece);
  std::cout << "--> Pitch: " << pitch_entropy << std::endl;
  std::cout << "--> Duration: " << dur_entropy << std::endl;

  render(piece, json_fname);
}

int main(void) {
  ChoraleMVS::BasicVP<ChoralePitch> pitch_vp(3);
  ChoraleMVS::BasicVP<ChoraleDuration> duration_vp(3);
  ChoraleMVS::BasicVP<ChoraleRest> rest_vp(3);
  IntervalViewpoint interval_vp(3);

  ChoraleMVS svs(2.0, "{pitch,duration,rest}");
  svs.add_viewpoint(&pitch_vp);
  svs.add_viewpoint(&duration_vp);
  svs.add_viewpoint(&rest_vp);

  ChoraleMVS mvs(6.0, "{pitch,interval,duration,rest}");
  mvs.add_viewpoint(&pitch_vp);
  mvs.add_viewpoint(&interval_vp);
  mvs.add_viewpoint(&duration_vp);
  mvs.add_viewpoint(&rest_vp);

  corpus_t corpus;
  parse("corpus/chorale_dataset.json", corpus);

  std::cout << "Training... " << std::flush;
  train(corpus, {&svs, &mvs});
  std::cout << "done." << std::endl;

  evaluate(corpus, 2.0, 8.0, {&svs, &mvs});
  //generate(mvs, 42, "out/gend.json");
}


