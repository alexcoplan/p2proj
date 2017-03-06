#include <iostream>
#include <fstream>
#include "json.hpp"
#include "event.hpp"
#include "chorale.hpp"
#include "viewpoint.hpp"

using json = nlohmann::json;
using corpus_t = std::vector<std::vector<ChoraleEvent>>;

void parse_subcorpus(
  const json &subcorp_j,
  corpus_t &subcorp
) {
  const auto num_chorales = subcorp_j.size();
  for (unsigned int i = 0; i < num_chorales; i++) {
    const auto &chorale_j = subcorp_j[i];
    
    std::vector<ChoraleEvent> chorale_events;

    const auto &notes_j = chorale_j["notes"];
    assert(notes_j.size() > 1);

    const auto &first_note_j = notes_j[0];
    unsigned int first_pitch   = first_note_j[0];
    unsigned int prev_offset   = first_note_j[1];
    unsigned int prev_duration = first_note_j[2];

    unsigned int num_sharps = chorale_j["key_sig_sharps"];
    KeySig ks(num_sharps);

    chorale_events.push_back(ChoraleEvent(
      ks, MidiPitch(first_pitch), QuantizedDuration(prev_duration), nullptr
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
        ks,
        MidiPitch(pitch), QuantizedDuration(duration), rest.shared_instance()
      ));

      prev_offset = offset;
      prev_duration = duration;
    }

    subcorp.push_back(chorale_events);
  }
}

void parse(
  const std::string corpus_path, 
  corpus_t &train_corpus, 
  corpus_t &test_corpus
) {
  std::ifstream corpus_file(corpus_path);
  json j;
  corpus_file >> j;

  std::cout << "Parsing corpus... " << std::endl << std::flush;

  const auto &corpus_j = j["corpus"];
  std::cerr << "loading train corpus" << std::endl;
  parse_subcorpus(corpus_j["train"], train_corpus);
  std::cerr << "loading test corpus" << std::endl;
  parse_subcorpus(corpus_j["validate"], test_corpus);

  std::cout << "done." << std::endl;
}

void train(const corpus_t &corpus, std::initializer_list<ChoraleMVS *> mvss) {
  for (const auto &piece : corpus) {
    for (auto mvs_ptr : mvss)
      mvs_ptr->learn(piece);
  }
}

void render(const std::vector<ChoraleEvent> &piece, 
            const std::map<std::string, std::vector<double>> entropies,
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

  json entropies_j;
  for (const auto &kv : entropies) {
    auto k = kv.first;
    auto v = kv.second;
    entropies_j[k] = v;
  }

  result_j["entropies"] = entropies_j;

  std::ofstream o(json_fname);
  o << std::setw(2) << result_j << std::endl;
}

// returns < pitch_entropies, duration_entropies >
std::pair<std::vector<double>, std::vector<double>>
evaluate(const corpus_t &corpus, double intra_bias, double inter_bias,
    std::initializer_list<ChoraleMVS *> mvss) {

  std::cout 
    << std::endl
    << "Evaluating MVSs at (intra: " << intra_bias << ", " 
    << "inter: " << inter_bias << ")" << std::endl;

  for (auto mvs_ptr : mvss) {
    mvs_ptr->set_intra_layer_bias(intra_bias);
    mvs_ptr->entropy_bias = inter_bias;
  }

  std::vector<double> pitch_entropies(mvss.size());
  std::vector<double> dur_entropies(mvss.size());
  for (auto &h : pitch_entropies)
    h = 0.0;
  for (auto &h : dur_entropies)
    h = 0.0;

  unsigned int i = 1;

  for (const auto &c : corpus) {
    if (++i % 5 == 0)
      std::cout << "=" << std::flush;

    unsigned int j = 0;
    for (auto mvs_ptr : mvss) {
      pitch_entropies[j] += mvs_ptr->avg_sequence_entropy<ChoralePitch>(c);
      dur_entropies[j]   += mvs_ptr->avg_sequence_entropy<ChoraleDuration>(c);
    }
  }

  for (auto &h : pitch_entropies)
    h /= corpus.size();
  for (auto &h : dur_entropies)
    h /= corpus.size();

  return std::pair<std::vector<double>, std::vector<double>>(
      pitch_entropies, dur_entropies);
}

void bias_grid_sweep(const corpus_t &corpus, ChoraleMVS &mvs, double max_intra,
    double max_inter, double step) {
  double min_inter, min_intra;
  min_inter = min_intra = 0.0;

  json inter_biases = json::array();
  json intra_biases = json::array();
  for (double b = min_inter; b <= max_inter; b += step)
    inter_biases.push_back(b);
  for (double b = min_intra; b <= max_intra; b += step)
    intra_biases.push_back(b);

  json entropy_values = json::array();

  for (double inter = min_inter; inter <= max_inter; inter += step) {
    json inner_values = json::array();
    for (double intra = min_intra; intra <= max_intra; intra += step) {
      auto measurements = evaluate(corpus, intra, inter, {&mvs});
      auto pitch_ents = measurements.first; // vector for each mvs
      auto dur_ents = measurements.second;

      inner_values.push_back(pitch_ents[0] + dur_ents[0]);
    }
    entropy_values.push_back(inner_values);
  }

  json data_j({
    {"inter_biases", inter_biases},
    {"intra_biases", intra_biases},
    {"entropy_values", entropy_values}
  });

  std::ofstream o("out/bias_sweep.json");
  o << data_j;
}

void generate(ChoraleMVS &mvs, 
              const unsigned int len, 
              const std::string &json_fname) {
  std::cout << "Generating piece of length " << len << "... " << std::flush;
  auto piece = mvs.random_walk(len);
  std::cout << "done." << std::endl;

  std::cout << "Entropy of generated piece: " << std::endl << std::flush;
  auto pitch_entropy = mvs.avg_sequence_entropy<ChoralePitch>(piece);
  auto dur_entropy = mvs.avg_sequence_entropy<ChoraleDuration>(piece);
  std::cout << "--> Pitch: " << pitch_entropy << std::endl;
  std::cout << "--> Duration: " << dur_entropy << std::endl;

  // TODO: include rest entropy calculations
  auto pitch_xents = mvs.cross_entropies<ChoralePitch>(piece);
  auto dur_xents = mvs.cross_entropies<ChoraleDuration>(piece);
  auto pitch_dents = mvs.dist_entropies<ChoralePitch>(piece);
  auto dur_dents = mvs.dist_entropies<ChoraleDuration>(piece);
  std::vector<double> total_xents(piece.size());
  std::vector<double> total_dents(piece.size());
  for (unsigned int i = 0; i < piece.size(); i++) {
    total_xents[i] = pitch_xents.at(i) + dur_xents.at(i);
    total_dents[i] = pitch_dents.at(i) + dur_dents.at(i);
  }

  std::map<std::string, std::vector<double>> entropy_map;
  entropy_map.insert({"cross_entropies", total_xents});
  entropy_map.insert({"dist_entropies", total_dents});

  render(piece, entropy_map, json_fname);
}

int main(void) {
  ChoraleMVS::BasicVP<ChoralePitch> pitch_vp(4);
  ChoraleMVS::BasicVP<ChoraleDuration> duration_vp(4);
  ChoraleMVS::BasicVP<ChoraleRest> rest_vp(4);
  IntervalViewpoint interval_vp(4);
  IntrefViewpoint intref_vp(4);

  std::cout << "Evaluating MVS with long-term model only against full MVS" 
    << std::endl;

  auto lt_config = MVSConfig::long_term_only(1.0);
  ChoraleMVS lt_only(lt_config);

  MVSConfig full_config;
  full_config.enable_short_term = true;
  full_config.mvs_name = "full mvs for evaluation";
  full_config.intra_layer_bias = 1.3;
  full_config.inter_layer_bias = 1.0;

  ChoraleMVS full_mvs(full_config);

  for (auto mvs_ptr : {&lt_only, &full_mvs}) {
    mvs_ptr->add_viewpoint(&pitch_vp);
    mvs_ptr->add_viewpoint(&interval_vp);
    mvs_ptr->add_viewpoint(&intref_vp);
    mvs_ptr->add_viewpoint(&duration_vp);
    mvs_ptr->add_viewpoint(&rest_vp);
  }

  corpus_t train_corp;
  corpus_t test_corp;
  parse("corpus/chorales_t3.json", train_corp, test_corp);

  std::cout << "Training... " << std::flush;
  train(train_corp, {&lt_only, &full_mvs});
  std::cout << "done." << std::endl;

  double max_intra = 2.0;
  double max_inter = 2.0;
  double step = 0.5;
  bias_grid_sweep(test_corp, full_mvs, max_intra, max_inter, step);

  //generate(full_mvs, 42, "out/gend.json");
}


