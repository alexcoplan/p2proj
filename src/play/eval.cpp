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

    // get key and time signatures
    unsigned int num_sharps = chorale_j["key_sig_sharps"];
    KeySig ks(num_sharps);
    unsigned int bar_length = chorale_j["time_sig_amt"];
    QuantizedDuration ts(bar_length);

    chorale_events.push_back(ChoraleEvent(
      ks, ts, MidiPitch(first_pitch), 
      QuantizedDuration(prev_duration), 
      QuantizedDuration(prev_offset)
    ));

    for (unsigned int j = 1; j < notes_j.size(); j++) {
      const auto &note_j = notes_j[j];

      unsigned int pitch    = note_j[0];
      unsigned int offset   = note_j[1];
      unsigned int duration = note_j[2];

      assert(offset >= prev_offset + prev_duration);
      auto rest_amt = offset - prev_offset - prev_duration;
      QuantizedDuration rest_dur{rest_amt};

      chorale_events.push_back(ChoraleEvent(
        ks, ts,
        MidiPitch(pitch), 
        QuantizedDuration(duration), 
        rest_amt
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
    unsigned int rest_amt = event.rest.raw_value();

    notes_j.push_back({ pitch, offset + rest_amt, duration });
    offset += duration + rest_amt;
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

struct EntropyMeasurement {
  double h_pitch;
  double h_duration;
  double h_rest;

  template<class T>
  double project();

  EntropyMeasurement() :
    h_pitch(0.0), h_duration(0.0), h_rest(0.0) {}
};

template<>
double EntropyMeasurement::project<ChoralePitch>() {
  return h_pitch;
}

template<>
double EntropyMeasurement::project<ChoraleDuration>() {
  return h_duration;
}

template<>
double EntropyMeasurement::project<ChoraleRest>() {
  return h_rest;
}

// returns < pitch_entropies, duration_entropies >
std::vector<EntropyMeasurement>
evaluate(const corpus_t &corpus, double intra_bias, double inter_bias,
    std::initializer_list<ChoraleMVS *> mvss) {

  for (auto mvs_ptr : mvss) {
    mvs_ptr->set_intra_layer_bias(intra_bias);
    mvs_ptr->entropy_bias = inter_bias;
  }

  std::vector<EntropyMeasurement> result(mvss.size());

  //unsigned int i = 1;

  for (const auto &c : corpus) {
    /*
    if (++i % 4 == 0)
      std::cout << "=" << std::flush;
      */

    unsigned int j = 0;
    for (auto mvs_ptr : mvss) {
      result[j].h_pitch    += mvs_ptr->avg_sequence_entropy<ChoralePitch>(c);
      result[j].h_duration += mvs_ptr->avg_sequence_entropy<ChoraleDuration>(c);
      result[j].h_rest     += mvs_ptr->avg_sequence_entropy<ChoraleRest>(c);
      j++;
    }
  }

  //std::cout << std::endl;

  for (auto &point : result) {
    point.h_pitch /= corpus.size();
    point.h_duration /= corpus.size();
    point.h_rest /= corpus.size();
  }

  return result;
}

class MVSOptimizer {
  template<typename T>
  using PredictorList = std::vector<ChoraleMVS::Pred<T> *>;

  PredictorList<ChoralePitch> pitch_pool;
  PredictorList<ChoraleDuration> dur_pool;
  PredictorList<ChoraleRest> rest_pool;

  const MVSConfig mvs_config;
  ChoraleMVS::GenVP<ChoralePitch> pitch_base;
  ChoraleMVS::GenVP<ChoraleDuration> duration_base;
  ChoraleMVS::GenVP<ChoraleRest> rest_base;

public:
  void add_to_pool(ChoraleMVS::Pred<ChoralePitch> *vp_ptr) {
    pitch_pool.push_back(vp_ptr);
  }

  void add_to_pool(ChoraleMVS::Pred<ChoraleDuration> *vp_ptr) {
    dur_pool.push_back(vp_ptr);
  }

  void add_to_pool(ChoraleMVS::Pred<ChoraleRest> *vp_ptr) {
    rest_pool.push_back(vp_ptr);
  }

  template<typename T>
  static std::string vps_to_string(const PredictorList<T> &vps) {
    if (vps.empty())
      return "{}";

    std::string result = "{ " + vps[0]->vp_name();
    for (unsigned int i = 1; i < vps.size(); i++)
      result += ", " + vps[i]->vp_name();

    return result + " }";
  }

  // this adds the base VPs for the types other than T to the argument mvs. e.g.
  // if T = ChoralePitch, then this adds base VPs for ChoraleDuration and
  // ChoraleRest to the arguments mvs
  template<typename T>
  void add_support_vps(ChoraleMVS &mvs);

  template<typename T>
  PredictorList<T> pool() const;

  template<typename T>
  ChoraleMVS::GenVP<T> *base_vp();

  template<typename T>
  void optimize(double eps_terminate, 
      const corpus_t &train_corp, 
      const corpus_t &test_corp);

  MVSOptimizer(const MVSConfig &cfg) : 
    mvs_config(cfg), 
    pitch_base(cfg.lt_history), 
    duration_base(cfg.lt_history),
    rest_base(cfg.lt_history) {}
};

template<> 
MVSOptimizer::PredictorList<ChoralePitch>
MVSOptimizer::pool<ChoralePitch>() const { return pitch_pool; }

template<>
MVSOptimizer::PredictorList<ChoraleDuration>
MVSOptimizer::pool<ChoraleDuration>() const { return dur_pool; }

template<>
MVSOptimizer::PredictorList<ChoraleRest>
MVSOptimizer::pool<ChoraleRest>() const { return rest_pool; }

template<>
ChoraleMVS::GenVP<ChoralePitch> *
MVSOptimizer::base_vp<ChoralePitch>() { return &pitch_base; }

template<>
ChoraleMVS::GenVP<ChoraleDuration> *
MVSOptimizer::base_vp<ChoraleDuration>() { return &duration_base; }

template<>
ChoraleMVS::GenVP<ChoraleRest> *
MVSOptimizer::base_vp<ChoraleRest>() { return &rest_base; }

template<>
void MVSOptimizer::add_support_vps<ChoralePitch>(ChoraleMVS &mvs) {
  mvs.add_viewpoint(&duration_base);
  mvs.add_viewpoint(&rest_base);
}

template<>
void MVSOptimizer::add_support_vps<ChoraleDuration>(ChoraleMVS &mvs) {
  mvs.add_viewpoint(&pitch_base);
  mvs.add_viewpoint(&rest_base);
}

template<>
void MVSOptimizer::add_support_vps<ChoraleRest>(ChoraleMVS &mvs) {
  mvs.add_viewpoint(&pitch_base);
  mvs.add_viewpoint(&duration_base);
}

template<typename T>
void MVSOptimizer::optimize(double eps_terminate, 
    const corpus_t &train_corp, 
    const corpus_t &test_corp)
{
  const double xent_inf = 1024.0; // no x-entropy will be bigger than this

  auto vp_pool = pool<T>();

  std::cout 
    << "*** Optimising " << T::type_name 
    << " cross-entropy..." << std::endl;
  std::cout << "viewpoint pool: " << std::endl;
  std::cout << vps_to_string(vp_pool) << std::endl;

  ChoraleMVS base_mvs(mvs_config);
  base_mvs.add_viewpoint(&pitch_base);
  base_mvs.add_viewpoint(&duration_base);
  base_mvs.add_viewpoint(&rest_base);

  train(train_corp, {&base_mvs});

  double prev_best_xent = evaluate(test_corp, 
      mvs_config.intra_layer_bias, 
      mvs_config.inter_layer_bias,
      {&base_mvs}).at(0).project<T>();
  double round_best_xent = xent_inf;

  std::cout << "base " 
    << T::type_name << " entropy: " << prev_best_xent 
    << std::endl << std::endl;

  PredictorList<T> vp_stack { base_vp<T>() };

  for (;;) {
    round_best_xent = xent_inf;
    ChoraleMVS::Pred<T> *best_addition = nullptr;

    for (auto vp_ptr : vp_pool) {
      // if this VP has already been added just skip it
      bool already_on_stack = false;
      for (auto on_stack : vp_stack) {
        if (on_stack == vp_ptr) {
          already_on_stack = true;
          break;
        }
      }
      if (already_on_stack)
        continue;

      vp_stack.push_back(vp_ptr);

      std::cout << "Evaluating system: " 
        << vps_to_string(vp_stack) 
        << std::endl;

      ChoraleMVS trial_mvs(mvs_config);
      add_support_vps<T>(trial_mvs);
      for (auto vp_ptr : vp_stack)
        trial_mvs.add_viewpoint(vp_ptr);

      train(train_corp, {&trial_mvs});

      auto this_xent = evaluate(test_corp,
          mvs_config.intra_layer_bias,
          mvs_config.inter_layer_bias,
          {&trial_mvs}).at(0).project<T>();

      std::cout << "--> xent: " << this_xent << std::endl;
      if (this_xent < round_best_xent) {
        std::cout << "--> round best!" << std::endl;
        round_best_xent = this_xent;
        best_addition = vp_ptr;
      }

      std::cout << std::endl;

      vp_stack.pop_back();
    }

    assert(best_addition != nullptr);

    double delta = prev_best_xent - round_best_xent;
    std::cout << "round delta: " << delta << " bits" << std::endl;
    std::cout << " round best: " << round_best_xent <<  " bits" << std::endl;
    std::cout << "    best vp: " << best_addition->vp_name() << std::endl;


    if (delta < eps_terminate) {
      std::cout << std::endl;
      std::cout << "Optimisation complete." << std::endl;
      std::cout << "Final system: " 
        << vps_to_string(vp_stack) << std::endl;
      std::cout << "Final xentropy: "
        << prev_best_xent << std::endl;

      return;
    }

    // take the best addition forward
    vp_stack.push_back(best_addition);
    std::cout << " new system: " << vps_to_string(vp_stack) 
      << std::endl << std::endl;
    prev_best_xent = round_best_xent;
  }
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

  double inter_steps = 1 + (max_inter - min_inter) / step;
  double intra_steps = 1 + (max_intra - min_intra) / step;
  unsigned int total_steps = ceil(inter_steps * intra_steps);
  unsigned int curr_step = 1;

  json entropy_values = json::array();

  for (double inter = min_inter; inter <= max_inter; inter += step) {
    json inner_values = json::array();
    for (double intra = min_intra; intra <= max_intra; intra += step) {
      std::cout 
        << std::endl
        << "Evaluating MVSs at (inter: " << inter << ", " 
        << "intra: " << intra << ")"
        << " - step " << (curr_step++) << "/" << total_steps
        << std::endl;

      auto all_measurements = evaluate(corpus, intra, inter, {&mvs});
      auto h_this_mvs = all_measurements.at(0);
      auto h_pitch = h_this_mvs.h_pitch;
      auto h_dur   = h_this_mvs.h_duration;
      auto h_rest  = h_this_mvs.h_rest;
      auto total = h_pitch + h_dur + h_rest;

      std::cout << "-->    Pitch entropy: " << h_pitch << std::endl;
      std::cout << "--> Duration entropy: " << h_dur << std::endl;
      std::cout << "-->     Rest entropy: " << h_rest << std::endl;
      std::cout << "-->    Total entropy: " << total << std::endl;

      inner_values.push_back(total);
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
              const QuantizedDuration &ts_dur,
              const std::string &json_fname) {
  ChoraleTimeSig timesig(ts_dur);
  std::cout 
    << "Generating piece of length " << len 
    << " in " << timesig << ".." << std::flush;
  auto piece = mvs.random_walk(len, ts_dur);
  std::cout << "done." << std::endl;

  std::cout << "Entropy of generated piece: " << std::endl << std::flush;
  auto pitch_entropy = mvs.avg_sequence_entropy<ChoralePitch>(piece);
  auto dur_entropy = mvs.avg_sequence_entropy<ChoraleDuration>(piece);
  auto rest_entropy = mvs.avg_sequence_entropy<ChoraleRest>(piece);
  std::cout << "-->    Pitch: " << pitch_entropy << std::endl;
  std::cout << "--> Duration: " << dur_entropy << std::endl;
  std::cout << "-->     Rest: " << rest_entropy << std::endl;
  std::cout << "-->    Total: "
    << (pitch_entropy + dur_entropy + rest_entropy) << std::endl;

  auto pitch_xents = mvs.cross_entropies<ChoralePitch>(piece);
  auto dur_xents   = mvs.cross_entropies<ChoraleDuration>(piece);
  auto rest_xents  = mvs.cross_entropies<ChoraleRest>(piece);

  auto pitch_dents = mvs.dist_entropies<ChoralePitch>(piece);
  auto dur_dents   = mvs.dist_entropies<ChoraleDuration>(piece);
  auto rest_dents  = mvs.dist_entropies<ChoraleRest>(piece);

  std::vector<double> total_xents(piece.size());
  std::vector<double> total_dents(piece.size());
  for (unsigned int i = 0; i < piece.size(); i++) {
    total_xents[i] = pitch_xents.at(i) + dur_xents.at(i) + rest_xents.at(i);
    total_dents[i] = pitch_dents.at(i) + dur_dents.at(i) + rest_dents.at(i);
  }

  std::map<std::string, std::vector<double>> entropy_map;
  entropy_map.insert({"cross_entropies", total_xents});
  entropy_map.insert({"dist_entropies", total_dents});

  render(piece, entropy_map, json_fname);
}

int main(void) {
  const QuantizedDuration three_four(12);
  const QuantizedDuration four_four(16);

  const unsigned int hist = 5;

  ChoraleMVS::GenVP<ChoralePitch> pitch_vp(hist);
  ChoraleMVS::GenVP<ChoraleDuration> duration_vp(hist);

  // inter-pitch crosses 
  ChoraleMVS::GenLinkedVP<ChoraleIntref, ChoraleInterval> 
    intref_p_seqint(hist);
  ChoraleMVS::GenLinkedVP<ChoraleInterval, ChoraleIntref> 
    seqint_p_intref(hist);

  ChoraleMVS::GenLinkedVP<ChoraleDuration, ChoraleInterval>
    dur_p_seqint(hist);
  ChoraleMVS::GenLinkedVP<ChoraleInterval, ChoraleDuration>
    seqint_p_dur(hist);

  ChoraleMVS::GenLinkedVP<ChoraleDuration, ChoraleIntref>
    dur_p_intref(hist);

  // pxd ~ pitch cross duration
  ChoraleMVS::GenLinkedVP<ChoralePitch, ChoraleDuration> 
    pitch_p_dur(hist);
  ChoraleMVS::GenLinkedVP<ChoraleDuration, ChoralePitch>
    dur_p_pitch(hist);

  // rest crossed with things
  ChoraleMVS::GenLinkedVP<ChoraleRest, ChoraleDuration>
    rest_p_dur(hist);
  ChoraleMVS::GenLinkedVP<ChoraleDuration, ChoraleRest>
    dur_p_rest(hist);

  ChoraleMVS::GenLinkedVP<ChoraleRest, ChoralePitch>
    rest_p_pitch(hist);

  ChoraleMVS::GenLinkedVP<ChoraleInterval, ChoraleRest>
    seqint_p_rest(hist);
  ChoraleMVS::GenLinkedVP<ChoraleRest, ChoraleInterval>
    rest_p_seqint(hist);

  ChoraleMVS::GenLinkedVP<ChoraleIntref, ChoraleRest>
    intref_p_rest(hist);
  ChoraleMVS::GenLinkedVP<ChoraleRest, ChoraleIntref>
    rest_p_intref(hist);

  ChoraleMVS::GenLinkedVP<ChoralePitch, ChoraleRest>
    pitch_p_rest(hist);
  ChoraleMVS::GenLinkedVP<ChoraleIntref, ChoraleDuration> 
    intref_p_dur(hist);

  // posinbar crossed with things
  ChoraleMVS::GenLinkedVP<ChoralePosinbar, ChoraleDuration>
    posinbar_p_dur(hist);

  ChoraleMVS::GenLinkedVP<ChoralePosinbar, ChoraleRest>
    posinbar_p_rest(hist);

  ChoraleMVS::GenLinkedVP<ChoralePosinbar, ChoraleIntref>
    posinbar_p_intref(hist);

  ChoraleMVS::GenLinkedVP<ChoralePosinbar, ChoraleInterval>
    posinbar_p_seqint(hist);

  ChoraleMVS::GenLinkedVP<ChoralePosinbar, ChoralePitch>
    posinbar_p_pitch(hist);

  // ioi cross with things
  ChoraleMVS::GenLinkedVP<ChoraleIOI, ChoralePitch>
    ioi_p_pitch(hist);

  ChoraleMVS::GenLinkedVP<ChoraleIOI, ChoraleDuration>
    ioi_p_dur(hist);

  ChoraleMVS::GenLinkedVP<ChoraleIOI, ChoraleRest>
    ioi_p_rest(hist);

  ChoraleMVS::GenLinkedVP<ChoraleIOI, ChoraleInterval>
    ioi_p_seqint(hist);

  ChoraleMVS::GenLinkedVP<ChoraleIOI, ChoraleIntref>
    ioi_p_intref(hist);

  // fib crossed with the predcitve types
  ChoraleMVS::GenLinkedVP<ChoraleFib, ChoralePitch>
    fib_p_pitch(hist);

  ChoraleMVS::GenLinkedVP<ChoraleFib, ChoraleInterval>
    fib_p_seqint(hist);

  ChoraleMVS::GenLinkedVP<ChoraleFib, ChoraleIntref>
    fib_p_intref(hist);

  ChoraleMVS::GenLinkedVP<ChoraleFib, ChoraleDuration>
    fib_p_dur(hist);

  ChoraleMVS::GenLinkedVP<ChoraleFib, ChoraleRest>
    fib_p_rest(hist);


  ChoraleMVS::GenVP<ChoraleRest> rest_vp(hist);
  ChoraleMVS::GenVP<ChoraleInterval> seqint_vp(hist);
  ChoraleMVS::GenVP<ChoraleIntref> intref_vp(hist);

  // triply-linked VPs
  ChoraleMVS::TripleLinkedVP<ChoralePosinbar, ChoraleRest, ChoraleDuration> 
   clock_duration(hist);

  ChoraleMVS::TripleLinkedVP<ChoralePosinbar, ChoraleDuration, ChoraleRest>
    clock_rest(hist);

  // fib triple links predcting duration
  ChoraleMVS::TripleLinkedVP<ChoraleFib, ChoraleRest, ChoraleDuration>
    fibxrest_p_dur(hist);
  ChoraleMVS::TripleLinkedVP<ChoraleFib, ChoraleIntref, ChoraleDuration>
    fibxintref_p_dur(hist);
  ChoraleMVS::TripleLinkedVP<ChoraleFib, ChoraleInterval, ChoraleDuration>
    fibxseqint_p_dur(hist);
  ChoraleMVS::TripleLinkedVP<ChoraleFib, ChoralePitch, ChoraleDuration>
    fibxpitch_p_dur(hist);
  ChoraleMVS::TripleLinkedVP<ChoraleFib, ChoraleIOI, ChoraleDuration>
    fibxioi_p_dur(hist);

  // fib triple links predicting rest
  ChoraleMVS::TripleLinkedVP<ChoraleFib, ChoraleDuration, ChoraleRest>
    fibxdur_p_rest(hist);
  ChoraleMVS::TripleLinkedVP<ChoraleFib, ChoraleIntref, ChoraleRest>
    fibxintref_p_rest(hist);
  ChoraleMVS::TripleLinkedVP<ChoraleFib, ChoraleInterval, ChoraleRest>
    fibxseqint_p_rest(hist);
  ChoraleMVS::TripleLinkedVP<ChoraleFib, ChoralePitch, ChoraleRest>
    fibxpitch_p_rest(hist);
  ChoraleMVS::TripleLinkedVP<ChoraleFib, ChoraleIOI, ChoraleRest>
    fibxioi_p_rest(hist);

  auto lt_config = MVSConfig::long_term_only(1.0);
  ChoraleMVS lt_only(lt_config);

  MVSConfig full_config;
  full_config.lt_history = hist;
  full_config.st_history = 3;
  full_config.enable_short_term = true;
  full_config.mvs_name = "full mvs for evaluation";
  full_config.intra_layer_bias = 0.0;
  full_config.inter_layer_bias = 0.0;

  // n.b. we don't need to add the base VPs of each type as the optimizer
  // includes these for us (it doesn't take them from the pool)
  MVSOptimizer optimizer(full_config);

  /*** pitch predictors into pool ***/

  // derived from pitch
  optimizer.add_to_pool(&seqint_vp);
  optimizer.add_to_pool(&intref_vp);
  // inter-pitch crosses
  optimizer.add_to_pool(&seqint_p_intref);
  optimizer.add_to_pool(&intref_p_seqint);
  // posinbar->(pitch type)
  optimizer.add_to_pool(&posinbar_p_intref);
  optimizer.add_to_pool(&posinbar_p_seqint);
  optimizer.add_to_pool(&posinbar_p_pitch);
  // dur->(pitch type)
  optimizer.add_to_pool(&dur_p_intref);
  optimizer.add_to_pool(&dur_p_seqint);
  optimizer.add_to_pool(&dur_p_pitch);
  // ioi->(pitch type)
  optimizer.add_to_pool(&ioi_p_intref);
  optimizer.add_to_pool(&ioi_p_seqint);
  optimizer.add_to_pool(&ioi_p_pitch);
  // rest->(pitch type)
  optimizer.add_to_pool(&rest_p_intref);
  optimizer.add_to_pool(&rest_p_seqint);
  optimizer.add_to_pool(&rest_p_pitch);
  // fib->(pitch type)
  optimizer.add_to_pool(&fib_p_intref);
  optimizer.add_to_pool(&fib_p_seqint);
  optimizer.add_to_pool(&fib_p_pitch);

  // TODO: also try dur and rest with clock VPs
  // this will be slow though
  //
  // may also want to consider triple-linking fib with existing linked vps since
  // its state space is so small that it won't hurt performance too much

  /*** duration predictors into pool ***/
  // (pitch type)->dur
  optimizer.add_to_pool(&pitch_p_dur);
  optimizer.add_to_pool(&intref_p_dur);
  optimizer.add_to_pool(&seqint_p_dur);
  // link-only types
  optimizer.add_to_pool(&posinbar_p_dur);
  optimizer.add_to_pool(&ioi_p_dur);
  optimizer.add_to_pool(&fib_p_dur);
  // rest
  optimizer.add_to_pool(&rest_p_dur);
  // triple fib links
  optimizer.add_to_pool(&fibxrest_p_dur);
  optimizer.add_to_pool(&fibxioi_p_dur);
  optimizer.add_to_pool(&fibxpitch_p_dur);
  optimizer.add_to_pool(&fibxseqint_p_dur);
  optimizer.add_to_pool(&fibxintref_p_dur);

  /*** rest predictors into pool ***/
  optimizer.add_to_pool(&pitch_p_rest);
  optimizer.add_to_pool(&intref_p_rest);
  optimizer.add_to_pool(&seqint_p_rest);
  // link-only types
  optimizer.add_to_pool(&posinbar_p_rest);
  optimizer.add_to_pool(&ioi_p_rest);
  optimizer.add_to_pool(&fib_p_rest);
  // dur
  optimizer.add_to_pool(&dur_p_rest);
  // fib triple links predicting rest
  optimizer.add_to_pool(&fibxdur_p_rest);
  optimizer.add_to_pool(&fibxioi_p_rest);
  optimizer.add_to_pool(&fibxpitch_p_rest);
  optimizer.add_to_pool(&fibxseqint_p_rest);
  optimizer.add_to_pool(&fibxintref_p_rest);

  corpus_t train_corp;
  corpus_t test_corp;
  parse("corpus/fixed_rests_t5.json", train_corp, test_corp);

  double eps_terminate = 0.001;
  optimizer.optimize<ChoraleRest>(eps_terminate, train_corp, test_corp);

  /*
  ChoraleMVS full_mvs(full_config);
  full_mvs.add_viewpoint(&pitch_vp);
  full_mvs.add_viewpoint(&duration_vp);
  full_mvs.add_viewpoint(&rest_vp);

  train(train_corp, {&full_mvs});

  double max_intra = 0.0;
  double max_inter = 0.0;
  double step = 0.25;
  bias_grid_sweep(test_corp, full_mvs, max_intra, max_inter, step);
  */

  /*
  // n.b. foe => first-order effectiveness
  // foe taken with (intra_layer 0.25, inter_layer 0.5, lt_hist 5, st_hist 3)
  for (auto mvs_ptr : {&lt_only, &full_mvs}) {
    // good system: 
    // pitch: { pitch, seqint dur->seqint, dur->pitch, pos->iref, pos->pitch }
    //   dur: { dur, pitch->dur, pos->dur, seqint->dur }

    // *** pitch predictors (primitive):
    mvs_ptr->add_viewpoint(&pitch_vp); // benchmark: pitch alone => 2.30912
    //mvs_ptr->add_viewpoint(&seqint_vp); // foe: 0.0939
    //mvs_ptr->add_viewpoint(&intref_vp); // foe: 0.05249
    
    // *** pitch  predictors (linked):
    mvs_ptr->add_viewpoint(&posinbar_p_intref); // foe: 0.20506
    //mvs_ptr->add_viewpoint(&posinbar_p_seqint); // foe: 0.2051
    mvs_ptr->add_viewpoint(&dur_p_seqint); // foe: 0.20356
    //mvs_ptr->add_viewpoint(&dur_p_pitch); // foe: 0.14405 (v good)
    mvs_ptr->add_viewpoint(&intref_p_seqint); // foe: 0.14133 (v good)
    mvs_ptr->add_viewpoint(&posinbar_p_pitch); // foe: 0.13485 (good)
    //mvs_ptr->add_viewpoint(&ioi_p_pitch); // foe: 0.0872
    //mvs_ptr->add_viewpoint(&seqint_p_intref); // foe: 0.07033

    // *** duration predictors
    mvs_ptr->add_viewpoint(&duration_vp); // benchmark: dur alone => 0.957828
    //mvs_ptr->add_viewpoint(&ioi_p_dur); // foe: 0.005793
    //mvs_ptr->add_viewpoint(&pitch_p_dur); // foe: 0.032973
    //mvs_ptr->add_viewpoint(&posinbar_p_dur); // foe: 0.093191
    //mvs_ptr->add_viewpoint(&clock_duration); // foe: 0.101675
    //mvs_ptr->add_viewpoint(&rest_p_dur); // foe: 0.014745
    //mvs_ptr->add_viewpoint(&intref_p_dur); // foe: 0.016286
    //mvs_ptr->add_viewpoint(&seqint_p_dur); // foe: -0.002191

    // *** rest predictors
    // { clock->rest, posinbar->rest, dur->rest }: 0.021333 effectiveness
    // { clock->rest, posinbar->rest, dur->rest, intref->rest }: 0.021076 :/
    // { dur->rest, posinbar->rest, intref->rest }: 0.018142 effectiveness
    // { posinbar->rest, dur->rest }: 0.01764 effectiveness
    mvs_ptr->add_viewpoint(&rest_vp); // benchmark: rest alone => 0.13079
    //mvs_ptr->add_viewpoint(&clock_rest); // foe: 0.019662
    //mvs_ptr->add_viewpoint(&dur_p_rest); // foe: 0.014718
    //mvs_ptr->add_viewpoint(&posinbar_p_rest); // foe: 0.012614
    //mvs_ptr->add_viewpoint(&intref_p_rest); // foe: 0.010797
    
    // *** rest predictors (not very useful)
    //mvs_ptr->add_viewpoint(&pitch_p_rest); // foe: 0.006659
    //mvs_ptr->add_viewpoint(&ioi_p_rest); // foe: -0.060714 (bad)
    //mvs_ptr->add_viewpoint(&seqint_p_rest); // foe: -0.116818 (bad!)
  }

  std::cout << "Training... " << std::flush;
  train(train_corp, {&lt_only, &full_mvs});
  std::cout << "done." << std::endl;

  double max_intra = 0.0;
  double max_inter = 0.0;
  double step = 0.25;
  bias_grid_sweep(test_corp, full_mvs, max_intra, max_inter, step);
  */
  
  //generate(full_mvs, 42, four_four, "out/gend.json");
}


