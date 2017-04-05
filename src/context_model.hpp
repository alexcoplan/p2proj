#ifndef AJC_HGUARD_CTXMODEL
#define AJC_HGUARD_CTXMODEL

#include <vector>
#include <list>
#include <utility>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <bitset>

typedef std::pair<unsigned, std::list<unsigned int>> Ngram;

struct GraphWriter {
  std::string node_decls;
  std::string edge_list;
  std::string (*decoder)(unsigned int);

  GraphWriter(std::string (*decode_fn)(unsigned int)) :
    decoder(decode_fn) {}
};


template<int b>
struct TrieNode {
  TrieNode *children[b];
  std::bitset<b> child_mask; // 1 where we have a child, 0 elsewhere
  TrieNode *parent;
  unsigned int count;

  TrieNode();
  ~TrieNode();
  TrieNode(const TrieNode &other);
  void get_ngrams(const unsigned int n, std::list<Ngram> &result);
  void write_latex(const std::string &fname, 
      std::string (*decoder)(unsigned int)) const;
  void gen_graphviz(std::string id_prefix, 
      std::string lab_prefix, GraphWriter &gw) const;
  void debug_summary();
};

template<int b>
class ContextModel {
  TrieNode<b> trie_root;
  unsigned int history;
  void addOrIncrement(const std::vector<unsigned int> &seq, 
                      const size_t i_begin, const size_t i_end);
  const TrieNode<b> *match_context(const std::vector<unsigned int> &seq, 
                                   const unsigned int i_start,
                                   const unsigned int i_end,
                                         unsigned int &i_matched) const;
  TrieNode<b> *match_context(const std::vector<unsigned int> &seq,
                             const unsigned int i_start,
                             const unsigned int i_end,
                                   unsigned int &i_matched);

  double ppm_a(const std::vector<unsigned int> &seq,
               const unsigned int i_start, 
               const unsigned int i_end,
               const std::bitset<b> &dead) const;

public:
  void set_history(unsigned int h);
  unsigned int get_history() const { return history; }
  void learn_sequence(const std::vector<unsigned int> &seq);
  void update_from_tail(const std::vector<unsigned int> &seq);
  void get_ngrams(const unsigned int n, std::list<Ngram> &result);
  unsigned int count_of(const std::vector<unsigned int> &seq) const;
  unsigned int count_of(const std::vector<unsigned int> &seq);
  double probability_of(const std::vector<unsigned int> &seq) const;
  double avg_sequence_entropy(const std::vector<unsigned int> &seq) const;
  void write_latex(const std::string &fname, 
      std::string (*decoder)(unsigned int)) const;
  void debug_summary();
  void clear_model(); // unlearn everything so far

  ContextModel(unsigned int history);
};

/**************************************************
 * ContextModel: public methods
 **************************************************/

template<int b>
ContextModel<b>::ContextModel(unsigned int h) : history(h) {}

template<int b>
void ContextModel<b>::debug_summary() {
  trie_root.debug_summary();
}

template<int b>
void ContextModel<b>::write_latex(const std::string &fname,
    std::string (*decoder)(unsigned int)) const {
  trie_root.write_latex(fname, decoder);
}

template<int b>
void ContextModel<b>::set_history(unsigned int h) {
  history = h;
}

template<int b>
void ContextModel<b>::clear_model() {
  for (unsigned int i = 0; i < b; i++) {
    if (trie_root.children[i] != nullptr) {
      delete trie_root.children[i];
      trie_root.children[i] = nullptr;
    }
  }

  trie_root.count = 0;
  trie_root.child_mask.reset();
}

template<int b>
unsigned int 
ContextModel<b>::count_of(const std::vector<unsigned int> &seq) const {
  const TrieNode<b> *node = &trie_root;
  for (auto event : seq) {
    if (node->children[event] == nullptr)
      return 0;
    node = node->children[event];
  }

  return node->count;
}

template<int b>
unsigned int ContextModel<b>::count_of(const std::vector<unsigned int> &seq) {
  return const_cast<const ContextModel<b> *>(this)->count_of(seq);
}

/* Public wrapper to calculate probability of n-gram */
template<int b> double
ContextModel<b>::probability_of(const std::vector<unsigned int> &seq) const {
  unsigned int i_begin = (seq.size() > history) ? seq.size() - history : 0;
  return ppm_a(seq, i_begin, seq.size() - 1, std::bitset<b>());
}

template<int b>
double ContextModel<b>::
avg_sequence_entropy(const std::vector<unsigned int> &seq) const {
  assert(seq.size() > 0);

  double total_entropy = 0.0;
  auto ramp = std::min(seq.size(), (size_t)history);

  for (unsigned int i_end = 0; i_end < ramp; i_end++) {
    double prob = ppm_a(seq, 0, i_end, std::bitset<b>());
    total_entropy -= std::log2(prob);
  }

  for (unsigned int i_end = history; i_end < seq.size(); i_end++) {
    auto i_begin = i_end - history + 1;
    double prob = ppm_a(seq, i_begin, i_end, std::bitset<b>());
    total_entropy -= std::log2(prob);
  }

  return total_entropy / static_cast<double>(seq.size()); 
}

/**************************************************
 * ContextModel: private methods
 **************************************************/

/** Find the TrieNode corresponding to a given context in the trie
 *
 * @param i_start: index in seq marking the start of the context
 * @param i_matched: set to the lowest i with i >= i_start s.t. the context
 *  e_i^(n-1) is successfully matched
 *
 * @return pointer to the node corresponding to the matched context */
template<int b> const TrieNode<b> *
ContextModel<b>::match_context(const std::vector<unsigned int> &seq,
                               const unsigned int i_start,
                               const unsigned int i_end,
                                     unsigned int &i_matched) const {
  assert(i_end <= seq.size() - 1);

  const TrieNode<b> *node = &trie_root;

  for (unsigned int i = i_start; i < i_end; i++) {
    node = &trie_root;

    unsigned int j = i;
    for (; j < i_end; j++) {
      unsigned int event = seq[j];
      if (node->children[event] == nullptr)
        break;

      node = node->children[event];
    }

    // if we matched the entire context from i to i_end
    // then we're done. 
    if (j == i_end) {
      i_matched = i;
      return node;
    }
  }

  // if we come out of the above loop then we didn't match anything :(
  i_matched = i_end;
  return node;
}

template<int b> TrieNode<b> *
ContextModel<b>::match_context(const std::vector<unsigned int> &seq,
                               const unsigned int i_start,
                               const unsigned int i_end,
                                     unsigned int &i_matched) {
  return const_cast<const TrieNode<b> *>(this)
          ->match_context(seq, i_start, i_end, i_matched);
}

/* Calculate probability of sequence using PPM method A */
template<int b> double 
ContextModel<b>::ppm_a(const std::vector<unsigned int> &seq, 
                       const unsigned int ctx_start,
                       const unsigned int ctx_end,
                       const std::bitset<b> &dead) const {
  // base case: use uniform distribution
  if (ctx_start > ctx_end) {
    assert(!dead.all());
    return 1.0 / (double)(b - dead.count());
  }

  unsigned int i_matched;
  const TrieNode<b> *ctx_node = 
    match_context(seq, ctx_start, ctx_end, i_matched);
  // we matched the context e_{i_matched}^{n-1}

  int sum = 0;
  std::bitset<b> seen_or_dead = ctx_node->child_mask | dead;
  std::bitset<b> novel_events = ~seen_or_dead;
  std::bitset<b> known_events = ctx_node->child_mask & ~dead;

  for (unsigned int i = 0; i < b; i++) {
    if (known_events[i])
      sum += ctx_node->children[i]->count;
  }

  double known_total = (double)sum;

  unsigned int event = seq[ctx_end];
  if (novel_events.any()) {
    if (known_events[event])
      return (double)(ctx_node->children[event]->count) / (1.0 + known_total);
    return ppm_a(seq, i_matched+1, ctx_end, seen_or_dead) / (1.0 + known_total);
  }

  // no novel events, so don't include escape probability
  return (double)(ctx_node->children[event]->count) / known_total;
}

// begin is inclusive, end is exclusive
template<int b>
void ContextModel<b>::addOrIncrement(const std::vector<unsigned int> &seq, 
                                     const size_t i_begin, 
                                     const size_t i_end) {
  TrieNode<b> *node = &trie_root;

  for (size_t i = i_begin; i < i_end; i++) {
    unsigned int event = seq[i];
    if (node->children[event] == nullptr) {
      node->children[event] = new TrieNode<b>();
      node->children[event]->parent = node;
      node->child_mask.set(event);
    }

    node = node->children[event];
  }
  
  node->count++;
}

template<int b>
void ContextModel<b>::learn_sequence(const std::vector<unsigned int> &seq) {
  // We train the context model by passing a window of size h over the training
  // sequence, and generating examples from the subsequence lying under the
  // window. 
  //
  // Note that the window initially starts off to the left of the sequence. To
  // implement this, we have a separate loop to handle this "ramp-up" phase.
  
  // ramp-up phase
  for (size_t cap = 1; cap < history; cap++) 
    for (size_t beg = 0; beg <= cap; beg++)
      addOrIncrement(seq, beg, cap);
    
  // main loop
  for (size_t end = history; end <= seq.size(); end++) 
    for (size_t beg = end - history; beg <= end; beg++) 
      addOrIncrement(seq, beg, end);
}

// takes h-, (h-1)-, ..., 1-grams from the end of a sequence
// and updates the context model with them.
//
// this is used for models which are dynamically trained on a sequence which is
// continually growing (such as the short-term model in a MVS)
template<int b> 
void ContextModel<b>::update_from_tail(const std::vector<unsigned int> &seq) {
  size_t pos = seq.size() >= history ? (seq.size() - history) : 0;
  for (; pos <= seq.size(); pos++) 
    addOrIncrement(seq, pos, seq.size());
}

template<int b> void
ContextModel<b>::get_ngrams(const unsigned int n, std::list<Ngram> &result) {
  trie_root.get_ngrams(n, result);
}

// TrieNode implementation

template<int b>
void TrieNode<b>::debug_summary() {
  std::cout << "TrieNode summary:" << std::endl;
  std::cout << "root count: " << count << std::endl;

  for (unsigned int i = 0; i < b; i++) {
    std::cout << i << ": " << (children[i]->count) << std::endl;
  }
}

template<int b>
TrieNode<b>::TrieNode() : 
  parent(nullptr), count(0) {
  for (unsigned int i = 0; i < b; i++) {
    children[i] = nullptr;
  }
}

template<int b>
TrieNode<b>::~TrieNode() {
  for (unsigned int i = 0; i < b; i++) 
    if (children[i] != nullptr)
      delete children[i];
}

template<int b>
TrieNode<b>::TrieNode(const TrieNode &other) {
  parent = nullptr;
  count = other.count;
  child_mask = other.child_mask;
  for (unsigned int i = 0; i < b; i++)  {
    if (other.children[i] != nullptr) {
      children[i] = new TrieNode<b>(*other.children[i]);
      children[i]->parent = this;
    }
    else
      children[i] = nullptr;
  }
}

template<int b>
void TrieNode<b>::get_ngrams(const unsigned int n, std::list<Ngram> &result) {
  assert(n > 0);
  if (n == 1) {
    for (unsigned int i = 0; i < b; i++) {
      if (children[i] == nullptr)
        continue;

      std::list<unsigned int> ngram; 
      ngram.push_back(i);
      result.push_back(
        Ngram(children[i]->count, ngram)
      );
    }
    return;
  } 

  for (unsigned int i = 0; i < b; i++) {
    if (children[i] == nullptr)
      continue;

    std::list<std::pair<unsigned int, std::list<unsigned int>>> child_ngrams;
    children[i]->get_ngrams(n-1, child_ngrams);
    for (auto sub_ngram : child_ngrams) {
      sub_ngram.second.push_front(i);
      result.push_back(sub_ngram);
    }
  }
}

/* GraphViz generation for visualising Tries */

template<int b>
void TrieNode<b>::gen_graphviz(
    std::string id_prefix, std::string lab_prefix, GraphWriter &gw) const {
  TrieNode *child = nullptr;
  for (unsigned int i = 0; i < b; i++) {
    child = children[i];
    if (child != nullptr) {
      // construct human-readable label
      std::string pretty_str = gw.decoder(i);
      std::string child_label = lab_prefix + pretty_str;

      std::string code_str = std::to_string(i); 
      std::string this_id = (id_prefix.length() == 0 ? "root" : id_prefix);
      std::string child_id = (id_prefix.length() == 0) ?
          "n" + code_str : id_prefix + "_" + code_str;
      gw.node_decls += child_id + " [label=\"" + child_label + ":" +
        std::to_string(child->count) + "\"];\n";
      gw.edge_list += this_id + " -> " + child_id + " [label=\"" + pretty_str
        + "\"];\n";

      child->gen_graphviz(child_id, child_label, gw);
    }
  }
}

template<int b>
void TrieNode<b>::write_latex(const std::string &fname, 
    std::string (*decode)(unsigned int)) const {
  GraphWriter gw(decode);

  gw.node_decls += "root [label=\"():" + std::to_string(count) + "\"];\n";
  gen_graphviz("", "", gw);

  std::ofstream texfile;
  texfile.open(fname);
  texfile << "\\documentclass[11pt]{article}" << std::endl;
  texfile << "\\usepackage[active,tightpage]{preview}" << std::endl;
  texfile << "\\usepackage{fontspec}" << std::endl;
  texfile << "\\usepackage{lilyglyphs}" << std::endl;
  texfile << "\\newcommand{\\flatten}[1]{#1\\hspace{0.08em}\\flat{}}" 
    << std::endl;
  texfile << "\\setlength\\PreviewBorder{10pt}" << std::endl;
  texfile << "\\usepackage{tikz}" << std::endl;
  texfile << "\\usetikzlibrary{shapes,arrows}" << std::endl;
  texfile << "\\usepackage{dot2texi}" << std::endl;
  texfile << "\\begin{document}" << std::endl;
  texfile << "\\begin{preview}" << std::endl;
  texfile << "\\begin{tikzpicture}[>=latex']" << std::endl;
  texfile << "\\tikzstyle{n} = [shape=rectangle]" << std::endl;
  texfile 
    << "\\begin{dot2tex}[dot,tikz,codeonly,autosize,options="
    << "-traw --tikzedgelabels]" 
    << std::endl;
  texfile << "digraph G {" << std::endl;
  texfile << "node [style=\"n\"];" << std::endl;
  texfile << "edge [lblstyle=\"auto\"]" << std::endl;
  texfile << gw.node_decls << std::endl << gw.edge_list;
  texfile << "}" << std::endl;
  texfile << "\\end{dot2tex}" << std::endl;
  texfile << "\\end{tikzpicture}" << std::endl;
  texfile << "\\end{preview}" << std::endl;
  texfile << "\\end{document}" << std::endl;
  texfile.close();
}

#endif // header guard
