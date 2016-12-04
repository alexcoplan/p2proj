#include <cstddef>
#include <vector>
#include <list>
#include <utility>
#include <cassert>
#include <iostream>
#include <fstream>

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
  void get_ngrams(const unsigned int n, std::list<Ngram> &result);
  void write_graphviz(const std::string &fname, 
      std::string (*decoder)(unsigned int));
  void gen_graphviz(std::string prefix, GraphWriter &gw);
  void debug_summary();
};

template<int b>
class ContextModel {
  TrieNode<b> trie_root;
  unsigned int history;
  void addOrIncrement(const std::vector<unsigned int> &seq, 
                      const size_t i_begin, const size_t i_end);
  TrieNode<b> *match_context(const std::vector<unsigned int> &seq, 
                             const unsigned int i_start,
                                   unsigned int &i_matched);
  double ppm_a(const std::vector<unsigned int> &seq,
               const unsigned int i_start, const std::bitset<b> &dead);

public:
  void learnSequence(const std::vector<unsigned int> &seq);
  void get_ngrams(const unsigned int n, std::list<Ngram> &result);
  unsigned int count_of(const std::vector<unsigned int> &seq);
  double probability_of(const std::vector<unsigned int> &seq);
  void write_graphviz(const std::string &fname, 
      std::string (*decoder)(unsigned int));
  void debug_summary();

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
void ContextModel<b>::write_graphviz(const std::string &fname,
    std::string (*decoder)(unsigned int)) {
  trie_root.write_graphviz(fname, decoder);
}

template<int b>
unsigned int ContextModel<b>::count_of(const std::vector<unsigned int> &seq) {
  TrieNode<b> *node = &trie_root;
  for (auto event : seq) {
    if (node->children[event] == NULL)
      return 0;
    node = node->children[event];
  }

  return node->count;
}

/* Public wrapper to calculate probability of n-gram */
template<int b> double
ContextModel<b>::probability_of(const std::vector<unsigned int> &seq) {
  return ppm_a(seq, 0, std::bitset<b>());
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
template<int b> TrieNode<b> *
ContextModel<b>::match_context(const std::vector<unsigned int> &seq,
                               const unsigned int i_start,
                                     unsigned int &i_matched) {
  TrieNode<b> *node = &trie_root;

  for (unsigned int i = i_start; i < seq.size() - 1; i++) {
    node = &trie_root;

    unsigned int j = i;
    for (; j < seq.size() - 1; j++) {
      unsigned int event = seq[j];
      if (node->children[event] == NULL)
        break;

      node = node->children[event];
    }

    // if we matched the entire context from i to n-1
    // then we're done. 
    if (j == seq.size() - 1) {
      i_matched = i;
      return node;
    }
  }

  // if we come out of the above loop then we didn't match anything :(
  i_matched = seq.size() - 1;
  return node;
}

/* Calculate probability of sequence using PPM method A */
template<int b> double 
ContextModel<b>::ppm_a(const std::vector<unsigned int> &seq, 
                       const unsigned int ctx_start,
                       const std::bitset<b> &dead) {
  // base case: use uniform distribution
  if (ctx_start == seq.size()) {
    assert(!dead.all());
    return 1.0 / (double)(b - dead.count());
  }

  unsigned int i_matched;
  TrieNode<b> *ctx_node = match_context(seq, ctx_start, i_matched);
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

  unsigned int event = seq[seq.size() - 1];
  if (novel_events.any()) {
    if (known_events[event])
      return (double)(ctx_node->children[event]->count) / (1.0 + known_total);
    return ppm_a(seq, i_matched+1, seen_or_dead) / (1.0 + known_total);
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
    if (node->children[event] == NULL) {
      node->children[event] = new TrieNode<b>();
      node->children[event]->parent = node;
      node->child_mask.set(event);
    }

    node = node->children[event];
  }
  
  node->count++;
}

template<int b>
void ContextModel<b>::learnSequence(const std::vector<unsigned int> &seq) {
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
  parent(NULL), count(0) {
  for (unsigned int i = 0; i < b; i++) {
    children[i] = NULL;
  }
}

template<int b>
TrieNode<b>::~TrieNode() {
  for (unsigned int i = 0; i < b; i++) 
    if (children[i] != NULL)
      delete children[i];
}

template<int b>
void TrieNode<b>::get_ngrams(const unsigned int n, std::list<Ngram> &result) {
  assert(n > 0);
  if (n == 1) {
    for (unsigned int i = 0; i < b; i++) {
      if (children[i] == NULL)
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
    if (children[i] == NULL)
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
void TrieNode<b>::gen_graphviz(std::string prefix, GraphWriter &gw) {
  TrieNode *child = NULL;
  for (unsigned int i = 0; i < b; i++) {
    child = children[i];
    if (child != NULL) {
      std::string symbol = gw.decoder(i);
      std::string this_node = (prefix.length() == 0 ? "root" : prefix);
      std::string child_name = prefix + symbol;
      gw.node_decls += child_name + " [label=\"" + child_name + ":" +
        std::to_string(child->count) + "\"];\n";
      gw.edge_list += this_node + " -> " + child_name + " [label=\"" + symbol
        + "\"];\n";

      child->gen_graphviz(child_name, gw);
    }
  }
}

template<int b>
void TrieNode<b>::write_graphviz(const std::string &fname, 
    std::string (*decode)(unsigned int)) {
  GraphWriter gw(decode);

  gw.node_decls += "root [label=\"():" + std::to_string(count) + "\"];\n";
  gen_graphviz("", gw);

  std::ofstream gvfile;
  gvfile.open(fname);
  gvfile << "digraph G {" << std::endl;
  gvfile << gw.node_decls << std::endl << gw.edge_list;
  gvfile << "}" << std::endl;
  gvfile.close();
}
