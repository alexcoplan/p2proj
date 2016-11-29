#include <cstddef>
#include <vector>
#include <list>
#include <utility>
#include <cassert>
#include <iostream>

typedef std::pair<unsigned, std::list<unsigned int>> Ngram;

template<int b>
struct TrieNode {
  TrieNode *children[b];
  TrieNode *parent;
  unsigned int count;
  unsigned int child_sum;
  unsigned int num_children; // used in PPM: need # of novel child events

  TrieNode();
  ~TrieNode();
  void get_ngrams(const unsigned int n, std::list<Ngram> &result);
  void debug_summary();
};

template<int b>
class ContextModel {
  TrieNode<b> trie_root;
  unsigned int history;
  void addOrIncrement(const std::vector<unsigned int> &seq, 
                      const size_t i_begin, const size_t i_end);
  double calculate_probability(const std::vector<unsigned int> &seq,
                               const unsigned int context);

public:
  void learnSequence(const std::vector<unsigned int> &seq);
  void get_ngrams(const unsigned int n, std::list<Ngram> &result);
  unsigned int count_of(const std::vector<unsigned int> &seq);
  double probability_of(const std::vector<unsigned int> &seq);
  void debug_summary();

  ContextModel(unsigned int history);
};

// ContextModel implementation

template<int b>
ContextModel<b>::ContextModel(unsigned int h) : history(h) {}

template<int b>
void ContextModel<b>::debug_summary() {
  trie_root.debug_summary();
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

/* Calculate probability of sequence using PPM method A */
template<int b> double 
ContextModel<b>::calculate_probability(const std::vector<unsigned int> &seq, 
                                       const unsigned int context) {
  assert(seq.size() > 0);

  TrieNode<b> *node = &trie_root;
  unsigned int i = 0;
  for (; i < context; i++) {
    unsigned int event = seq[i];
    if (node->children[event] == NULL) {
      i++;
      break;
    }
    node = node->children[event];
  }

  // we have matched i context events
  // due to PPM, any context that was not matched is irrelevant
  // since the counts were effectively zero, so we multiply with
  // an escape probability of one. Thus, it suffices to calculate
  // p(e' | e_1^i)
  unsigned int target = seq[seq.size() - 1];
  TrieNode<b> *child = node->children[target];
  if (child == NULL) {
    if (i == 0) { // no context
      if (node->count == 0)
        return 1.0 / (double)b; // PPM base case: blend with 1/b
      
      // divide escape probability of root equally among novel events
      double p_escape = 1.0 / (1.0 + (double)node->count);
      return p_escape / ((double)(b - node->num_children));
    }

    // event is novel for this context
    int delta_c = node->count - node->child_sum;
    assert(delta_c >= 0);

    double p_escape = (1.0 + (double)delta_c) / (1.0 + (double)node->count);
    return p_escape * calculate_probability(seq, i-1);
  }

  // event is known for this context
  return (double)(child->count) / (1.0 + (double)node->count);
}

/* Public wrapper to calculate probability of n-gram */
template<int b> double
ContextModel<b>::probability_of(const std::vector<unsigned int> &seq) {
  return calculate_probability(seq, seq.size() - 1);
}

// begin is inclusive, end is exclusive
template<int b>
void ContextModel<b>::addOrIncrement(const std::vector<unsigned int> &seq, 
    const size_t i_begin, const size_t i_end) {
  TrieNode<b> *node = &trie_root;

  for (size_t i = i_begin; i < i_end; i++) {
    unsigned int event = seq[i];
    if (node->children[event] == NULL) {
      node->children[event] = new TrieNode<b>();
      node->children[event]->parent = node;
      node->num_children++;
    }

    node = node->children[event];
  }
  
  // we check for null parent in case this is the root
  if (node->parent != NULL)
    node->parent->child_sum++;

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
TrieNode<b>::TrieNode() : count(0), child_sum(0), num_children(0) {
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

