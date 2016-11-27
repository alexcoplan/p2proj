#include <cstddef>
#include <vector>
#include <list>
#include <utility>
#include <cassert>
#include <iostream>

template<int b>
struct TrieNode {
  TrieNode *children[b];
  unsigned int count;

  TrieNode();
  ~TrieNode();
  void get_ngrams(const unsigned int n, std::list<std::pair<unsigned int, std::list<unsigned int>>> &result);
  void debug_summary();
};

template<int b>
class ContextModel {
  TrieNode<b> trie_root;
  unsigned int history;

public:
  void addOrIncrement(const std::vector<unsigned int> &seq, const size_t i_begin, const size_t i_end);
  void learnSequence(const std::vector<unsigned int> &seq);
  void get_ngrams(const unsigned int n, std::list<std::pair<unsigned int, std::list<unsigned int>>> &result);
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

// begin is inclusive, end is exclusive
template<int b>
void ContextModel<b>::addOrIncrement(const std::vector<unsigned int> &seq, const size_t i_begin, const size_t i_end) {
  TrieNode<b> *node = &trie_root;

  for (size_t i = i_begin; i < i_end; i++) {
    unsigned int event = seq[i];
    if (node->children[event] == NULL) 
      node->children[event] = new TrieNode<b>();

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
ContextModel<b>::get_ngrams(const unsigned int n, 
     std::list<std::pair<unsigned int, std::list<unsigned int>>> &result) {
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
TrieNode<b>::TrieNode() : count(0) {
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
void TrieNode<b>::get_ngrams(const unsigned int n, 
     std::list<std::pair<unsigned int, std::list<unsigned int>>> &result) {
  assert(n > 0);
  if (n == 1) {
    for (unsigned int i = 0; i < b; i++) {
      if (children[i] == NULL)
        continue;

      std::list<unsigned int> ngram; 
      ngram.push_back(i);
      result.push_back(std::pair<unsigned int, std::list<unsigned int>>(children[i]->count, ngram));
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
