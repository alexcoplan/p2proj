#include <cassert>
#include <string>
#include <iostream>
#include "context_model.hpp"

#define HISTORY 3
#define NUM_NOTES 4

unsigned int encode(const char c) {
  switch(c) {
    case 'G' :
      return 0;
    case 'A' :
      return 1;
    case 'B' : 
      return 2;
    case 'D' : 
      return 3;
    default :
      assert(! "Bad character");
      return 4;
  }
}

char decode(const unsigned int x) {
  assert(x < 4);
  char a[4] = { 'G', 'A', 'B', 'D' };
  return a[x];
}

int main()
{
  const std::string input = "GGDBAGGABA";
  const std::vector<char> chars(input.begin(), input.end());
  for (auto c : chars)
    std::cout << c;
  
  std::cout << std::endl;

  std::vector<unsigned int> nums(chars.size());
  std::transform(chars.begin(), chars.end(), nums.begin(), [](char c) { return encode(c); });

  std::cout << "Converted: ";
  for (auto n : nums) 
    std::cout << n;

  std::cout << std::endl;

  ContextModel<NUM_NOTES> model(HISTORY);
  model.learnSequence(nums);

  std::list<std::pair<unsigned int, std::list<unsigned int>>> ngrams;
  model.get_ngrams(2, ngrams);
  std::cout << ngrams.size() << std::endl;
  for (auto ngram : ngrams) {
    for (auto x : ngram.second) 
      std::cout << decode(x);
    std::cout << ": " << ngram.first << std::endl;
  }
}
