#include "context_model.hpp"

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

std::string decode_to_str(const unsigned int x) {
  return std::string(1, decode(x));
}

std::vector<unsigned int> encode_string(const std::string &str) {
  const std::vector<char> chars(str.begin(), str.end());
  std::vector<unsigned int> encoded(chars.size());
  std::transform(chars.begin(), chars.end(), encoded.begin(),
      [](char c) { return encode(c); });
  return encoded;
}

int main(void)
{
  ContextModel<4> model(3);
  model.learn_sequence( encode_string("GGDBAGGABA") );
  model.write_latex("out/gout.gv", decode_to_str);
}
