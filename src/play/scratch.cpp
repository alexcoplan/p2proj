#include <iostream>
#include <fstream>
#include "json.hpp"

using json = nlohmann::json;

int main(void) {
  std::ifstream corpus_file("corpus/chorale_dataset.json");
  json j;
  corpus_file >> j;

  json first_chorale = j["corpus"][0];
  std::cout << "Chorale title: " << first_chorale["title"] << std::endl;
}
