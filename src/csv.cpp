#include <sstream>
#include <fstream>
#include <tuple>

#include "csv.h"

auto ReadCsv(std::string &location) -> std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>>
{

  std::fstream in(location, std::ios::in);
  std::string line;
  std::string name;
  std::string label;
  std::vector<std::tuple<std::string, int64_t>> csv;

  while (getline(in, line))
  {
    std::stringstream s(line);
    getline(s, name, ',');
    getline(s, label, ',');

    csv.push_back(std::make_tuple("../" + name, std::stoi(label)));
  }

  return csv;
}
