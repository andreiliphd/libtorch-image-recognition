#include <string>
#include <vector>

auto ReadCsv(std::string &location) -> std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>>;