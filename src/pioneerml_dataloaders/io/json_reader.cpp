#include "pioneerml_dataloaders/io/json_reader.h"

#include <fstream>
#include <stdexcept>

namespace pioneerml::io {

nlohmann::json JsonReader::ReadFile(const std::string& filepath) const {
  std::ifstream in(filepath);
  if (!in.is_open()) {
    throw std::runtime_error("JsonReader: could not open file: " + filepath);
  }

  nlohmann::json j;
  try {
    in >> j;
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("JsonReader: parse error: ") + e.what());
  }
  return j;
}

}  // namespace pioneerml::io
