#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace pioneerml::io {

class JsonReader {
 public:
  nlohmann::json ReadFile(const std::string& filepath) const;
};

}  // namespace pioneerml::io
