#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace pioneerml::utils::json {

class JsonUtils {
 public:
  static void RequireObject(const nlohmann::json& value, const std::string& context);
  static const nlohmann::json& RequireArrayField(
      const nlohmann::json& value,
      const std::string& field_name,
      const std::string& context);
  static void ValidateAllowedKeys(
      const nlohmann::json& object,
      const std::vector<std::string>& allowed_keys,
      const std::string& context);
  static std::string RequireStringFieldAnyOf(
      const nlohmann::json& object,
      const std::vector<std::string>& keys,
      const std::string& context,
      const std::string& required_name);
  static bool TryGetStringFieldAnyOf(
      const nlohmann::json& object,
      const std::vector<std::string>& keys,
      std::string* out_value);
};

}  // namespace pioneerml::utils::json
