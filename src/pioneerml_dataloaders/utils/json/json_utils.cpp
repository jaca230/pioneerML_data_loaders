#include "pioneerml_dataloaders/utils/json/json_utils.h"

#include <stdexcept>

namespace pioneerml::utils::json {

void JsonUtils::RequireObject(const nlohmann::json& value, const std::string& context) {
  if (!value.is_object()) {
    throw std::runtime_error(context + " must be a JSON object.");
  }
}

const nlohmann::json& JsonUtils::RequireArrayField(const nlohmann::json& value,
                                                   const std::string& field_name,
                                                   const std::string& context) {
  if (!value.contains(field_name) || !value.at(field_name).is_array()) {
    throw std::runtime_error(context + " missing required array: " + field_name);
  }
  return value.at(field_name);
}

void JsonUtils::ValidateAllowedKeys(const nlohmann::json& object,
                                    const std::vector<std::string>& allowed_keys,
                                    const std::string& context) {
  for (auto it = object.begin(); it != object.end(); ++it) {
    const auto& key = it.key();
    bool allowed = false;
    for (const auto& allowed_key : allowed_keys) {
      if (key == allowed_key) {
        allowed = true;
        break;
      }
    }
    if (!allowed) {
      throw std::runtime_error(context + " has unsupported key: " + key);
    }
  }
}

std::string JsonUtils::RequireStringFieldAnyOf(const nlohmann::json& object,
                                               const std::vector<std::string>& keys,
                                               const std::string& context,
                                               const std::string& required_name) {
  std::string value;
  if (!TryGetStringFieldAnyOf(object, keys, &value)) {
    throw std::runtime_error(context + " missing required key: " + required_name);
  }
  return value;
}

bool JsonUtils::TryGetStringFieldAnyOf(const nlohmann::json& object,
                                       const std::vector<std::string>& keys,
                                       std::string* out_value) {
  for (const auto& key : keys) {
    if (object.contains(key) && !object.at(key).is_null()) {
      *out_value = object.at(key).get<std::string>();
      return true;
    }
  }
  return false;
}

}  // namespace pioneerml::utils::json
