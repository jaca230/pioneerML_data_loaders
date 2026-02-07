#include "pioneerml_dataloaders/configurable/input_adapters/base_input_adapter.h"

#include <stdexcept>

#include "pioneerml_dataloaders/utils/json/json_utils.h"
#include "pioneerml_dataloaders/utils/parquet/parquet_utils.h"
#include "pioneerml_dataloaders/utils/timing/scoped_timer.h"

namespace pioneerml::input_adapters {

std::shared_ptr<arrow::Table> InputAdapter::BuildUnifiedTableFromFilesSpec(
    const nlohmann::json& input_spec,
    const std::vector<JsonFieldSpec>& field_specs,
    const std::string& context) const {
  utils::timing::ScopedTimer total_timer("input_adapter.build_unified_table");
  utils::json::JsonUtils::RequireObject(input_spec, context + " input_spec");
  const auto& files =
      utils::json::JsonUtils::RequireArrayField(input_spec, "files", context + " input_spec");
  if (files.empty()) {
    throw std::runtime_error(context + " input_spec.files cannot be empty.");
  }

  std::vector<std::string> allowed_keys;
  for (const auto& spec : field_specs) {
    for (const auto& alias : spec.aliases) {
      allowed_keys.push_back(alias);
    }
  }

  std::vector<std::shared_ptr<arrow::Table>> shard_tables;
  shard_tables.reserve(files.size());
  for (size_t i = 0; i < files.size(); ++i) {
    utils::timing::ScopedTimer shard_timer("input_adapter.build_unified_table.shard");
    const auto& item = files.at(i);
    const std::string item_context = context + " files[" + std::to_string(i) + "]";
    utils::json::JsonUtils::RequireObject(item, item_context);
    utils::json::JsonUtils::ValidateAllowedKeys(item, allowed_keys, item_context);

    std::vector<std::string> paths;
    std::vector<std::vector<std::string>> columns_by_path;
    for (const auto& field_spec : field_specs) {
      if (field_spec.required) {
        paths.push_back(utils::json::JsonUtils::RequireStringFieldAnyOf(
            item, field_spec.aliases, item_context, field_spec.canonical_name));
        columns_by_path.push_back(field_spec.projected_columns);
      } else {
        std::string value;
        if (utils::json::JsonUtils::TryGetStringFieldAnyOf(item, field_spec.aliases, &value)) {
          paths.push_back(value);
          columns_by_path.push_back(field_spec.projected_columns);
        }
      }
    }
    shard_tables.push_back(utils::parquet::LoadAndMergeTablesByColumns(paths, columns_by_path));
  }

  utils::timing::ScopedTimer concat_timer("input_adapter.build_unified_table.concat_shards");
  return utils::parquet::ConcatenateTablesByRows(shard_tables);
}

}  // namespace pioneerml::input_adapters
