#include "pioneerml_dataloaders/configurable/output_adapters/graph/group_classifier_output_adapter.h"

#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <stdexcept>

namespace pioneerml::output_adapters::graph {

GroupClassifierOutputAdapter::GroupOffsets GroupClassifierOutputAdapter::ComputeGroupOffsets(
    const arrow::Array& node_ptr,
    const arrow::Array& time_group_ids) const {
  const auto& node_ptr_arr = static_cast<const arrow::NumericArray<arrow::Int64Type>&>(node_ptr);
  const auto& tg_arr = static_cast<const arrow::NumericArray<arrow::Int64Type>&>(time_group_ids);

  const int64_t num_events = node_ptr_arr.length() - 1;
  std::vector<int64_t> counts(num_events, 0);

  const int64_t* ptr = node_ptr_arr.raw_values();
  const int64_t* tg = tg_arr.raw_values();

  for (int64_t evt = 0; evt < num_events; ++evt) {
    const int64_t start = ptr[evt];
    const int64_t end = ptr[evt + 1];
    int64_t max_group = -1;
    for (int64_t i = start; i < end; ++i) {
      max_group = std::max(max_group, tg[i]);
    }
    counts[evt] = (max_group >= 0) ? (max_group + 1) : 0;
  }

  std::vector<int64_t> offsets(num_events + 1, 0);
  for (int64_t i = 0; i < num_events; ++i) {
    offsets[i + 1] = offsets[i] + counts[i];
  }

  return {std::move(offsets), std::move(counts)};
}

std::shared_ptr<arrow::Array> GroupClassifierOutputAdapter::BuildGroupListColumn(
    const float* pred_raw,
    int64_t num_groups,
    const std::vector<int64_t>& offsets,
    const std::vector<int64_t>& counts,
    int class_index) const {
  arrow::ListBuilder list_builder(arrow::default_memory_pool(),
                                  std::make_shared<arrow::FloatBuilder>());
  auto* value_builder =
      static_cast<arrow::FloatBuilder*>(list_builder.value_builder());

  const int64_t num_events = static_cast<int64_t>(counts.size());
  for (int64_t evt = 0; evt < num_events; ++evt) {
    const int64_t count = counts[evt];
    auto status = list_builder.Append();
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    const int64_t start = offsets[evt];
    for (int64_t g = 0; g < count; ++g) {
      const int64_t idx = (start + g) * 3 + class_index;
      if (idx >= num_groups * 3) {
        throw std::runtime_error("Prediction length does not match group offsets.");
      }
      status = value_builder->Append(pred_raw[idx]);
      if (!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
    }
  }

  std::shared_ptr<arrow::Array> out;
  if (!list_builder.Finish(&out).ok()) {
    throw std::runtime_error("Failed to finish list column.");
  }
  return out;
}

std::shared_ptr<arrow::Table> GroupClassifierOutputAdapter::BuildEventTable(
    const std::shared_ptr<arrow::Array>& group_pred,
    const std::shared_ptr<arrow::Array>& group_pred_energy,
    const std::shared_ptr<arrow::Array>& node_ptr,
    const std::shared_ptr<arrow::Array>& time_group_ids) const {
  if (!group_pred || !node_ptr || !time_group_ids) {
    throw std::runtime_error("Missing required arrays for output adapter.");
  }

  const auto& pred_arr = static_cast<const arrow::NumericArray<arrow::FloatType>&>(
      *group_pred);
  const float* pred_raw = pred_arr.raw_values();

  auto offsets = ComputeGroupOffsets(*node_ptr, *time_group_ids);
  const int64_t total_groups = offsets.offsets.back();

  auto pred_pion = BuildGroupListColumn(pred_raw, total_groups, offsets.offsets,
                                        offsets.counts, 0);
  auto pred_muon = BuildGroupListColumn(pred_raw, total_groups, offsets.offsets,
                                        offsets.counts, 1);
  auto pred_mip = BuildGroupListColumn(pred_raw, total_groups, offsets.offsets,
                                       offsets.counts, 2);

  std::vector<std::shared_ptr<arrow::Field>> fields = {
      arrow::field("pred_pion", pred_pion->type()),
      arrow::field("pred_muon", pred_muon->type()),
      arrow::field("pred_mip", pred_mip->type()),
  };
  std::vector<std::shared_ptr<arrow::Array>> columns = {pred_pion, pred_muon, pred_mip};

  if (group_pred_energy) {
    const auto& energy_arr =
        static_cast<const arrow::NumericArray<arrow::FloatType>&>(*group_pred_energy);
    const float* energy_raw = energy_arr.raw_values();
    auto e_pion = BuildGroupListColumn(energy_raw, total_groups, offsets.offsets,
                                       offsets.counts, 0);
    auto e_muon = BuildGroupListColumn(energy_raw, total_groups, offsets.offsets,
                                       offsets.counts, 1);
    auto e_mip = BuildGroupListColumn(energy_raw, total_groups, offsets.offsets,
                                      offsets.counts, 2);
    fields.push_back(arrow::field("pred_pion_energy", e_pion->type()));
    fields.push_back(arrow::field("pred_muon_energy", e_muon->type()));
    fields.push_back(arrow::field("pred_mip_energy", e_mip->type()));
    columns.push_back(e_pion);
    columns.push_back(e_muon);
    columns.push_back(e_mip);
  }

  auto schema = std::make_shared<arrow::Schema>(fields);
  return arrow::Table::Make(schema, columns);
}

void GroupClassifierOutputAdapter::WriteParquet(
    const std::string& output_path,
    const std::shared_ptr<arrow::Array>& group_pred,
    const std::shared_ptr<arrow::Array>& group_pred_energy,
    const std::shared_ptr<arrow::Array>& node_ptr,
    const std::shared_ptr<arrow::Array>& time_group_ids) const {
  auto table = BuildEventTable(group_pred, group_pred_energy, node_ptr, time_group_ids);
  auto out_result = arrow::io::FileOutputStream::Open(output_path);
  if (!out_result.ok()) {
    throw std::runtime_error(out_result.status().ToString());
  }
  auto out = out_result.MoveValueUnsafe();
  auto result = parquet::arrow::WriteTable(
      *table, arrow::default_memory_pool(), out);
  if (!result.ok()) {
    throw std::runtime_error(result.ToString());
  }
}

}  // namespace pioneerml::output_adapters::graph
