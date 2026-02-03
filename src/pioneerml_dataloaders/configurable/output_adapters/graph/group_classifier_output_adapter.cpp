#include "pioneerml_dataloaders/configurable/output_adapters/graph/group_classifier_output_adapter.h"

#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace pioneerml::output_adapters::graph {

GroupClassifierOutputAdapter::GroupIndexLists GroupClassifierOutputAdapter::BuildGraphIndexLists(
    const arrow::Array& graph_event_ids,
    const arrow::Array& graph_group_ids) const {
  const auto& evt_arr =
      static_cast<const arrow::NumericArray<arrow::Int64Type>&>(graph_event_ids);
  const auto& grp_arr =
      static_cast<const arrow::NumericArray<arrow::Int64Type>&>(graph_group_ids);

  const int64_t num_graphs = evt_arr.length();
  if (grp_arr.length() != num_graphs) {
    throw std::runtime_error("graph_event_ids and graph_group_ids length mismatch.");
  }

  const int64_t* evt = evt_arr.raw_values();
  const int64_t* grp = grp_arr.raw_values();
  int64_t max_event = -1;
  for (int64_t i = 0; i < num_graphs; ++i) {
    max_event = std::max(max_event, evt[i]);
  }
  const int64_t num_events = (max_event >= 0) ? (max_event + 1) : 0;

  std::vector<std::vector<std::pair<int64_t, int64_t>>> per_event_pairs(num_events);
  for (int64_t i = 0; i < num_graphs; ++i) {
    const int64_t event_id = evt[i];
    if (event_id < 0 || event_id >= num_events) {
      throw std::runtime_error("Invalid event id in graph_event_ids.");
    }
    per_event_pairs[event_id].emplace_back(grp[i], i);
  }

  GroupIndexLists out;
  out.graphs_by_event.resize(num_events);
  for (int64_t e = 0; e < num_events; ++e) {
    auto& pairs = per_event_pairs[e];
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    auto& graphs = out.graphs_by_event[e];
    graphs.reserve(pairs.size());
    for (const auto& pair : pairs) {
      graphs.push_back(pair.second);
    }
  }
  return out;
}

std::shared_ptr<arrow::Array> GroupClassifierOutputAdapter::BuildGroupListColumn(
    const float* pred_raw,
    int64_t num_groups,
    const std::vector<std::vector<int64_t>>& graphs_by_event,
    int class_index) const {
  arrow::ListBuilder list_builder(arrow::default_memory_pool(),
                                  std::make_shared<arrow::FloatBuilder>());
  auto* value_builder =
      static_cast<arrow::FloatBuilder*>(list_builder.value_builder());

  const int64_t num_events = static_cast<int64_t>(graphs_by_event.size());
  for (int64_t evt = 0; evt < num_events; ++evt) {
    const auto& graphs = graphs_by_event[evt];
    const int64_t count = static_cast<int64_t>(graphs.size());
    auto status = list_builder.Append();
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    for (int64_t g = 0; g < count; ++g) {
      const int64_t graph_idx = graphs[g];
      const int64_t idx = graph_idx * 3 + class_index;
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
    const std::shared_ptr<arrow::Array>& graph_event_ids,
    const std::shared_ptr<arrow::Array>& graph_group_ids) const {
  if (!group_pred || !graph_event_ids || !graph_group_ids) {
    throw std::runtime_error("Missing required arrays for output adapter.");
  }

  const auto& pred_arr = static_cast<const arrow::NumericArray<arrow::FloatType>&>(
      *group_pred);
  const float* pred_raw = pred_arr.raw_values();

  auto graphs_by_event = BuildGraphIndexLists(*graph_event_ids, *graph_group_ids);
  const int64_t total_groups = pred_arr.length() / 3;

  auto pred_pion = BuildGroupListColumn(pred_raw, total_groups, graphs_by_event.graphs_by_event, 0);
  auto pred_muon = BuildGroupListColumn(pred_raw, total_groups, graphs_by_event.graphs_by_event, 1);
  auto pred_mip = BuildGroupListColumn(pred_raw, total_groups, graphs_by_event.graphs_by_event, 2);

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
    auto e_pion = BuildGroupListColumn(energy_raw, total_groups, graphs_by_event.graphs_by_event, 0);
    auto e_muon = BuildGroupListColumn(energy_raw, total_groups, graphs_by_event.graphs_by_event, 1);
    auto e_mip = BuildGroupListColumn(energy_raw, total_groups, graphs_by_event.graphs_by_event, 2);
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
    const std::shared_ptr<arrow::Array>& graph_event_ids,
    const std::shared_ptr<arrow::Array>& graph_group_ids) const {
  auto table = BuildEventTable(group_pred, group_pred_energy, graph_event_ids, graph_group_ids);
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
