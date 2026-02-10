#include "pioneerml_dataloaders/configurable/output_adapters/graph/group_splitter_output_adapter.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

namespace pioneerml::output_adapters::graph {

std::shared_ptr<arrow::Table> GroupSplitterOutputAdapter::BuildEventTable(
    const std::shared_ptr<arrow::Array>& node_pred,
    const std::shared_ptr<arrow::Array>& node_ptr,
    const std::shared_ptr<arrow::Array>& graph_event_ids,
    const std::shared_ptr<arrow::Array>& graph_group_ids) const {
  if (!node_pred || !node_ptr || !graph_event_ids || !graph_group_ids) {
    throw std::runtime_error("Missing required arrays for GroupSplitter output adapter.");
  }

  const auto& pred_arr = static_cast<const arrow::NumericArray<arrow::FloatType>&>(*node_pred);
  const auto& node_ptr_arr = static_cast<const arrow::NumericArray<arrow::Int64Type>&>(*node_ptr);
  const auto& event_arr = static_cast<const arrow::NumericArray<arrow::Int64Type>&>(*graph_event_ids);
  const auto& group_arr = static_cast<const arrow::NumericArray<arrow::Int64Type>&>(*graph_group_ids);
  const int64_t total_nodes = pred_arr.length() / 3;
  const int64_t num_graphs = node_ptr_arr.length() - 1;
  if (event_arr.length() != num_graphs || group_arr.length() != num_graphs) {
    throw std::runtime_error("graph_event_ids / graph_group_ids length mismatch with node_ptr.");
  }

  const float* pred_raw = pred_arr.raw_values();
  const int64_t* node_ptr_raw = node_ptr_arr.raw_values();
  const int64_t* event_raw = event_arr.raw_values();
  const int64_t* group_raw = group_arr.raw_values();

  int64_t max_event_id = -1;
  for (int64_t graph_idx = 0; graph_idx < num_graphs; ++graph_idx) {
    max_event_id = std::max(max_event_id, event_raw[graph_idx]);
  }
  const int64_t num_events = (max_event_id >= 0) ? (max_event_id + 1) : 0;
  std::vector<std::vector<std::pair<int64_t, int64_t>>> per_event(
      static_cast<size_t>(num_events));
  for (int64_t graph_idx = 0; graph_idx < num_graphs; ++graph_idx) {
    const int64_t event_id = event_raw[graph_idx];
    if (event_id < 0 || event_id >= num_events) {
      throw std::runtime_error("Invalid event id while building GroupSplitter output.");
    }
    per_event[static_cast<size_t>(event_id)].emplace_back(group_raw[graph_idx], graph_idx);
  }
  for (auto& pairs : per_event) {
    std::sort(
        pairs.begin(),
        pairs.end(),
        [](const std::pair<int64_t, int64_t>& a, const std::pair<int64_t, int64_t>& b) {
          if (a.first != b.first) {
            return a.first < b.first;
          }
          return a.second < b.second;
        });
  }

  arrow::Int64Builder event_builder;
  arrow::ListBuilder tg_builder(arrow::default_memory_pool(), std::make_shared<arrow::Int64Builder>());
  auto* tg_values = static_cast<arrow::Int64Builder*>(tg_builder.value_builder());
  arrow::ListBuilder pred_pion_builder(
      arrow::default_memory_pool(), std::make_shared<arrow::FloatBuilder>());
  auto* pred_pion_values = static_cast<arrow::FloatBuilder*>(pred_pion_builder.value_builder());
  arrow::ListBuilder pred_muon_builder(
      arrow::default_memory_pool(), std::make_shared<arrow::FloatBuilder>());
  auto* pred_muon_values = static_cast<arrow::FloatBuilder*>(pred_muon_builder.value_builder());
  arrow::ListBuilder pred_mip_builder(
      arrow::default_memory_pool(), std::make_shared<arrow::FloatBuilder>());
  auto* pred_mip_values = static_cast<arrow::FloatBuilder*>(pred_mip_builder.value_builder());

  for (int64_t event_id = 0; event_id < num_events; ++event_id) {
    auto status = event_builder.Append(event_id);
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    status = tg_builder.Append();
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    status = pred_pion_builder.Append();
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    status = pred_muon_builder.Append();
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    status = pred_mip_builder.Append();
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }

    const auto& pairs = per_event[static_cast<size_t>(event_id)];
    for (const auto& [group_id, graph_idx] : pairs) {
      const int64_t start = node_ptr_raw[graph_idx];
      const int64_t end = node_ptr_raw[graph_idx + 1];
      if (start < 0 || end < start || end > total_nodes) {
        throw std::runtime_error("Invalid node_ptr while building GroupSplitter output.");
      }
      for (int64_t node = start; node < end; ++node) {
        status = tg_values->Append(group_id);
        if (!status.ok()) {
          throw std::runtime_error(status.ToString());
        }
        status = pred_pion_values->Append(pred_raw[node * 3]);
        if (!status.ok()) {
          throw std::runtime_error(status.ToString());
        }
        status = pred_muon_values->Append(pred_raw[node * 3 + 1]);
        if (!status.ok()) {
          throw std::runtime_error(status.ToString());
        }
        status = pred_mip_values->Append(pred_raw[node * 3 + 2]);
        if (!status.ok()) {
          throw std::runtime_error(status.ToString());
        }
      }
    }
  }

  std::shared_ptr<arrow::Array> event_ids;
  std::shared_ptr<arrow::Array> tg_list;
  std::shared_ptr<arrow::Array> pred_pion;
  std::shared_ptr<arrow::Array> pred_muon;
  std::shared_ptr<arrow::Array> pred_mip;
  if (!event_builder.Finish(&event_ids).ok() || !tg_builder.Finish(&tg_list).ok() ||
      !pred_pion_builder.Finish(&pred_pion).ok() || !pred_muon_builder.Finish(&pred_muon).ok() ||
      !pred_mip_builder.Finish(&pred_mip).ok()) {
    throw std::runtime_error("Failed to finalize GroupSplitter event-aligned output arrays.");
  }

  std::vector<std::shared_ptr<arrow::Field>> fields = {
      arrow::field("event_id", arrow::int64()),
      arrow::field("time_group_ids", tg_list->type()),
      arrow::field("pred_hit_pion", pred_pion->type()),
      arrow::field("pred_hit_muon", pred_muon->type()),
      arrow::field("pred_hit_mip", pred_mip->type()),
  };
  std::vector<std::shared_ptr<arrow::Array>> columns = {
      event_ids,
      tg_list,
      pred_pion,
      pred_muon,
      pred_mip,
  };

  auto schema = std::make_shared<arrow::Schema>(fields);
  return arrow::Table::Make(schema, columns);
}

void GroupSplitterOutputAdapter::WriteParquet(
    const std::string& output_path,
    const std::shared_ptr<arrow::Array>& node_pred,
    const std::shared_ptr<arrow::Array>& node_ptr,
    const std::shared_ptr<arrow::Array>& graph_event_ids,
    const std::shared_ptr<arrow::Array>& graph_group_ids) const {
  auto table = BuildEventTable(node_pred, node_ptr, graph_event_ids, graph_group_ids);
  auto out_result = arrow::io::FileOutputStream::Open(output_path);
  if (!out_result.ok()) {
    throw std::runtime_error(out_result.status().ToString());
  }
  auto out = out_result.MoveValueUnsafe();
  auto result = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out);
  if (!result.ok()) {
    throw std::runtime_error(result.ToString());
  }
}

}  // namespace pioneerml::output_adapters::graph
