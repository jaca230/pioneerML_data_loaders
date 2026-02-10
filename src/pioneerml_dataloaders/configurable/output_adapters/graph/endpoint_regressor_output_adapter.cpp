#include "pioneerml_dataloaders/configurable/output_adapters/graph/endpoint_regressor_output_adapter.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

namespace pioneerml::output_adapters::graph {

EndpointRegressorOutputAdapter::GroupIndexLists EndpointRegressorOutputAdapter::BuildGraphIndexLists(
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

  std::vector<std::vector<std::pair<int64_t, int64_t>>> per_event_pairs(
      static_cast<size_t>(num_events));
  for (int64_t i = 0; i < num_graphs; ++i) {
    const int64_t event_id = evt[i];
    if (event_id < 0 || event_id >= num_events) {
      throw std::runtime_error("Invalid event id in graph_event_ids.");
    }
    per_event_pairs[static_cast<size_t>(event_id)].emplace_back(grp[i], i);
  }

  GroupIndexLists out;
  out.graphs_by_event.resize(static_cast<size_t>(num_events));
  out.group_ids_by_event.resize(static_cast<size_t>(num_events));
  for (int64_t e = 0; e < num_events; ++e) {
    auto& pairs = per_event_pairs[static_cast<size_t>(e)];
    std::sort(
        pairs.begin(),
        pairs.end(),
        [](const std::pair<int64_t, int64_t>& a, const std::pair<int64_t, int64_t>& b) {
          if (a.first != b.first) {
            return a.first < b.first;
          }
          return a.second < b.second;
        });
    auto& graphs = out.graphs_by_event[static_cast<size_t>(e)];
    auto& groups = out.group_ids_by_event[static_cast<size_t>(e)];
    graphs.reserve(pairs.size());
    groups.reserve(pairs.size());
    for (const auto& pair : pairs) {
      groups.push_back(pair.first);
      graphs.push_back(pair.second);
    }
  }
  return out;
}

std::shared_ptr<arrow::Array> EndpointRegressorOutputAdapter::BuildGroupListColumn(
    const float* pred_raw,
    int64_t num_groups,
    const std::vector<std::vector<int64_t>>& graphs_by_event,
    int feature_index) const {
  arrow::ListBuilder list_builder(arrow::default_memory_pool(), std::make_shared<arrow::FloatBuilder>());
  auto* value_builder = static_cast<arrow::FloatBuilder*>(list_builder.value_builder());

  const int64_t num_events = static_cast<int64_t>(graphs_by_event.size());
  for (int64_t evt = 0; evt < num_events; ++evt) {
    auto status = list_builder.Append();
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    const auto& graphs = graphs_by_event[static_cast<size_t>(evt)];
    for (int64_t graph_idx : graphs) {
      const int64_t idx = graph_idx * 6 + feature_index;
      if (idx >= num_groups * 6) {
        throw std::runtime_error("Prediction length does not match graph index list.");
      }
      status = value_builder->Append(pred_raw[idx]);
      if (!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
    }
  }

  std::shared_ptr<arrow::Array> out;
  if (!list_builder.Finish(&out).ok()) {
    throw std::runtime_error("Failed to finalize endpoint list column.");
  }
  return out;
}

std::shared_ptr<arrow::Array> EndpointRegressorOutputAdapter::BuildGroupIdListColumn(
    const std::vector<std::vector<int64_t>>& group_ids_by_event) const {
  arrow::ListBuilder list_builder(arrow::default_memory_pool(), std::make_shared<arrow::Int64Builder>());
  auto* value_builder = static_cast<arrow::Int64Builder*>(list_builder.value_builder());
  for (const auto& groups : group_ids_by_event) {
    auto status = list_builder.Append();
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    for (int64_t group_id : groups) {
      status = value_builder->Append(group_id);
      if (!status.ok()) {
        throw std::runtime_error(status.ToString());
      }
    }
  }
  std::shared_ptr<arrow::Array> out;
  if (!list_builder.Finish(&out).ok()) {
    throw std::runtime_error("Failed to finalize endpoint time-group id column.");
  }
  return out;
}

std::shared_ptr<arrow::Table> EndpointRegressorOutputAdapter::BuildEventTable(
    const std::shared_ptr<arrow::Array>& group_pred,
    const std::shared_ptr<arrow::Array>& graph_event_ids,
    const std::shared_ptr<arrow::Array>& graph_group_ids) const {
  if (!group_pred || !graph_event_ids || !graph_group_ids) {
    throw std::runtime_error("Missing required arrays for EndpointRegressor output adapter.");
  }

  const auto& pred_arr = static_cast<const arrow::NumericArray<arrow::FloatType>&>(*group_pred);
  const float* pred_raw = pred_arr.raw_values();
  const int64_t total_groups = pred_arr.length() / 6;

  auto index_lists = BuildGraphIndexLists(*graph_event_ids, *graph_group_ids);

  auto time_group_ids = BuildGroupIdListColumn(index_lists.group_ids_by_event);
  auto start_x = BuildGroupListColumn(pred_raw, total_groups, index_lists.graphs_by_event, 0);
  auto start_y = BuildGroupListColumn(pred_raw, total_groups, index_lists.graphs_by_event, 1);
  auto start_z = BuildGroupListColumn(pred_raw, total_groups, index_lists.graphs_by_event, 2);
  auto end_x = BuildGroupListColumn(pred_raw, total_groups, index_lists.graphs_by_event, 3);
  auto end_y = BuildGroupListColumn(pred_raw, total_groups, index_lists.graphs_by_event, 4);
  auto end_z = BuildGroupListColumn(pred_raw, total_groups, index_lists.graphs_by_event, 5);

  arrow::Int64Builder event_builder;
  for (int64_t event_id = 0; event_id < static_cast<int64_t>(index_lists.graphs_by_event.size()); ++event_id) {
    auto status = event_builder.Append(event_id);
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
  }
  std::shared_ptr<arrow::Array> event_ids;
  if (!event_builder.Finish(&event_ids).ok()) {
    throw std::runtime_error("Failed to finalize endpoint event_id column.");
  }

  std::vector<std::shared_ptr<arrow::Field>> fields = {
      arrow::field("event_id", arrow::int64()),
      arrow::field("time_group_ids", time_group_ids->type()),
      arrow::field("pred_group_start_x", start_x->type()),
      arrow::field("pred_group_start_y", start_y->type()),
      arrow::field("pred_group_start_z", start_z->type()),
      arrow::field("pred_group_end_x", end_x->type()),
      arrow::field("pred_group_end_y", end_y->type()),
      arrow::field("pred_group_end_z", end_z->type()),
  };
  std::vector<std::shared_ptr<arrow::Array>> columns = {
      event_ids,
      time_group_ids,
      start_x,
      start_y,
      start_z,
      end_x,
      end_y,
      end_z,
  };

  auto schema = std::make_shared<arrow::Schema>(fields);
  return arrow::Table::Make(schema, columns);
}

void EndpointRegressorOutputAdapter::WriteParquet(
    const std::string& output_path,
    const std::shared_ptr<arrow::Array>& group_pred,
    const std::shared_ptr<arrow::Array>& graph_event_ids,
    const std::shared_ptr<arrow::Array>& graph_group_ids) const {
  auto table = BuildEventTable(group_pred, graph_event_ids, graph_group_ids);
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
