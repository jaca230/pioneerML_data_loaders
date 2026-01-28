#include "pioneerml_dataloaders/configurable/dataloaders/graph/group_classifier_loader.h"

#include "pioneerml_dataloaders/configurable/data_derivers/time_grouper.h"
#include "pioneerml_dataloaders/configurable/data_derivers/group_summary_deriver.h"
#include "pioneerml_dataloaders/io/parquet_reader.h"
#include "pioneerml_dataloaders/batch/group_classifier_batch.h"
#include "pioneerml_dataloaders/utils/parquet/parquet_utils.h"
#include "pioneerml_dataloaders/utils/timing/scoped_timer.h"

#include <arrow/api.h>
#include <arrow/buffer_builder.h>
#include <arrow/status.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace pioneerml::dataloaders::graph {

namespace {
}  // namespace

GroupClassifierLoader::GroupClassifierLoader() {
  input_columns_ = {
      "hits_x",
      "hits_y",
      "hits_z",
      "hits_edep",
      "hits_strip_type",
      "hits_pdg_id",
      "hits_time",
      "hits_time_group",
  };
  target_columns_ = {
      "pion_in_group",
      "muon_in_group",
      "mip_in_group",
      "total_pion_energy",
      "total_muon_energy",
      "total_mip_energy",
  };
  ConfigureDerivers(nullptr);
}

void GroupClassifierLoader::LoadConfig(const nlohmann::json& cfg) {
  if (cfg.contains("time_window_ns")) {
    time_window_ns_ = cfg.at("time_window_ns").get<double>();
  }
  if (cfg.contains("compute_time_groups")) {
    compute_time_groups_ = cfg.at("compute_time_groups").get<bool>();
  }

  const nlohmann::json* derivers_cfg = nullptr;
  if (cfg.contains("derivers")) {
    derivers_cfg = &cfg.at("derivers");
  }
  ConfigureDerivers(derivers_cfg);
}

void GroupClassifierLoader::ConfigureDerivers(const nlohmann::json* derivers_cfg) {
  derivers_.clear();

  auto time_grouper = std::make_shared<data_derivers::TimeGrouper>(time_window_ns_);
  if (derivers_cfg && derivers_cfg->contains("time_grouper")) {
    time_grouper->LoadConfig(derivers_cfg->at("time_grouper"));
  }
  AddDeriver("hits_time_group", time_grouper);

  auto group_summary = std::make_shared<data_derivers::GroupSummaryDeriver>();
  if (derivers_cfg && derivers_cfg->contains("group_summary")) {
    group_summary->LoadConfig(derivers_cfg->at("group_summary"));
  }
  AddDeriver(
      {"pion_in_group",
       "muon_in_group",
       "mip_in_group",
       "total_pion_energy",
       "total_muon_energy",
       "total_mip_energy"},
      group_summary);
}

TrainingBundle GroupClassifierLoader::LoadTraining(const std::string& parquet_path) const {
  auto table = AddDerivedColumns(LoadTable(parquet_path));
  auto required = MergeColumns(input_columns_, target_columns_);
  ValidateColumns(*table, required, {}, true, "GroupClassifierLoader training");
  auto batch = BuildGraph(*table);
  return SplitInputsTargets(std::move(batch));
}

TrainingBundle GroupClassifierLoader::LoadTraining(
    const std::vector<std::string>& parquet_paths) const {
  auto table = LoadAndConcatenateTables(parquet_paths, true);
  auto required = MergeColumns(input_columns_, target_columns_);
  ValidateColumns(*table, required, {}, true, "GroupClassifierLoader training");
  auto batch = BuildGraph(*table);
  return SplitInputsTargets(std::move(batch));
}

InferenceBundle GroupClassifierLoader::LoadInference(const std::string& parquet_path) const {
  auto table = AddDerivedColumns(LoadTable(parquet_path));
  ValidateColumns(*table, input_columns_, target_columns_, true, "GroupClassifierLoader inputs");
  InferenceBundle out;
  out.inputs = BuildGraph(*table);
  return out;
}

InferenceBundle GroupClassifierLoader::LoadInference(
    const std::vector<std::string>& parquet_paths) const {
  auto table = LoadAndConcatenateTables(parquet_paths, true);
  ValidateColumns(*table, input_columns_, target_columns_, true, "GroupClassifierLoader inputs");
  InferenceBundle out;
  out.inputs = BuildGraph(*table);
  return out;
}

std::shared_ptr<arrow::Table> GroupClassifierLoader::LoadTable(const std::string& parquet_path) const {
  io::ParquetReader reader;
  return reader.ReadTable(parquet_path);
}

std::unique_ptr<BaseBatch> GroupClassifierLoader::BuildGraph(const arrow::Table& table) const {
  utils::timing::ScopedTimer total_timer("group_classifier.build_graph");
  auto input_cols = BindColumns(table, input_columns_, true, true, "GroupClassifierLoader inputs");
  auto target_cols = BindColumns(table, target_columns_, false, true, "GroupClassifierLoader targets");
  auto ensure_ok = [](const arrow::Status& st) {
    if (!st.ok()) throw std::runtime_error(st.ToString());
  };

  const bool has_targets = target_cols.size() == target_columns_.size();
  const auto& time_groups_col = *input_cols.at("hits_time_group")->chunk(0);
  utils::parquet::ParquetUtils parquet_utils;

  const auto& hits_x = *input_cols.at("hits_x")->chunk(0);
  const auto& hits_y = *input_cols.at("hits_y")->chunk(0);
  const auto& hits_z = *input_cols.at("hits_z")->chunk(0);
  const auto& hits_edep = *input_cols.at("hits_edep")->chunk(0);
  const auto& hits_view = *input_cols.at("hits_strip_type")->chunk(0);
  const auto& hits_pdg = *input_cols.at("hits_pdg_id")->chunk(0);
  const auto& hits_time = *input_cols.at("hits_time")->chunk(0);

  int64_t total_nodes = 0;
  int64_t total_edges = 0;
  {
    utils::timing::ScopedTimer count_timer("group_classifier.count_nodes_edges");
    for (int64_t row = 0; row < table.num_rows(); ++row) {
      int64_t n = parquet_utils.ListLength(hits_z, row);
      total_nodes += n;
      total_edges += n * (n - 1);
    }
  }

  auto pool = arrow::default_memory_pool();
  arrow::TypedBufferBuilder<float> node_feat_builder(pool);
  arrow::TypedBufferBuilder<int64_t> edge_index_builder(pool);
  arrow::TypedBufferBuilder<float> edge_attr_builder(pool);
  arrow::TypedBufferBuilder<int64_t> time_group_builder(pool);
  arrow::TypedBufferBuilder<int64_t> node_ptr_builder(pool);
  arrow::TypedBufferBuilder<int64_t> edge_ptr_builder(pool);
  arrow::TypedBufferBuilder<float> u_builder(pool);
  arrow::TypedBufferBuilder<float> y_builder(pool);
  arrow::TypedBufferBuilder<float> y_energy_builder(pool);

  ensure_ok(node_feat_builder.Reserve(total_nodes * 4));
  ensure_ok(edge_index_builder.Reserve(total_edges * 2));
  ensure_ok(edge_attr_builder.Reserve(total_edges * 4));
  ensure_ok(time_group_builder.Reserve(total_nodes));
  ensure_ok(node_ptr_builder.Reserve(table.num_rows() + 1));
  ensure_ok(edge_ptr_builder.Reserve(table.num_rows() + 1));
  ensure_ok(u_builder.Reserve(table.num_rows()));
  if (has_targets) {
    ensure_ok(y_builder.Reserve(table.num_rows() * 3));
    ensure_ok(y_energy_builder.Reserve(table.num_rows() * 3));
  }

  ensure_ok(node_ptr_builder.Append(0));
  ensure_ok(edge_ptr_builder.Append(0));

  int64_t node_count = 0;
  int64_t edge_count = 0;

  {
    utils::timing::ScopedTimer build_timer("group_classifier.build_rows");
    for (int64_t row = 0; row < table.num_rows(); ++row) {
      auto xs = parquet_utils.ListToVector<arrow::DoubleType, double>(hits_x, row);
      auto ys = parquet_utils.ListToVector<arrow::DoubleType, double>(hits_y, row);
      auto zs = parquet_utils.ListToVector<arrow::DoubleType, double>(hits_z, row);
      auto edeps = parquet_utils.ListToVector<arrow::DoubleType, double>(hits_edep, row);
      auto views = parquet_utils.ListToVector<arrow::Int32Type, int64_t>(hits_view, row);
      auto pdgs = parquet_utils.ListToVector<arrow::Int32Type, int64_t>(hits_pdg, row);
      size_t n = zs.size();
      std::vector<int64_t> time_groups;
      if (compute_time_groups_) {
        auto values = std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(
            parquet_utils.ListValues(time_groups_col));
        const int64_t* raw = values->raw_values();
        auto range = parquet_utils.ListRange(time_groups_col, row);
        for (int64_t i = range.first; i < range.second; ++i) {
          time_groups.push_back(raw[i]);
        }
      } else {
        time_groups.assign(n, 0);
      }

      int64_t node_start = node_count;

      for (size_t i = 0; i < n; ++i) {
        double coord = (views[i] == 0 ? xs[i] : ys[i]);
        ensure_ok(node_feat_builder.Append(static_cast<float>(coord)));
        ensure_ok(node_feat_builder.Append(static_cast<float>(zs[i])));
        ensure_ok(node_feat_builder.Append(static_cast<float>(edeps[i])));
        ensure_ok(node_feat_builder.Append(static_cast<float>(views[i])));
        ensure_ok(time_group_builder.Append(i < time_groups.size() ? time_groups[i] : 0));
      }

      for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        for (int64_t j = 0; j < static_cast<int64_t>(n); ++j) {
          if (i == j) continue;
          ensure_ok(edge_index_builder.Append(node_start + i));
          ensure_ok(edge_index_builder.Append(node_start + j));
          double coord_i = (views[i] == 0 ? xs[i] : ys[i]);
          double coord_j = (views[j] == 0 ? xs[j] : ys[j]);
          double dx = coord_j - coord_i;
          double dz = zs[j] - zs[i];
          double dE = edeps[j] - edeps[i];
          double same_view = (views[i] == views[j]) ? 1.0 : 0.0;
          ensure_ok(edge_attr_builder.Append(static_cast<float>(dx)));
          ensure_ok(edge_attr_builder.Append(static_cast<float>(dz)));
          ensure_ok(edge_attr_builder.Append(static_cast<float>(dE)));
          ensure_ok(edge_attr_builder.Append(static_cast<float>(same_view)));
          edge_count++;
        }
      }

      node_count += static_cast<int64_t>(n);
      ensure_ok(node_ptr_builder.Append(node_count));
      ensure_ok(edge_ptr_builder.Append(edge_count));

      if (has_targets) {
        int64_t pion = *parquet_utils.GetScalarPtr<int32_t>(*target_cols.at("pion_in_group"), row);
        int64_t muon = *parquet_utils.GetScalarPtr<int32_t>(*target_cols.at("muon_in_group"), row);
        int64_t mip = *parquet_utils.GetScalarPtr<int32_t>(*target_cols.at("mip_in_group"), row);
        ensure_ok(y_builder.Append(static_cast<float>(pion)));
        ensure_ok(y_builder.Append(static_cast<float>(muon)));
        ensure_ok(y_builder.Append(static_cast<float>(mip)));

        double e_pi = *parquet_utils.GetScalarPtr<double>(*target_cols.at("total_pion_energy"), row);
        double e_mu = *parquet_utils.GetScalarPtr<double>(*target_cols.at("total_muon_energy"), row);
        double e_mip = *parquet_utils.GetScalarPtr<double>(*target_cols.at("total_mip_energy"), row);
        ensure_ok(y_energy_builder.Append(static_cast<float>(e_pi)));
        ensure_ok(y_energy_builder.Append(static_cast<float>(e_mu)));
        ensure_ok(y_energy_builder.Append(static_cast<float>(e_mip)));
      }

      double u = 0.0;
      for (size_t i = 0; i < n; ++i) u += edeps[i];
      ensure_ok(u_builder.Append(static_cast<float>(u)));
    }
  }

  utils::timing::ScopedTimer finalize_timer("group_classifier.finalize_arrays");
  auto make_array = [](std::shared_ptr<arrow::Buffer> buf, std::shared_ptr<arrow::DataType> type, int64_t length) {
    auto data = arrow::ArrayData::Make(type, length, {nullptr, std::move(buf)});
    return arrow::MakeArray(std::move(data));
  };

  std::shared_ptr<arrow::Buffer> buf;
  auto out = std::make_unique<GroupClassifierInputs>();

  const int64_t node_feat_len = node_feat_builder.length();
  const int64_t edge_index_len = edge_index_builder.length();
  const int64_t edge_attr_len = edge_attr_builder.length();
  const int64_t time_group_len = time_group_builder.length();
  const int64_t node_ptr_len = node_ptr_builder.length();
  const int64_t edge_ptr_len = edge_ptr_builder.length();
  const int64_t u_len = u_builder.length();

  ensure_ok(node_feat_builder.Finish(&buf));
  out->node_features = make_array(buf, arrow::float32(), node_feat_len);
  ensure_ok(edge_index_builder.Finish(&buf));
  out->edge_index = make_array(buf, arrow::int64(), edge_index_len);
  ensure_ok(edge_attr_builder.Finish(&buf));
  out->edge_attr = make_array(buf, arrow::float32(), edge_attr_len);
  ensure_ok(time_group_builder.Finish(&buf));
  out->time_group_ids = make_array(buf, arrow::int64(), time_group_len);
  ensure_ok(node_ptr_builder.Finish(&buf));
  out->node_ptr = make_array(buf, arrow::int64(), node_ptr_len);
  ensure_ok(edge_ptr_builder.Finish(&buf));
  out->edge_ptr = make_array(buf, arrow::int64(), edge_ptr_len);
  ensure_ok(u_builder.Finish(&buf));
  out->u = make_array(buf, arrow::float32(), u_len);
  if (has_targets) {
    const int64_t y_len = y_builder.length();
    const int64_t y_energy_len = y_energy_builder.length();
    ensure_ok(y_builder.Finish(&buf));
    out->y = make_array(buf, arrow::float32(), y_len);
    ensure_ok(y_energy_builder.Finish(&buf));
    out->y_energy = make_array(buf, arrow::float32(), y_energy_len);
  } else {
    out->y = nullptr;
    out->y_energy = nullptr;
  }

  out->num_graphs = static_cast<size_t>(table.num_rows());
  return out;
}

TrainingBundle GroupClassifierLoader::SplitInputsTargets(std::unique_ptr<BaseBatch> batch_base) const {
  auto* typed = dynamic_cast<GroupClassifierInputs*>(batch_base.get());
  if (!typed) {
    throw std::runtime_error("Unexpected batch type in SplitInputsTargets");
  }
  if (!typed->y || !typed->y_energy) {
    throw std::runtime_error("Training targets are missing. Use LoadInference or provide label columns.");
  }
  auto targets = std::make_unique<GroupClassifierTargets>();
  targets->num_graphs = typed->num_graphs;
  targets->y = typed->y;
  targets->y_energy = typed->y_energy;

  typed->y.reset();
  typed->y_energy.reset();

  TrainingBundle result;
  result.inputs = std::move(batch_base);
  result.targets = std::move(targets);
  return result;
}

}  // namespace pioneerml::dataloaders::graph
