#include "pioneerml_dataloaders/configurable/dataloaders/graph/group_classifier_loader.h"

#include "pioneerml_dataloaders/configurable/data_derivers/time_grouper.h"
#include "pioneerml_dataloaders/configurable/data_derivers/group_summary_deriver.h"
#include "pioneerml_dataloaders/io/parquet_reader.h"
#include "pioneerml_dataloaders/batch/group_classifier_batch.h"
#include "pioneerml_dataloaders/utils/parallel/parallel.h"
#include "pioneerml_dataloaders/utils/timing/scoped_timer.h"

#include <arrow/api.h>
#include <arrow/buffer.h>
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
  const bool has_targets = target_cols.size() == target_columns_.size();

  const auto& hits_x = static_cast<const arrow::ListArray&>(*input_cols.at("hits_x")->chunk(0));
  const auto& hits_y = static_cast<const arrow::ListArray&>(*input_cols.at("hits_y")->chunk(0));
  const auto& hits_z = static_cast<const arrow::ListArray&>(*input_cols.at("hits_z")->chunk(0));
  const auto& hits_edep = static_cast<const arrow::ListArray&>(*input_cols.at("hits_edep")->chunk(0));
  const auto& hits_view = static_cast<const arrow::ListArray&>(*input_cols.at("hits_strip_type")->chunk(0));
  const auto& hits_time_group = static_cast<const arrow::ListArray&>(
      *input_cols.at("hits_time_group")->chunk(0));

  auto x_values = std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(hits_x.values());
  auto y_values = std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(hits_y.values());
  auto z_values = std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(hits_z.values());
  auto edep_values = std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(hits_edep.values());
  auto view_values = std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(hits_view.values());
  auto tg_values = std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(hits_time_group.values());

  const double* x_raw = x_values->raw_values();
  const double* y_raw = y_values->raw_values();
  const double* z_raw = z_values->raw_values();
  const double* edep_raw = edep_values->raw_values();
  const int32_t* view_raw = view_values->raw_values();
  const int64_t* tg_raw = tg_values->raw_values();

  const int32_t* z_offsets = hits_z.raw_value_offsets();
  const int32_t* tg_offsets = hits_time_group.raw_value_offsets();

  const int64_t rows = table.num_rows();
  std::vector<int64_t> node_counts(rows, 0);
  std::vector<int64_t> edge_counts(rows, 0);
  std::vector<int64_t> group_counts(rows, 0);

  {
    utils::timing::ScopedTimer count_timer("group_classifier.count_nodes_edges");
    utils::parallel::Parallel::For(0, rows, [&](int64_t row) {
      int64_t n = static_cast<int64_t>(z_offsets[row + 1] - z_offsets[row]);
      node_counts[row] = n;
      edge_counts[row] = n * (n - 1);
    });
  }

  auto node_offsets = utils::parallel::Parallel::PrefixSum(node_counts);
  auto edge_offsets = utils::parallel::Parallel::PrefixSum(edge_counts);
  const int64_t total_nodes = node_offsets.back();
  const int64_t total_edges = edge_offsets.back();

  auto alloc_buffer = [](int64_t bytes) -> std::shared_ptr<arrow::Buffer> {
    auto result = arrow::AllocateBuffer(bytes);
    if (!result.ok()) {
      throw std::runtime_error(result.status().ToString());
    }
    return std::shared_ptr<arrow::Buffer>(std::move(result).ValueOrDie());
  };

  auto node_feat_buf = alloc_buffer(total_nodes * static_cast<int64_t>(sizeof(float)) * 4);
  auto edge_index_buf = alloc_buffer(total_edges * static_cast<int64_t>(sizeof(int64_t)) * 2);
  auto edge_attr_buf = alloc_buffer(total_edges * static_cast<int64_t>(sizeof(float)) * 4);
  auto time_group_buf = alloc_buffer(total_nodes * static_cast<int64_t>(sizeof(int64_t)));
  auto node_ptr_buf = alloc_buffer((rows + 1) * static_cast<int64_t>(sizeof(int64_t)));
  auto edge_ptr_buf = alloc_buffer((rows + 1) * static_cast<int64_t>(sizeof(int64_t)));
  auto group_ptr_buf = alloc_buffer((rows + 1) * static_cast<int64_t>(sizeof(int64_t)));
  auto u_buf = alloc_buffer(rows * static_cast<int64_t>(sizeof(float)));

  auto* node_feat = reinterpret_cast<float*>(node_feat_buf->mutable_data());
  auto* edge_index = reinterpret_cast<int64_t*>(edge_index_buf->mutable_data());
  auto* edge_attr = reinterpret_cast<float*>(edge_attr_buf->mutable_data());
  auto* time_group_ids = reinterpret_cast<int64_t*>(time_group_buf->mutable_data());
  auto* node_ptr = reinterpret_cast<int64_t*>(node_ptr_buf->mutable_data());
  auto* edge_ptr = reinterpret_cast<int64_t*>(edge_ptr_buf->mutable_data());
  auto* group_ptr = reinterpret_cast<int64_t*>(group_ptr_buf->mutable_data());
  auto* u = reinterpret_cast<float*>(u_buf->mutable_data());

  for (int64_t row = 0; row <= rows; ++row) {
    node_ptr[row] = node_offsets[row];
    edge_ptr[row] = edge_offsets[row];
  }

  {
    utils::timing::ScopedTimer build_timer("group_classifier.build_rows");
    utils::parallel::Parallel::For(0, rows, [&](int64_t row) {
      const int64_t n = node_counts[row];
      const int64_t node_offset = node_offsets[row];
      const int64_t edge_offset = edge_offsets[row];
      const int32_t start = z_offsets[row];
      const int32_t tg_start = tg_offsets[row];
      const int64_t tg_len = static_cast<int64_t>(tg_offsets[row + 1] - tg_offsets[row]);

      double sum_edep = 0.0;

      int64_t max_group = -1;
      for (int64_t i = 0; i < n; ++i) {
        const int64_t idx = node_offset + i;
        const int64_t base = idx * 4;
        const int32_t view = view_raw[start + i];
        const double coord = (view == 0 ? x_raw[start + i] : y_raw[start + i]);

        node_feat[base] = static_cast<float>(coord);
        node_feat[base + 1] = static_cast<float>(z_raw[start + i]);
        node_feat[base + 2] = static_cast<float>(edep_raw[start + i]);
        node_feat[base + 3] = static_cast<float>(view);

        if (tg_len < n) {
          throw std::runtime_error("hits_time_group length mismatch with hits.");
        }
        const int64_t tg_val = tg_raw[tg_start + i];
        time_group_ids[idx] = tg_val;
        if (tg_val > max_group) {
          max_group = tg_val;
        }

        sum_edep += edep_raw[start + i];
      }

      u[row] = static_cast<float>(sum_edep);

      const int64_t groups_for_row = std::max<int64_t>(0, max_group + 1);
      if (groups_for_row == 0 && n > 0) {
        throw std::runtime_error("No time groups found for non-empty event.");
      }
      group_counts[row] = groups_for_row;

      int64_t edge_local = 0;
      for (int64_t i = 0; i < n; ++i) {
        const int32_t view_i = view_raw[start + i];
        const double coord_i = (view_i == 0 ? x_raw[start + i] : y_raw[start + i]);
        for (int64_t j = 0; j < n; ++j) {
          if (i == j) continue;
          const int64_t edge_idx = edge_offset + edge_local;
          const int64_t edge_base = edge_idx * 2;
          const int64_t attr_base = edge_idx * 4;

          edge_index[edge_base] = node_offset + i;
          edge_index[edge_base + 1] = node_offset + j;

          const int32_t view_j = view_raw[start + j];
          const double coord_j = (view_j == 0 ? x_raw[start + j] : y_raw[start + j]);

          edge_attr[attr_base] = static_cast<float>(coord_j - coord_i);
          edge_attr[attr_base + 1] = static_cast<float>(z_raw[start + j] - z_raw[start + i]);
          edge_attr[attr_base + 2] = static_cast<float>(edep_raw[start + j] - edep_raw[start + i]);
          edge_attr[attr_base + 3] = (view_i == view_j) ? 1.0f : 0.0f;

          edge_local++;
        }
      }
    });
  }

  utils::timing::ScopedTimer finalize_timer("group_classifier.finalize_arrays");
  auto make_array = [](std::shared_ptr<arrow::Buffer> buf,
                       std::shared_ptr<arrow::DataType> type,
                       int64_t length) {
    auto data = arrow::ArrayData::Make(type, length, {nullptr, std::move(buf)});
    return arrow::MakeArray(std::move(data));
  };

  auto out = std::make_unique<GroupClassifierInputs>();
  group_ptr[0] = 0;
  for (int64_t row = 0; row < rows; ++row) {
    group_ptr[row + 1] = group_ptr[row] + group_counts[row];
  }
  const int64_t total_groups = group_ptr[rows];

  std::shared_ptr<arrow::Buffer> y_buf;
  std::shared_ptr<arrow::Buffer> y_energy_buf;
  float* y = nullptr;
  float* y_energy = nullptr;
  if (has_targets) {
    y_buf = alloc_buffer(total_groups * static_cast<int64_t>(sizeof(float)) * 3);
    y_energy_buf = alloc_buffer(total_groups * static_cast<int64_t>(sizeof(float)) * 3);
    y = reinterpret_cast<float*>(y_buf->mutable_data());
    y_energy = reinterpret_cast<float*>(y_energy_buf->mutable_data());

    const auto& pion_list = static_cast<const arrow::ListArray&>(
        *target_cols.at("pion_in_group")->chunk(0));
    const auto& muon_list = static_cast<const arrow::ListArray&>(
        *target_cols.at("muon_in_group")->chunk(0));
    const auto& mip_list = static_cast<const arrow::ListArray&>(
        *target_cols.at("mip_in_group")->chunk(0));
    const auto& pion_energy_list = static_cast<const arrow::ListArray&>(
        *target_cols.at("total_pion_energy")->chunk(0));
    const auto& muon_energy_list = static_cast<const arrow::ListArray&>(
        *target_cols.at("total_muon_energy")->chunk(0));
    const auto& mip_energy_list = static_cast<const arrow::ListArray&>(
        *target_cols.at("total_mip_energy")->chunk(0));

    auto pion_values =
        std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(
            pion_list.values());
    auto muon_values =
        std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(
            muon_list.values());
    auto mip_values =
        std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(
            mip_list.values());
    auto pion_energy_values =
        std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(
            pion_energy_list.values());
    auto muon_energy_values =
        std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(
            muon_energy_list.values());
    auto mip_energy_values =
        std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(
            mip_energy_list.values());

    const int32_t* pion_raw = pion_values->raw_values();
    const int32_t* muon_raw = muon_values->raw_values();
    const int32_t* mip_raw = mip_values->raw_values();
    const double* e_pi_raw = pion_energy_values->raw_values();
    const double* e_mu_raw = muon_energy_values->raw_values();
    const double* e_mip_raw = mip_energy_values->raw_values();

    const int32_t* pion_offsets = pion_list.raw_value_offsets();
    const int32_t* muon_offsets = muon_list.raw_value_offsets();
    const int32_t* mip_offsets = mip_list.raw_value_offsets();
    const int32_t* e_pi_offsets = pion_energy_list.raw_value_offsets();
    const int32_t* e_mu_offsets = muon_energy_list.raw_value_offsets();
    const int32_t* e_mip_offsets = mip_energy_list.raw_value_offsets();

    utils::parallel::Parallel::For(0, rows, [&](int64_t row) {
      auto check_offsets = [&](const int32_t* offsets) {
        return offsets[row + 1] - offsets[row];
      };
      const int64_t count = check_offsets(pion_offsets);
      if (count != check_offsets(muon_offsets) ||
          count != check_offsets(mip_offsets) ||
          count != check_offsets(e_pi_offsets) ||
          count != check_offsets(e_mu_offsets) ||
          count != check_offsets(e_mip_offsets)) {
        throw std::runtime_error("Target list columns have mismatched lengths.");
      }
      if (count != group_counts[row]) {
        throw std::runtime_error("Target list length does not match time groups.");
      }
      const int64_t base_group = group_ptr[row];
      const int32_t start = pion_offsets[row];
      for (int64_t g = 0; g < count; ++g) {
        const int64_t base = (base_group + g) * 3;
        const int32_t idx = start + static_cast<int32_t>(g);
        y[base] = static_cast<float>(pion_raw[idx]);
        y[base + 1] = static_cast<float>(muon_raw[idx]);
        y[base + 2] = static_cast<float>(mip_raw[idx]);
        y_energy[base] = static_cast<float>(e_pi_raw[idx]);
        y_energy[base + 1] = static_cast<float>(e_mu_raw[idx]);
        y_energy[base + 2] = static_cast<float>(e_mip_raw[idx]);
      }
    });
  }

  out->node_features = make_array(node_feat_buf, arrow::float32(), total_nodes * 4);
  out->edge_index = make_array(edge_index_buf, arrow::int64(), total_edges * 2);
  out->edge_attr = make_array(edge_attr_buf, arrow::float32(), total_edges * 4);
  out->time_group_ids = make_array(time_group_buf, arrow::int64(), total_nodes);
  out->node_ptr = make_array(node_ptr_buf, arrow::int64(), rows + 1);
  out->edge_ptr = make_array(edge_ptr_buf, arrow::int64(), rows + 1);
  out->group_ptr = make_array(group_ptr_buf, arrow::int64(), rows + 1);
  out->u = make_array(u_buf, arrow::float32(), rows);

  if (has_targets) {
    out->y = make_array(y_buf, arrow::float32(), total_groups * 3);
    out->y_energy = make_array(y_energy_buf, arrow::float32(), total_groups * 3);
  } else {
    out->y = nullptr;
    out->y_energy = nullptr;
  }

  out->num_graphs = static_cast<size_t>(rows);
  out->num_groups = static_cast<size_t>(total_groups);
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
  targets->num_groups = typed->num_groups;
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
