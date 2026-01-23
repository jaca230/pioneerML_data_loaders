#include "pioneerml_dataloaders/dataloaders/graph/group_classifier_loader.h"

#include "pioneerml_dataloaders/data_derivers/time_grouper.h"
#include "pioneerml_dataloaders/data_derivers/particle_mask_deriver.h"
#include "pioneerml_dataloaders/io/reader_utils.h"

#include <arrow/api.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace pioneerml::dataloaders::graph {

namespace {

struct ColumnRefs {
  std::shared_ptr<arrow::ChunkedArray> event_id;
  std::shared_ptr<arrow::ChunkedArray> pion_in_group;
  std::shared_ptr<arrow::ChunkedArray> muon_in_group;
  std::shared_ptr<arrow::ChunkedArray> mip_in_group;
  std::shared_ptr<arrow::ChunkedArray> total_pion_energy;
  std::shared_ptr<arrow::ChunkedArray> total_muon_energy;
  std::shared_ptr<arrow::ChunkedArray> total_mip_energy;
  std::shared_ptr<arrow::ChunkedArray> hits_x;
  std::shared_ptr<arrow::ChunkedArray> hits_y;
  std::shared_ptr<arrow::ChunkedArray> hits_z;
  std::shared_ptr<arrow::ChunkedArray> hits_edep;
  std::shared_ptr<arrow::ChunkedArray> hits_strip_type;
  std::shared_ptr<arrow::ChunkedArray> hits_pdg_id;
  std::shared_ptr<arrow::ChunkedArray> hits_time;
};

ColumnRefs BindColumns(const std::shared_ptr<arrow::Table>& table) {
  auto col = [&](const std::string& name) { return table->GetColumnByName(name); };
  ColumnRefs refs;
  refs.event_id = col("event_id");
  refs.pion_in_group = col("pion_in_group");
  refs.muon_in_group = col("muon_in_group");
  refs.mip_in_group = col("mip_in_group");
  refs.total_pion_energy = col("total_pion_energy");
  refs.total_muon_energy = col("total_muon_energy");
  refs.total_mip_energy = col("total_mip_energy");
  refs.hits_x = col("hits_x");
  refs.hits_y = col("hits_y");
  refs.hits_z = col("hits_z");
  refs.hits_edep = col("hits_edep");
  refs.hits_strip_type = col("hits_strip_type");
  refs.hits_pdg_id = col("hits_pdg_id");
  refs.hits_time = col("hits_time");
  return refs;
}

}  // namespace

GroupClassifierLoader::GroupClassifierLoader(GroupClassifierConfig cfg) : cfg_(cfg) {}

std::shared_ptr<arrow::Table> GroupClassifierLoader::LoadTable(const std::string& parquet_path) const {
  return io::ReadParquet(parquet_path);
}

std::unique_ptr<GraphBatch> GroupClassifierLoader::BuildGraph(const arrow::Table& table) const {
  auto cols = BindColumns(std::shared_ptr<arrow::Table>(const_cast<arrow::Table*>(&table), [](arrow::Table*) {}));
  data_derivers::TimeGrouper grouper(cfg_.time_window_ns);
  data_derivers::ParticleMaskDeriver mask_deriver;

  auto time_groups_array = grouper.DeriveColumn(table);
  auto mask_array = mask_deriver.DeriveColumn(table);
  const auto* time_groups_list = time_groups_array ? static_cast<const arrow::ListArray*>(time_groups_array.get()) : nullptr;
  const auto* masks_list = mask_array ? static_cast<const arrow::ListArray*>(mask_array.get()) : nullptr;

  auto batch = std::make_unique<GraphBatch>();
  batch->node_ptr.reserve(table.num_rows() + 1);
  batch->edge_ptr.reserve(table.num_rows() + 1);
  batch->node_ptr.push_back(0);
  batch->edge_ptr.push_back(0);

  const auto& hits_x = static_cast<const arrow::ListArray&>(*cols.hits_x->chunk(0));
  const auto& hits_y = static_cast<const arrow::ListArray&>(*cols.hits_y->chunk(0));
  const auto& hits_z = static_cast<const arrow::ListArray&>(*cols.hits_z->chunk(0));
  const auto& hits_edep = static_cast<const arrow::ListArray&>(*cols.hits_edep->chunk(0));
  const auto& hits_view = static_cast<const arrow::ListArray&>(*cols.hits_strip_type->chunk(0));
  const auto& hits_pdg = static_cast<const arrow::ListArray&>(*cols.hits_pdg_id->chunk(0));
  const auto& hits_time = static_cast<const arrow::ListArray&>(*cols.hits_time->chunk(0));

  for (int64_t row = 0; row < table.num_rows(); ++row) {
    auto xs = io::ListToVector<arrow::DoubleType, double>(hits_x, row);
    auto ys = io::ListToVector<arrow::DoubleType, double>(hits_y, row);
    auto zs = io::ListToVector<arrow::DoubleType, double>(hits_z, row);
    auto edeps = io::ListToVector<arrow::DoubleType, double>(hits_edep, row);
    auto views = io::ListToVector<arrow::Int32Type, int64_t>(hits_view, row);
    auto pdgs = io::ListToVector<arrow::Int32Type, int64_t>(hits_pdg, row);
    size_t n = zs.size();
    std::vector<int64_t> time_groups;
    std::vector<int64_t> masks;
    if (time_groups_list && cfg_.compute_time_groups) {
      auto offsets = time_groups_list->raw_value_offsets();
      auto values = std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(time_groups_list->values());
      const int64_t* raw = values->raw_values();
      for (int32_t i = offsets[row]; i < offsets[row + 1]; ++i) {
        time_groups.push_back(raw[i]);
      }
    } else {
      time_groups.assign(n, 0);
    }
    if (masks_list) {
      auto offsets = masks_list->raw_value_offsets();
      auto values = std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(masks_list->values());
      const int64_t* raw = values->raw_values();
      for (int32_t i = offsets[row]; i < offsets[row + 1]; ++i) {
        masks.push_back(raw[i]);
      }
    }

    int64_t node_start = batch->node_ptr.back();

    for (size_t i = 0; i < n; ++i) {
      double coord = (views[i] == 0 ? xs[i] : ys[i]);
      batch->node_features.push_back(static_cast<float>(coord));
      batch->node_features.push_back(static_cast<float>(zs[i]));
      batch->node_features.push_back(static_cast<float>(edeps[i]));
      batch->node_features.push_back(static_cast<float>(views[i]));
      batch->hit_mask.push_back(1);
      batch->time_group_ids.push_back(0);  // fill below
      batch->y_node.push_back(pdgs[i]);
      batch->particle_mask.push_back(i < masks.size() ? masks[i] : mask_deriver.ComputeSingle(static_cast<int>(pdgs[i])));
    }

    for (size_t i = 0; i < n && i < time_groups.size(); ++i) {
      batch->time_group_ids[node_start + i] = time_groups[i];
    }

    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
      for (int64_t j = 0; j < static_cast<int64_t>(n); ++j) {
        if (i == j) continue;
        batch->edge_index.push_back(node_start + i);
        batch->edge_index.push_back(node_start + j);
        double dx = batch->node_features[(node_start + j) * 4 + 0] - batch->node_features[(node_start + i) * 4 + 0];
        double dz = batch->node_features[(node_start + j) * 4 + 1] - batch->node_features[(node_start + i) * 4 + 1];
        double dE = batch->node_features[(node_start + j) * 4 + 2] - batch->node_features[(node_start + i) * 4 + 2];
        double same_view = (views[i] == views[j]) ? 1.0 : 0.0;
        batch->edge_attr.push_back(static_cast<float>(dx));
        batch->edge_attr.push_back(static_cast<float>(dz));
        batch->edge_attr.push_back(static_cast<float>(dE));
        batch->edge_attr.push_back(static_cast<float>(same_view));
      }
    }

    batch->node_ptr.push_back(node_start + static_cast<int64_t>(n));
    batch->edge_ptr.push_back(static_cast<int64_t>(batch->edge_index.size() / 2));

    int64_t pion = *io::GetScalarPtr<int32_t>(*cols.pion_in_group, row);
    int64_t muon = *io::GetScalarPtr<int32_t>(*cols.muon_in_group, row);
    int64_t mip = *io::GetScalarPtr<int32_t>(*cols.mip_in_group, row);
    batch->y.push_back(static_cast<float>(pion));
    batch->y.push_back(static_cast<float>(muon));
    batch->y.push_back(static_cast<float>(mip));

    double e_pi = *io::GetScalarPtr<double>(*cols.total_pion_energy, row);
    double e_mu = *io::GetScalarPtr<double>(*cols.total_muon_energy, row);
    double e_mip = *io::GetScalarPtr<double>(*cols.total_mip_energy, row);
    batch->y_energy.push_back(static_cast<float>(e_pi));
    batch->y_energy.push_back(static_cast<float>(e_mu));
    batch->y_energy.push_back(static_cast<float>(e_mip));

    double u = 0.0;
    for (size_t i = 0; i < n; ++i) u += edeps[i];
    batch->u.push_back(static_cast<float>(u));
  }

  batch->num_graphs = static_cast<size_t>(table.num_rows());
  return batch;
}

}  // namespace pioneerml::dataloaders::graph
