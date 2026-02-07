#include "pioneerml_dataloaders/configurable/dataloaders/graph/group_classifier_loader.h"

#include "pioneerml_dataloaders/configurable/data_derivers/time_grouper.h"
#include "pioneerml_dataloaders/configurable/data_derivers/group_summary_deriver.h"
#include "pioneerml_dataloaders/batch/group_classifier_batch.h"
#include "pioneerml_dataloaders/utils/parallel/parallel.h"
#include "pioneerml_dataloaders/utils/timing/scoped_timer.h"

#include <arrow/api.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace pioneerml::dataloaders::graph {

GroupClassifierLoader::GroupClassifierLoader() {
  input_columns_ = {
      "hits_x",
      "hits_y",
      "hits_z",
      "hits_edep",
      "hits_strip_type",
      "hits_time",
      "hits_time_group",
  };
  target_columns_ = {
      "pion_in_group",
      "muon_in_group",
      "mip_in_group",
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
       "mip_in_group"},
      group_summary);
}

void GroupClassifierLoader::CountGroupsForRows(
    const int32_t* z_offsets,
    const int32_t* tg_offsets,
    const int64_t* tg_raw,
    int64_t rows,
    std::vector<int64_t>* group_counts,
    std::vector<std::vector<int64_t>>* group_node_counts) const {
  utils::parallel::Parallel::For(0, rows, [&](int64_t row) {
    const int64_t n = static_cast<int64_t>(z_offsets[row + 1] - z_offsets[row]);
    if (n == 0) {
      (*group_counts)[row] = 0;
      return;
    }
    const int32_t tg_start = tg_offsets[row];
    int64_t max_group = -1;
    for (int64_t i = 0; i < n; ++i) {
      max_group = std::max(max_group, tg_raw[tg_start + i]);
    }
    const int64_t groups_for_row = std::max<int64_t>(0, max_group + 1);
    (*group_counts)[row] = groups_for_row;
    std::vector<int64_t> counts(groups_for_row, 0);
    for (int64_t i = 0; i < n; ++i) {
      const int64_t tg_val = tg_raw[tg_start + i];
      if (tg_val < 0 || tg_val >= groups_for_row) {
        throw std::runtime_error("Invalid time group id encountered.");
      }
      counts[tg_val] += 1;
    }
    (*group_node_counts)[row] = std::move(counts);
  });
}

void GroupClassifierLoader::EncodeTargets(const ColumnMap& target_cols,
                                          const std::vector<int64_t>& group_counts,
                                          const std::vector<int64_t>& graph_offsets,
                                          int64_t rows,
                                          float* y) const {
  const auto& pion_list =
      static_cast<const arrow::ListArray&>(*target_cols.at("pion_in_group")->chunk(0));
  const auto& muon_list =
      static_cast<const arrow::ListArray&>(*target_cols.at("muon_in_group")->chunk(0));
  const auto& mip_list =
      static_cast<const arrow::ListArray&>(*target_cols.at("mip_in_group")->chunk(0));

  auto pion_values =
      std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(pion_list.values());
  auto muon_values =
      std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(muon_list.values());
  auto mip_values =
      std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(mip_list.values());

  const int32_t* pion_raw = pion_values->raw_values();
  const int32_t* muon_raw = muon_values->raw_values();
  const int32_t* mip_raw = mip_values->raw_values();

  const int32_t* pion_offsets = pion_list.raw_value_offsets();
  const int32_t* muon_offsets = muon_list.raw_value_offsets();
  const int32_t* mip_offsets = mip_list.raw_value_offsets();

  utils::parallel::Parallel::For(0, rows, [&](int64_t row) {
    auto check_offsets = [&](const int32_t* offsets) {
      return offsets[row + 1] - offsets[row];
    };
    const int64_t count = check_offsets(pion_offsets);
    if (count != check_offsets(muon_offsets) || count != check_offsets(mip_offsets)) {
      throw std::runtime_error("Target list columns have mismatched lengths.");
    }
    if (count != group_counts[row]) {
      throw std::runtime_error("Target list length does not match time groups.");
    }
    const int64_t base_group = graph_offsets[row];
    const int32_t start = pion_offsets[row];
    for (int64_t g = 0; g < count; ++g) {
      const int64_t base = (base_group + g) * 3;
      const int32_t idx = start + static_cast<int32_t>(g);
      y[base] = static_cast<float>(pion_raw[idx]);
      y[base + 1] = static_cast<float>(muon_raw[idx]);
      y[base + 2] = static_cast<float>(mip_raw[idx]);
    }
  });
}

TrainingBundle GroupClassifierLoader::LoadTraining(
    const std::shared_ptr<arrow::Table>& table) const {
  auto prepared = PrepareTable(table, true);
  auto required = utils::parquet::MergeColumns(input_columns_, target_columns_);
  utils::parquet::ValidateColumns(
      *prepared, required, {}, true, "GroupClassifierLoader training");
  auto batch = BuildGraph(*prepared);
  return SplitInputsTargets(std::move(batch));
}

InferenceBundle GroupClassifierLoader::LoadInference(
    const std::shared_ptr<arrow::Table>& table) const {
  auto prepared = PrepareTable(table, true);
  utils::parquet::ValidateColumns(
      *prepared, input_columns_, target_columns_, true, "GroupClassifierLoader inputs");
  InferenceBundle out;
  out.inputs = BuildGraph(*prepared);
  return out;
}

std::unique_ptr<BaseBatch> GroupClassifierLoader::BuildGraph(const arrow::Table& table) const {
  utils::timing::ScopedTimer total_timer("group_classifier.build_graph");
  BuildContext ctx;
  BuildBuffers bufs;
  {
    utils::timing::ScopedTimer timer("group_classifier.phase0_initialize");
    BuildGraphPhase0Initialize(table, &ctx);
  }
  {
    utils::timing::ScopedTimer timer("group_classifier.phase1_count");
    BuildGraphPhase1Count(&ctx);
  }
  {
    utils::timing::ScopedTimer timer("group_classifier.phase2_offsets");
    BuildGraphPhase2Offsets(&ctx);
  }
  {
    utils::timing::ScopedTimer timer("group_classifier.phase3_allocate");
    BuildGraphPhase3Allocate(ctx, &bufs);
  }
  {
    utils::timing::ScopedTimer timer("group_classifier.phase4_populate");
    BuildGraphPhase4Populate(ctx, &bufs);
  }
  {
    utils::timing::ScopedTimer timer("group_classifier.phase5_finalize");
    return BuildGraphPhase5Finalize(ctx, &bufs);
  }
}

void GroupClassifierLoader::BuildGraphPhase0Initialize(const arrow::Table& table,
                                                       BuildContext* ctx) const {
  ctx->input_cols = utils::parquet::BindColumns(
      table, input_columns_, true, true, "GroupClassifierLoader inputs");
  ctx->target_cols = utils::parquet::BindColumns(
      table, target_columns_, false, true, "GroupClassifierLoader targets");
  ctx->has_targets = ctx->target_cols.size() == target_columns_.size();

  ctx->hits_x =
      &static_cast<const arrow::ListArray&>(*ctx->input_cols.at("hits_x")->chunk(0));
  ctx->hits_y =
      &static_cast<const arrow::ListArray&>(*ctx->input_cols.at("hits_y")->chunk(0));
  ctx->hits_z =
      &static_cast<const arrow::ListArray&>(*ctx->input_cols.at("hits_z")->chunk(0));
  ctx->hits_edep =
      &static_cast<const arrow::ListArray&>(*ctx->input_cols.at("hits_edep")->chunk(0));
  ctx->hits_view =
      &static_cast<const arrow::ListArray&>(*ctx->input_cols.at("hits_strip_type")->chunk(0));
  ctx->hits_time_group = &static_cast<const arrow::ListArray&>(
      *ctx->input_cols.at("hits_time_group")->chunk(0));

  ctx->x_values = MakeNumericAccessor(ctx->hits_x->values(), "group classifier x");
  ctx->y_values = MakeNumericAccessor(ctx->hits_y->values(), "group classifier y");
  ctx->z_values = MakeNumericAccessor(ctx->hits_z->values(), "group classifier z");
  ctx->edep_values = MakeNumericAccessor(ctx->hits_edep->values(), "group classifier edep");

  auto view_values =
      std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(ctx->hits_view->values());
  auto tg_values = std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(
      ctx->hits_time_group->values());

  ctx->view_raw = view_values->raw_values();
  ctx->tg_raw = tg_values->raw_values();
  ctx->z_offsets = ctx->hits_z->raw_value_offsets();
  ctx->tg_offsets = ctx->hits_time_group->raw_value_offsets();

  ctx->rows = table.num_rows();
  ctx->group_counts.assign(ctx->rows, 0);
  ctx->group_node_counts.resize(static_cast<size_t>(ctx->rows));
}

void GroupClassifierLoader::BuildGraphPhase1Count(BuildContext* ctx) const {
  CountGroupsForRows(ctx->z_offsets,
                     ctx->tg_offsets,
                     ctx->tg_raw,
                     ctx->rows,
                     &ctx->group_counts,
                     &ctx->group_node_counts);
}

void GroupClassifierLoader::BuildGraphPhase2Offsets(BuildContext* ctx) const {
  ctx->graph_offsets = BuildOffsets(ctx->group_counts);
  ctx->total_graphs = ctx->graph_offsets.back();

  ctx->node_counts.assign(static_cast<size_t>(ctx->total_graphs), 0);
  ctx->edge_counts.assign(static_cast<size_t>(ctx->total_graphs), 0);
  for (int64_t row = 0; row < ctx->rows; ++row) {
    const int64_t groups_for_row = ctx->group_counts[row];
    if (groups_for_row == 0) {
      continue;
    }
    const int64_t graph_base = ctx->graph_offsets[row];
    const auto& counts = ctx->group_node_counts[static_cast<size_t>(row)];
    for (int64_t g = 0; g < groups_for_row; ++g) {
      const int64_t graph_idx = graph_base + g;
      const int64_t count = counts[static_cast<size_t>(g)];
      ctx->node_counts[static_cast<size_t>(graph_idx)] = count;
      ctx->edge_counts[static_cast<size_t>(graph_idx)] = count * (count - 1);
    }
  }

  ctx->node_offsets = BuildOffsets(ctx->node_counts);
  ctx->edge_offsets = BuildOffsets(ctx->edge_counts);
  ctx->total_nodes = ctx->node_offsets.back();
  ctx->total_edges = ctx->edge_offsets.back();
  ctx->total_groups = ctx->total_graphs;
}

void GroupClassifierLoader::BuildGraphPhase3Allocate(const BuildContext& ctx,
                                                     BuildBuffers* bufs) const {
  bufs->node_feat_buf =
      AllocBuffer(ctx.total_nodes * static_cast<int64_t>(sizeof(float)) * 4);
  bufs->edge_index_buf =
      AllocBuffer(ctx.total_edges * static_cast<int64_t>(sizeof(int64_t)) * 2);
  bufs->edge_attr_buf =
      AllocBuffer(ctx.total_edges * static_cast<int64_t>(sizeof(float)) * 4);
  bufs->time_group_buf =
      AllocBuffer(ctx.total_nodes * static_cast<int64_t>(sizeof(int64_t)));
  bufs->node_ptr_buf =
      AllocBuffer((ctx.total_graphs + 1) * static_cast<int64_t>(sizeof(int64_t)));
  bufs->edge_ptr_buf =
      AllocBuffer((ctx.total_graphs + 1) * static_cast<int64_t>(sizeof(int64_t)));
  bufs->group_ptr_buf =
      AllocBuffer((ctx.total_graphs + 1) * static_cast<int64_t>(sizeof(int64_t)));
  bufs->u_buf = AllocBuffer(ctx.total_graphs * static_cast<int64_t>(sizeof(float)));
  bufs->graph_event_ids_buf =
      AllocBuffer(ctx.total_graphs * static_cast<int64_t>(sizeof(int64_t)));
  bufs->graph_group_ids_buf =
      AllocBuffer(ctx.total_graphs * static_cast<int64_t>(sizeof(int64_t)));

  bufs->node_feat = reinterpret_cast<float*>(bufs->node_feat_buf->mutable_data());
  bufs->edge_index = reinterpret_cast<int64_t*>(bufs->edge_index_buf->mutable_data());
  bufs->edge_attr = reinterpret_cast<float*>(bufs->edge_attr_buf->mutable_data());
  bufs->time_group_ids = reinterpret_cast<int64_t*>(bufs->time_group_buf->mutable_data());
  bufs->node_ptr = reinterpret_cast<int64_t*>(bufs->node_ptr_buf->mutable_data());
  bufs->edge_ptr = reinterpret_cast<int64_t*>(bufs->edge_ptr_buf->mutable_data());
  bufs->group_ptr = reinterpret_cast<int64_t*>(bufs->group_ptr_buf->mutable_data());
  bufs->u = reinterpret_cast<float*>(bufs->u_buf->mutable_data());
  bufs->graph_event_ids =
      reinterpret_cast<int64_t*>(bufs->graph_event_ids_buf->mutable_data());
  bufs->graph_group_ids =
      reinterpret_cast<int64_t*>(bufs->graph_group_ids_buf->mutable_data());

  FillPointerArrayFromOffsets(ctx.node_offsets, bufs->node_ptr);
  FillPointerArrayFromOffsets(ctx.edge_offsets, bufs->edge_ptr);
  for (int64_t graph = 0; graph <= ctx.total_graphs; ++graph) {
    bufs->group_ptr[graph] = graph;
  }
}

void GroupClassifierLoader::BuildGraphPhase4Populate(const BuildContext& ctx,
                                                     BuildBuffers* bufs) const {
  utils::parallel::Parallel::For(0, ctx.rows, [&](int64_t row) {
    const int64_t n = static_cast<int64_t>(ctx.z_offsets[row + 1] - ctx.z_offsets[row]);
    const int32_t start = ctx.z_offsets[row];
    const int32_t tg_start = ctx.tg_offsets[row];
    const int64_t tg_len =
        static_cast<int64_t>(ctx.tg_offsets[row + 1] - ctx.tg_offsets[row]);
    if (tg_len < n) {
      throw std::runtime_error("hits_time_group length mismatch with hits.");
    }

    const int64_t groups_for_row = ctx.group_counts[row];
    if (groups_for_row == 0) {
      return;
    }

    std::vector<std::vector<int64_t>> group_nodes(static_cast<size_t>(groups_for_row));
    for (int64_t g = 0; g < groups_for_row; ++g) {
      group_nodes[static_cast<size_t>(g)].reserve(
          static_cast<size_t>(ctx.group_node_counts[static_cast<size_t>(row)]
                                  [static_cast<size_t>(g)]));
    }
    for (int64_t i = 0; i < n; ++i) {
      const int64_t tg_val = ctx.tg_raw[tg_start + i];
      if (tg_val < 0 || tg_val >= groups_for_row) {
        throw std::runtime_error("Invalid time group id encountered.");
      }
      group_nodes[static_cast<size_t>(tg_val)].push_back(i);
    }

      const int64_t graph_base = ctx.graph_offsets[row];
      for (int64_t g = 0; g < groups_for_row; ++g) {
        const int64_t graph_idx = graph_base + g;
        bufs->graph_event_ids[graph_idx] = row;
        bufs->graph_group_ids[graph_idx] = g;

      const int64_t node_offset = ctx.node_offsets[graph_idx];
        const int64_t edge_offset = ctx.edge_offsets[graph_idx];
        const auto& nodes = group_nodes[static_cast<size_t>(g)];
        const int64_t k = static_cast<int64_t>(nodes.size());
        std::vector<float> coord_local(static_cast<size_t>(k), 0.0f);
        std::vector<float> z_local(static_cast<size_t>(k), 0.0f);
        std::vector<float> e_local(static_cast<size_t>(k), 0.0f);
        std::vector<int32_t> view_local(static_cast<size_t>(k), 0);

        double sum_edep = 0.0;
        for (int64_t local = 0; local < k; ++local) {
          const int64_t i = nodes[static_cast<size_t>(local)];
          const int64_t idx = node_offset + local;
          const int64_t base = idx * 4;
          const int32_t view = ctx.view_raw[start + i];
          const double coord =
              ResolveCoordinateForView(ctx.x_values, ctx.y_values, view, start + i);
          const float z_value = static_cast<float>(
              ctx.z_values.IsValid(start + i) ? ctx.z_values.Value(start + i) : 0.0);
          const float e_value = static_cast<float>(
              ctx.edep_values.IsValid(start + i) ? ctx.edep_values.Value(start + i) : 0.0);
          coord_local[static_cast<size_t>(local)] = static_cast<float>(coord);
          z_local[static_cast<size_t>(local)] = z_value;
          e_local[static_cast<size_t>(local)] = e_value;
          view_local[static_cast<size_t>(local)] = view;

          bufs->node_feat[base] = coord_local[static_cast<size_t>(local)];
          bufs->node_feat[base + 1] = z_value;
          bufs->node_feat[base + 2] = e_value;
          bufs->node_feat[base + 3] = static_cast<float>(view);
          bufs->time_group_ids[idx] = g;
          sum_edep += e_value;
        }

        bufs->u[graph_idx] = static_cast<float>(sum_edep);

        int64_t edge_local = 0;
        for (int64_t a = 0; a < k; ++a) {
          const int32_t view_i = view_local[static_cast<size_t>(a)];
          const float coord_i = coord_local[static_cast<size_t>(a)];
          const float z_i = z_local[static_cast<size_t>(a)];
          const float e_i = e_local[static_cast<size_t>(a)];
          for (int64_t b = 0; b < k; ++b) {
            if (a == b) {
              continue;
            }
            const int32_t view_j = view_local[static_cast<size_t>(b)];
            const float coord_j = coord_local[static_cast<size_t>(b)];
            const float z_j = z_local[static_cast<size_t>(b)];
            const float e_j = e_local[static_cast<size_t>(b)];
            const int64_t edge_idx = edge_offset + edge_local;
            const int64_t edge_base = edge_idx * 2;
            const int64_t attr_base = edge_idx * 4;

            bufs->edge_index[edge_base] = node_offset + a;
            bufs->edge_index[edge_base + 1] = node_offset + b;

            bufs->edge_attr[attr_base] = coord_j - coord_i;
            bufs->edge_attr[attr_base + 1] = z_j - z_i;
            bufs->edge_attr[attr_base + 2] = e_j - e_i;
            bufs->edge_attr[attr_base + 3] = (view_i == view_j) ? 1.0f : 0.0f;

            edge_local++;
          }
        }
      }
    });
  }

std::unique_ptr<BaseBatch> GroupClassifierLoader::BuildGraphPhase5Finalize(
    const BuildContext& ctx,
    BuildBuffers* bufs) const {
  if (ctx.has_targets) {
    bufs->y_buf =
        AllocBuffer(ctx.total_groups * static_cast<int64_t>(sizeof(float)) * 3);
    bufs->y = reinterpret_cast<float*>(bufs->y_buf->mutable_data());
    EncodeTargets(
        ctx.target_cols, ctx.group_counts, ctx.graph_offsets, ctx.rows, bufs->y);
  }

  auto out = std::make_unique<GroupClassifierInputs>();
  out->node_features = MakeArray(bufs->node_feat_buf, arrow::float32(), ctx.total_nodes * 4);
  out->edge_index = MakeArray(bufs->edge_index_buf, arrow::int64(), ctx.total_edges * 2);
  out->edge_attr = MakeArray(bufs->edge_attr_buf, arrow::float32(), ctx.total_edges * 4);
  out->time_group_ids = MakeArray(bufs->time_group_buf, arrow::int64(), ctx.total_nodes);
  out->node_ptr = MakeArray(bufs->node_ptr_buf, arrow::int64(), ctx.total_graphs + 1);
  out->edge_ptr = MakeArray(bufs->edge_ptr_buf, arrow::int64(), ctx.total_graphs + 1);
  out->group_ptr = MakeArray(bufs->group_ptr_buf, arrow::int64(), ctx.total_graphs + 1);
  out->u = MakeArray(bufs->u_buf, arrow::float32(), ctx.total_graphs);
  out->graph_event_ids = MakeArray(bufs->graph_event_ids_buf, arrow::int64(), ctx.total_graphs);
  out->graph_group_ids = MakeArray(bufs->graph_group_ids_buf, arrow::int64(), ctx.total_graphs);
  if (ctx.has_targets) {
    out->y = MakeArray(bufs->y_buf, arrow::float32(), ctx.total_groups * 3);
  } else {
    out->y = nullptr;
  }
  out->num_graphs = static_cast<size_t>(ctx.total_graphs);
  out->num_groups = static_cast<size_t>(ctx.total_groups);
  return out;
}

TrainingBundle GroupClassifierLoader::SplitInputsTargets(std::unique_ptr<BaseBatch> batch_base) const {
  auto* typed = dynamic_cast<GroupClassifierInputs*>(batch_base.get());
  if (!typed) {
    throw std::runtime_error("Unexpected batch type in SplitInputsTargets");
  }
  if (!typed->y) {
    throw std::runtime_error("Training targets are missing. Use LoadInference or provide label columns.");
  }
  auto targets = std::make_unique<GroupClassifierTargets>();
  targets->num_groups = typed->num_groups;
  targets->y = typed->y;

  typed->y.reset();

  TrainingBundle result;
  result.inputs = std::move(batch_base);
  result.targets = std::move(targets);
  return result;
}

}  // namespace pioneerml::dataloaders::graph
