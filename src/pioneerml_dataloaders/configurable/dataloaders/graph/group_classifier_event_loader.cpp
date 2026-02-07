#include "pioneerml_dataloaders/configurable/dataloaders/graph/group_classifier_event_loader.h"

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

GroupClassifierEventLoader::GroupClassifierEventLoader() {
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

void GroupClassifierEventLoader::LoadConfig(const nlohmann::json& cfg) {
  if (cfg.contains("time_window_ns")) {
    time_window_ns_ = cfg.at("time_window_ns").get<double>();
  }

  const nlohmann::json* derivers_cfg = nullptr;
  if (cfg.contains("derivers")) {
    derivers_cfg = &cfg.at("derivers");
  }
  ConfigureDerivers(derivers_cfg);
}

void GroupClassifierEventLoader::ConfigureDerivers(const nlohmann::json* derivers_cfg) {
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

void GroupClassifierEventLoader::CountNodeEdgePerRow(const int32_t* z_offsets,
                                                     int64_t rows,
                                                     std::vector<int64_t>* node_counts,
                                                     std::vector<int64_t>* edge_counts) const {
  utils::parallel::Parallel::For(0, rows, [&](int64_t row) {
    int64_t n = static_cast<int64_t>(z_offsets[row + 1] - z_offsets[row]);
    (*node_counts)[row] = n;
    (*edge_counts)[row] = n * (n - 1);
  });
}

void GroupClassifierEventLoader::CountGroupsForRows(const int32_t* z_offsets,
                                                    const int32_t* tg_offsets,
                                                    const int64_t* tg_raw,
                                                    int64_t rows,
                                                    std::vector<int64_t>* group_counts) const {
  utils::parallel::Parallel::For(0, rows, [&](int64_t row) {
    const int64_t n = static_cast<int64_t>(z_offsets[row + 1] - z_offsets[row]);
    if (n == 0) {
      (*group_counts)[row] = 0;
      return;
    }
    const int32_t tg_start = tg_offsets[row];
    int64_t max_group = -1;
    for (int64_t i = 0; i < n; ++i) {
      const int64_t tg_val = tg_raw[tg_start + i];
      if (tg_val > max_group) {
        max_group = tg_val;
      }
    }
    const int64_t groups_for_row = std::max<int64_t>(0, max_group + 1);
    if (groups_for_row == 0 && n > 0) {
      throw std::runtime_error("No time groups found for non-empty event.");
    }
    (*group_counts)[row] = groups_for_row;
  });
}

void GroupClassifierEventLoader::EncodeTargets(const ColumnMap& target_cols,
                                               const std::vector<int64_t>& group_counts,
                                               const std::vector<int64_t>& row_group_offsets,
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
    const int64_t base_group = row_group_offsets[row];
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

TrainingBundle GroupClassifierEventLoader::LoadTraining(
    const std::shared_ptr<arrow::Table>& table) const {
  auto prepared = PrepareTable(table, true);
  auto required = utils::parquet::MergeColumns(input_columns_, target_columns_);
  utils::parquet::ValidateColumns(
      *prepared, required, {}, true, "GroupClassifierEventLoader training");
  auto batch = BuildGraph(*prepared);
  return SplitInputsTargets(std::move(batch));
}

InferenceBundle GroupClassifierEventLoader::LoadInference(
    const std::shared_ptr<arrow::Table>& table) const {
  auto prepared = PrepareTable(table, true);
  utils::parquet::ValidateColumns(
      *prepared, input_columns_, target_columns_, true, "GroupClassifierEventLoader inputs");
  InferenceBundle out;
  out.inputs = BuildGraph(*prepared);
  return out;
}

std::unique_ptr<BaseBatch> GroupClassifierEventLoader::BuildGraph(const arrow::Table& table) const {
  utils::timing::ScopedTimer total_timer("group_classifier_event.build_graph");
  BuildContext ctx;
  BuildBuffers bufs;
  {
    utils::timing::ScopedTimer timer("group_classifier_event.phase0_initialize");
    BuildGraphPhase0Initialize(table, &ctx);
  }
  {
    utils::timing::ScopedTimer timer("group_classifier_event.phase1_count");
    BuildGraphPhase1Count(&ctx);
  }
  {
    utils::timing::ScopedTimer timer("group_classifier_event.phase2_offsets");
    BuildGraphPhase2Offsets(&ctx);
  }
  {
    utils::timing::ScopedTimer timer("group_classifier_event.phase3_allocate");
    BuildGraphPhase3Allocate(ctx, &bufs);
  }
  {
    utils::timing::ScopedTimer timer("group_classifier_event.phase4_populate");
    BuildGraphPhase4Populate(ctx, &bufs);
  }
  {
    utils::timing::ScopedTimer timer("group_classifier_event.phase5_finalize");
    return BuildGraphPhase5Finalize(ctx, &bufs);
  }
}

void GroupClassifierEventLoader::BuildGraphPhase0Initialize(const arrow::Table& table,
                                                            BuildContext* ctx) const {
  ctx->input_cols = utils::parquet::BindColumns(
      table, input_columns_, true, true, "GroupClassifierEventLoader inputs");
  ctx->target_cols = utils::parquet::BindColumns(
      table, target_columns_, false, true, "GroupClassifierEventLoader targets");
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

  ctx->x_values = MakeNumericAccessor(ctx->hits_x->values(), "group classifier event x");
  ctx->y_values = MakeNumericAccessor(ctx->hits_y->values(), "group classifier event y");
  ctx->z_values = MakeNumericAccessor(ctx->hits_z->values(), "group classifier event z");
  ctx->edep_values =
      MakeNumericAccessor(ctx->hits_edep->values(), "group classifier event edep");

  auto view_values =
      std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(ctx->hits_view->values());
  auto tg_values = std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(
      ctx->hits_time_group->values());
  ctx->view_raw = view_values->raw_values();
  ctx->tg_raw = tg_values->raw_values();
  ctx->z_offsets = ctx->hits_z->raw_value_offsets();
  ctx->tg_offsets = ctx->hits_time_group->raw_value_offsets();

  ctx->rows = table.num_rows();
  ctx->node_counts.assign(static_cast<size_t>(ctx->rows), 0);
  ctx->edge_counts.assign(static_cast<size_t>(ctx->rows), 0);
  ctx->group_counts.assign(static_cast<size_t>(ctx->rows), 0);
}

void GroupClassifierEventLoader::BuildGraphPhase1Count(BuildContext* ctx) const {
  CountNodeEdgePerRow(ctx->z_offsets, ctx->rows, &ctx->node_counts, &ctx->edge_counts);
  CountGroupsForRows(
      ctx->z_offsets, ctx->tg_offsets, ctx->tg_raw, ctx->rows, &ctx->group_counts);
}

void GroupClassifierEventLoader::BuildGraphPhase2Offsets(BuildContext* ctx) const {
  ctx->node_offsets = BuildOffsets(ctx->node_counts);
  ctx->edge_offsets = BuildOffsets(ctx->edge_counts);
  ctx->total_nodes = ctx->node_offsets.back();
  ctx->total_edges = ctx->edge_offsets.back();

  ctx->row_group_offsets.assign(static_cast<size_t>(ctx->rows + 1), 0);
  for (int64_t row = 0; row < ctx->rows; ++row) {
    ctx->row_group_offsets[static_cast<size_t>(row + 1)] =
        ctx->row_group_offsets[static_cast<size_t>(row)] +
        ctx->group_counts[static_cast<size_t>(row)];
  }
  ctx->total_groups = ctx->row_group_offsets.back();
}

void GroupClassifierEventLoader::BuildGraphPhase3Allocate(const BuildContext& ctx,
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
      AllocBuffer((ctx.rows + 1) * static_cast<int64_t>(sizeof(int64_t)));
  bufs->edge_ptr_buf =
      AllocBuffer((ctx.rows + 1) * static_cast<int64_t>(sizeof(int64_t)));
  bufs->group_ptr_buf =
      AllocBuffer((ctx.rows + 1) * static_cast<int64_t>(sizeof(int64_t)));
  bufs->u_buf = AllocBuffer(ctx.rows * static_cast<int64_t>(sizeof(float)));

  bufs->node_feat = reinterpret_cast<float*>(bufs->node_feat_buf->mutable_data());
  bufs->edge_index = reinterpret_cast<int64_t*>(bufs->edge_index_buf->mutable_data());
  bufs->edge_attr = reinterpret_cast<float*>(bufs->edge_attr_buf->mutable_data());
  bufs->time_group_ids = reinterpret_cast<int64_t*>(bufs->time_group_buf->mutable_data());
  bufs->node_ptr = reinterpret_cast<int64_t*>(bufs->node_ptr_buf->mutable_data());
  bufs->edge_ptr = reinterpret_cast<int64_t*>(bufs->edge_ptr_buf->mutable_data());
  bufs->group_ptr = reinterpret_cast<int64_t*>(bufs->group_ptr_buf->mutable_data());
  bufs->u = reinterpret_cast<float*>(bufs->u_buf->mutable_data());

  FillPointerArrayFromOffsets(ctx.node_offsets, bufs->node_ptr);
  FillPointerArrayFromOffsets(ctx.edge_offsets, bufs->edge_ptr);
}

void GroupClassifierEventLoader::BuildGraphPhase4Populate(const BuildContext& ctx,
                                                          BuildBuffers* bufs) const {
  utils::parallel::Parallel::For(0, ctx.rows, [&](int64_t row) {
    const int64_t n = ctx.node_counts[static_cast<size_t>(row)];
    const int64_t node_offset = ctx.node_offsets[static_cast<size_t>(row)];
    const int64_t edge_offset = ctx.edge_offsets[static_cast<size_t>(row)];
    const int32_t start = ctx.z_offsets[row];
    const int32_t tg_start = ctx.tg_offsets[row];
    const int64_t tg_len =
        static_cast<int64_t>(ctx.tg_offsets[row + 1] - ctx.tg_offsets[row]);

    std::vector<float> coord_local(static_cast<size_t>(n), 0.0f);
    std::vector<float> z_local(static_cast<size_t>(n), 0.0f);
    std::vector<float> e_local(static_cast<size_t>(n), 0.0f);
    std::vector<int32_t> view_local(static_cast<size_t>(n), 0);

    double sum_edep = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      const int64_t idx = node_offset + i;
      const int64_t base = idx * 4;
      const int32_t view = ctx.view_raw[start + i];
      const double coord =
          ResolveCoordinateForView(ctx.x_values, ctx.y_values, view, start + i);
      const float z_value = static_cast<float>(
          ctx.z_values.IsValid(start + i) ? ctx.z_values.Value(start + i) : 0.0);
      const float e_value = static_cast<float>(
          ctx.edep_values.IsValid(start + i) ? ctx.edep_values.Value(start + i) : 0.0);
      coord_local[static_cast<size_t>(i)] = static_cast<float>(coord);
      z_local[static_cast<size_t>(i)] = z_value;
      e_local[static_cast<size_t>(i)] = e_value;
      view_local[static_cast<size_t>(i)] = view;

      bufs->node_feat[base] = coord_local[static_cast<size_t>(i)];
      bufs->node_feat[base + 1] = z_value;
      bufs->node_feat[base + 2] = e_value;
      bufs->node_feat[base + 3] = static_cast<float>(view);

      if (tg_len < n) {
        throw std::runtime_error("hits_time_group length mismatch with hits.");
      }
      bufs->time_group_ids[idx] = ctx.tg_raw[tg_start + i];
      sum_edep += e_value;
    }
    bufs->u[row] = static_cast<float>(sum_edep);

    int64_t edge_local = 0;
    for (int64_t i = 0; i < n; ++i) {
      const int32_t view_i = view_local[static_cast<size_t>(i)];
      const float coord_i = coord_local[static_cast<size_t>(i)];
      const float z_i = z_local[static_cast<size_t>(i)];
      const float e_i = e_local[static_cast<size_t>(i)];
      for (int64_t j = 0; j < n; ++j) {
        if (i == j) {
          continue;
        }
        const int64_t edge_idx = edge_offset + edge_local;
        const int64_t edge_base = edge_idx * 2;
        const int64_t attr_base = edge_idx * 4;

        bufs->edge_index[edge_base] = node_offset + i;
        bufs->edge_index[edge_base + 1] = node_offset + j;

        const int32_t view_j = view_local[static_cast<size_t>(j)];
        const float coord_j = coord_local[static_cast<size_t>(j)];
        const float z_j = z_local[static_cast<size_t>(j)];
        const float e_j = e_local[static_cast<size_t>(j)];

        bufs->edge_attr[attr_base] = coord_j - coord_i;
        bufs->edge_attr[attr_base + 1] = z_j - z_i;
        bufs->edge_attr[attr_base + 2] = e_j - e_i;
        bufs->edge_attr[attr_base + 3] = (view_i == view_j) ? 1.0f : 0.0f;
        edge_local++;
      }
    }
  });
}

std::unique_ptr<BaseBatch> GroupClassifierEventLoader::BuildGraphPhase5Finalize(
    const BuildContext& ctx,
    BuildBuffers* bufs) const {
  bufs->group_ptr[0] = 0;
  for (int64_t row = 0; row < ctx.rows; ++row) {
    bufs->group_ptr[row + 1] =
        bufs->group_ptr[row] + ctx.group_counts[static_cast<size_t>(row)];
  }

  if (ctx.has_targets) {
    bufs->y_buf =
        AllocBuffer(ctx.total_groups * static_cast<int64_t>(sizeof(float)) * 3);
    bufs->y = reinterpret_cast<float*>(bufs->y_buf->mutable_data());
    EncodeTargets(
        ctx.target_cols, ctx.group_counts, ctx.row_group_offsets, ctx.rows, bufs->y);
  }

  auto out = std::make_unique<GroupClassifierInputs>();
  out->node_features = MakeArray(bufs->node_feat_buf, arrow::float32(), ctx.total_nodes * 4);
  out->edge_index = MakeArray(bufs->edge_index_buf, arrow::int64(), ctx.total_edges * 2);
  out->edge_attr = MakeArray(bufs->edge_attr_buf, arrow::float32(), ctx.total_edges * 4);
  out->time_group_ids = MakeArray(bufs->time_group_buf, arrow::int64(), ctx.total_nodes);
  out->node_ptr = MakeArray(bufs->node_ptr_buf, arrow::int64(), ctx.rows + 1);
  out->edge_ptr = MakeArray(bufs->edge_ptr_buf, arrow::int64(), ctx.rows + 1);
  out->group_ptr = MakeArray(bufs->group_ptr_buf, arrow::int64(), ctx.rows + 1);
  out->u = MakeArray(bufs->u_buf, arrow::float32(), ctx.rows);
  if (ctx.has_targets) {
    out->y = MakeArray(bufs->y_buf, arrow::float32(), ctx.total_groups * 3);
  } else {
    out->y = nullptr;
  }
  out->num_graphs = static_cast<size_t>(ctx.rows);
  out->num_groups = static_cast<size_t>(ctx.total_groups);
  return out;
}

TrainingBundle GroupClassifierEventLoader::SplitInputsTargets(std::unique_ptr<BaseBatch> batch_base) const {
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
