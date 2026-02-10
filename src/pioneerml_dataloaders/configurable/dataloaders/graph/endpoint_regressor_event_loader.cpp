#include "pioneerml_dataloaders/configurable/dataloaders/graph/endpoint_regressor_event_loader.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#include <arrow/api.h>

#include "pioneerml_dataloaders/batch/endpoint_regressor_batch.h"
#include "pioneerml_dataloaders/configurable/data_derivers/time_group_summary_deriver.h"
#include "pioneerml_dataloaders/utils/parallel/parallel.h"
#include "pioneerml_dataloaders/utils/timing/scoped_timer.h"

namespace pioneerml::dataloaders::graph {
namespace {

int ClassFromPdg(int pdg) {
  if (pdg == 211) {
    return 0;
  }
  if (pdg == -13) {
    return 1;
  }
  if (pdg == -11 || pdg == 11) {
    return 2;
  }
  return -1;
}

}  // namespace

EndpointRegressorEventLoader::EndpointRegressorEventLoader() {
  input_columns_ = {
      "hits_x",
      "hits_y",
      "hits_z",
      "hits_edep",
      "hits_strip_type",
      "hits_time",
      "hits_time_group",
      "hits_pdg_id",
  };
  target_columns_ = {
      "group_start_x",
      "group_start_y",
      "group_start_z",
      "group_end_x",
      "group_end_y",
      "group_end_z",
  };
  ConfigureDerivers(nullptr);
}

void EndpointRegressorEventLoader::LoadConfig(const nlohmann::json& cfg) {
  if (cfg.contains("time_window_ns")) {
    time_window_ns_ = cfg.at("time_window_ns").get<double>();
  }
  if (cfg.contains("use_group_probs")) {
    use_group_probs_ = cfg.at("use_group_probs").get<bool>();
  }
  if (cfg.contains("use_splitter_probs")) {
    use_splitter_probs_ = cfg.at("use_splitter_probs").get<bool>();
  }

  const nlohmann::json* derivers_cfg = nullptr;
  if (cfg.contains("derivers")) {
    derivers_cfg = &cfg.at("derivers");
  }
  ConfigureDerivers(derivers_cfg);
}

void EndpointRegressorEventLoader::ConfigureDerivers(const nlohmann::json* derivers_cfg) {
  derivers_.clear();

  auto summary = std::make_shared<data_derivers::TimeGroupSummaryDeriver>(
      time_window_ns_,
      std::vector<std::string>{
          "hits_time_group",
          "hits_pdg_id",
          "hits_particle_mask",
          "group_start_x",
          "group_start_y",
          "group_start_z",
          "group_end_x",
          "group_end_y",
          "group_end_z",
      });
  if (derivers_cfg && derivers_cfg->contains("time_group_summary")) {
    summary->LoadConfig(derivers_cfg->at("time_group_summary"));
  }
  AddDeriver({
                 "hits_time_group",
                 "hits_pdg_id",
                 "hits_particle_mask",
                 "group_start_x",
                 "group_start_y",
                 "group_start_z",
                 "group_end_x",
                 "group_end_y",
                 "group_end_z",
             },
             summary);
}

void EndpointRegressorEventLoader::CountRows(const int32_t* offsets,
                                        const int32_t* tg_offsets,
                                        const int64_t* tg_raw,
                                        const int32_t* pdg_offsets,
                                        int64_t rows,
                                        std::vector<int64_t>* node_counts,
                                        std::vector<int64_t>* edge_counts,
                                        std::vector<int64_t>* group_counts) const {
  utils::parallel::Parallel::For(0, rows, [&](int64_t row) {
    const int64_t n = static_cast<int64_t>(offsets[row + 1] - offsets[row]);
    (*node_counts)[static_cast<size_t>(row)] = n;
    (*edge_counts)[static_cast<size_t>(row)] = n * (n - 1);

    if ((tg_offsets[row + 1] - tg_offsets[row]) != n) {
      throw std::runtime_error("hits_time_group length mismatch with hits.");
    }
    if ((pdg_offsets[row + 1] - pdg_offsets[row]) != n) {
      throw std::runtime_error("hits_pdg_id length mismatch with hits.");
    }

    if (n == 0) {
      (*group_counts)[static_cast<size_t>(row)] = 0;
      return;
    }

    const int32_t tg_start = tg_offsets[row];
    int64_t max_group = -1;
    for (int64_t i = 0; i < n; ++i) {
      const int64_t tg_val = tg_raw[tg_start + i];
      if (tg_val < 0) {
        throw std::runtime_error("Invalid negative time group id encountered.");
      }
      max_group = std::max(max_group, tg_val);
    }
    const int64_t groups_for_row = std::max<int64_t>(0, max_group + 1);
    if (groups_for_row == 0 && n > 0) {
      throw std::runtime_error("No time groups found for non-empty event.");
    }
    (*group_counts)[static_cast<size_t>(row)] = groups_for_row;
  });
}

void EndpointRegressorEventLoader::EncodeTargets(const ColumnMap& target_cols,
                                            const std::vector<int64_t>& group_counts,
                                            const std::vector<int64_t>& row_group_offsets,
                                            int64_t rows,
                                            float* y) const {
  const auto& start_x_list = static_cast<const arrow::ListArray&>(*target_cols.at("group_start_x")->chunk(0));
  const auto& start_y_list = static_cast<const arrow::ListArray&>(*target_cols.at("group_start_y")->chunk(0));
  const auto& start_z_list = static_cast<const arrow::ListArray&>(*target_cols.at("group_start_z")->chunk(0));
  const auto& end_x_list = static_cast<const arrow::ListArray&>(*target_cols.at("group_end_x")->chunk(0));
  const auto& end_y_list = static_cast<const arrow::ListArray&>(*target_cols.at("group_end_y")->chunk(0));
  const auto& end_z_list = static_cast<const arrow::ListArray&>(*target_cols.at("group_end_z")->chunk(0));

  auto sx = MakeNumericAccessor(start_x_list.values(), "group_start_x");
  auto sy = MakeNumericAccessor(start_y_list.values(), "group_start_y");
  auto sz = MakeNumericAccessor(start_z_list.values(), "group_start_z");
  auto ex = MakeNumericAccessor(end_x_list.values(), "group_end_x");
  auto ey = MakeNumericAccessor(end_y_list.values(), "group_end_y");
  auto ez = MakeNumericAccessor(end_z_list.values(), "group_end_z");

  const int32_t* sx_off = start_x_list.raw_value_offsets();
  const int32_t* sy_off = start_y_list.raw_value_offsets();
  const int32_t* sz_off = start_z_list.raw_value_offsets();
  const int32_t* ex_off = end_x_list.raw_value_offsets();
  const int32_t* ey_off = end_y_list.raw_value_offsets();
  const int32_t* ez_off = end_z_list.raw_value_offsets();

  for (int64_t row = 0; row < rows; ++row) {
    const int64_t count = static_cast<int64_t>(sx_off[row + 1] - sx_off[row]);
    if (count != static_cast<int64_t>(sy_off[row + 1] - sy_off[row]) ||
        count != static_cast<int64_t>(sz_off[row + 1] - sz_off[row]) ||
        count != static_cast<int64_t>(ex_off[row + 1] - ex_off[row]) ||
        count != static_cast<int64_t>(ey_off[row + 1] - ey_off[row]) ||
        count != static_cast<int64_t>(ez_off[row + 1] - ez_off[row])) {
      throw std::runtime_error("Endpoint target list columns have mismatched lengths.");
    }
    if (count != group_counts[static_cast<size_t>(row)]) {
      throw std::runtime_error("Endpoint target list length does not match time groups.");
    }

    const int64_t base_group = row_group_offsets[static_cast<size_t>(row)];
    const int32_t start = sx_off[row];
    for (int64_t g = 0; g < count; ++g) {
      const int32_t idx = start + static_cast<int32_t>(g);
      const int64_t out_base = (base_group + g) * 6;
      y[out_base] = static_cast<float>(sx.Value(idx));
      y[out_base + 1] = static_cast<float>(sy.Value(idx));
      y[out_base + 2] = static_cast<float>(sz.Value(idx));
      y[out_base + 3] = static_cast<float>(ex.Value(idx));
      y[out_base + 4] = static_cast<float>(ey.Value(idx));
      y[out_base + 5] = static_cast<float>(ez.Value(idx));
    }
  }
}

TrainingBundle EndpointRegressorEventLoader::LoadTraining(
    const std::shared_ptr<arrow::Table>& table) const {
  auto prepared = PrepareTable(table, true);
  auto required = utils::parquet::MergeColumns(input_columns_, target_columns_);
  utils::parquet::ValidateColumns(*prepared,
                                  required,
                                  {"pred_pion", "pred_muon", "pred_mip", "pred_hit_pion",
                                   "pred_hit_muon", "pred_hit_mip"},
                                  true,
                                  "EndpointRegressorEventLoader training");
  auto batch = BuildGraph(*prepared);
  return SplitInputsTargets(std::move(batch));
}

InferenceBundle EndpointRegressorEventLoader::LoadInference(
    const std::shared_ptr<arrow::Table>& table) const {
  auto prepared = PrepareTable(table, true);
  utils::parquet::ValidateColumns(*prepared,
                                  input_columns_,
                                  {"pred_pion", "pred_muon", "pred_mip", "pred_hit_pion",
                                   "pred_hit_muon", "pred_hit_mip", "group_start_x",
                                   "group_start_y", "group_start_z", "group_end_x",
                                   "group_end_y", "group_end_z"},
                                  true,
                                  "EndpointRegressorEventLoader inputs");
  InferenceBundle out;
  out.inputs = BuildGraph(*prepared);
  return out;
}

std::unique_ptr<BaseBatch> EndpointRegressorEventLoader::BuildGraph(const arrow::Table& table) const {
  utils::timing::ScopedTimer total_timer("endpoint_regressor.build_graph");
  BuildContext ctx;
  BuildBuffers bufs;
  {
    utils::timing::ScopedTimer timer("endpoint_regressor.phase0_initialize");
    BuildGraphPhase0Initialize(table, &ctx);
  }
  {
    utils::timing::ScopedTimer timer("endpoint_regressor.phase1_count");
    BuildGraphPhase1Count(&ctx);
  }
  {
    utils::timing::ScopedTimer timer("endpoint_regressor.phase2_offsets");
    BuildGraphPhase2Offsets(&ctx);
  }
  {
    utils::timing::ScopedTimer timer("endpoint_regressor.phase3_allocate");
    BuildGraphPhase3Allocate(ctx, &bufs);
  }
  {
    utils::timing::ScopedTimer timer("endpoint_regressor.phase4_populate");
    BuildGraphPhase4Populate(ctx, &bufs);
  }
  {
    utils::timing::ScopedTimer timer("endpoint_regressor.phase5_finalize");
    return BuildGraphPhase5Finalize(ctx, &bufs);
  }
}

void EndpointRegressorEventLoader::BuildGraphPhase0Initialize(const arrow::Table& table,
                                                         BuildContext* ctx) const {
  ctx->input_cols = utils::parquet::BindColumns(
      table, input_columns_, true, true, "EndpointRegressorEventLoader inputs");
  ctx->target_cols = utils::parquet::BindColumns(
      table, target_columns_, false, true, "EndpointRegressorEventLoader targets");
  ctx->prob_cols = utils::parquet::BindColumns(
      table, {"pred_pion", "pred_muon", "pred_mip"}, false, true, "EndpointRegressorEventLoader probs");
  ctx->splitter_cols = utils::parquet::BindColumns(
      table,
      {"pred_hit_pion", "pred_hit_muon", "pred_hit_mip", "time_group_ids"},
      false,
      true,
      "EndpointRegressorEventLoader splitter probs");

  ctx->has_targets = ctx->target_cols.size() == target_columns_.size();
  ctx->has_prob_columns = ctx->prob_cols.size() == 3;
  ctx->has_splitter_columns = ctx->splitter_cols.count("pred_hit_pion") > 0 &&
                              ctx->splitter_cols.count("pred_hit_muon") > 0 &&
                              ctx->splitter_cols.count("pred_hit_mip") > 0;

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
  ctx->hits_pdg_id = &static_cast<const arrow::ListArray&>(
      *ctx->input_cols.at("hits_pdg_id")->chunk(0));

  ctx->x_values = MakeNumericAccessor(ctx->hits_x->values(), "endpoint regressor x");
  ctx->y_values = MakeNumericAccessor(ctx->hits_y->values(), "endpoint regressor y");
  ctx->z_values = MakeNumericAccessor(ctx->hits_z->values(), "endpoint regressor z");
  ctx->edep_values = MakeNumericAccessor(ctx->hits_edep->values(), "endpoint regressor edep");

  auto view_values =
      std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(ctx->hits_view->values());
  auto tg_values = std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(
      ctx->hits_time_group->values());
  auto pdg_values = std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(
      ctx->hits_pdg_id->values());
  ctx->view_raw = view_values->raw_values();
  ctx->tg_raw = tg_values->raw_values();
  ctx->pdg_raw = pdg_values->raw_values();
  ctx->offsets = ctx->hits_z->raw_value_offsets();
  ctx->tg_offsets = ctx->hits_time_group->raw_value_offsets();
  ctx->pdg_offsets = ctx->hits_pdg_id->raw_value_offsets();

  ctx->rows = table.num_rows();
  ctx->node_counts.assign(static_cast<size_t>(ctx->rows), 0);
  ctx->edge_counts.assign(static_cast<size_t>(ctx->rows), 0);
  ctx->group_counts.assign(static_cast<size_t>(ctx->rows), 0);
}

void EndpointRegressorEventLoader::BuildGraphPhase1Count(BuildContext* ctx) const {
  CountRows(ctx->offsets,
            ctx->tg_offsets,
            ctx->tg_raw,
            ctx->pdg_offsets,
            ctx->rows,
            &ctx->node_counts,
            &ctx->edge_counts,
            &ctx->group_counts);
}

void EndpointRegressorEventLoader::BuildGraphPhase2Offsets(BuildContext* ctx) const {
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

void EndpointRegressorEventLoader::BuildGraphPhase3Allocate(const BuildContext& ctx,
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
  bufs->group_probs_buf =
      AllocBuffer(ctx.total_groups * static_cast<int64_t>(sizeof(float)) * 3);
  bufs->splitter_probs_buf =
      AllocBuffer(ctx.total_nodes * static_cast<int64_t>(sizeof(float)) * 3);
  bufs->graph_event_ids_buf =
      AllocBuffer(ctx.rows * static_cast<int64_t>(sizeof(int64_t)));
  bufs->y_buf =
      AllocBuffer(ctx.total_groups * static_cast<int64_t>(sizeof(float)) * 6);

  bufs->node_feat = reinterpret_cast<float*>(bufs->node_feat_buf->mutable_data());
  bufs->edge_index = reinterpret_cast<int64_t*>(bufs->edge_index_buf->mutable_data());
  bufs->edge_attr = reinterpret_cast<float*>(bufs->edge_attr_buf->mutable_data());
  bufs->time_group_ids = reinterpret_cast<int64_t*>(bufs->time_group_buf->mutable_data());
  bufs->node_ptr = reinterpret_cast<int64_t*>(bufs->node_ptr_buf->mutable_data());
  bufs->edge_ptr = reinterpret_cast<int64_t*>(bufs->edge_ptr_buf->mutable_data());
  bufs->group_ptr = reinterpret_cast<int64_t*>(bufs->group_ptr_buf->mutable_data());
  bufs->u = reinterpret_cast<float*>(bufs->u_buf->mutable_data());
  bufs->group_probs = reinterpret_cast<float*>(bufs->group_probs_buf->mutable_data());
  bufs->splitter_probs = reinterpret_cast<float*>(bufs->splitter_probs_buf->mutable_data());
  bufs->graph_event_ids = reinterpret_cast<int64_t*>(bufs->graph_event_ids_buf->mutable_data());
  bufs->y = reinterpret_cast<float*>(bufs->y_buf->mutable_data());

  bufs->group_truth.assign(static_cast<size_t>(ctx.total_groups * 3), 0U);
  bufs->node_truth.assign(static_cast<size_t>(ctx.total_nodes * 3), 0.0f);
  std::fill(bufs->group_probs, bufs->group_probs + (ctx.total_groups * 3), 0.0f);
  std::fill(bufs->splitter_probs, bufs->splitter_probs + (ctx.total_nodes * 3), 0.0f);
  std::fill(bufs->y, bufs->y + (ctx.total_groups * 6), 0.0f);

  FillPointerArrayFromOffsets(ctx.node_offsets, bufs->node_ptr);
  FillPointerArrayFromOffsets(ctx.edge_offsets, bufs->edge_ptr);
}

void EndpointRegressorEventLoader::BuildGraphPhase4Populate(const BuildContext& ctx,
                                                       BuildBuffers* bufs) const {
  utils::parallel::Parallel::For(0, ctx.rows, [&](int64_t row) {
    const int64_t n = static_cast<int64_t>(ctx.offsets[row + 1] - ctx.offsets[row]);
    if (n == 0) {
      bufs->graph_event_ids[row] = row;
      return;
    }

    const int64_t node_offset = ctx.node_offsets[static_cast<size_t>(row)];
    const int64_t edge_offset = ctx.edge_offsets[static_cast<size_t>(row)];
    const int64_t group_offset = ctx.row_group_offsets[static_cast<size_t>(row)];

    const int32_t start = ctx.offsets[row];
    const int32_t tg_start = ctx.tg_offsets[row];

    std::vector<float> coord_local(static_cast<size_t>(n), 0.0f);
    std::vector<float> z_local(static_cast<size_t>(n), 0.0f);
    std::vector<float> e_local(static_cast<size_t>(n), 0.0f);
    std::vector<int32_t> view_local(static_cast<size_t>(n), 0);

    double sum_edep = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      const int64_t raw_idx = start + i;
      const int64_t node_idx = node_offset + i;
      const int64_t base = node_idx * 4;
      const int32_t view = ctx.view_raw[raw_idx];
      const double coord = ResolveCoordinateForView(ctx.x_values, ctx.y_values, view, raw_idx);
      const float z =
          static_cast<float>(ctx.z_values.IsValid(raw_idx) ? ctx.z_values.Value(raw_idx) : 0.0);
      const float e = static_cast<float>(
          ctx.edep_values.IsValid(raw_idx) ? ctx.edep_values.Value(raw_idx) : 0.0);

      coord_local[static_cast<size_t>(i)] = static_cast<float>(coord);
      z_local[static_cast<size_t>(i)] = z;
      e_local[static_cast<size_t>(i)] = e;
      view_local[static_cast<size_t>(i)] = view;

      bufs->node_feat[base] = coord_local[static_cast<size_t>(i)];
      bufs->node_feat[base + 1] = z;
      bufs->node_feat[base + 2] = e;
      bufs->node_feat[base + 3] = static_cast<float>(view);

      const int64_t local_group = ctx.tg_raw[tg_start + i];
      if (local_group < 0 || local_group >= ctx.group_counts[static_cast<size_t>(row)]) {
        throw std::runtime_error("Invalid time group id encountered while populating graph.");
      }
      bufs->time_group_ids[node_idx] = local_group;

      const int cls = ClassFromPdg(ctx.pdg_raw[raw_idx]);
      if (cls >= 0) {
        bufs->node_truth[static_cast<size_t>(node_idx * 3 + cls)] = 1.0f;
        const int64_t global_group = group_offset + local_group;
        bufs->group_truth[static_cast<size_t>(global_group * 3 + cls)] = 1U;
      }

      sum_edep += e;
    }

    bufs->graph_event_ids[row] = row;
    bufs->u[row] = static_cast<float>(sum_edep);

    int64_t edge_local = 0;
    for (int64_t a = 0; a < n; ++a) {
      const int32_t view_i = view_local[static_cast<size_t>(a)];
      const float z_i = z_local[static_cast<size_t>(a)];
      const float e_i = e_local[static_cast<size_t>(a)];
      const float coord_i = coord_local[static_cast<size_t>(a)];

      for (int64_t b = 0; b < n; ++b) {
        if (a == b) {
          continue;
        }
        const int32_t view_j = view_local[static_cast<size_t>(b)];
        const float z_j = z_local[static_cast<size_t>(b)];
        const float e_j = e_local[static_cast<size_t>(b)];
        const float coord_j = coord_local[static_cast<size_t>(b)];

        const int64_t edge_idx = edge_offset + edge_local;
        const int64_t edge_base = edge_idx * 2;
        const int64_t attr_base = edge_idx * 4;
        bufs->edge_index[edge_base] = node_offset + a;
        bufs->edge_index[edge_base + 1] = node_offset + b;
        bufs->edge_attr[attr_base] = static_cast<float>(coord_j - coord_i);
        bufs->edge_attr[attr_base + 1] = static_cast<float>(z_j - z_i);
        bufs->edge_attr[attr_base + 2] = static_cast<float>(e_j - e_i);
        bufs->edge_attr[attr_base + 3] = (view_i == view_j) ? 1.0f : 0.0f;
        edge_local++;
      }
    }
  });
}

std::unique_ptr<BaseBatch> EndpointRegressorEventLoader::BuildGraphPhase5Finalize(
    const BuildContext& ctx,
    BuildBuffers* bufs) const {
  FillPointerArrayFromOffsets(ctx.row_group_offsets, bufs->group_ptr);

  if (use_group_probs_) {
    for (int64_t group_idx = 0; group_idx < ctx.total_groups; ++group_idx) {
      const int64_t base = group_idx * 3;
      bufs->group_probs[base] =
          static_cast<float>(bufs->group_truth[static_cast<size_t>(base)]);
      bufs->group_probs[base + 1] =
          static_cast<float>(bufs->group_truth[static_cast<size_t>(base + 1)]);
      bufs->group_probs[base + 2] =
          static_cast<float>(bufs->group_truth[static_cast<size_t>(base + 2)]);
    }
  }

  if (use_splitter_probs_) {
    std::copy(bufs->node_truth.begin(), bufs->node_truth.end(), bufs->splitter_probs);
  }

  if (ctx.has_prob_columns) {
    const auto& pion_pred_list =
        static_cast<const arrow::ListArray&>(*ctx.prob_cols.at("pred_pion")->chunk(0));
    const auto& muon_pred_list =
        static_cast<const arrow::ListArray&>(*ctx.prob_cols.at("pred_muon")->chunk(0));
    const auto& mip_pred_list =
        static_cast<const arrow::ListArray&>(*ctx.prob_cols.at("pred_mip")->chunk(0));

    auto pion_pred_values = MakeNumericAccessor(pion_pred_list.values(), "pred_pion");
    auto muon_pred_values = MakeNumericAccessor(muon_pred_list.values(), "pred_muon");
    auto mip_pred_values = MakeNumericAccessor(mip_pred_list.values(), "pred_mip");

    const int32_t* p_offsets = pion_pred_list.raw_value_offsets();
    const int32_t* m_offsets = muon_pred_list.raw_value_offsets();
    const int32_t* i_offsets = mip_pred_list.raw_value_offsets();

    for (int64_t row = 0; row < ctx.rows; ++row) {
      const int64_t count_p = static_cast<int64_t>(p_offsets[row + 1] - p_offsets[row]);
      const int64_t count_m = static_cast<int64_t>(m_offsets[row + 1] - m_offsets[row]);
      const int64_t count_i = static_cast<int64_t>(i_offsets[row + 1] - i_offsets[row]);
      const int64_t expected = ctx.group_counts[static_cast<size_t>(row)];
      const int64_t count = std::min(expected, std::min(count_p, std::min(count_m, count_i)));
      const int64_t group_base = ctx.row_group_offsets[static_cast<size_t>(row)];
      const int32_t start_idx = p_offsets[row];
      for (int64_t g = 0; g < count; ++g) {
        const int32_t list_idx = start_idx + static_cast<int32_t>(g);
        const int64_t base = (group_base + g) * 3;
        bufs->group_probs[base] = static_cast<float>(pion_pred_values.Value(list_idx));
        bufs->group_probs[base + 1] = static_cast<float>(muon_pred_values.Value(list_idx));
        bufs->group_probs[base + 2] = static_cast<float>(mip_pred_values.Value(list_idx));
      }
    }
  }

  if (ctx.has_splitter_columns) {
    const auto& pion_pred_list =
        static_cast<const arrow::ListArray&>(*ctx.splitter_cols.at("pred_hit_pion")->chunk(0));
    const auto& muon_pred_list =
        static_cast<const arrow::ListArray&>(*ctx.splitter_cols.at("pred_hit_muon")->chunk(0));
    const auto& mip_pred_list =
        static_cast<const arrow::ListArray&>(*ctx.splitter_cols.at("pred_hit_mip")->chunk(0));

    auto pion_pred_values = MakeNumericAccessor(pion_pred_list.values(), "pred_hit_pion");
    auto muon_pred_values = MakeNumericAccessor(muon_pred_list.values(), "pred_hit_muon");
    auto mip_pred_values = MakeNumericAccessor(mip_pred_list.values(), "pred_hit_mip");

    const int32_t* p_offsets = pion_pred_list.raw_value_offsets();
    const int32_t* m_offsets = muon_pred_list.raw_value_offsets();
    const int32_t* i_offsets = mip_pred_list.raw_value_offsets();

    for (int64_t row = 0; row < ctx.rows; ++row) {
      const int64_t expected_nodes =
          static_cast<int64_t>(ctx.offsets[row + 1] - ctx.offsets[row]);
      const int64_t p_count = static_cast<int64_t>(p_offsets[row + 1] - p_offsets[row]);
      const int64_t m_count = static_cast<int64_t>(m_offsets[row + 1] - m_offsets[row]);
      const int64_t i_count = static_cast<int64_t>(i_offsets[row + 1] - i_offsets[row]);
      const int64_t count = std::min(expected_nodes, std::min(p_count, std::min(m_count, i_count)));
      const int64_t node_base = ctx.node_offsets[static_cast<size_t>(row)];
      const int32_t start_idx = p_offsets[row];
      for (int64_t i = 0; i < count; ++i) {
        const int32_t list_idx = start_idx + static_cast<int32_t>(i);
        const int64_t out_base = (node_base + i) * 3;
        bufs->splitter_probs[out_base] = static_cast<float>(pion_pred_values.Value(list_idx));
        bufs->splitter_probs[out_base + 1] = static_cast<float>(muon_pred_values.Value(list_idx));
        bufs->splitter_probs[out_base + 2] = static_cast<float>(mip_pred_values.Value(list_idx));
      }
    }
  }

  if (ctx.has_targets) {
    EncodeTargets(ctx.target_cols, ctx.group_counts, ctx.row_group_offsets, ctx.rows, bufs->y);
  }

  auto out = std::make_unique<EndpointRegressorInputs>();
  out->node_features = MakeArray(bufs->node_feat_buf, arrow::float32(), ctx.total_nodes * 4);
  out->edge_index = MakeArray(bufs->edge_index_buf, arrow::int64(), ctx.total_edges * 2);
  out->edge_attr = MakeArray(bufs->edge_attr_buf, arrow::float32(), ctx.total_edges * 4);
  out->time_group_ids = MakeArray(bufs->time_group_buf, arrow::int64(), ctx.total_nodes);
  out->u = MakeArray(bufs->u_buf, arrow::float32(), ctx.rows);
  out->group_probs = MakeArray(bufs->group_probs_buf, arrow::float32(), ctx.total_groups * 3);
  out->splitter_probs =
      MakeArray(bufs->splitter_probs_buf, arrow::float32(), ctx.total_nodes * 3);
  out->node_ptr = MakeArray(bufs->node_ptr_buf, arrow::int64(), ctx.rows + 1);
  out->edge_ptr = MakeArray(bufs->edge_ptr_buf, arrow::int64(), ctx.rows + 1);
  out->group_ptr = MakeArray(bufs->group_ptr_buf, arrow::int64(), ctx.rows + 1);
  out->graph_event_ids = MakeArray(bufs->graph_event_ids_buf, arrow::int64(), ctx.rows);
  out->y = MakeArray(bufs->y_buf, arrow::float32(), ctx.total_groups * 6);
  out->num_graphs = static_cast<size_t>(ctx.rows);
  out->num_groups = static_cast<size_t>(ctx.total_groups);

  if (!ctx.has_targets) {
    out->y.reset();
  }

  return out;
}

TrainingBundle EndpointRegressorEventLoader::SplitInputsTargets(
    std::unique_ptr<BaseBatch> batch_base) const {
  auto* typed = dynamic_cast<EndpointRegressorInputs*>(batch_base.get());
  if (!typed) {
    throw std::runtime_error("Unexpected batch type in SplitInputsTargets");
  }
  if (!typed->y) {
    throw std::runtime_error(
        "Training targets are missing. Use LoadInference or provide endpoint target columns.");
  }

  auto targets = std::make_unique<EndpointRegressorTargets>();
  targets->num_groups = typed->num_groups;
  targets->y = typed->y;

  typed->y.reset();

  TrainingBundle result;
  result.inputs = std::move(batch_base);
  result.targets = std::move(targets);
  return result;
}

}  // namespace pioneerml::dataloaders::graph
