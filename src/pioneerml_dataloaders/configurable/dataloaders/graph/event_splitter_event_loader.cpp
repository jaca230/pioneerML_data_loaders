#include "pioneerml_dataloaders/configurable/dataloaders/graph/event_splitter_event_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <arrow/api.h>

#include "pioneerml_dataloaders/batch/event_splitter_event_batch.h"
#include "pioneerml_dataloaders/configurable/data_derivers/time_grouper.h"
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

struct StepKey {
  int32_t mc_event_id{0};
  int32_t step_id{0};

  bool operator==(const StepKey& other) const {
    return mc_event_id == other.mc_event_id && step_id == other.step_id;
  }
};

struct StepKeyHash {
  std::size_t operator()(const StepKey& key) const noexcept {
    const std::size_t h1 = std::hash<int32_t>{}(key.mc_event_id);
    const std::size_t h2 = std::hash<int32_t>{}(key.step_id);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6U) + (h1 >> 2U));
  }
};

struct StepInfo {
  int32_t pdg{0};
  double edep{0.0};
};

template <typename KeyType>
KeyType ArgMaxByWeight(const std::unordered_map<KeyType, double>& values,
                       KeyType default_value) {
  if (values.empty()) {
    return default_value;
  }
  bool initialized = false;
  KeyType best_key = default_value;
  double best_weight = 0.0;
  for (const auto& kv : values) {
    if (!initialized || kv.second > best_weight ||
        (std::abs(kv.second - best_weight) < 1e-12 && kv.first < best_key)) {
      initialized = true;
      best_key = kv.first;
      best_weight = kv.second;
    }
  }
  return initialized ? best_key : default_value;
}

}  // namespace

EventSplitterEventLoader::EventSplitterEventLoader() {
  input_columns_ = {
      "event_id",
      "hits_x",
      "hits_y",
      "hits_z",
      "hits_edep",
      "hits_strip_type",
      "hits_time",
      "hits_time_group",
  };
  target_columns_ = {
      "hits_contrib_mc_event_id",
      "hits_contrib_step_id",
      "steps_mc_event_id",
      "steps_step_id",
      "steps_pdg_id",
      "steps_edep",
  };
  ConfigureDerivers(nullptr);
}

void EventSplitterEventLoader::LoadConfig(const nlohmann::json& cfg) {
  if (cfg.contains("time_window_ns")) {
    time_window_ns_ = cfg.at("time_window_ns").get<double>();
  }
  if (cfg.contains("use_group_probs")) {
    use_group_probs_ = cfg.at("use_group_probs").get<bool>();
  }
  if (cfg.contains("use_splitter_probs")) {
    use_splitter_probs_ = cfg.at("use_splitter_probs").get<bool>();
  }
  if (cfg.contains("use_endpoint_preds")) {
    use_endpoint_preds_ = cfg.at("use_endpoint_preds").get<bool>();
  }
  if (cfg.contains("derive_time_groups")) {
    derive_time_groups_ = cfg.at("derive_time_groups").get<bool>();
  }

  const nlohmann::json* derivers_cfg = nullptr;
  if (cfg.contains("derivers")) {
    derivers_cfg = &cfg.at("derivers");
  }
  ConfigureDerivers(derivers_cfg);
}

void EventSplitterEventLoader::ConfigureDerivers(const nlohmann::json* derivers_cfg) {
  derivers_.clear();
  if (!derive_time_groups_) {
    return;
  }

  auto time_grouper =
      std::make_shared<data_derivers::TimeGrouper>(time_window_ns_, "hits_time");
  if (derivers_cfg && derivers_cfg->contains("time_grouper")) {
    time_grouper->LoadConfig(derivers_cfg->at("time_grouper"));
  }
  AddDeriver("hits_time_group", time_grouper);
}

void EventSplitterEventLoader::CountRows(
    const int32_t* offsets,
    const int32_t* tg_offsets,
    const int64_t* tg_raw,
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

TrainingBundle EventSplitterEventLoader::LoadTraining(
    const std::shared_ptr<arrow::Table>& table) const {
  auto prepared = PrepareTable(table, true);
  auto required = utils::parquet::MergeColumns(input_columns_, target_columns_);
  utils::parquet::ValidateColumns(*prepared,
                                  required,
                                  {"pred_pion", "pred_muon", "pred_mip", "pred_hit_pion",
                                   "pred_hit_muon", "pred_hit_mip", "time_group_ids",
                                   "pred_group_start_x", "pred_group_start_y",
                                   "pred_group_start_z", "pred_group_end_x",
                                   "pred_group_end_y", "pred_group_end_z"},
                                  true,
                                  "EventSplitterEventLoader training");
  auto batch = BuildGraph(*prepared);
  return SplitInputsTargets(std::move(batch));
}

InferenceBundle EventSplitterEventLoader::LoadInference(
    const std::shared_ptr<arrow::Table>& table) const {
  auto prepared = PrepareTable(table, true);
  utils::parquet::ValidateColumns(*prepared,
                                  input_columns_,
                                  {"pred_pion", "pred_muon", "pred_mip", "pred_hit_pion",
                                   "pred_hit_muon", "pred_hit_mip", "time_group_ids",
                                   "pred_group_start_x", "pred_group_start_y",
                                   "pred_group_start_z", "pred_group_end_x",
                                   "pred_group_end_y", "pred_group_end_z",
                                   "hits_contrib_mc_event_id", "hits_contrib_step_id",
                                   "steps_mc_event_id", "steps_step_id", "steps_pdg_id",
                                   "steps_edep"},
                                  true,
                                  "EventSplitterEventLoader inputs");
  InferenceBundle out;
  out.inputs = BuildGraph(*prepared);
  return out;
}

std::unique_ptr<BaseBatch> EventSplitterEventLoader::BuildGraph(const arrow::Table& table) const {
  utils::timing::ScopedTimer total_timer("event_splitter_event.build_graph");
  BuildContext ctx;
  BuildBuffers bufs;
  {
    utils::timing::ScopedTimer timer("event_splitter_event.phase0_initialize");
    BuildGraphPhase0Initialize(table, &ctx);
  }
  {
    utils::timing::ScopedTimer timer("event_splitter_event.phase1_count");
    BuildGraphPhase1Count(&ctx);
  }
  {
    utils::timing::ScopedTimer timer("event_splitter_event.phase2_offsets");
    BuildGraphPhase2Offsets(&ctx);
  }
  {
    utils::timing::ScopedTimer timer("event_splitter_event.phase3_allocate");
    BuildGraphPhase3Allocate(ctx, &bufs);
  }
  {
    utils::timing::ScopedTimer timer("event_splitter_event.phase4_populate");
    BuildGraphPhase4Populate(ctx, &bufs);
  }
  {
    utils::timing::ScopedTimer timer("event_splitter_event.phase5_finalize");
    return BuildGraphPhase5Finalize(ctx, &bufs);
  }
}

void EventSplitterEventLoader::BuildGraphPhase0Initialize(const arrow::Table& table,
                                                          BuildContext* ctx) const {
  ctx->input_cols = utils::parquet::BindColumns(
      table, input_columns_, true, true, "EventSplitterEventLoader inputs");
  ctx->target_cols = utils::parquet::BindColumns(
      table, target_columns_, false, true, "EventSplitterEventLoader targets");
  ctx->prob_cols = utils::parquet::BindColumns(
      table, {"pred_pion", "pred_muon", "pred_mip"}, false, true, "EventSplitterEventLoader probs");
  ctx->splitter_cols = utils::parquet::BindColumns(
      table,
      {"pred_hit_pion", "pred_hit_muon", "pred_hit_mip", "time_group_ids"},
      false,
      true,
      "EventSplitterEventLoader splitter probs");
  ctx->endpoint_cols = utils::parquet::BindColumns(
      table,
      {"pred_group_start_x", "pred_group_start_y", "pred_group_start_z",
       "pred_group_end_x", "pred_group_end_y", "pred_group_end_z"},
      false,
      true,
      "EventSplitterEventLoader endpoint preds");

  ctx->has_targets = ctx->target_cols.size() == target_columns_.size();
  ctx->has_prob_columns = ctx->prob_cols.size() == 3;
  ctx->has_splitter_columns = ctx->splitter_cols.count("pred_hit_pion") > 0 &&
                              ctx->splitter_cols.count("pred_hit_muon") > 0 &&
                              ctx->splitter_cols.count("pred_hit_mip") > 0;
  ctx->has_endpoint_columns = ctx->endpoint_cols.size() == 6;

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

  ctx->x_values = MakeNumericAccessor(ctx->hits_x->values(), "event splitter event x");
  ctx->y_values = MakeNumericAccessor(ctx->hits_y->values(), "event splitter event y");
  ctx->z_values = MakeNumericAccessor(ctx->hits_z->values(), "event splitter event z");
  ctx->edep_values = MakeNumericAccessor(ctx->hits_edep->values(), "event splitter event edep");

  auto view_values =
      std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(ctx->hits_view->values());
  auto tg_values = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(
      ctx->hits_time_group->values());
  if (!view_values || !tg_values) {
    throw std::runtime_error("Expected hits_strip_type=int32 and hits_time_group=int64.");
  }

  auto event_id = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(
      ctx->input_cols.at("event_id")->chunk(0));
  if (!event_id) {
    throw std::runtime_error("Expected event_id=int32.");
  }

  ctx->view_raw = view_values->raw_values();
  ctx->tg_raw = tg_values->raw_values();
  ctx->event_id_raw = event_id->raw_values();
  ctx->offsets = ctx->hits_z->raw_value_offsets();
  ctx->tg_offsets = ctx->hits_time_group->raw_value_offsets();

  if (ctx->has_targets) {
    ctx->contrib_mc_outer = &static_cast<const arrow::ListArray&>(
        *ctx->target_cols.at("hits_contrib_mc_event_id")->chunk(0));
    ctx->contrib_step_outer = &static_cast<const arrow::ListArray&>(
        *ctx->target_cols.at("hits_contrib_step_id")->chunk(0));

    ctx->contrib_mc_inner =
        &static_cast<const arrow::ListArray&>(*ctx->contrib_mc_outer->values());
    ctx->contrib_step_inner =
        &static_cast<const arrow::ListArray&>(*ctx->contrib_step_outer->values());

    auto contrib_mc_values = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(
        ctx->contrib_mc_inner->values());
    auto contrib_step_values = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(
        ctx->contrib_step_inner->values());
    if (!contrib_mc_values || !contrib_step_values) {
      throw std::runtime_error(
          "Expected hits_contrib_mc_event_id/hits_contrib_step_id inner values=int32.");
    }

    ctx->contrib_mc_outer_offsets = ctx->contrib_mc_outer->raw_value_offsets();
    ctx->contrib_step_outer_offsets = ctx->contrib_step_outer->raw_value_offsets();
    ctx->contrib_mc_inner_offsets = ctx->contrib_mc_inner->raw_value_offsets();
    ctx->contrib_step_inner_offsets = ctx->contrib_step_inner->raw_value_offsets();
    ctx->contrib_mc_values_raw = contrib_mc_values->raw_values();
    ctx->contrib_step_values_raw = contrib_step_values->raw_values();

    ctx->steps_mc = &static_cast<const arrow::ListArray&>(
        *ctx->target_cols.at("steps_mc_event_id")->chunk(0));
    ctx->steps_step = &static_cast<const arrow::ListArray&>(
        *ctx->target_cols.at("steps_step_id")->chunk(0));
    ctx->steps_pdg = &static_cast<const arrow::ListArray&>(
        *ctx->target_cols.at("steps_pdg_id")->chunk(0));
    ctx->steps_edep = &static_cast<const arrow::ListArray&>(
        *ctx->target_cols.at("steps_edep")->chunk(0));

    auto steps_mc_values = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(
        ctx->steps_mc->values());
    auto steps_step_values = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(
        ctx->steps_step->values());
    auto steps_pdg_values = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(
        ctx->steps_pdg->values());
    if (!steps_mc_values || !steps_step_values || !steps_pdg_values) {
      throw std::runtime_error("Expected steps_mc_event_id/steps_step_id/steps_pdg_id values=int32.");
    }

    ctx->steps_edep_values = MakeNumericAccessor(ctx->steps_edep->values(), "event splitter event steps_edep");
    ctx->steps_offsets = ctx->steps_mc->raw_value_offsets();
    ctx->steps_step_offsets = ctx->steps_step->raw_value_offsets();
    ctx->steps_pdg_offsets = ctx->steps_pdg->raw_value_offsets();
    ctx->steps_edep_offsets = ctx->steps_edep->raw_value_offsets();
    ctx->steps_mc_values_raw = steps_mc_values->raw_values();
    ctx->steps_step_values_raw = steps_step_values->raw_values();
    ctx->steps_pdg_values_raw = steps_pdg_values->raw_values();
  }

  ctx->rows = table.num_rows();
  ctx->node_counts.assign(static_cast<size_t>(ctx->rows), 0);
  ctx->edge_counts.assign(static_cast<size_t>(ctx->rows), 0);
  ctx->group_counts.assign(static_cast<size_t>(ctx->rows), 0);
}

void EventSplitterEventLoader::BuildGraphPhase1Count(BuildContext* ctx) const {
  CountRows(
      ctx->offsets, ctx->tg_offsets, ctx->tg_raw, ctx->rows, &ctx->node_counts, &ctx->edge_counts, &ctx->group_counts);
}

void EventSplitterEventLoader::BuildGraphPhase2Offsets(BuildContext* ctx) const {
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

void EventSplitterEventLoader::BuildGraphPhase3Allocate(const BuildContext& ctx,
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
  bufs->group_probs_buf =
      AllocBuffer(ctx.total_groups * static_cast<int64_t>(sizeof(float)) * 3);
  bufs->splitter_probs_buf =
      AllocBuffer(ctx.total_nodes * static_cast<int64_t>(sizeof(float)) * 3);
  bufs->endpoint_preds_buf =
      AllocBuffer(ctx.total_groups * static_cast<int64_t>(sizeof(float)) * 6);
  bufs->graph_event_ids_buf =
      AllocBuffer(ctx.rows * static_cast<int64_t>(sizeof(int64_t)));
  bufs->y_edge_buf =
      AllocBuffer(ctx.total_edges * static_cast<int64_t>(sizeof(float)));

  bufs->node_feat = reinterpret_cast<float*>(bufs->node_feat_buf->mutable_data());
  bufs->edge_index = reinterpret_cast<int64_t*>(bufs->edge_index_buf->mutable_data());
  bufs->edge_attr = reinterpret_cast<float*>(bufs->edge_attr_buf->mutable_data());
  bufs->time_group_ids = reinterpret_cast<int64_t*>(bufs->time_group_buf->mutable_data());
  bufs->node_ptr = reinterpret_cast<int64_t*>(bufs->node_ptr_buf->mutable_data());
  bufs->edge_ptr = reinterpret_cast<int64_t*>(bufs->edge_ptr_buf->mutable_data());
  bufs->group_ptr = reinterpret_cast<int64_t*>(bufs->group_ptr_buf->mutable_data());
  bufs->group_probs = reinterpret_cast<float*>(bufs->group_probs_buf->mutable_data());
  bufs->splitter_probs = reinterpret_cast<float*>(bufs->splitter_probs_buf->mutable_data());
  bufs->endpoint_preds = reinterpret_cast<float*>(bufs->endpoint_preds_buf->mutable_data());
  bufs->graph_event_ids = reinterpret_cast<int64_t*>(bufs->graph_event_ids_buf->mutable_data());
  bufs->y_edge = reinterpret_cast<float*>(bufs->y_edge_buf->mutable_data());

  bufs->group_truth.assign(static_cast<size_t>(ctx.total_groups * 3), 0U);
  bufs->node_class.assign(static_cast<size_t>(ctx.total_nodes), -1);
  std::fill(bufs->group_probs, bufs->group_probs + (ctx.total_groups * 3), 0.0f);
  std::fill(bufs->splitter_probs, bufs->splitter_probs + (ctx.total_nodes * 3), 0.0f);
  std::fill(bufs->endpoint_preds, bufs->endpoint_preds + (ctx.total_groups * 6), 0.0f);
  std::fill(bufs->y_edge, bufs->y_edge + ctx.total_edges, 0.0f);

  FillPointerArrayFromOffsets(ctx.node_offsets, bufs->node_ptr);
  FillPointerArrayFromOffsets(ctx.edge_offsets, bufs->edge_ptr);
}

void EventSplitterEventLoader::BuildGraphPhase4Populate(const BuildContext& ctx,
                                                        BuildBuffers* bufs) const {
  utils::parallel::Parallel::For(0, ctx.rows, [&](int64_t row) {
    const int64_t n = static_cast<int64_t>(ctx.offsets[row + 1] - ctx.offsets[row]);
    bufs->graph_event_ids[row] = static_cast<int64_t>(ctx.event_id_raw[row]);
    if (n == 0) {
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
    std::vector<int32_t> mc_event_local(static_cast<size_t>(n), -1);

    std::unordered_map<StepKey, StepInfo, StepKeyHash> step_map;
    if (ctx.has_targets) {
      const int64_t mc_hits = static_cast<int64_t>(ctx.contrib_mc_outer_offsets[row + 1] -
                                                   ctx.contrib_mc_outer_offsets[row]);
      const int64_t step_hits = static_cast<int64_t>(ctx.contrib_step_outer_offsets[row + 1] -
                                                     ctx.contrib_step_outer_offsets[row]);
      if (mc_hits != n || step_hits != n) {
        throw std::runtime_error(
            "hits_contrib_mc_event_id/hits_contrib_step_id lengths do not match hits.");
      }

      const int32_t steps_start = ctx.steps_offsets[row];
      const int32_t steps_end = ctx.steps_offsets[row + 1];
      if ((ctx.steps_step_offsets[row + 1] - ctx.steps_step_offsets[row]) !=
              (steps_end - steps_start) ||
          (ctx.steps_pdg_offsets[row + 1] - ctx.steps_pdg_offsets[row]) !=
              (steps_end - steps_start) ||
          (ctx.steps_edep_offsets[row + 1] - ctx.steps_edep_offsets[row]) !=
              (steps_end - steps_start)) {
        throw std::runtime_error("Step list columns length mismatch in EventSplitterEventLoader.");
      }

      step_map.reserve(static_cast<size_t>(steps_end - steps_start));
      for (int32_t idx = steps_start; idx < steps_end; ++idx) {
        StepKey key{ctx.steps_mc_values_raw[idx], ctx.steps_step_values_raw[idx]};
        StepInfo info;
        info.pdg = ctx.steps_pdg_values_raw[idx];
        info.edep = ctx.steps_edep_values.IsValid(idx) ? ctx.steps_edep_values.Value(idx) : 0.0;
        step_map.emplace(key, info);
      }
    }

    for (int64_t i = 0; i < n; ++i) {
      const int64_t raw_idx = start + i;
      const int64_t node_idx = node_offset + i;
      const int64_t base = node_idx * 4;
      const int32_t view = ctx.view_raw[raw_idx];
      const double coord = ResolveCoordinateForView(ctx.x_values, ctx.y_values, view, raw_idx);
      const float z = static_cast<float>(ctx.z_values.IsValid(raw_idx) ? ctx.z_values.Value(raw_idx) : 0.0);
      const float e = static_cast<float>(ctx.edep_values.IsValid(raw_idx) ? ctx.edep_values.Value(raw_idx) : 0.0);

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

      if (ctx.has_targets) {
        const int64_t outer_idx = static_cast<int64_t>(ctx.contrib_mc_outer_offsets[row]) + i;
        const int32_t contrib_start = ctx.contrib_mc_inner_offsets[outer_idx];
        const int32_t contrib_end = ctx.contrib_mc_inner_offsets[outer_idx + 1];
        const int32_t step_start = ctx.contrib_step_inner_offsets[outer_idx];
        const int32_t step_end = ctx.contrib_step_inner_offsets[outer_idx + 1];

        if ((contrib_end - contrib_start) != (step_end - step_start)) {
          throw std::runtime_error(
              "Contrib mc_event/step inner list lengths mismatch in EventSplitterEventLoader.");
        }

        std::unordered_map<int32_t, double> mc_energy;
        std::unordered_map<int32_t, double> pdg_energy;
        for (int32_t c = 0; c < (contrib_end - contrib_start); ++c) {
          const int32_t mc_id = ctx.contrib_mc_values_raw[contrib_start + c];
          const int32_t step_id = ctx.contrib_step_values_raw[step_start + c];

          double weight = 1.0;
          int32_t pdg = 0;
          auto it = step_map.find(StepKey{mc_id, step_id});
          if (it != step_map.end()) {
            pdg = it->second.pdg;
            weight = std::max(0.0, it->second.edep);
            if (weight <= 0.0) {
              weight = 1e-6;
            }
          }

          mc_energy[mc_id] += weight;
          if (pdg != 0) {
            pdg_energy[pdg] += weight;
          }
        }

        const int32_t selected_mc = ArgMaxByWeight<int32_t>(mc_energy, -1);
        const int32_t selected_pdg = ArgMaxByWeight<int32_t>(pdg_energy, 0);
        const int cls = ClassFromPdg(selected_pdg);

        mc_event_local[static_cast<size_t>(i)] = selected_mc;
        bufs->node_class[static_cast<size_t>(node_idx)] = static_cast<int8_t>(cls);

        if (cls >= 0) {
          const int64_t global_group = group_offset + local_group;
          bufs->group_truth[static_cast<size_t>(global_group * 3 + cls)] = 1U;
        }
      }
    }

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

        if (ctx.has_targets) {
          const int32_t mc_a = mc_event_local[static_cast<size_t>(a)];
          const int32_t mc_b = mc_event_local[static_cast<size_t>(b)];
          bufs->y_edge[edge_idx] = (mc_a >= 0 && mc_a == mc_b) ? 1.0f : 0.0f;
        }
        edge_local++;
      }
    }
  });
}

std::unique_ptr<BaseBatch> EventSplitterEventLoader::BuildGraphPhase5Finalize(
    const BuildContext& ctx,
    BuildBuffers* bufs) const {
  FillPointerArrayFromOffsets(ctx.row_group_offsets, bufs->group_ptr);

  if (!ctx.has_prob_columns && use_group_probs_ && ctx.has_targets) {
    for (int64_t group_idx = 0; group_idx < ctx.total_groups; ++group_idx) {
      const int64_t base = group_idx * 3;
      bufs->group_probs[base] = static_cast<float>(bufs->group_truth[static_cast<size_t>(base)]);
      bufs->group_probs[base + 1] =
          static_cast<float>(bufs->group_truth[static_cast<size_t>(base + 1)]);
      bufs->group_probs[base + 2] =
          static_cast<float>(bufs->group_truth[static_cast<size_t>(base + 2)]);
    }
  }

  if (ctx.has_prob_columns && use_group_probs_) {
    const auto& pion_pred_list = static_cast<const arrow::ListArray&>(
        *ctx.prob_cols.at("pred_pion")->chunk(0));
    const auto& muon_pred_list = static_cast<const arrow::ListArray&>(
        *ctx.prob_cols.at("pred_muon")->chunk(0));
    const auto& mip_pred_list = static_cast<const arrow::ListArray&>(
        *ctx.prob_cols.at("pred_mip")->chunk(0));

    auto pion_pred_values = MakeNumericAccessor(pion_pred_list.values(), "pred_pion");
    auto muon_pred_values = MakeNumericAccessor(muon_pred_list.values(), "pred_muon");
    auto mip_pred_values = MakeNumericAccessor(mip_pred_list.values(), "pred_mip");

    const int32_t* p_offsets = pion_pred_list.raw_value_offsets();
    const int32_t* m_offsets = muon_pred_list.raw_value_offsets();
    const int32_t* i_offsets = mip_pred_list.raw_value_offsets();

    for (int64_t row = 0; row < ctx.rows; ++row) {
      const int64_t count = static_cast<int64_t>(p_offsets[row + 1] - p_offsets[row]);
      if (count != static_cast<int64_t>(m_offsets[row + 1] - m_offsets[row]) ||
          count != static_cast<int64_t>(i_offsets[row + 1] - i_offsets[row])) {
        throw std::runtime_error("pred_pion/pred_muon/pred_mip list lengths mismatch.");
      }
      if (count != ctx.group_counts[static_cast<size_t>(row)]) {
        throw std::runtime_error("Prediction list length does not match time groups.");
      }

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

  if (ctx.has_splitter_columns && use_splitter_probs_) {
    const auto& p_list = static_cast<const arrow::ListArray&>(
        *ctx.splitter_cols.at("pred_hit_pion")->chunk(0));
    const auto& m_list = static_cast<const arrow::ListArray&>(
        *ctx.splitter_cols.at("pred_hit_muon")->chunk(0));
    const auto& i_list = static_cast<const arrow::ListArray&>(
        *ctx.splitter_cols.at("pred_hit_mip")->chunk(0));

    auto p_values = MakeNumericAccessor(p_list.values(), "pred_hit_pion");
    auto m_values = MakeNumericAccessor(m_list.values(), "pred_hit_muon");
    auto i_values = MakeNumericAccessor(i_list.values(), "pred_hit_mip");

    const int32_t* p_offsets = p_list.raw_value_offsets();
    const int32_t* m_offsets = m_list.raw_value_offsets();
    const int32_t* i_offsets = i_list.raw_value_offsets();

    const bool has_splitter_tg = ctx.splitter_cols.count("time_group_ids") > 0;
    const arrow::ListArray* splitter_tg_list = nullptr;
    const int32_t* splitter_tg_offsets = nullptr;
    const int64_t* splitter_tg_raw = nullptr;
    if (has_splitter_tg) {
      splitter_tg_list = &static_cast<const arrow::ListArray&>(
          *ctx.splitter_cols.at("time_group_ids")->chunk(0));
      auto splitter_tg_values = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(
          splitter_tg_list->values());
      if (!splitter_tg_values) {
        throw std::runtime_error("Expected splitter time_group_ids values=int64.");
      }
      splitter_tg_offsets = splitter_tg_list->raw_value_offsets();
      splitter_tg_raw = splitter_tg_values->raw_values();
    }

    for (int64_t row = 0; row < ctx.rows; ++row) {
      const int64_t count = static_cast<int64_t>(p_offsets[row + 1] - p_offsets[row]);
      if (count != static_cast<int64_t>(m_offsets[row + 1] - m_offsets[row]) ||
          count != static_cast<int64_t>(i_offsets[row + 1] - i_offsets[row])) {
        throw std::runtime_error("Splitter prediction list lengths mismatch.");
      }
      if (count != ctx.node_counts[static_cast<size_t>(row)]) {
        throw std::runtime_error("Splitter prediction list length does not match hits.");
      }

      const int64_t node_base = ctx.node_offsets[static_cast<size_t>(row)];
      const int32_t start_idx = p_offsets[row];
      for (int64_t n = 0; n < count; ++n) {
        const int32_t list_idx = start_idx + static_cast<int32_t>(n);
        const int64_t base = (node_base + n) * 3;
        bufs->splitter_probs[base] = static_cast<float>(p_values.Value(list_idx));
        bufs->splitter_probs[base + 1] = static_cast<float>(m_values.Value(list_idx));
        bufs->splitter_probs[base + 2] = static_cast<float>(i_values.Value(list_idx));
      }

      if (has_splitter_tg) {
        if ((splitter_tg_offsets[row + 1] - splitter_tg_offsets[row]) != count) {
          throw std::runtime_error("Splitter time_group_ids length does not match hits.");
        }
        const int32_t tg_start = splitter_tg_offsets[row];
        const int32_t source_tg_start = ctx.tg_offsets[row];
        for (int64_t n = 0; n < count; ++n) {
          if (splitter_tg_raw[tg_start + n] != ctx.tg_raw[source_tg_start + n]) {
            throw std::runtime_error(
                "Splitter time_group_ids are not aligned with main hits_time_group.");
          }
        }
      }
    }
  } else if (use_splitter_probs_ && ctx.has_targets) {
    for (int64_t node_idx = 0; node_idx < ctx.total_nodes; ++node_idx) {
      const int cls = static_cast<int>(bufs->node_class[static_cast<size_t>(node_idx)]);
      if (cls >= 0 && cls < 3) {
        bufs->splitter_probs[node_idx * 3 + cls] = 1.0f;
      }
    }
  }

  if (ctx.has_endpoint_columns && use_endpoint_preds_) {
    const auto& sx_list = static_cast<const arrow::ListArray&>(
        *ctx.endpoint_cols.at("pred_group_start_x")->chunk(0));
    const auto& sy_list = static_cast<const arrow::ListArray&>(
        *ctx.endpoint_cols.at("pred_group_start_y")->chunk(0));
    const auto& sz_list = static_cast<const arrow::ListArray&>(
        *ctx.endpoint_cols.at("pred_group_start_z")->chunk(0));
    const auto& ex_list = static_cast<const arrow::ListArray&>(
        *ctx.endpoint_cols.at("pred_group_end_x")->chunk(0));
    const auto& ey_list = static_cast<const arrow::ListArray&>(
        *ctx.endpoint_cols.at("pred_group_end_y")->chunk(0));
    const auto& ez_list = static_cast<const arrow::ListArray&>(
        *ctx.endpoint_cols.at("pred_group_end_z")->chunk(0));

    auto sx_values = MakeNumericAccessor(sx_list.values(), "pred_group_start_x");
    auto sy_values = MakeNumericAccessor(sy_list.values(), "pred_group_start_y");
    auto sz_values = MakeNumericAccessor(sz_list.values(), "pred_group_start_z");
    auto ex_values = MakeNumericAccessor(ex_list.values(), "pred_group_end_x");
    auto ey_values = MakeNumericAccessor(ey_list.values(), "pred_group_end_y");
    auto ez_values = MakeNumericAccessor(ez_list.values(), "pred_group_end_z");

    const int32_t* sx_offsets = sx_list.raw_value_offsets();
    const int32_t* sy_offsets = sy_list.raw_value_offsets();
    const int32_t* sz_offsets = sz_list.raw_value_offsets();
    const int32_t* ex_offsets = ex_list.raw_value_offsets();
    const int32_t* ey_offsets = ey_list.raw_value_offsets();
    const int32_t* ez_offsets = ez_list.raw_value_offsets();

    for (int64_t row = 0; row < ctx.rows; ++row) {
      const int64_t count = static_cast<int64_t>(sx_offsets[row + 1] - sx_offsets[row]);
      if (count != static_cast<int64_t>(sy_offsets[row + 1] - sy_offsets[row]) ||
          count != static_cast<int64_t>(sz_offsets[row + 1] - sz_offsets[row]) ||
          count != static_cast<int64_t>(ex_offsets[row + 1] - ex_offsets[row]) ||
          count != static_cast<int64_t>(ey_offsets[row + 1] - ey_offsets[row]) ||
          count != static_cast<int64_t>(ez_offsets[row + 1] - ez_offsets[row])) {
        throw std::runtime_error("Endpoint prediction list lengths mismatch.");
      }
      if (count != ctx.group_counts[static_cast<size_t>(row)]) {
        throw std::runtime_error("Endpoint prediction list length does not match time groups.");
      }

      const int64_t group_base = ctx.row_group_offsets[static_cast<size_t>(row)];
      const int32_t start_idx = sx_offsets[row];
      for (int64_t g = 0; g < count; ++g) {
        const int32_t list_idx = start_idx + static_cast<int32_t>(g);
        const int64_t base = (group_base + g) * 6;
        bufs->endpoint_preds[base] = static_cast<float>(sx_values.Value(list_idx));
        bufs->endpoint_preds[base + 1] = static_cast<float>(sy_values.Value(list_idx));
        bufs->endpoint_preds[base + 2] = static_cast<float>(sz_values.Value(list_idx));
        bufs->endpoint_preds[base + 3] = static_cast<float>(ex_values.Value(list_idx));
        bufs->endpoint_preds[base + 4] = static_cast<float>(ey_values.Value(list_idx));
        bufs->endpoint_preds[base + 5] = static_cast<float>(ez_values.Value(list_idx));
      }
    }
  }

  auto out = std::make_unique<EventSplitterEventInputs>();
  out->node_features = MakeArray(bufs->node_feat_buf, arrow::float32(), ctx.total_nodes * 4);
  out->edge_index = MakeArray(bufs->edge_index_buf, arrow::int64(), ctx.total_edges * 2);
  out->edge_attr = MakeArray(bufs->edge_attr_buf, arrow::float32(), ctx.total_edges * 4);
  out->time_group_ids = MakeArray(bufs->time_group_buf, arrow::int64(), ctx.total_nodes);
  out->group_probs = MakeArray(bufs->group_probs_buf, arrow::float32(), ctx.total_groups * 3);
  out->splitter_probs = MakeArray(bufs->splitter_probs_buf, arrow::float32(), ctx.total_nodes * 3);
  out->endpoint_preds = MakeArray(bufs->endpoint_preds_buf, arrow::float32(), ctx.total_groups * 6);
  out->node_ptr = MakeArray(bufs->node_ptr_buf, arrow::int64(), ctx.rows + 1);
  out->edge_ptr = MakeArray(bufs->edge_ptr_buf, arrow::int64(), ctx.rows + 1);
  out->group_ptr = MakeArray(bufs->group_ptr_buf, arrow::int64(), ctx.rows + 1);
  out->graph_event_ids = MakeArray(bufs->graph_event_ids_buf, arrow::int64(), ctx.rows);
  out->y_edge = MakeArray(bufs->y_edge_buf, arrow::float32(), ctx.total_edges);
  out->num_graphs = static_cast<size_t>(ctx.rows);
  out->num_groups = static_cast<size_t>(ctx.total_groups);

  if (!ctx.has_targets) {
    out->y_edge.reset();
  }

  return out;
}

TrainingBundle EventSplitterEventLoader::SplitInputsTargets(
    std::unique_ptr<BaseBatch> batch_base) const {
  auto* typed = dynamic_cast<EventSplitterEventInputs*>(batch_base.get());
  if (!typed) {
    throw std::runtime_error("Unexpected batch type in SplitInputsTargets");
  }
  if (!typed->y_edge) {
    throw std::runtime_error("Training targets are missing. Use LoadInference or provide label columns.");
  }

  auto targets = std::make_unique<EventSplitterEventTargets>();
  targets->num_graphs = typed->num_graphs;
  targets->y_edge = typed->y_edge;

  typed->y_edge.reset();

  TrainingBundle result;
  result.inputs = std::move(batch_base);
  result.targets = std::move(targets);
  return result;
}

}  // namespace pioneerml::dataloaders::graph
