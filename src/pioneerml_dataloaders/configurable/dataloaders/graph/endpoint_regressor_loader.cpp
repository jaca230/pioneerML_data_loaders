#include "pioneerml_dataloaders/configurable/dataloaders/graph/endpoint_regressor_loader.h"

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

EndpointRegressorLoader::EndpointRegressorLoader() {
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

void EndpointRegressorLoader::LoadConfig(const nlohmann::json& cfg) {
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

void EndpointRegressorLoader::ConfigureDerivers(const nlohmann::json* derivers_cfg) {
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

void EndpointRegressorLoader::CountGroupsForRows(
    const int32_t* offsets,
    const int32_t* tg_offsets,
    const int64_t* tg_raw,
    const int32_t* pdg_offsets,
    int64_t rows,
    std::vector<int64_t>* group_counts,
    std::vector<std::vector<int64_t>>* group_node_counts) const {
  utils::parallel::Parallel::For(0, rows, [&](int64_t row) {
    const int64_t n = static_cast<int64_t>(offsets[row + 1] - offsets[row]);
    if (n == 0) {
      (*group_counts)[static_cast<size_t>(row)] = 0;
      return;
    }
    if ((tg_offsets[row + 1] - tg_offsets[row]) != n) {
      throw std::runtime_error("hits_time_group length mismatch with hits.");
    }
    if ((pdg_offsets[row + 1] - pdg_offsets[row]) != n) {
      throw std::runtime_error("hits_pdg_id length mismatch with hits.");
    }

    const int32_t tg_start = tg_offsets[row];
    int64_t max_group = -1;
    for (int64_t i = 0; i < n; ++i) {
      const int64_t tg = tg_raw[tg_start + i];
      if (tg < 0) {
        throw std::runtime_error("Invalid negative time group id encountered.");
      }
      max_group = std::max(max_group, tg);
    }
    const int64_t groups_for_row = std::max<int64_t>(0, max_group + 1);
    (*group_counts)[static_cast<size_t>(row)] = groups_for_row;

    std::vector<int64_t> counts(groups_for_row, 0);
    for (int64_t i = 0; i < n; ++i) {
      const int64_t group = tg_raw[tg_start + i];
      if (group < 0 || group >= groups_for_row) {
        throw std::runtime_error("Invalid time group id encountered.");
      }
      counts[static_cast<size_t>(group)] += 1;
    }
    (*group_node_counts)[static_cast<size_t>(row)] = std::move(counts);
  });
}

EndpointRegressorLoader::TargetReaders EndpointRegressorLoader::BuildTargetReaders(
    const ColumnMap& target_cols) const {
  TargetReaders out;
  const auto& sx_list = static_cast<const arrow::ListArray&>(*target_cols.at("group_start_x")->chunk(0));
  const auto& sy_list = static_cast<const arrow::ListArray&>(*target_cols.at("group_start_y")->chunk(0));
  const auto& sz_list = static_cast<const arrow::ListArray&>(*target_cols.at("group_start_z")->chunk(0));
  const auto& ex_list = static_cast<const arrow::ListArray&>(*target_cols.at("group_end_x")->chunk(0));
  const auto& ey_list = static_cast<const arrow::ListArray&>(*target_cols.at("group_end_y")->chunk(0));
  const auto& ez_list = static_cast<const arrow::ListArray&>(*target_cols.at("group_end_z")->chunk(0));

  out.start_x = MakeNumericAccessor(sx_list.values(), "group_start_x");
  out.start_y = MakeNumericAccessor(sy_list.values(), "group_start_y");
  out.start_z = MakeNumericAccessor(sz_list.values(), "group_start_z");
  out.end_x = MakeNumericAccessor(ex_list.values(), "group_end_x");
  out.end_y = MakeNumericAccessor(ey_list.values(), "group_end_y");
  out.end_z = MakeNumericAccessor(ez_list.values(), "group_end_z");

  out.sx_offsets = sx_list.raw_value_offsets();
  out.sy_offsets = sy_list.raw_value_offsets();
  out.sz_offsets = sz_list.raw_value_offsets();
  out.ex_offsets = ex_list.raw_value_offsets();
  out.ey_offsets = ey_list.raw_value_offsets();
  out.ez_offsets = ez_list.raw_value_offsets();
  return out;
}

void EndpointRegressorLoader::ValidateTargetListLengths(const TargetReaders& readers,
                                                        const std::vector<int64_t>& group_counts,
                                                        int64_t rows) const {
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t count = static_cast<int64_t>(readers.sx_offsets[row + 1] - readers.sx_offsets[row]);
    if (count != static_cast<int64_t>(readers.sy_offsets[row + 1] - readers.sy_offsets[row]) ||
        count != static_cast<int64_t>(readers.sz_offsets[row + 1] - readers.sz_offsets[row]) ||
        count != static_cast<int64_t>(readers.ex_offsets[row + 1] - readers.ex_offsets[row]) ||
        count != static_cast<int64_t>(readers.ey_offsets[row + 1] - readers.ey_offsets[row]) ||
        count != static_cast<int64_t>(readers.ez_offsets[row + 1] - readers.ez_offsets[row])) {
      throw std::runtime_error("Endpoint target list columns have mismatched lengths.");
    }
    if (count != group_counts[static_cast<size_t>(row)]) {
      throw std::runtime_error("Endpoint target list length does not match time groups.");
    }
  }
}

TrainingBundle EndpointRegressorLoader::LoadTraining(
    const std::shared_ptr<arrow::Table>& table) const {
  auto prepared = PrepareTable(table, true);
  auto required = utils::parquet::MergeColumns(input_columns_, target_columns_);
  utils::parquet::ValidateColumns(*prepared,
                                  required,
                                  {"pred_pion", "pred_muon", "pred_mip", "pred_hit_pion",
                                   "pred_hit_muon", "pred_hit_mip"},
                                  true,
                                  "EndpointRegressorLoader training");
  auto batch = BuildGraph(*prepared);
  return SplitInputsTargets(std::move(batch));
}

InferenceBundle EndpointRegressorLoader::LoadInference(
    const std::shared_ptr<arrow::Table>& table) const {
  auto prepared = PrepareTable(table, true);
  utils::parquet::ValidateColumns(*prepared,
                                  input_columns_,
                                  {"pred_pion", "pred_muon", "pred_mip", "pred_hit_pion",
                                   "pred_hit_muon", "pred_hit_mip", "group_start_x",
                                   "group_start_y", "group_start_z", "group_end_x",
                                   "group_end_y", "group_end_z"},
                                  true,
                                  "EndpointRegressorLoader inputs");
  InferenceBundle out;
  out.inputs = BuildGraph(*prepared);
  return out;
}

std::unique_ptr<BaseBatch> EndpointRegressorLoader::BuildGraph(const arrow::Table& table) const {
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

void EndpointRegressorLoader::BuildGraphPhase0Initialize(const arrow::Table& table,
                                                         BuildContext* ctx) const {
  ctx->input_cols = utils::parquet::BindColumns(
      table, input_columns_, true, true, "EndpointRegressorLoader inputs");
  ctx->target_cols = utils::parquet::BindColumns(
      table, target_columns_, false, true, "EndpointRegressorLoader targets");
  ctx->prob_cols = utils::parquet::BindColumns(
      table, {"pred_pion", "pred_muon", "pred_mip"}, false, true, "EndpointRegressorLoader probs");
  ctx->splitter_cols = utils::parquet::BindColumns(
      table,
      {"pred_hit_pion", "pred_hit_muon", "pred_hit_mip", "time_group_ids"},
      false,
      true,
      "EndpointRegressorLoader splitter probs");

  ctx->has_targets = ctx->target_cols.size() == target_columns_.size();
  ctx->has_prob_columns = ctx->prob_cols.size() == 3;
  ctx->has_splitter_columns = ctx->splitter_cols.count("pred_hit_pion") > 0 &&
                              ctx->splitter_cols.count("pred_hit_muon") > 0 &&
                              ctx->splitter_cols.count("pred_hit_mip") > 0;

  ctx->hits_x = &static_cast<const arrow::ListArray&>(*ctx->input_cols.at("hits_x")->chunk(0));
  ctx->hits_y = &static_cast<const arrow::ListArray&>(*ctx->input_cols.at("hits_y")->chunk(0));
  ctx->hits_z = &static_cast<const arrow::ListArray&>(*ctx->input_cols.at("hits_z")->chunk(0));
  ctx->hits_edep = &static_cast<const arrow::ListArray&>(*ctx->input_cols.at("hits_edep")->chunk(0));
  ctx->hits_view = &static_cast<const arrow::ListArray&>(*ctx->input_cols.at("hits_strip_type")->chunk(0));
  ctx->hits_time_group =
      &static_cast<const arrow::ListArray&>(*ctx->input_cols.at("hits_time_group")->chunk(0));
  ctx->hits_pdg_id = &static_cast<const arrow::ListArray&>(*ctx->input_cols.at("hits_pdg_id")->chunk(0));

  ctx->x_values = MakeNumericAccessor(ctx->hits_x->values(), "endpoint regressor x");
  ctx->y_values = MakeNumericAccessor(ctx->hits_y->values(), "endpoint regressor y");
  ctx->z_values = MakeNumericAccessor(ctx->hits_z->values(), "endpoint regressor z");
  ctx->edep_values = MakeNumericAccessor(ctx->hits_edep->values(), "endpoint regressor edep");

  auto view_values =
      std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(ctx->hits_view->values());
  auto tg_values =
      std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(ctx->hits_time_group->values());
  auto pdg_values =
      std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(ctx->hits_pdg_id->values());
  ctx->view_raw = view_values->raw_values();
  ctx->tg_raw = tg_values->raw_values();
  ctx->pdg_raw = pdg_values->raw_values();
  ctx->offsets = ctx->hits_z->raw_value_offsets();
  ctx->tg_offsets = ctx->hits_time_group->raw_value_offsets();
  ctx->pdg_offsets = ctx->hits_pdg_id->raw_value_offsets();

  ctx->rows = table.num_rows();
  ctx->group_counts.assign(static_cast<size_t>(ctx->rows), 0);
  ctx->group_node_counts.resize(static_cast<size_t>(ctx->rows));
}

void EndpointRegressorLoader::BuildGraphPhase1Count(BuildContext* ctx) const {
  CountGroupsForRows(ctx->offsets,
                     ctx->tg_offsets,
                     ctx->tg_raw,
                     ctx->pdg_offsets,
                     ctx->rows,
                     &ctx->group_counts,
                     &ctx->group_node_counts);
}

void EndpointRegressorLoader::BuildGraphPhase2Offsets(BuildContext* ctx) const {
  ctx->graph_offsets = BuildOffsets(ctx->group_counts);
  ctx->total_graphs = ctx->graph_offsets.back();

  ctx->node_counts.assign(static_cast<size_t>(ctx->total_graphs), 0);
  ctx->edge_counts.assign(static_cast<size_t>(ctx->total_graphs), 0);
  for (int64_t row = 0; row < ctx->rows; ++row) {
    const int64_t graph_base = ctx->graph_offsets[row];
    const auto& counts = ctx->group_node_counts[static_cast<size_t>(row)];
    for (int64_t g = 0; g < static_cast<int64_t>(counts.size()); ++g) {
      const int64_t graph_idx = graph_base + g;
      const int64_t node_count = counts[static_cast<size_t>(g)];
      ctx->node_counts[static_cast<size_t>(graph_idx)] = node_count;
      ctx->edge_counts[static_cast<size_t>(graph_idx)] = node_count * (node_count - 1);
    }
  }

  ctx->node_offsets = BuildOffsets(ctx->node_counts);
  ctx->edge_offsets = BuildOffsets(ctx->edge_counts);
  ctx->total_nodes = ctx->node_offsets.back();
  ctx->total_edges = ctx->edge_offsets.back();
}

void EndpointRegressorLoader::BuildGraphPhase3Allocate(const BuildContext& ctx,
                                                       BuildBuffers* bufs) const {
  bufs->node_feat_buf = AllocBuffer(ctx.total_nodes * static_cast<int64_t>(sizeof(float)) * 4);
  bufs->edge_index_buf = AllocBuffer(ctx.total_edges * static_cast<int64_t>(sizeof(int64_t)) * 2);
  bufs->edge_attr_buf = AllocBuffer(ctx.total_edges * static_cast<int64_t>(sizeof(float)) * 4);
  bufs->time_group_buf = AllocBuffer(ctx.total_nodes * static_cast<int64_t>(sizeof(int64_t)));
  bufs->node_ptr_buf = AllocBuffer((ctx.total_graphs + 1) * static_cast<int64_t>(sizeof(int64_t)));
  bufs->edge_ptr_buf = AllocBuffer((ctx.total_graphs + 1) * static_cast<int64_t>(sizeof(int64_t)));
  bufs->group_ptr_buf = AllocBuffer((ctx.total_graphs + 1) * static_cast<int64_t>(sizeof(int64_t)));
  bufs->u_buf = AllocBuffer(ctx.total_graphs * static_cast<int64_t>(sizeof(float)));
  bufs->group_probs_buf = AllocBuffer(ctx.total_graphs * static_cast<int64_t>(sizeof(float)) * 3);
  bufs->splitter_probs_buf = AllocBuffer(ctx.total_nodes * static_cast<int64_t>(sizeof(float)) * 3);
  bufs->graph_event_ids_buf =
      AllocBuffer(ctx.total_graphs * static_cast<int64_t>(sizeof(int64_t)));
  bufs->graph_group_ids_buf =
      AllocBuffer(ctx.total_graphs * static_cast<int64_t>(sizeof(int64_t)));
  bufs->y_buf = AllocBuffer(ctx.total_graphs * static_cast<int64_t>(sizeof(float)) * 6);

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
  bufs->graph_group_ids = reinterpret_cast<int64_t*>(bufs->graph_group_ids_buf->mutable_data());
  bufs->y = reinterpret_cast<float*>(bufs->y_buf->mutable_data());

  bufs->group_truth.assign(static_cast<size_t>(ctx.total_graphs * 3), 0U);
  bufs->node_truth.assign(static_cast<size_t>(ctx.total_nodes * 3), 0.0f);
  std::fill(bufs->group_probs, bufs->group_probs + (ctx.total_graphs * 3), 0.0f);
  std::fill(bufs->splitter_probs, bufs->splitter_probs + (ctx.total_nodes * 3), 0.0f);
  std::fill(bufs->y, bufs->y + (ctx.total_graphs * 6), 0.0f);

  FillPointerArrayFromOffsets(ctx.node_offsets, bufs->node_ptr);
  FillPointerArrayFromOffsets(ctx.edge_offsets, bufs->edge_ptr);
  for (int64_t i = 0; i <= ctx.total_graphs; ++i) {
    bufs->group_ptr[i] = i;
  }
  std::fill(bufs->time_group_ids, bufs->time_group_ids + ctx.total_nodes, static_cast<int64_t>(0));
}

void EndpointRegressorLoader::BuildGraphPhase4Populate(const BuildContext& ctx,
                                                       BuildBuffers* bufs) const {
  TargetReaders targets;
  if (ctx.has_targets) {
    targets = BuildTargetReaders(ctx.target_cols);
    ValidateTargetListLengths(targets, ctx.group_counts, ctx.rows);
  }

  const arrow::ListArray* pion_prob_list = nullptr;
  const arrow::ListArray* muon_prob_list = nullptr;
  const arrow::ListArray* mip_prob_list = nullptr;
  NumericAccessor pion_prob_values;
  NumericAccessor muon_prob_values;
  NumericAccessor mip_prob_values;
  const int32_t* p_prob_offsets = nullptr;
  const int32_t* m_prob_offsets = nullptr;
  const int32_t* i_prob_offsets = nullptr;
  if (ctx.has_prob_columns && use_group_probs_) {
    pion_prob_list = &static_cast<const arrow::ListArray&>(*ctx.prob_cols.at("pred_pion")->chunk(0));
    muon_prob_list = &static_cast<const arrow::ListArray&>(*ctx.prob_cols.at("pred_muon")->chunk(0));
    mip_prob_list = &static_cast<const arrow::ListArray&>(*ctx.prob_cols.at("pred_mip")->chunk(0));
    pion_prob_values = MakeNumericAccessor(pion_prob_list->values(), "pred_pion");
    muon_prob_values = MakeNumericAccessor(muon_prob_list->values(), "pred_muon");
    mip_prob_values = MakeNumericAccessor(mip_prob_list->values(), "pred_mip");
    p_prob_offsets = pion_prob_list->raw_value_offsets();
    m_prob_offsets = muon_prob_list->raw_value_offsets();
    i_prob_offsets = mip_prob_list->raw_value_offsets();
  }

  const arrow::ListArray* pion_splitter_list = nullptr;
  const arrow::ListArray* muon_splitter_list = nullptr;
  const arrow::ListArray* mip_splitter_list = nullptr;
  NumericAccessor pion_splitter_values;
  NumericAccessor muon_splitter_values;
  NumericAccessor mip_splitter_values;
  const int64_t* splitter_tg_raw64 = nullptr;
  const int32_t* splitter_tg_raw32 = nullptr;
  bool splitter_tg_is_int64 = false;
  bool splitter_tg_is_int32 = false;
  const int32_t* p_splitter_offsets = nullptr;
  const int32_t* m_splitter_offsets = nullptr;
  const int32_t* i_splitter_offsets = nullptr;
  const int32_t* s_tg_offsets = nullptr;
  if (ctx.has_splitter_columns && use_splitter_probs_) {
    pion_splitter_list =
        &static_cast<const arrow::ListArray&>(*ctx.splitter_cols.at("pred_hit_pion")->chunk(0));
    muon_splitter_list =
        &static_cast<const arrow::ListArray&>(*ctx.splitter_cols.at("pred_hit_muon")->chunk(0));
    mip_splitter_list =
        &static_cast<const arrow::ListArray&>(*ctx.splitter_cols.at("pred_hit_mip")->chunk(0));
    pion_splitter_values = MakeNumericAccessor(pion_splitter_list->values(), "pred_hit_pion");
    muon_splitter_values = MakeNumericAccessor(muon_splitter_list->values(), "pred_hit_muon");
    mip_splitter_values = MakeNumericAccessor(mip_splitter_list->values(), "pred_hit_mip");
    p_splitter_offsets = pion_splitter_list->raw_value_offsets();
    m_splitter_offsets = muon_splitter_list->raw_value_offsets();
    i_splitter_offsets = mip_splitter_list->raw_value_offsets();
    if (ctx.splitter_cols.count("time_group_ids") > 0) {
      const auto& splitter_time_group_list =
          static_cast<const arrow::ListArray&>(*ctx.splitter_cols.at("time_group_ids")->chunk(0));
      const auto splitter_values = splitter_time_group_list.values();
      if (splitter_values->type_id() == arrow::Type::INT64) {
        auto tg_values =
            std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(splitter_values);
        splitter_tg_raw64 = tg_values->raw_values();
        splitter_tg_is_int64 = true;
      } else if (splitter_values->type_id() == arrow::Type::INT32) {
        auto tg_values =
            std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(splitter_values);
        splitter_tg_raw32 = tg_values->raw_values();
        splitter_tg_is_int32 = true;
      } else {
        throw std::runtime_error("Unsupported splitter time_group_ids type.");
      }
      s_tg_offsets = splitter_time_group_list.raw_value_offsets();
    }
  }

  utils::parallel::Parallel::For(0, ctx.rows, [&](int64_t row) {
    const int64_t n = static_cast<int64_t>(ctx.offsets[row + 1] - ctx.offsets[row]);
    if (n == 0) {
      return;
    }

    const int64_t groups_for_row = ctx.group_counts[row];
    const int64_t graph_base = ctx.graph_offsets[row];
    const int32_t start = ctx.offsets[row];
    const int32_t tg_start = ctx.tg_offsets[row];

    if (ctx.has_prob_columns && use_group_probs_) {
      const int64_t count = static_cast<int64_t>(p_prob_offsets[row + 1] - p_prob_offsets[row]);
      if (count != static_cast<int64_t>(m_prob_offsets[row + 1] - m_prob_offsets[row]) ||
          count != static_cast<int64_t>(i_prob_offsets[row + 1] - i_prob_offsets[row])) {
        throw std::runtime_error("pred_pion/pred_muon/pred_mip list lengths mismatch.");
      }
      if (count != groups_for_row) {
        throw std::runtime_error("Group probability list length does not match time groups.");
      }
    }

    if (ctx.has_splitter_columns && use_splitter_probs_) {
      const int64_t count = static_cast<int64_t>(p_splitter_offsets[row + 1] - p_splitter_offsets[row]);
      if (count != static_cast<int64_t>(m_splitter_offsets[row + 1] - m_splitter_offsets[row]) ||
          count != static_cast<int64_t>(i_splitter_offsets[row + 1] - i_splitter_offsets[row])) {
        throw std::runtime_error("pred_hit_pion/pred_hit_muon/pred_hit_mip list lengths mismatch.");
      }
      if (count != n) {
        throw std::runtime_error("Splitter probability list length does not match hits.");
      }
      if (s_tg_offsets != nullptr &&
          count != static_cast<int64_t>(s_tg_offsets[row + 1] - s_tg_offsets[row])) {
        throw std::runtime_error("Splitter time_group_ids length does not match hits.");
      }
    }

    std::vector<std::vector<int64_t>> group_nodes(static_cast<size_t>(groups_for_row));
    for (int64_t g = 0; g < groups_for_row; ++g) {
      group_nodes[static_cast<size_t>(g)].reserve(
          static_cast<size_t>(ctx.group_node_counts[static_cast<size_t>(row)][static_cast<size_t>(g)]));
    }
    for (int64_t i = 0; i < n; ++i) {
      group_nodes[static_cast<size_t>(ctx.tg_raw[tg_start + i])].push_back(i);
    }
    std::vector<std::vector<int64_t>> splitter_positions_by_group;
    if (ctx.has_splitter_columns && use_splitter_probs_ && s_tg_offsets != nullptr) {
      splitter_positions_by_group.resize(static_cast<size_t>(groups_for_row));
      const int32_t s_start = s_tg_offsets[row];
      for (int64_t i = 0; i < n; ++i) {
        const int32_t s_idx = s_start + static_cast<int32_t>(i);
        int64_t gid = 0;
        if (splitter_tg_is_int64) {
          gid = splitter_tg_raw64[s_idx];
        } else if (splitter_tg_is_int32) {
          gid = static_cast<int64_t>(splitter_tg_raw32[s_idx]);
        } else {
          throw std::runtime_error("Splitter time_group_ids type flags are not initialized.");
        }
        if (gid < 0 || gid >= groups_for_row) {
          throw std::runtime_error("Invalid splitter time_group_ids value.");
        }
        splitter_positions_by_group[static_cast<size_t>(gid)].push_back(i);
      }
      for (int64_t g = 0; g < groups_for_row; ++g) {
        const auto expected = group_nodes[static_cast<size_t>(g)].size();
        const auto observed = splitter_positions_by_group[static_cast<size_t>(g)].size();
        if (expected != observed) {
          throw std::runtime_error("Splitter time_group_ids are not aligned with event hits.");
        }
      }
    }

    for (int64_t group = 0; group < groups_for_row; ++group) {
      const int64_t graph_idx = graph_base + group;
      bufs->graph_event_ids[graph_idx] = row;
      bufs->graph_group_ids[graph_idx] = group;

      const int64_t node_offset = ctx.node_offsets[static_cast<size_t>(graph_idx)];
      const int64_t edge_offset = ctx.edge_offsets[static_cast<size_t>(graph_idx)];
      const auto& nodes = group_nodes[static_cast<size_t>(group)];
      const int64_t k = static_cast<int64_t>(nodes.size());

      std::vector<float> coord_local(static_cast<size_t>(k), 0.0f);
      std::vector<float> z_local(static_cast<size_t>(k), 0.0f);
      std::vector<float> e_local(static_cast<size_t>(k), 0.0f);
      std::vector<int32_t> view_local(static_cast<size_t>(k), 0);

      double sum_edep = 0.0;
      for (int64_t local = 0; local < k; ++local) {
        const int64_t src_idx = nodes[static_cast<size_t>(local)];
        const int64_t raw_idx = start + src_idx;
        const int64_t node_idx = node_offset + local;
        const int64_t feat_base = node_idx * 4;

        const int32_t view = ctx.view_raw[raw_idx];
        const double coord = ResolveCoordinateForView(ctx.x_values, ctx.y_values, view, raw_idx);
        const float z = static_cast<float>(ctx.z_values.IsValid(raw_idx) ? ctx.z_values.Value(raw_idx) : 0.0);
        const float e = static_cast<float>(ctx.edep_values.IsValid(raw_idx) ? ctx.edep_values.Value(raw_idx) : 0.0);

        coord_local[static_cast<size_t>(local)] = static_cast<float>(coord);
        z_local[static_cast<size_t>(local)] = z;
        e_local[static_cast<size_t>(local)] = e;
        view_local[static_cast<size_t>(local)] = view;

        bufs->node_feat[feat_base] = coord_local[static_cast<size_t>(local)];
        bufs->node_feat[feat_base + 1] = z;
        bufs->node_feat[feat_base + 2] = e;
        bufs->node_feat[feat_base + 3] = static_cast<float>(view);
        bufs->time_group_ids[node_idx] = 0;

        const int cls = ClassFromPdg(ctx.pdg_raw[raw_idx]);
        if (cls >= 0) {
          bufs->node_truth[static_cast<size_t>(node_idx * 3 + cls)] = 1.0f;
          bufs->group_truth[static_cast<size_t>(graph_idx * 3 + cls)] = 1U;
        }

        if (ctx.has_splitter_columns && use_splitter_probs_) {
          int32_t list_idx = p_splitter_offsets[row] + static_cast<int32_t>(src_idx);
          if (!splitter_positions_by_group.empty()) {
            const int64_t splitter_rel =
                splitter_positions_by_group[static_cast<size_t>(group)][static_cast<size_t>(local)];
            list_idx = p_splitter_offsets[row] + static_cast<int32_t>(splitter_rel);
          }
          const int64_t out_base = node_idx * 3;
          bufs->splitter_probs[out_base] = static_cast<float>(pion_splitter_values.Value(list_idx));
          bufs->splitter_probs[out_base + 1] = static_cast<float>(muon_splitter_values.Value(list_idx));
          bufs->splitter_probs[out_base + 2] = static_cast<float>(mip_splitter_values.Value(list_idx));
        }
        sum_edep += e;
      }
      bufs->u[graph_idx] = static_cast<float>(sum_edep);

      if (ctx.has_prob_columns && use_group_probs_) {
        const int32_t list_idx = p_prob_offsets[row] + static_cast<int32_t>(group);
        const int64_t out_base = graph_idx * 3;
        bufs->group_probs[out_base] = static_cast<float>(pion_prob_values.Value(list_idx));
        bufs->group_probs[out_base + 1] = static_cast<float>(muon_prob_values.Value(list_idx));
        bufs->group_probs[out_base + 2] = static_cast<float>(mip_prob_values.Value(list_idx));
      }

      if (ctx.has_targets) {
        const int32_t list_idx = targets.sx_offsets[row] + static_cast<int32_t>(group);
        const int64_t out_base = graph_idx * 6;
        bufs->y[out_base] = static_cast<float>(targets.start_x.Value(list_idx));
        bufs->y[out_base + 1] = static_cast<float>(targets.start_y.Value(list_idx));
        bufs->y[out_base + 2] = static_cast<float>(targets.start_z.Value(list_idx));
        bufs->y[out_base + 3] = static_cast<float>(targets.end_x.Value(list_idx));
        bufs->y[out_base + 4] = static_cast<float>(targets.end_y.Value(list_idx));
        bufs->y[out_base + 5] = static_cast<float>(targets.end_z.Value(list_idx));
      }

      int64_t edge_local = 0;
      for (int64_t a = 0; a < k; ++a) {
        const int32_t view_i = view_local[static_cast<size_t>(a)];
        const float z_i = z_local[static_cast<size_t>(a)];
        const float e_i = e_local[static_cast<size_t>(a)];
        const float coord_i = coord_local[static_cast<size_t>(a)];

        for (int64_t b = 0; b < k; ++b) {
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
    }
  });
}

std::unique_ptr<BaseBatch> EndpointRegressorLoader::BuildGraphPhase5Finalize(
    const BuildContext& ctx,
    BuildBuffers* bufs) const {
  if (use_group_probs_ && !ctx.has_prob_columns) {
    for (int64_t graph_idx = 0; graph_idx < ctx.total_graphs; ++graph_idx) {
      const int64_t base = graph_idx * 3;
      bufs->group_probs[base] = static_cast<float>(bufs->group_truth[static_cast<size_t>(base)]);
      bufs->group_probs[base + 1] = static_cast<float>(bufs->group_truth[static_cast<size_t>(base + 1)]);
      bufs->group_probs[base + 2] = static_cast<float>(bufs->group_truth[static_cast<size_t>(base + 2)]);
    }
  }
  if (use_splitter_probs_ && !ctx.has_splitter_columns) {
    std::copy(bufs->node_truth.begin(), bufs->node_truth.end(), bufs->splitter_probs);
  }

  auto out = std::make_unique<EndpointRegressorInputs>();
  out->node_features = MakeArray(bufs->node_feat_buf, arrow::float32(), ctx.total_nodes * 4);
  out->edge_index = MakeArray(bufs->edge_index_buf, arrow::int64(), ctx.total_edges * 2);
  out->edge_attr = MakeArray(bufs->edge_attr_buf, arrow::float32(), ctx.total_edges * 4);
  out->time_group_ids = MakeArray(bufs->time_group_buf, arrow::int64(), ctx.total_nodes);
  out->u = MakeArray(bufs->u_buf, arrow::float32(), ctx.total_graphs);
  out->group_probs = MakeArray(bufs->group_probs_buf, arrow::float32(), ctx.total_graphs * 3);
  out->splitter_probs = MakeArray(bufs->splitter_probs_buf, arrow::float32(), ctx.total_nodes * 3);
  out->node_ptr = MakeArray(bufs->node_ptr_buf, arrow::int64(), ctx.total_graphs + 1);
  out->edge_ptr = MakeArray(bufs->edge_ptr_buf, arrow::int64(), ctx.total_graphs + 1);
  out->group_ptr = MakeArray(bufs->group_ptr_buf, arrow::int64(), ctx.total_graphs + 1);
  out->graph_event_ids = MakeArray(bufs->graph_event_ids_buf, arrow::int64(), ctx.total_graphs);
  out->graph_group_ids = MakeArray(bufs->graph_group_ids_buf, arrow::int64(), ctx.total_graphs);
  out->y = MakeArray(bufs->y_buf, arrow::float32(), ctx.total_graphs * 6);
  out->num_graphs = static_cast<size_t>(ctx.total_graphs);
  out->num_groups = static_cast<size_t>(ctx.total_graphs);
  if (!ctx.has_targets) {
    out->y.reset();
  }
  return out;
}

TrainingBundle EndpointRegressorLoader::SplitInputsTargets(std::unique_ptr<BaseBatch> batch_base) const {
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
