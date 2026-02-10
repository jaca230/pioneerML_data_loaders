#include "pioneerml_dataloaders/configurable/dataloaders/graph/group_splitter_loader.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <arrow/api.h>

#include "pioneerml_dataloaders/batch/group_splitter_batch.h"
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

GroupSplitterLoader::GroupSplitterLoader() {
  input_columns_ = {
      "hits_x",
      "hits_y",
      "hits_z",
      "hits_edep",
      "hits_strip_type",
      "hits_time",
      "hits_time_group",
  };
  target_columns_ = {"hits_pdg_id"};
  ConfigureDerivers(nullptr);
}

void GroupSplitterLoader::LoadConfig(const nlohmann::json& cfg) {
  if (cfg.contains("time_window_ns")) {
    time_window_ns_ = cfg.at("time_window_ns").get<double>();
  }
  if (cfg.contains("use_group_probs")) {
    use_group_probs_ = cfg.at("use_group_probs").get<bool>();
  }

  const nlohmann::json* derivers_cfg = nullptr;
  if (cfg.contains("derivers")) {
    derivers_cfg = &cfg.at("derivers");
  }
  ConfigureDerivers(derivers_cfg);
}

void GroupSplitterLoader::ConfigureDerivers(const nlohmann::json* derivers_cfg) {
  derivers_.clear();

  auto time_group_summary =
      std::make_shared<data_derivers::TimeGroupSummaryDeriver>(
          time_window_ns_,
          std::vector<std::string>{
              "hits_time_group",
              "hits_pdg_id",
              "hits_particle_mask",
          });
  if (derivers_cfg && derivers_cfg->contains("time_group_summary")) {
    time_group_summary->LoadConfig(derivers_cfg->at("time_group_summary"));
  }
  AddDeriver({"hits_time_group", "hits_pdg_id", "hits_particle_mask"},
             time_group_summary);
}

void GroupSplitterLoader::CountGroupsForRows(
    const int32_t* offsets,
    const int32_t* tg_offsets,
    const int64_t* tg_raw,
    const int32_t* pdg_offsets,
    int64_t rows,
    bool has_targets,
    std::vector<int64_t>* group_counts,
    std::vector<std::vector<int64_t>>* group_node_counts) const {
  utils::parallel::Parallel::For(0, rows, [&](int64_t row) {
    const int64_t n = static_cast<int64_t>(offsets[row + 1] - offsets[row]);
    if (n == 0) {
      (*group_counts)[row] = 0;
      return;
    }
    if ((tg_offsets[row + 1] - tg_offsets[row]) != n) {
      throw std::runtime_error("hits_time_group length mismatch with hits.");
    }
    if (has_targets && (pdg_offsets[row + 1] - pdg_offsets[row]) != n) {
      throw std::runtime_error("hits_pdg_id length mismatch with hits.");
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
      const int64_t group = tg_raw[tg_start + i];
      if (group < 0 || group >= groups_for_row) {
        throw std::runtime_error("Invalid time group id encountered.");
      }
      counts[group] += 1;
    }
    (*group_node_counts)[row] = std::move(counts);
  });
}

TrainingBundle GroupSplitterLoader::LoadTraining(
    const std::shared_ptr<arrow::Table>& table) const {
  auto prepared = PrepareTable(table, true);
  auto required = utils::parquet::MergeColumns(input_columns_, target_columns_);
  utils::parquet::ValidateColumns(*prepared,
                                  required,
                                  {"pred_pion", "pred_muon", "pred_mip"},
                                  true,
                                  "GroupSplitterLoader training");
  auto batch = BuildGraph(*prepared);
  return SplitInputsTargets(std::move(batch));
}

InferenceBundle GroupSplitterLoader::LoadInference(
    const std::shared_ptr<arrow::Table>& table) const {
  auto prepared = PrepareTable(table, true);
  utils::parquet::ValidateColumns(*prepared,
                                  input_columns_,
                                  {"pred_pion", "pred_muon", "pred_mip", "hits_pdg_id",
                                   "hits_particle_mask"},
                                  true,
                                  "GroupSplitterLoader inputs");
  InferenceBundle out;
  out.inputs = BuildGraph(*prepared);
  return out;
}

std::unique_ptr<BaseBatch> GroupSplitterLoader::BuildGraph(const arrow::Table& table) const {
  utils::timing::ScopedTimer total_timer("group_splitter.build_graph");
  BuildContext ctx;
  BuildBuffers bufs;
  {
    utils::timing::ScopedTimer timer("group_splitter.phase0_initialize");
    BuildGraphPhase0Initialize(table, &ctx);
  }
  {
    utils::timing::ScopedTimer timer("group_splitter.phase1_count");
    BuildGraphPhase1Count(&ctx);
  }
  {
    utils::timing::ScopedTimer timer("group_splitter.phase2_offsets");
    BuildGraphPhase2Offsets(&ctx);
  }
  {
    utils::timing::ScopedTimer timer("group_splitter.phase3_allocate");
    BuildGraphPhase3Allocate(ctx, &bufs);
  }
  {
    utils::timing::ScopedTimer timer("group_splitter.phase4_populate");
    BuildGraphPhase4Populate(ctx, &bufs);
  }
  {
    utils::timing::ScopedTimer timer("group_splitter.phase5_finalize");
    return BuildGraphPhase5Finalize(ctx, &bufs);
  }
}

void GroupSplitterLoader::BuildGraphPhase0Initialize(const arrow::Table& table,
                                                     BuildContext* ctx) const {
  ctx->input_cols = utils::parquet::BindColumns(
      table, input_columns_, true, true, "GroupSplitterLoader inputs");
  ctx->target_cols = utils::parquet::BindColumns(
      table, target_columns_, false, true, "GroupSplitterLoader targets");
  ctx->prob_cols = utils::parquet::BindColumns(
      table, {"pred_pion", "pred_muon", "pred_mip"}, false, true, "GroupSplitterLoader probs");

  ctx->has_targets = ctx->target_cols.size() == target_columns_.size();
  ctx->has_prob_columns = ctx->prob_cols.size() == 3;

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

  ctx->x_values = MakeNumericAccessor(ctx->hits_x->values(), "group splitter x");
  ctx->y_values = MakeNumericAccessor(ctx->hits_y->values(), "group splitter y");
  ctx->z_values = MakeNumericAccessor(ctx->hits_z->values(), "group splitter z");
  ctx->edep_values = MakeNumericAccessor(ctx->hits_edep->values(), "group splitter edep");

  auto view_values =
      std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(ctx->hits_view->values());
  auto tg_values = std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(
      ctx->hits_time_group->values());
  ctx->view_raw = view_values->raw_values();
  ctx->tg_raw = tg_values->raw_values();
  ctx->offsets = ctx->hits_z->raw_value_offsets();
  ctx->tg_offsets = ctx->hits_time_group->raw_value_offsets();

  if (ctx->has_targets) {
    const auto& hits_pdg = static_cast<const arrow::ListArray&>(
        *ctx->target_cols.at("hits_pdg_id")->chunk(0));
    auto pdg_values =
        std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(hits_pdg.values());
    ctx->pdg_offsets = hits_pdg.raw_value_offsets();
    ctx->pdg_raw = pdg_values->raw_values();
  }

  ctx->rows = table.num_rows();
  ctx->group_counts.assign(static_cast<size_t>(ctx->rows), 0);
  ctx->group_node_counts.resize(static_cast<size_t>(ctx->rows));
}

void GroupSplitterLoader::BuildGraphPhase1Count(BuildContext* ctx) const {
  CountGroupsForRows(ctx->offsets,
                     ctx->tg_offsets,
                     ctx->tg_raw,
                     ctx->pdg_offsets,
                     ctx->rows,
                     ctx->has_targets,
                     &ctx->group_counts,
                     &ctx->group_node_counts);
}

void GroupSplitterLoader::BuildGraphPhase2Offsets(BuildContext* ctx) const {
  ctx->graph_offsets = BuildOffsets(ctx->group_counts);
  ctx->total_graphs = ctx->graph_offsets.back();

  ctx->node_counts.assign(static_cast<size_t>(ctx->total_graphs), 0);
  ctx->edge_counts.assign(static_cast<size_t>(ctx->total_graphs), 0);
  for (int64_t row = 0; row < ctx->rows; ++row) {
    const int64_t graph_base = ctx->graph_offsets[row];
    const auto& counts = ctx->group_node_counts[static_cast<size_t>(row)];
    for (int64_t g = 0; g < static_cast<int64_t>(counts.size()); ++g) {
      const int64_t graph_idx = graph_base + g;
      ctx->node_counts[static_cast<size_t>(graph_idx)] = counts[static_cast<size_t>(g)];
      ctx->edge_counts[static_cast<size_t>(graph_idx)] =
          counts[static_cast<size_t>(g)] * (counts[static_cast<size_t>(g)] - 1);
    }
  }

  ctx->node_offsets = BuildOffsets(ctx->node_counts);
  ctx->edge_offsets = BuildOffsets(ctx->edge_counts);
  ctx->total_nodes = ctx->node_offsets.back();
  ctx->total_edges = ctx->edge_offsets.back();
}

void GroupSplitterLoader::BuildGraphPhase3Allocate(const BuildContext& ctx,
                                                   BuildBuffers* bufs) const {
  bufs->node_feat_buf =
      AllocBuffer(ctx.total_nodes * static_cast<int64_t>(sizeof(float)) * 4);
  bufs->edge_index_buf =
      AllocBuffer(ctx.total_edges * static_cast<int64_t>(sizeof(int64_t)) * 2);
  bufs->edge_attr_buf =
      AllocBuffer(ctx.total_edges * static_cast<int64_t>(sizeof(float)) * 4);
  bufs->node_ptr_buf =
      AllocBuffer((ctx.total_graphs + 1) * static_cast<int64_t>(sizeof(int64_t)));
  bufs->edge_ptr_buf =
      AllocBuffer((ctx.total_graphs + 1) * static_cast<int64_t>(sizeof(int64_t)));
  bufs->u_buf = AllocBuffer(ctx.total_graphs * static_cast<int64_t>(sizeof(float)));
  bufs->group_probs_buf =
      AllocBuffer(ctx.total_graphs * static_cast<int64_t>(sizeof(float)) * 3);
  bufs->graph_event_ids_buf =
      AllocBuffer(ctx.total_graphs * static_cast<int64_t>(sizeof(int64_t)));
  bufs->graph_group_ids_buf =
      AllocBuffer(ctx.total_graphs * static_cast<int64_t>(sizeof(int64_t)));
  bufs->y_node_buf =
      AllocBuffer(ctx.total_nodes * static_cast<int64_t>(sizeof(float)) * 3);

  bufs->node_feat = reinterpret_cast<float*>(bufs->node_feat_buf->mutable_data());
  bufs->edge_index = reinterpret_cast<int64_t*>(bufs->edge_index_buf->mutable_data());
  bufs->edge_attr = reinterpret_cast<float*>(bufs->edge_attr_buf->mutable_data());
  bufs->node_ptr = reinterpret_cast<int64_t*>(bufs->node_ptr_buf->mutable_data());
  bufs->edge_ptr = reinterpret_cast<int64_t*>(bufs->edge_ptr_buf->mutable_data());
  bufs->u = reinterpret_cast<float*>(bufs->u_buf->mutable_data());
  bufs->group_probs = reinterpret_cast<float*>(bufs->group_probs_buf->mutable_data());
  bufs->graph_event_ids =
      reinterpret_cast<int64_t*>(bufs->graph_event_ids_buf->mutable_data());
  bufs->graph_group_ids =
      reinterpret_cast<int64_t*>(bufs->graph_group_ids_buf->mutable_data());
  bufs->y_node = reinterpret_cast<float*>(bufs->y_node_buf->mutable_data());
  bufs->group_truth.assign(static_cast<size_t>(ctx.total_graphs * 3), 0U);

  std::fill(bufs->group_probs, bufs->group_probs + (ctx.total_graphs * 3), 0.0f);
  std::fill(bufs->y_node, bufs->y_node + (ctx.total_nodes * 3), 0.0f);

  FillPointerArrayFromOffsets(ctx.node_offsets, bufs->node_ptr);
  FillPointerArrayFromOffsets(ctx.edge_offsets, bufs->edge_ptr);
}

void GroupSplitterLoader::BuildGraphPhase4Populate(const BuildContext& ctx,
                                                   BuildBuffers* bufs) const {
  utils::parallel::Parallel::For(0, ctx.rows, [&](int64_t row) {
    const int64_t n = static_cast<int64_t>(ctx.offsets[row + 1] - ctx.offsets[row]);
    if (n == 0) {
      return;
    }
    const int32_t start = ctx.offsets[row];
    const int32_t tg_start = ctx.tg_offsets[row];
    const int64_t groups_for_row = ctx.group_counts[row];
    if (groups_for_row == 0) {
      return;
    }

    std::vector<std::vector<int64_t>> group_nodes(static_cast<size_t>(groups_for_row));
    for (int64_t g = 0; g < groups_for_row; ++g) {
      group_nodes[static_cast<size_t>(g)].reserve(static_cast<size_t>(
          ctx.group_node_counts[static_cast<size_t>(row)][static_cast<size_t>(g)]));
    }
    for (int64_t i = 0; i < n; ++i) {
      group_nodes[static_cast<size_t>(ctx.tg_raw[tg_start + i])].push_back(i);
    }

    const int64_t graph_base = ctx.graph_offsets[row];
    for (int64_t group = 0; group < groups_for_row; ++group) {
      const int64_t graph_idx = graph_base + group;
      bufs->graph_event_ids[graph_idx] = row;
      bufs->graph_group_ids[graph_idx] = group;

      const int64_t node_offset = ctx.node_offsets[graph_idx];
      const int64_t edge_offset = ctx.edge_offsets[graph_idx];
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
        const int64_t base = node_idx * 4;

        const int32_t view = ctx.view_raw[raw_idx];
        const double coord =
            ResolveCoordinateForView(ctx.x_values, ctx.y_values, view, raw_idx);
        const float z = static_cast<float>(
            ctx.z_values.IsValid(raw_idx) ? ctx.z_values.Value(raw_idx) : 0.0);
        const float e = static_cast<float>(
            ctx.edep_values.IsValid(raw_idx) ? ctx.edep_values.Value(raw_idx) : 0.0);
        coord_local[static_cast<size_t>(local)] = static_cast<float>(coord);
        z_local[static_cast<size_t>(local)] = z;
        e_local[static_cast<size_t>(local)] = e;
        view_local[static_cast<size_t>(local)] = view;

        bufs->node_feat[base] = coord_local[static_cast<size_t>(local)];
        bufs->node_feat[base + 1] = z;
        bufs->node_feat[base + 2] = e;
        bufs->node_feat[base + 3] = static_cast<float>(view);

        if (ctx.has_targets) {
          const int cls = ClassFromPdg(ctx.pdg_raw[raw_idx]);
          if (cls >= 0) {
            bufs->y_node[node_idx * 3 + cls] = 1.0f;
            bufs->group_truth[static_cast<size_t>(graph_idx * 3 + cls)] = 1U;
          }
        }
        sum_edep += e;
      }
      bufs->u[graph_idx] = static_cast<float>(sum_edep);

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

std::unique_ptr<BaseBatch> GroupSplitterLoader::BuildGraphPhase5Finalize(
    const BuildContext& ctx,
    BuildBuffers* bufs) const {
  if (!ctx.has_prob_columns && use_group_probs_ && ctx.has_targets) {
    for (int64_t graph_idx = 0; graph_idx < ctx.total_graphs; ++graph_idx) {
      const int64_t base = graph_idx * 3;
      bufs->group_probs[base] =
          static_cast<float>(bufs->group_truth[static_cast<size_t>(base)]);
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
      if (count != ctx.group_counts[row]) {
        throw std::runtime_error("Prediction list length does not match time groups.");
      }

      const int64_t graph_base = ctx.graph_offsets[row];
      const int32_t start_idx = p_offsets[row];
      for (int64_t g = 0; g < count; ++g) {
        const int32_t list_idx = start_idx + static_cast<int32_t>(g);
        const int64_t base = (graph_base + g) * 3;
        bufs->group_probs[base] = static_cast<float>(pion_pred_values.Value(list_idx));
        bufs->group_probs[base + 1] = static_cast<float>(muon_pred_values.Value(list_idx));
        bufs->group_probs[base + 2] = static_cast<float>(mip_pred_values.Value(list_idx));
      }
    }
  }

  auto out = std::make_unique<GroupSplitterInputs>();
  out->node_features = MakeArray(bufs->node_feat_buf, arrow::float32(), ctx.total_nodes * 4);
  out->edge_index = MakeArray(bufs->edge_index_buf, arrow::int64(), ctx.total_edges * 2);
  out->edge_attr = MakeArray(bufs->edge_attr_buf, arrow::float32(), ctx.total_edges * 4);
  out->u = MakeArray(bufs->u_buf, arrow::float32(), ctx.total_graphs);
  out->group_probs = MakeArray(bufs->group_probs_buf, arrow::float32(), ctx.total_graphs * 3);
  out->node_ptr = MakeArray(bufs->node_ptr_buf, arrow::int64(), ctx.total_graphs + 1);
  out->edge_ptr = MakeArray(bufs->edge_ptr_buf, arrow::int64(), ctx.total_graphs + 1);
  out->graph_event_ids = MakeArray(bufs->graph_event_ids_buf, arrow::int64(), ctx.total_graphs);
  out->graph_group_ids = MakeArray(bufs->graph_group_ids_buf, arrow::int64(), ctx.total_graphs);
  out->y_node = MakeArray(bufs->y_node_buf, arrow::float32(), ctx.total_nodes * 3);
  out->num_graphs = static_cast<size_t>(ctx.total_graphs);
  if (!ctx.has_targets) {
    out->y_node.reset();
  }
  return out;
}

TrainingBundle GroupSplitterLoader::SplitInputsTargets(std::unique_ptr<BaseBatch> batch_base) const {
  auto* typed = dynamic_cast<GroupSplitterInputs*>(batch_base.get());
  if (!typed) {
    throw std::runtime_error("Unexpected batch type in SplitInputsTargets");
  }
  if (!typed->y_node) {
    throw std::runtime_error("Training targets are missing. Use LoadInference or provide label columns.");
  }

  auto targets = std::make_unique<GroupSplitterTargets>();
  targets->num_graphs = typed->num_graphs;
  targets->y_node = typed->y_node;

  typed->y_node.reset();

  TrainingBundle result;
  result.inputs = std::move(batch_base);
  result.targets = std::move(targets);
  return result;
}

}  // namespace pioneerml::dataloaders::graph
