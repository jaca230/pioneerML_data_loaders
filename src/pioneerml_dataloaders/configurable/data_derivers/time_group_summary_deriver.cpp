#include "pioneerml_dataloaders/configurable/data_derivers/time_group_summary_deriver.h"

#include <arrow/api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pioneerml_dataloaders/utils/parallel/parallel.h"

namespace pioneerml::data_derivers {
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

int64_t MaskFromPdg(int pdg) {
  if (pdg == 211) {
    return 0b00001;
  }
  if (pdg == -13) {
    return 0b00010;
  }
  if (pdg == -11) {
    return 0b00100;
  }
  if (pdg == 11) {
    return 0b01000;
  }
  return 0b10000;
}

template <typename BuilderType, typename ValueType>
void AppendListValues(arrow::ListBuilder* list_builder,
                      BuilderType* value_builder,
                      const std::vector<ValueType>& values,
                      const std::string& context) {
  auto st = list_builder->Append();
  if (!st.ok()) {
    throw std::runtime_error(context + ": failed to append list (" + st.ToString() + ")");
  }
  for (const auto& v : values) {
    st = value_builder->Append(v);
    if (!st.ok()) {
      throw std::runtime_error(context + ": failed to append value (" + st.ToString() + ")");
    }
  }
}

class IntegerReader {
 public:
  IntegerReader() = default;

  explicit IntegerReader(const std::shared_ptr<arrow::Array>& arr) { Bind(arr); }

  void Bind(const std::shared_ptr<arrow::Array>& arr) {
    arr_ = arr;
    i32_ = nullptr;
    i64_ = nullptr;
    if (!arr_) {
      return;
    }
    switch (arr_->type_id()) {
      case arrow::Type::INT32:
        i32_ = std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(arr_)->raw_values();
        break;
      case arrow::Type::INT64:
        i64_ = std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(arr_)->raw_values();
        break;
      default:
        throw std::runtime_error("Expected int32/int64 array.");
    }
  }

  bool IsBound() const { return arr_ != nullptr; }
  bool IsValid(int64_t idx) const { return arr_ && arr_->IsValid(idx); }

  int64_t Value(int64_t idx) const {
    if (i64_) {
      return i64_[idx];
    }
    if (i32_) {
      return static_cast<int64_t>(i32_[idx]);
    }
    return 0;
  }

 private:
  std::shared_ptr<arrow::Array> arr_;
  const int32_t* i32_{nullptr};
  const int64_t* i64_{nullptr};
};

class FloatReader {
 public:
  FloatReader() = default;

  explicit FloatReader(const std::shared_ptr<arrow::Array>& arr) { Bind(arr); }

  void Bind(const std::shared_ptr<arrow::Array>& arr) {
    arr_ = arr;
    f32_ = nullptr;
    f64_ = nullptr;
    if (!arr_) {
      return;
    }
    switch (arr_->type_id()) {
      case arrow::Type::FLOAT:
        f32_ = std::static_pointer_cast<arrow::NumericArray<arrow::FloatType>>(arr_)->raw_values();
        break;
      case arrow::Type::DOUBLE:
        f64_ = std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(arr_)->raw_values();
        break;
      default:
        throw std::runtime_error("Expected float/double array.");
    }
  }

  bool IsBound() const { return arr_ != nullptr; }
  bool IsValid(int64_t idx) const { return arr_ && arr_->IsValid(idx); }

  double Value(int64_t idx) const {
    if (f64_) {
      return f64_[idx];
    }
    if (f32_) {
      return static_cast<double>(f32_[idx]);
    }
    return 0.0;
  }

 private:
  std::shared_ptr<arrow::Array> arr_;
  const float* f32_{nullptr};
  const double* f64_{nullptr};
};

struct StepKey {
  int64_t mc_event_id{0};
  int64_t step_id{0};

  bool operator==(const StepKey& other) const {
    return mc_event_id == other.mc_event_id && step_id == other.step_id;
  }
};

struct StepKeyHash {
  std::size_t operator()(const StepKey& k) const noexcept {
    const std::size_t h1 = std::hash<int64_t>{}(k.mc_event_id);
    const std::size_t h2 = std::hash<int64_t>{}(k.step_id);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6U) + (h1 >> 2U));
  }
};

struct StepTruth {
  int32_t pdg{0};
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double time{0.0};
  double edep{0.0};
};

struct GroupStats {
  int32_t has_pion{0};
  int32_t has_muon{0};
  int32_t has_mip{0};
  double pion_energy{0.0};
  double muon_energy{0.0};
  double mip_energy{0.0};
  std::vector<StepTruth> all_points;
  std::vector<StepTruth> non_electron_points;
  std::unordered_set<StepKey, StepKeyHash> seen_all;
  std::unordered_set<StepKey, StepKeyHash> seen_non_electron;
};

std::vector<std::shared_ptr<arrow::Array>> MakeNullOutputs(int64_t rows, size_t count) {
  std::vector<std::shared_ptr<arrow::Array>> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    out.push_back(std::make_shared<arrow::NullArray>(rows));
  }
  return out;
}

}  // namespace

void TimeGroupSummaryDeriver::LoadConfig(const nlohmann::json& cfg) {
  if (cfg.contains("window_ns")) {
    window_ns_ = cfg.at("window_ns").get<double>();
  }
  if (cfg.contains("time_column")) {
    time_column_ = cfg.at("time_column").get<std::string>();
  }
  if (cfg.contains("edep_column")) {
    edep_column_ = cfg.at("edep_column").get<std::string>();
  }
  if (cfg.contains("fallback_pdg_column")) {
    fallback_pdg_column_ = cfg.at("fallback_pdg_column").get<std::string>();
  }
  if (cfg.contains("contrib_mc_event_id_column")) {
    contrib_mc_event_id_column_ = cfg.at("contrib_mc_event_id_column").get<std::string>();
  }
  if (cfg.contains("contrib_step_id_column")) {
    contrib_step_id_column_ = cfg.at("contrib_step_id_column").get<std::string>();
  }
  if (cfg.contains("steps_mc_event_id_column")) {
    steps_mc_event_id_column_ = cfg.at("steps_mc_event_id_column").get<std::string>();
  }
  if (cfg.contains("steps_step_id_column")) {
    steps_step_id_column_ = cfg.at("steps_step_id_column").get<std::string>();
  }
  if (cfg.contains("steps_pdg_id_column")) {
    steps_pdg_id_column_ = cfg.at("steps_pdg_id_column").get<std::string>();
  }
  if (cfg.contains("steps_x_column")) {
    steps_x_column_ = cfg.at("steps_x_column").get<std::string>();
  }
  if (cfg.contains("steps_y_column")) {
    steps_y_column_ = cfg.at("steps_y_column").get<std::string>();
  }
  if (cfg.contains("steps_z_column")) {
    steps_z_column_ = cfg.at("steps_z_column").get<std::string>();
  }
  if (cfg.contains("steps_edep_column")) {
    steps_edep_column_ = cfg.at("steps_edep_column").get<std::string>();
  }
  if (cfg.contains("steps_time_column")) {
    steps_time_column_ = cfg.at("steps_time_column").get<std::string>();
  }
  if (cfg.contains("output_columns")) {
    output_columns_ = cfg.at("output_columns").get<std::vector<std::string>>();
  }
}

std::vector<std::shared_ptr<arrow::Array>> TimeGroupSummaryDeriver::DeriveColumns(
    const arrow::Table& table) const {
  if (output_columns_.empty()) {
    return {};
  }
  std::unordered_set<std::string> requested(output_columns_.begin(), output_columns_.end());
  const bool need_hits_time_group = requested.count("hits_time_group") > 0;
  const bool need_hits_pdg_id = requested.count("hits_pdg_id") > 0;
  const bool need_hits_particle_mask = requested.count("hits_particle_mask") > 0;
  const bool need_group_presence = requested.count("pion_in_group") > 0 ||
                                   requested.count("muon_in_group") > 0 ||
                                   requested.count("mip_in_group") > 0;
  const bool need_endpoints = requested.count("group_start_x") > 0 ||
                              requested.count("group_start_y") > 0 ||
                              requested.count("group_start_z") > 0 ||
                              requested.count("group_end_x") > 0 ||
                              requested.count("group_end_y") > 0 ||
                              requested.count("group_end_z") > 0;
  const bool need_arc = requested.count("group_true_arc_length") > 0;
  const bool need_group_energy = requested.count("pion_energy_per_group") > 0 ||
                                 requested.count("muon_energy_per_group") > 0 ||
                                 requested.count("mip_energy_per_group") > 0;
  const bool need_any_group_stats = need_group_presence || need_group_energy || need_endpoints || need_arc;

  if (!need_hits_time_group && !need_hits_pdg_id && !need_hits_particle_mask && !need_any_group_stats) {
    return {};
  }

  auto time_col = table.GetColumnByName(time_column_);
  auto edep_col = table.GetColumnByName(edep_column_);
  if (!time_col || !edep_col) {
    return MakeNullOutputs(table.num_rows(), output_columns_.size());
  }

  const auto& time_list = static_cast<const arrow::ListArray&>(*time_col->chunk(0));
  const auto& edep_list = static_cast<const arrow::ListArray&>(*edep_col->chunk(0));
  const int32_t* time_offsets = time_list.raw_value_offsets();
  const int32_t* edep_offsets = edep_list.raw_value_offsets();

  FloatReader time_reader(time_list.values());
  FloatReader edep_reader(edep_list.values());

  IntegerReader fallback_pdg_reader;
  const int32_t* fallback_pdg_offsets = nullptr;
  if (auto fallback_pdg_col = table.GetColumnByName(fallback_pdg_column_)) {
    const auto& fallback_pdg_list =
        static_cast<const arrow::ListArray&>(*fallback_pdg_col->chunk(0));
    fallback_pdg_offsets = fallback_pdg_list.raw_value_offsets();
    fallback_pdg_reader.Bind(fallback_pdg_list.values());
  }

  const arrow::ListArray* contrib_mc_outer = nullptr;
  const arrow::ListArray* contrib_step_outer = nullptr;
  const arrow::ListArray* contrib_mc_inner = nullptr;
  const arrow::ListArray* contrib_step_inner = nullptr;
  const int32_t* contrib_mc_outer_offsets = nullptr;
  const int32_t* contrib_step_outer_offsets = nullptr;
  const int32_t* contrib_mc_inner_offsets = nullptr;
  const int32_t* contrib_step_inner_offsets = nullptr;
  IntegerReader contrib_mc_values_reader;
  IntegerReader contrib_step_values_reader;
  bool has_contrib = false;
  if (auto contrib_mc_col = table.GetColumnByName(contrib_mc_event_id_column_)) {
    if (auto contrib_step_col = table.GetColumnByName(contrib_step_id_column_)) {
      contrib_mc_outer = &static_cast<const arrow::ListArray&>(*contrib_mc_col->chunk(0));
      contrib_step_outer = &static_cast<const arrow::ListArray&>(*contrib_step_col->chunk(0));
      contrib_mc_inner = &static_cast<const arrow::ListArray&>(*contrib_mc_outer->values());
      contrib_step_inner = &static_cast<const arrow::ListArray&>(*contrib_step_outer->values());
      contrib_mc_outer_offsets = contrib_mc_outer->raw_value_offsets();
      contrib_step_outer_offsets = contrib_step_outer->raw_value_offsets();
      contrib_mc_inner_offsets = contrib_mc_inner->raw_value_offsets();
      contrib_step_inner_offsets = contrib_step_inner->raw_value_offsets();
      contrib_mc_values_reader.Bind(contrib_mc_inner->values());
      contrib_step_values_reader.Bind(contrib_step_inner->values());
      has_contrib = true;
    }
  }

  IntegerReader steps_mc_reader;
  IntegerReader steps_step_reader;
  IntegerReader steps_pdg_reader;
  FloatReader steps_x_reader;
  FloatReader steps_y_reader;
  FloatReader steps_z_reader;
  FloatReader steps_edep_reader;
  FloatReader steps_time_reader;
  const int32_t* steps_offsets = nullptr;
  bool has_steps = false;
  if (auto steps_mc_col = table.GetColumnByName(steps_mc_event_id_column_)) {
    auto steps_step_col = table.GetColumnByName(steps_step_id_column_);
    auto steps_pdg_col = table.GetColumnByName(steps_pdg_id_column_);
    auto steps_x_col = table.GetColumnByName(steps_x_column_);
    auto steps_y_col = table.GetColumnByName(steps_y_column_);
    auto steps_z_col = table.GetColumnByName(steps_z_column_);
    auto steps_edep_col = table.GetColumnByName(steps_edep_column_);
    auto steps_time_col = table.GetColumnByName(steps_time_column_);
    if (steps_step_col && steps_pdg_col && steps_x_col && steps_y_col && steps_z_col &&
        steps_edep_col && steps_time_col) {
      const auto& steps_mc_list = static_cast<const arrow::ListArray&>(*steps_mc_col->chunk(0));
      const auto& steps_step_list =
          static_cast<const arrow::ListArray&>(*steps_step_col->chunk(0));
      const auto& steps_pdg_list =
          static_cast<const arrow::ListArray&>(*steps_pdg_col->chunk(0));
      const auto& steps_x_list = static_cast<const arrow::ListArray&>(*steps_x_col->chunk(0));
      const auto& steps_y_list = static_cast<const arrow::ListArray&>(*steps_y_col->chunk(0));
      const auto& steps_z_list = static_cast<const arrow::ListArray&>(*steps_z_col->chunk(0));
      const auto& steps_edep_list =
          static_cast<const arrow::ListArray&>(*steps_edep_col->chunk(0));
      const auto& steps_time_list =
          static_cast<const arrow::ListArray&>(*steps_time_col->chunk(0));

      steps_offsets = steps_mc_list.raw_value_offsets();
      auto* step_offsets = steps_step_list.raw_value_offsets();
      auto* pdg_offsets = steps_pdg_list.raw_value_offsets();
      auto* x_offsets = steps_x_list.raw_value_offsets();
      auto* y_offsets = steps_y_list.raw_value_offsets();
      auto* z_offsets = steps_z_list.raw_value_offsets();
      auto* e_offsets = steps_edep_list.raw_value_offsets();
      auto* t_offsets = steps_time_list.raw_value_offsets();
      for (int64_t row = 0; row < table.num_rows(); ++row) {
        const int32_t len = steps_offsets[row + 1] - steps_offsets[row];
        if ((step_offsets[row + 1] - step_offsets[row]) != len ||
            (pdg_offsets[row + 1] - pdg_offsets[row]) != len ||
            (x_offsets[row + 1] - x_offsets[row]) != len ||
            (y_offsets[row + 1] - y_offsets[row]) != len ||
            (z_offsets[row + 1] - z_offsets[row]) != len ||
            (e_offsets[row + 1] - e_offsets[row]) != len ||
            (t_offsets[row + 1] - t_offsets[row]) != len) {
          throw std::runtime_error("Step list column lengths do not match.");
        }
      }

      steps_mc_reader.Bind(steps_mc_list.values());
      steps_step_reader.Bind(steps_step_list.values());
      steps_pdg_reader.Bind(steps_pdg_list.values());
      steps_x_reader.Bind(steps_x_list.values());
      steps_y_reader.Bind(steps_y_list.values());
      steps_z_reader.Bind(steps_z_list.values());
      steps_edep_reader.Bind(steps_edep_list.values());
      steps_time_reader.Bind(steps_time_list.values());
      has_steps = true;
    }
  }

  auto pool = arrow::default_memory_pool();

  std::shared_ptr<arrow::Int64Builder> tg_values_builder;
  std::unique_ptr<arrow::ListBuilder> tg_builder;
  if (need_hits_time_group) {
    tg_values_builder = std::make_shared<arrow::Int64Builder>(pool);
    tg_builder = std::make_unique<arrow::ListBuilder>(pool, tg_values_builder);
  }

  std::shared_ptr<arrow::Int32Builder> pdg_values_builder;
  std::unique_ptr<arrow::ListBuilder> pdg_builder;
  if (need_hits_pdg_id) {
    pdg_values_builder = std::make_shared<arrow::Int32Builder>(pool);
    pdg_builder = std::make_unique<arrow::ListBuilder>(pool, pdg_values_builder);
  }

  std::shared_ptr<arrow::Int64Builder> mask_values_builder;
  std::unique_ptr<arrow::ListBuilder> mask_builder;
  if (need_hits_particle_mask) {
    mask_values_builder = std::make_shared<arrow::Int64Builder>(pool);
    mask_builder = std::make_unique<arrow::ListBuilder>(pool, mask_values_builder);
  }

  std::shared_ptr<arrow::Int32Builder> pion_present_builder;
  std::unique_ptr<arrow::ListBuilder> pion_builder;
  std::shared_ptr<arrow::Int32Builder> muon_present_builder;
  std::unique_ptr<arrow::ListBuilder> muon_builder;
  std::shared_ptr<arrow::Int32Builder> mip_present_builder;
  std::unique_ptr<arrow::ListBuilder> mip_builder;
  if (need_group_presence) {
    pion_present_builder = std::make_shared<arrow::Int32Builder>(pool);
    pion_builder = std::make_unique<arrow::ListBuilder>(pool, pion_present_builder);
    muon_present_builder = std::make_shared<arrow::Int32Builder>(pool);
    muon_builder = std::make_unique<arrow::ListBuilder>(pool, muon_present_builder);
    mip_present_builder = std::make_shared<arrow::Int32Builder>(pool);
    mip_builder = std::make_unique<arrow::ListBuilder>(pool, mip_present_builder);
  }

  std::shared_ptr<arrow::DoubleBuilder> start_x_values_builder;
  std::unique_ptr<arrow::ListBuilder> start_x_builder;
  std::shared_ptr<arrow::DoubleBuilder> start_y_values_builder;
  std::unique_ptr<arrow::ListBuilder> start_y_builder;
  std::shared_ptr<arrow::DoubleBuilder> start_z_values_builder;
  std::unique_ptr<arrow::ListBuilder> start_z_builder;
  std::shared_ptr<arrow::DoubleBuilder> end_x_values_builder;
  std::unique_ptr<arrow::ListBuilder> end_x_builder;
  std::shared_ptr<arrow::DoubleBuilder> end_y_values_builder;
  std::unique_ptr<arrow::ListBuilder> end_y_builder;
  std::shared_ptr<arrow::DoubleBuilder> end_z_values_builder;
  std::unique_ptr<arrow::ListBuilder> end_z_builder;
  if (need_endpoints) {
    start_x_values_builder = std::make_shared<arrow::DoubleBuilder>(pool);
    start_x_builder = std::make_unique<arrow::ListBuilder>(pool, start_x_values_builder);
    start_y_values_builder = std::make_shared<arrow::DoubleBuilder>(pool);
    start_y_builder = std::make_unique<arrow::ListBuilder>(pool, start_y_values_builder);
    start_z_values_builder = std::make_shared<arrow::DoubleBuilder>(pool);
    start_z_builder = std::make_unique<arrow::ListBuilder>(pool, start_z_values_builder);
    end_x_values_builder = std::make_shared<arrow::DoubleBuilder>(pool);
    end_x_builder = std::make_unique<arrow::ListBuilder>(pool, end_x_values_builder);
    end_y_values_builder = std::make_shared<arrow::DoubleBuilder>(pool);
    end_y_builder = std::make_unique<arrow::ListBuilder>(pool, end_y_values_builder);
    end_z_values_builder = std::make_shared<arrow::DoubleBuilder>(pool);
    end_z_builder = std::make_unique<arrow::ListBuilder>(pool, end_z_values_builder);
  }

  std::shared_ptr<arrow::DoubleBuilder> arc_values_builder;
  std::unique_ptr<arrow::ListBuilder> arc_builder;
  if (need_arc) {
    arc_values_builder = std::make_shared<arrow::DoubleBuilder>(pool);
    arc_builder = std::make_unique<arrow::ListBuilder>(pool, arc_values_builder);
  }

  std::shared_ptr<arrow::DoubleBuilder> pion_energy_values_builder;
  std::unique_ptr<arrow::ListBuilder> pion_energy_builder;
  std::shared_ptr<arrow::DoubleBuilder> muon_energy_values_builder;
  std::unique_ptr<arrow::ListBuilder> muon_energy_builder;
  std::shared_ptr<arrow::DoubleBuilder> mip_energy_values_builder;
  std::unique_ptr<arrow::ListBuilder> mip_energy_builder;
  if (need_group_energy) {
    pion_energy_values_builder = std::make_shared<arrow::DoubleBuilder>(pool);
    pion_energy_builder = std::make_unique<arrow::ListBuilder>(pool, pion_energy_values_builder);
    muon_energy_values_builder = std::make_shared<arrow::DoubleBuilder>(pool);
    muon_energy_builder = std::make_unique<arrow::ListBuilder>(pool, muon_energy_values_builder);
    mip_energy_values_builder = std::make_shared<arrow::DoubleBuilder>(pool);
    mip_energy_builder = std::make_unique<arrow::ListBuilder>(pool, mip_energy_values_builder);
  }

  struct RowDerived {
    std::vector<int64_t> hit_groups;
    std::vector<int32_t> hit_pdgs;
    std::vector<int64_t> hit_masks;
    std::vector<int32_t> pion_in_group;
    std::vector<int32_t> muon_in_group;
    std::vector<int32_t> mip_in_group;
    std::vector<double> start_x;
    std::vector<double> start_y;
    std::vector<double> start_z;
    std::vector<double> end_x;
    std::vector<double> end_y;
    std::vector<double> end_z;
    std::vector<double> true_arc_length;
    std::vector<double> pion_energy;
    std::vector<double> muon_energy;
    std::vector<double> mip_energy;
  };
  std::vector<RowDerived> per_row(static_cast<size_t>(table.num_rows()));

  utils::parallel::Parallel::For(0, table.num_rows(), [&](int64_t row) {
    RowDerived row_out;
    const int32_t hit_start = time_offsets[row];
    const int32_t hit_end = time_offsets[row + 1];
    const int64_t hit_count = static_cast<int64_t>(hit_end - hit_start);
    if ((edep_offsets[row + 1] - edep_offsets[row]) != hit_count) {
      throw std::runtime_error("hits_time/hits_edep list column lengths do not match.");
    }
    if (fallback_pdg_offsets &&
        (fallback_pdg_offsets[row + 1] - fallback_pdg_offsets[row]) != hit_count) {
      throw std::runtime_error("hits_time/fallback hits_pdg_id list lengths do not match.");
    }
    if (has_contrib) {
      const int64_t mc_hit_count =
          static_cast<int64_t>(contrib_mc_outer_offsets[row + 1] - contrib_mc_outer_offsets[row]);
      const int64_t step_hit_count = static_cast<int64_t>(
          contrib_step_outer_offsets[row + 1] - contrib_step_outer_offsets[row]);
      if (mc_hit_count != hit_count || step_hit_count != hit_count) {
        throw std::runtime_error(
            "hits_time/hits_contrib_mc_event_id/hits_contrib_step_id lengths do not match.");
      }
    }

    row_out.hit_groups.assign(static_cast<size_t>(hit_count), 0);
    int64_t group_count = 0;
    if (hit_count > 0) {
      std::vector<int64_t> order(static_cast<size_t>(hit_count), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return time_reader.Value(hit_start + a) < time_reader.Value(hit_start + b);
      });
      int64_t current_group = 0;
      row_out.hit_groups[static_cast<size_t>(order[0])] = 0;
      for (int64_t i = 1; i < hit_count; ++i) {
        const int64_t prev = order[static_cast<size_t>(i - 1)];
        const int64_t curr = order[static_cast<size_t>(i)];
        const double prev_t = time_reader.Value(hit_start + prev);
        const double curr_t = time_reader.Value(hit_start + curr);
        if (std::abs(curr_t - prev_t) > window_ns_) {
          ++current_group;
        }
        row_out.hit_groups[static_cast<size_t>(curr)] = current_group;
      }
      group_count = current_group + 1;
    }

    std::vector<GroupStats> group_stats(static_cast<size_t>(group_count));
    if (need_endpoints || need_arc) {
      for (auto& g : group_stats) {
        g.seen_all.reserve(64);
        g.seen_non_electron.reserve(64);
      }
    }

    std::unordered_map<StepKey, StepTruth, StepKeyHash> step_truth_by_key;
    if (has_steps) {
      const int32_t step_start = steps_offsets[row];
      const int32_t step_end = steps_offsets[row + 1];
      step_truth_by_key.reserve(static_cast<size_t>(step_end - step_start));
      for (int32_t idx = step_start; idx < step_end; ++idx) {
        StepKey key{steps_mc_reader.Value(idx), steps_step_reader.Value(idx)};
        StepTruth truth;
        truth.pdg = static_cast<int32_t>(steps_pdg_reader.Value(idx));
        truth.x = steps_x_reader.IsValid(idx) ? steps_x_reader.Value(idx) : 0.0;
        truth.y = steps_y_reader.IsValid(idx) ? steps_y_reader.Value(idx) : 0.0;
        truth.z = steps_z_reader.IsValid(idx) ? steps_z_reader.Value(idx) : 0.0;
        truth.time = steps_time_reader.IsValid(idx) ? steps_time_reader.Value(idx) : 0.0;
        truth.edep = steps_edep_reader.IsValid(idx) ? steps_edep_reader.Value(idx) : 0.0;
        step_truth_by_key.emplace(std::move(key), std::move(truth));
      }
    }

    row_out.hit_pdgs.assign(static_cast<size_t>(hit_count), 0);
    row_out.hit_masks.assign(static_cast<size_t>(hit_count), MaskFromPdg(0));
    for (int64_t i = 0; i < hit_count; ++i) {
      const int32_t raw_idx = hit_start + static_cast<int32_t>(i);
      const int64_t group_id = row_out.hit_groups[static_cast<size_t>(i)];
      int32_t selected_pdg = 0;
      std::unordered_map<int32_t, double> pdg_energy_sum;
      if (has_contrib && has_steps) {
        const int64_t outer_idx = static_cast<int64_t>(contrib_mc_outer_offsets[row]) + i;
        const int32_t contrib_start = contrib_mc_inner_offsets[outer_idx];
        const int32_t contrib_end = contrib_mc_inner_offsets[outer_idx + 1];
        const int32_t contrib_step_start = contrib_step_inner_offsets[outer_idx];
        const int32_t contrib_step_end = contrib_step_inner_offsets[outer_idx + 1];
        if ((contrib_end - contrib_start) != (contrib_step_end - contrib_step_start)) {
          throw std::runtime_error(
              "hits_contrib_mc_event_id/hits_contrib_step_id inner list lengths do not match.");
        }
        for (int32_t c = 0; c < (contrib_end - contrib_start); ++c) {
          const StepKey key{
              contrib_mc_values_reader.Value(contrib_start + c),
              contrib_step_values_reader.Value(contrib_step_start + c),
          };
          auto it = step_truth_by_key.find(key);
          if (it == step_truth_by_key.end()) {
            continue;
          }
          const StepTruth& truth = it->second;
          pdg_energy_sum[truth.pdg] += std::max(0.0, truth.edep);
          if ((need_endpoints || need_arc) && group_id >= 0 && group_id < group_count) {
            auto& g = group_stats[static_cast<size_t>(group_id)];
            if (g.seen_all.insert(key).second) {
              g.all_points.push_back(truth);
            }
            if (truth.pdg != 11 && truth.pdg != -11 && g.seen_non_electron.insert(key).second) {
              g.non_electron_points.push_back(truth);
            }
          }
        }
      }
      if (!pdg_energy_sum.empty()) {
        int32_t best_pdg = 0;
        double best_energy = -1.0;
        for (const auto& kv : pdg_energy_sum) {
          if (kv.second > best_energy ||
              (std::abs(kv.second - best_energy) < 1e-12 && kv.first < best_pdg)) {
            best_pdg = kv.first;
            best_energy = kv.second;
          }
        }
        selected_pdg = best_pdg;
      } else if (fallback_pdg_reader.IsBound()) {
        selected_pdg = static_cast<int32_t>(fallback_pdg_reader.Value(raw_idx));
      }
      row_out.hit_pdgs[static_cast<size_t>(i)] = selected_pdg;
      row_out.hit_masks[static_cast<size_t>(i)] = MaskFromPdg(selected_pdg);
      if (need_any_group_stats && group_id >= 0 && group_id < group_count) {
        auto& g = group_stats[static_cast<size_t>(group_id)];
        const int cls = ClassFromPdg(selected_pdg);
        const double hit_edep = edep_reader.IsValid(raw_idx) ? edep_reader.Value(raw_idx) : 0.0;
        if (cls == 0) {
          g.has_pion = 1;
          g.pion_energy += hit_edep;
        } else if (cls == 1) {
          g.has_muon = 1;
          g.muon_energy += hit_edep;
        } else if (cls == 2) {
          g.has_mip = 1;
          g.mip_energy += hit_edep;
        }
      }
    }

    if (need_group_presence) {
      row_out.pion_in_group.assign(static_cast<size_t>(group_count), 0);
      row_out.muon_in_group.assign(static_cast<size_t>(group_count), 0);
      row_out.mip_in_group.assign(static_cast<size_t>(group_count), 0);
    }
    if (need_endpoints) {
      row_out.start_x.assign(static_cast<size_t>(group_count), 0.0);
      row_out.start_y.assign(static_cast<size_t>(group_count), 0.0);
      row_out.start_z.assign(static_cast<size_t>(group_count), 0.0);
      row_out.end_x.assign(static_cast<size_t>(group_count), 0.0);
      row_out.end_y.assign(static_cast<size_t>(group_count), 0.0);
      row_out.end_z.assign(static_cast<size_t>(group_count), 0.0);
    }
    if (need_arc) {
      row_out.true_arc_length.assign(static_cast<size_t>(group_count), 0.0);
    }
    if (need_group_energy) {
      row_out.pion_energy.assign(static_cast<size_t>(group_count), 0.0);
      row_out.muon_energy.assign(static_cast<size_t>(group_count), 0.0);
      row_out.mip_energy.assign(static_cast<size_t>(group_count), 0.0);
    }

    for (int64_t gidx = 0; gidx < group_count; ++gidx) {
      auto& g = group_stats[static_cast<size_t>(gidx)];
      if (need_group_presence) {
        row_out.pion_in_group[static_cast<size_t>(gidx)] = g.has_pion;
        row_out.muon_in_group[static_cast<size_t>(gidx)] = g.has_muon;
        row_out.mip_in_group[static_cast<size_t>(gidx)] = g.has_mip;
      }
      if (need_group_energy) {
        row_out.pion_energy[static_cast<size_t>(gidx)] = g.pion_energy;
        row_out.muon_energy[static_cast<size_t>(gidx)] = g.muon_energy;
        row_out.mip_energy[static_cast<size_t>(gidx)] = g.mip_energy;
      }
      if (!(need_endpoints || need_arc)) {
        continue;
      }
      const auto& points = g.non_electron_points.empty() ? g.all_points : g.non_electron_points;
      if (points.empty()) {
        continue;
      }
      std::vector<StepTruth> sorted_points = points;
      std::sort(sorted_points.begin(),
                sorted_points.end(),
                [](const StepTruth& a, const StepTruth& b) { return a.time < b.time; });
      if (need_endpoints) {
        row_out.start_x[static_cast<size_t>(gidx)] = sorted_points.front().x;
        row_out.start_y[static_cast<size_t>(gidx)] = sorted_points.front().y;
        row_out.start_z[static_cast<size_t>(gidx)] = sorted_points.front().z;
        row_out.end_x[static_cast<size_t>(gidx)] = sorted_points.back().x;
        row_out.end_y[static_cast<size_t>(gidx)] = sorted_points.back().y;
        row_out.end_z[static_cast<size_t>(gidx)] = sorted_points.back().z;
      }
      if (need_arc) {
        double arc = 0.0;
        for (size_t i = 1; i < sorted_points.size(); ++i) {
          const double dx = sorted_points[i].x - sorted_points[i - 1].x;
          const double dy = sorted_points[i].y - sorted_points[i - 1].y;
          const double dz = sorted_points[i].z - sorted_points[i - 1].z;
          arc += std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        row_out.true_arc_length[static_cast<size_t>(gidx)] = arc;
      }
    }
    per_row[static_cast<size_t>(row)] = std::move(row_out);
  });

  for (int64_t row = 0; row < table.num_rows(); ++row) {
    const auto& r = per_row[static_cast<size_t>(row)];
    if (need_hits_time_group) {
      AppendListValues(
          tg_builder.get(), tg_values_builder.get(), r.hit_groups, "time_group_summary/hits_time_group");
    }
    if (need_hits_pdg_id) {
      AppendListValues(
          pdg_builder.get(), pdg_values_builder.get(), r.hit_pdgs, "time_group_summary/hits_pdg_id");
    }
    if (need_hits_particle_mask) {
      AppendListValues(mask_builder.get(),
                       mask_values_builder.get(),
                       r.hit_masks,
                       "time_group_summary/hits_particle_mask");
    }
    if (need_group_presence) {
      AppendListValues(pion_builder.get(),
                       pion_present_builder.get(),
                       r.pion_in_group,
                       "time_group_summary/pion_in_group");
      AppendListValues(muon_builder.get(),
                       muon_present_builder.get(),
                       r.muon_in_group,
                       "time_group_summary/muon_in_group");
      AppendListValues(mip_builder.get(),
                       mip_present_builder.get(),
                       r.mip_in_group,
                       "time_group_summary/mip_in_group");
    }
    if (need_endpoints) {
      AppendListValues(start_x_builder.get(),
                       start_x_values_builder.get(),
                       r.start_x,
                       "time_group_summary/group_start_x");
      AppendListValues(start_y_builder.get(),
                       start_y_values_builder.get(),
                       r.start_y,
                       "time_group_summary/group_start_y");
      AppendListValues(start_z_builder.get(),
                       start_z_values_builder.get(),
                       r.start_z,
                       "time_group_summary/group_start_z");
      AppendListValues(
          end_x_builder.get(), end_x_values_builder.get(), r.end_x, "time_group_summary/group_end_x");
      AppendListValues(
          end_y_builder.get(), end_y_values_builder.get(), r.end_y, "time_group_summary/group_end_y");
      AppendListValues(
          end_z_builder.get(), end_z_values_builder.get(), r.end_z, "time_group_summary/group_end_z");
    }
    if (need_arc) {
      AppendListValues(arc_builder.get(),
                       arc_values_builder.get(),
                       r.true_arc_length,
                       "time_group_summary/group_true_arc");
    }
    if (need_group_energy) {
      AppendListValues(pion_energy_builder.get(),
                       pion_energy_values_builder.get(),
                       r.pion_energy,
                       "time_group_summary/pion_energy");
      AppendListValues(muon_energy_builder.get(),
                       muon_energy_values_builder.get(),
                       r.muon_energy,
                       "time_group_summary/muon_energy");
      AppendListValues(mip_energy_builder.get(),
                       mip_energy_values_builder.get(),
                       r.mip_energy,
                       "time_group_summary/mip_energy");
    }
  }

  std::unordered_map<std::string, std::shared_ptr<arrow::Array>> finished_arrays;
  finished_arrays.reserve(output_columns_.size());
  std::shared_ptr<arrow::Array> arr;
  auto finish_and_store = [&](const std::string& name, arrow::ArrayBuilder* builder) {
    if (!builder) {
      return;
    }
    auto st = builder->Finish(&arr);
    if (!st.ok()) {
      throw std::runtime_error("Failed to finish " + name + ": " + st.ToString());
    }
    finished_arrays[name] = arr;
  };
  finish_and_store("hits_time_group", tg_builder.get());
  finish_and_store("hits_pdg_id", pdg_builder.get());
  finish_and_store("hits_particle_mask", mask_builder.get());
  finish_and_store("pion_in_group", pion_builder.get());
  finish_and_store("muon_in_group", muon_builder.get());
  finish_and_store("mip_in_group", mip_builder.get());
  finish_and_store("group_start_x", start_x_builder.get());
  finish_and_store("group_start_y", start_y_builder.get());
  finish_and_store("group_start_z", start_z_builder.get());
  finish_and_store("group_end_x", end_x_builder.get());
  finish_and_store("group_end_y", end_y_builder.get());
  finish_and_store("group_end_z", end_z_builder.get());
  finish_and_store("group_true_arc_length", arc_builder.get());
  finish_and_store("pion_energy_per_group", pion_energy_builder.get());
  finish_and_store("muon_energy_per_group", muon_energy_builder.get());
  finish_and_store("mip_energy_per_group", mip_energy_builder.get());

  std::vector<std::shared_ptr<arrow::Array>> out;
  out.reserve(output_columns_.size());
  for (const auto& name : output_columns_) {
    auto it = finished_arrays.find(name);
    if (it == finished_arrays.end()) {
      throw std::runtime_error("Unsupported requested output column: " + name);
    }
    out.push_back(it->second);
  }
  return out;
}

}  // namespace pioneerml::data_derivers
