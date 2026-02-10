#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pioneerml_dataloaders/configurable/data_derivers/base_deriver.h"

namespace pioneerml::data_derivers {

// Single-pass deriver for hit-level and time-group-level truth summaries.
//
// Derived columns (in order):
// 1)  hits_time_group           : list<int64>
// 2)  hits_pdg_id               : list<int32>
// 3)  hits_particle_mask        : list<int64>
// 4)  pion_in_group             : list<int32>
// 5)  muon_in_group             : list<int32>
// 6)  mip_in_group              : list<int32>
// 7)  group_start_x             : list<float64>
// 8)  group_start_y             : list<float64>
// 9)  group_start_z             : list<float64>
// 10) group_end_x               : list<float64>
// 11) group_end_y               : list<float64>
// 12) group_end_z               : list<float64>
// 13) group_true_arc_length     : list<float64>
// 14) pion_energy_per_group     : list<float64>
// 15) muon_energy_per_group     : list<float64>
// 16) mip_energy_per_group      : list<float64>
class TimeGroupSummaryDeriver : public BaseDeriver {
 public:
  explicit TimeGroupSummaryDeriver(
      double window_ns = 1.0,
      std::vector<std::string> output_columns = {
          "hits_time_group",
          "hits_pdg_id",
          "hits_particle_mask",
          "pion_in_group",
          "muon_in_group",
          "mip_in_group",
          "group_start_x",
          "group_start_y",
          "group_start_z",
          "group_end_x",
          "group_end_y",
          "group_end_z",
          "group_true_arc_length",
          "pion_energy_per_group",
          "muon_energy_per_group",
          "mip_energy_per_group",
      })
      : window_ns_(window_ns), output_columns_(std::move(output_columns)) {}

  void LoadConfig(const nlohmann::json& cfg) override;

  std::vector<std::shared_ptr<arrow::Array>> DeriveColumns(
      const arrow::Table& table) const override;

 private:
  double window_ns_{1.0};
  std::vector<std::string> output_columns_;
  std::string time_column_{"hits_time"};
  std::string edep_column_{"hits_edep"};
  std::string fallback_pdg_column_{"hits_pdg_id"};

  std::string contrib_mc_event_id_column_{"hits_contrib_mc_event_id"};
  std::string contrib_step_id_column_{"hits_contrib_step_id"};

  std::string steps_mc_event_id_column_{"steps_mc_event_id"};
  std::string steps_step_id_column_{"steps_step_id"};
  std::string steps_pdg_id_column_{"steps_pdg_id"};
  std::string steps_x_column_{"steps_x"};
  std::string steps_y_column_{"steps_y"};
  std::string steps_z_column_{"steps_z"};
  std::string steps_edep_column_{"steps_edep"};
  std::string steps_time_column_{"steps_time"};
};

}  // namespace pioneerml::data_derivers
