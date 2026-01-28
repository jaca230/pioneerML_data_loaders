#include "pioneerml_dataloaders/configurable/data_derivers/group_summary_deriver.h"

#include <arrow/api.h>

#include <stdexcept>

namespace pioneerml::data_derivers {

void GroupSummaryDeriver::LoadConfig(const nlohmann::json& cfg) {
  if (cfg.contains("pdg_column")) {
    pdg_column_ = cfg.at("pdg_column").get<std::string>();
  }
  if (cfg.contains("edep_column")) {
    edep_column_ = cfg.at("edep_column").get<std::string>();
  }
}

namespace {

enum class ClassIndex {
  kPion = 0,
  kMuon = 1,
  kMip = 2,
};

int ClassFromPdg(int pdg) {
  if (pdg == 211) return static_cast<int>(ClassIndex::kPion);
  if (pdg == -13) return static_cast<int>(ClassIndex::kMuon);
  if (pdg == -11 || pdg == 11) return static_cast<int>(ClassIndex::kMip);
  return -1;
}

}  // namespace

std::shared_ptr<arrow::Array> GroupSummaryDeriver::DeriveColumn(
    const arrow::Table& table) const {
  auto arrays = DeriveColumns(table);
  if (arrays.empty()) {
    return std::make_shared<arrow::NullArray>(table.num_rows());
  }
  return arrays.front();
}

std::vector<std::shared_ptr<arrow::Array>> GroupSummaryDeriver::DeriveColumns(
    const arrow::Table& table) const {
  auto pdg_col = table.GetColumnByName(pdg_column_);
  auto edep_col = table.GetColumnByName(edep_column_);
  if (!pdg_col || !edep_col) {
    std::vector<std::shared_ptr<arrow::Array>> empty;
    empty.reserve(6);
    for (int i = 0; i < 6; ++i) {
      empty.push_back(std::make_shared<arrow::NullArray>(table.num_rows()));
    }
    return empty;
  }

  const auto& pdg_list = static_cast<const arrow::ListArray&>(*pdg_col->chunk(0));
  const auto& edep_list = static_cast<const arrow::ListArray&>(*edep_col->chunk(0));

  auto pdg_values = std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(pdg_list.values());
  auto edep_values = std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(edep_list.values());

  const int32_t* pdg_raw = pdg_values->raw_values();
  const double* edep_raw = edep_values->raw_values();

  arrow::Int32Builder pion_present(arrow::default_memory_pool());
  arrow::Int32Builder muon_present(arrow::default_memory_pool());
  arrow::Int32Builder mip_present(arrow::default_memory_pool());
  arrow::DoubleBuilder pion_energy(arrow::default_memory_pool());
  arrow::DoubleBuilder muon_energy(arrow::default_memory_pool());
  arrow::DoubleBuilder mip_energy(arrow::default_memory_pool());

  auto pdg_offsets = pdg_list.raw_value_offsets();
  auto edep_offsets = edep_list.raw_value_offsets();

  for (int64_t row = 0; row < table.num_rows(); ++row) {
    if (pdg_offsets[row + 1] != edep_offsets[row + 1]) {
      throw std::runtime_error("List column lengths do not match for PDG/edep");
    }
    auto start = pdg_offsets[row];
    auto end = pdg_offsets[row + 1];

    int32_t has_pion = 0;
    int32_t has_muon = 0;
    int32_t has_mip = 0;
    double sum_pion = 0.0;
    double sum_muon = 0.0;
    double sum_mip = 0.0;

    for (int32_t i = start; i < end; ++i) {
      int cls = ClassFromPdg(pdg_raw[i]);
      if (cls < 0) {
        continue;
      }
      switch (static_cast<ClassIndex>(cls)) {
        case ClassIndex::kPion:
          has_pion = 1;
          sum_pion += edep_raw[i];
          break;
        case ClassIndex::kMuon:
          has_muon = 1;
          sum_muon += edep_raw[i];
          break;
        case ClassIndex::kMip:
          has_mip = 1;
          sum_mip += edep_raw[i];
          break;
      }
    }

    auto st0 = pion_present.Append(has_pion);
    auto st1 = muon_present.Append(has_muon);
    auto st2 = mip_present.Append(has_mip);
    auto st3 = pion_energy.Append(sum_pion);
    auto st4 = muon_energy.Append(sum_muon);
    auto st5 = mip_energy.Append(sum_mip);
    if (!st0.ok() || !st1.ok() || !st2.ok() || !st3.ok() || !st4.ok() || !st5.ok()) {
      throw std::runtime_error("Failed to append group summary values.");
    }
  }

  std::vector<std::shared_ptr<arrow::Array>> out;
  out.reserve(6);

  std::shared_ptr<arrow::Array> arr;
  if (!pion_present.Finish(&arr).ok()) throw std::runtime_error("Failed to finish pion_present");
  out.push_back(arr);
  if (!muon_present.Finish(&arr).ok()) throw std::runtime_error("Failed to finish muon_present");
  out.push_back(arr);
  if (!mip_present.Finish(&arr).ok()) throw std::runtime_error("Failed to finish mip_present");
  out.push_back(arr);
  if (!pion_energy.Finish(&arr).ok()) throw std::runtime_error("Failed to finish pion_energy");
  out.push_back(arr);
  if (!muon_energy.Finish(&arr).ok()) throw std::runtime_error("Failed to finish muon_energy");
  out.push_back(arr);
  if (!mip_energy.Finish(&arr).ok()) throw std::runtime_error("Failed to finish mip_energy");
  out.push_back(arr);

  return out;
}

}  // namespace pioneerml::data_derivers
