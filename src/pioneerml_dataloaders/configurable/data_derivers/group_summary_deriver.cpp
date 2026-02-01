#include "pioneerml_dataloaders/configurable/data_derivers/group_summary_deriver.h"

#include <arrow/api.h>

#include <stdexcept>

#include "pioneerml_dataloaders/utils/parallel/parallel.h"

namespace pioneerml::data_derivers {

void GroupSummaryDeriver::LoadConfig(const nlohmann::json& cfg) {
  if (cfg.contains("pdg_column")) {
    pdg_column_ = cfg.at("pdg_column").get<std::string>();
  }
  if (cfg.contains("edep_column")) {
    edep_column_ = cfg.at("edep_column").get<std::string>();
  }
  if (cfg.contains("time_group_column")) {
    time_group_column_ = cfg.at("time_group_column").get<std::string>();
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

std::vector<std::shared_ptr<arrow::Array>> GroupSummaryDeriver::DeriveColumns(
    const arrow::Table& table) const {
  auto pdg_col = table.GetColumnByName(pdg_column_);
  auto edep_col = table.GetColumnByName(edep_column_);
  auto tg_col = table.GetColumnByName(time_group_column_);
  if (!pdg_col || !edep_col || !tg_col) {
    std::vector<std::shared_ptr<arrow::Array>> empty;
    empty.reserve(6);
    for (int i = 0; i < 6; ++i) {
      empty.push_back(std::make_shared<arrow::NullArray>(table.num_rows()));
    }
    return empty;
  }

  const auto& pdg_list = static_cast<const arrow::ListArray&>(*pdg_col->chunk(0));
  const auto& edep_list = static_cast<const arrow::ListArray&>(*edep_col->chunk(0));
  const auto& tg_list = static_cast<const arrow::ListArray&>(*tg_col->chunk(0));

  auto pdg_values = std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(pdg_list.values());
  auto edep_values = std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(edep_list.values());
  auto tg_values = std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(tg_list.values());

  const int32_t* pdg_raw = pdg_values->raw_values();
  const double* edep_raw = edep_values->raw_values();
  const int64_t* tg_raw = tg_values->raw_values();

  auto pdg_offsets = pdg_list.raw_value_offsets();
  auto edep_offsets = edep_list.raw_value_offsets();
  auto tg_offsets = tg_list.raw_value_offsets();

  struct SummaryRow {
    std::vector<int32_t> has_pion;
    std::vector<int32_t> has_muon;
    std::vector<int32_t> has_mip;
    std::vector<double> sum_pion;
    std::vector<double> sum_muon;
    std::vector<double> sum_mip;
  };
  std::vector<SummaryRow> summaries(table.num_rows());

  utils::parallel::Parallel::For(0, table.num_rows(), [&](int64_t row) {
    if (pdg_offsets[row + 1] != edep_offsets[row + 1] ||
        pdg_offsets[row + 1] != tg_offsets[row + 1]) {
      throw std::runtime_error("List column lengths do not match for PDG/edep/time_group");
    }
    auto start = pdg_offsets[row];
    auto end = pdg_offsets[row + 1];
    SummaryRow summary;

    if (end > start) {
      int64_t max_group = -1;
      for (int32_t i = start; i < end; ++i) {
        max_group = std::max(max_group, tg_raw[i]);
      }
      const int64_t groups = max_group + 1;
      summary.has_pion.assign(groups, 0);
      summary.has_muon.assign(groups, 0);
      summary.has_mip.assign(groups, 0);
      summary.sum_pion.assign(groups, 0.0);
      summary.sum_muon.assign(groups, 0.0);
      summary.sum_mip.assign(groups, 0.0);
    }

    for (int32_t i = start; i < end; ++i) {
      int cls = ClassFromPdg(pdg_raw[i]);
      if (cls < 0) {
        continue;
      }
      const int64_t group = tg_raw[i];
      switch (static_cast<ClassIndex>(cls)) {
        case ClassIndex::kPion:
          summary.has_pion[group] = 1;
          summary.sum_pion[group] += edep_raw[i];
          break;
        case ClassIndex::kMuon:
          summary.has_muon[group] = 1;
          summary.sum_muon[group] += edep_raw[i];
          break;
        case ClassIndex::kMip:
          summary.has_mip[group] = 1;
          summary.sum_mip[group] += edep_raw[i];
          break;
      }
    }
    summaries[row] = summary;
  });

  arrow::ListBuilder pion_present(arrow::default_memory_pool(),
                                  std::make_shared<arrow::Int32Builder>());
  arrow::ListBuilder muon_present(arrow::default_memory_pool(),
                                  std::make_shared<arrow::Int32Builder>());
  arrow::ListBuilder mip_present(arrow::default_memory_pool(),
                                 std::make_shared<arrow::Int32Builder>());
  arrow::ListBuilder pion_energy(arrow::default_memory_pool(),
                                 std::make_shared<arrow::DoubleBuilder>());
  arrow::ListBuilder muon_energy(arrow::default_memory_pool(),
                                 std::make_shared<arrow::DoubleBuilder>());
  arrow::ListBuilder mip_energy(arrow::default_memory_pool(),
                                std::make_shared<arrow::DoubleBuilder>());

  auto* pion_present_values =
      static_cast<arrow::Int32Builder*>(pion_present.value_builder());
  auto* muon_present_values =
      static_cast<arrow::Int32Builder*>(muon_present.value_builder());
  auto* mip_present_values =
      static_cast<arrow::Int32Builder*>(mip_present.value_builder());
  auto* pion_energy_values =
      static_cast<arrow::DoubleBuilder*>(pion_energy.value_builder());
  auto* muon_energy_values =
      static_cast<arrow::DoubleBuilder*>(muon_energy.value_builder());
  auto* mip_energy_values =
      static_cast<arrow::DoubleBuilder*>(mip_energy.value_builder());

  for (int64_t row = 0; row < table.num_rows(); ++row) {
    const auto& summary = summaries[row];
    if (!pion_present.Append().ok() || !muon_present.Append().ok() ||
        !mip_present.Append().ok() || !pion_energy.Append().ok() ||
        !muon_energy.Append().ok() || !mip_energy.Append().ok()) {
      throw std::runtime_error("Failed to append group summary list.");
    }

    for (size_t i = 0; i < summary.has_pion.size(); ++i) {
      if (!pion_present_values->Append(summary.has_pion[i]).ok() ||
          !muon_present_values->Append(summary.has_muon[i]).ok() ||
          !mip_present_values->Append(summary.has_mip[i]).ok() ||
          !pion_energy_values->Append(summary.sum_pion[i]).ok() ||
          !muon_energy_values->Append(summary.sum_muon[i]).ok() ||
          !mip_energy_values->Append(summary.sum_mip[i]).ok()) {
        throw std::runtime_error("Failed to append group summary values.");
      }
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
