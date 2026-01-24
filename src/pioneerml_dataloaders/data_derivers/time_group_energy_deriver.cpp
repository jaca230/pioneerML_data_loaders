#include "pioneerml_dataloaders/data_derivers/time_group_energy_deriver.h"

#include <arrow/api.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace pioneerml::data_derivers {

namespace {

int ClassIndex(int pdg) {
  if (pdg == 211) return 0;        // pion
  if (pdg == -13) return 1;        // muon
  if (pdg == -11 || pdg == 11) return 2;  // MIP (positron/electron)
  return -1;
}

}  // namespace

std::shared_ptr<arrow::Array> TimeGroupEnergyDeriver::DeriveColumn(const arrow::Table& table) const {
  auto pdg_col = table.GetColumnByName(pdg_column_);
  auto edep_col = table.GetColumnByName(edep_column_);
  auto tg_col = table.GetColumnByName(time_group_column_);
  if (!pdg_col || !edep_col || !tg_col) {
    return std::make_shared<arrow::NullArray>(table.num_rows());
  }

  const auto& pdg_list = static_cast<const arrow::ListArray&>(*pdg_col->chunk(0));
  const auto& edep_list = static_cast<const arrow::ListArray&>(*edep_col->chunk(0));
  const auto& tg_list = static_cast<const arrow::ListArray&>(*tg_col->chunk(0));

  auto pdg_values = std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(pdg_list.values());
  auto edep_values = std::static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(edep_list.values());
  auto tg_values = std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(tg_list.values());

  const int32_t* pdg_raw = pdg_values->raw_values();
  const double* edep_raw = edep_values->raw_values();
  const int32_t* tg_raw = tg_values->raw_values();

  auto pool = arrow::default_memory_pool();
  auto pion_vals = std::make_shared<arrow::DoubleBuilder>(pool);
  auto mu_vals = std::make_shared<arrow::DoubleBuilder>(pool);
  auto mip_vals = std::make_shared<arrow::DoubleBuilder>(pool);

  auto pion_list = std::make_shared<arrow::ListBuilder>(pool, pion_vals);
  auto mu_list = std::make_shared<arrow::ListBuilder>(pool, mu_vals);
  auto mip_list = std::make_shared<arrow::ListBuilder>(pool, mip_vals);

  auto struct_type = arrow::struct_({
      arrow::field("pion_energy_per_group", arrow::list(arrow::float64())),
      arrow::field("muon_energy_per_group", arrow::list(arrow::float64())),
      arrow::field("mip_energy_per_group", arrow::list(arrow::float64())),
  });

  arrow::StructBuilder struct_builder(
      struct_type, pool,
      std::vector<std::shared_ptr<arrow::ArrayBuilder>>{pion_list, mu_list, mip_list});

  int64_t rows = table.num_rows();
  auto pdg_offsets = pdg_list.raw_value_offsets();
  auto edep_offsets = edep_list.raw_value_offsets();
  auto tg_offsets = tg_list.raw_value_offsets();

  for (int64_t row = 0; row < rows; ++row) {
    // Defensive: ensure offsets align
    if (pdg_offsets[row + 1] != edep_offsets[row + 1] || pdg_offsets[row + 1] != tg_offsets[row + 1]) {
      throw std::runtime_error("List column lengths do not match for PDG/edep/time_group");
    }
    auto start = pdg_offsets[row];
    auto end = pdg_offsets[row + 1];

    std::unordered_map<int32_t, std::array<double, 3>> accum;
    accum.reserve(static_cast<size_t>(end - start));

    for (int32_t i = start; i < end; ++i) {
      int cls = ClassIndex(pdg_raw[i]);
      if (cls < 0) continue;
      auto tg = tg_raw[i];
      auto& entry = accum[tg];
      entry[cls] += edep_raw[i];
    }

    // Append lists in ascending time-group order so per-class lists stay aligned.
    std::vector<int32_t> keys;
    keys.reserve(accum.size());
    for (const auto& kv : accum) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());

    auto st_struct = struct_builder.Append();
    if (!st_struct.ok()) throw std::runtime_error(st_struct.ToString());

    auto append_list = [&](arrow::ListBuilder& lb, arrow::DoubleBuilder& vb, int cls) {
      auto st_list = lb.Append();
      if (!st_list.ok()) throw std::runtime_error(st_list.ToString());
      for (auto tg : keys) {
        const auto& sums = accum[tg];
        auto st_val = vb.Append(sums[cls]);
        if (!st_val.ok()) throw std::runtime_error(st_val.ToString());
      }
    };

    append_list(*pion_list, *pion_vals, 0);
    append_list(*mu_list, *mu_vals, 1);
    append_list(*mip_list, *mip_vals, 2);
  }

  std::shared_ptr<arrow::Array> out;
  auto st_finish = struct_builder.Finish(&out);
  if (!st_finish.ok()) throw std::runtime_error(st_finish.ToString());
  return out;
}

}  // namespace pioneerml::data_derivers
