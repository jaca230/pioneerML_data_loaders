#include "pioneerml_dataloaders/configurable/data_derivers/particle_mask_deriver.h"

#include <arrow/api.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include "pioneerml_dataloaders/utils/parallel/parallel.h"

namespace pioneerml::data_derivers {

void ParticleMaskDeriver::LoadConfig(const nlohmann::json& cfg) {
  if (cfg.contains("pdg_column")) {
    pdg_column_ = cfg.at("pdg_column").get<std::string>();
  }
}

std::vector<std::shared_ptr<arrow::Array>> ParticleMaskDeriver::DeriveColumns(
    const arrow::Table& table) const {
  auto col = table.GetColumnByName(pdg_column_);
  if (!col) {
    return {std::make_shared<arrow::NullArray>(table.num_rows())};
  }
  const auto& list = static_cast<const arrow::ListArray&>(*col->chunk(0));

  arrow::ListBuilder list_builder(arrow::default_memory_pool(), std::make_shared<arrow::Int64Builder>(arrow::default_memory_pool()));
  auto* int_builder = static_cast<arrow::Int64Builder*>(list_builder.value_builder());

  int64_t rows = table.num_rows();
  auto offsets = list.raw_value_offsets();
  auto values = std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(list.values());
  const int32_t* raw = values->raw_values();

  std::vector<std::vector<int64_t>> masks(rows);
  utils::parallel::Parallel::For(0, rows, [&](int64_t row) {
    auto start = offsets[row];
    auto end = offsets[row + 1];
    std::vector<int64_t> row_masks;
    row_masks.reserve(end - start);
    for (int32_t i = start; i < end; ++i) {
      row_masks.push_back(ComputeSingle(raw[i]));
    }
    masks[row] = std::move(row_masks);
  });

  for (int64_t row = 0; row < rows; ++row) {
    auto st = list_builder.Append();
    if (!st.ok()) {
      throw std::runtime_error(st.ToString());
    }
    for (auto mask : masks[row]) {
      st = int_builder->Append(mask);
      if (!st.ok()) {
        throw std::runtime_error(st.ToString());
      }
    }
  }

  std::shared_ptr<arrow::Array> out;
  auto st_finish = list_builder.Finish(&out);
  if (!st_finish.ok()) {
    throw std::runtime_error(st_finish.ToString());
  }
  return {std::move(out)};
}

int64_t ParticleMaskDeriver::ComputeSingle(int pdg_id) const {
  switch (pdg_id) {
    case 211:
      return kPion;
    case -13:
      return kMuon;
    case -11:
      return kPositron;
    case 11:
      return kElectron;
    default:
      return kOther;
  }
}

}  // namespace pioneerml::data_derivers
