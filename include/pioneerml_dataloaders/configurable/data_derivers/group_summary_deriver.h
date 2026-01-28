#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pioneerml_dataloaders/configurable/data_derivers/base_deriver.h"

namespace pioneerml::data_derivers {

class GroupSummaryDeriver : public BaseDeriver, public MultiDeriver {
 public:
  explicit GroupSummaryDeriver(std::string pdg_column = "hits_pdg_id",
                               std::string edep_column = "hits_edep")
      : pdg_column_(std::move(pdg_column)), edep_column_(std::move(edep_column)) {}

  void LoadConfig(const nlohmann::json& cfg) override;

  std::shared_ptr<arrow::Array> DeriveColumn(const arrow::Table& table) const override;

  std::vector<std::shared_ptr<arrow::Array>> DeriveColumns(
      const arrow::Table& table) const override;

 private:
  std::string pdg_column_;
  std::string edep_column_;
};

}  // namespace pioneerml::data_derivers
