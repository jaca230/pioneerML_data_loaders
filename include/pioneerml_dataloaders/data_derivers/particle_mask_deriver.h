#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pioneerml_dataloaders/data_derivers/base_deriver.h"

namespace pioneerml::data_derivers {

// Convert PDG IDs to compact bitmasks for fast downstream use.
class ParticleMaskDeriver : public BaseDeriver {
 public:
  enum Mask : int64_t {
    kPion = 0b00001,
    kMuon = 0b00010,
    kPositron = 0b00100,
    kElectron = 0b01000,
    kOther = 0b10000
  };

  explicit ParticleMaskDeriver(std::string pdg_column = "hits_pdg_id")
      : pdg_column_(std::move(pdg_column)) {}

  std::shared_ptr<arrow::Array> DeriveColumn(const arrow::Table& table) const override;
  int64_t ComputeSingle(int pdg_id) const;

 private:
  std::string pdg_column_;
};

}  // namespace pioneerml::data_derivers
