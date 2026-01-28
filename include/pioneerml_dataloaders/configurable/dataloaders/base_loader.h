#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <arrow/api.h>

#include "pioneerml_dataloaders/batch/base_batch.h"
#include "pioneerml_dataloaders/configurable/data_derivers/base_deriver.h"

namespace pioneerml::dataloaders {

struct TrainingBundle {
  std::unique_ptr<BaseBatch> inputs;
  std::unique_ptr<BaseBatch> targets;
};

struct InferenceBundle {
  std::unique_ptr<BaseBatch> inputs;
};

// Contract for loading parquet shards into a model-ready batch abstraction.
class DataLoader : public pioneerml::configurable::Configurable {
 public:
  virtual ~DataLoader() = default;

  // Load inputs + targets for training.
  virtual TrainingBundle LoadTraining(const std::string& parquet_path) const = 0;
  virtual TrainingBundle LoadTraining(
      const std::vector<std::string>& parquet_paths) const;

  // Load inputs only for inference.
  virtual InferenceBundle LoadInference(const std::string& parquet_path) const = 0;
  virtual InferenceBundle LoadInference(
      const std::vector<std::string>& parquet_paths) const;

 protected:
  // Optional helper for derived loaders that need the raw table.
  virtual std::shared_ptr<arrow::Table> LoadTable(const std::string& parquet_path) const = 0;

  struct DeriverSpec {
    std::vector<std::string> names;
    std::shared_ptr<data_derivers::BaseDeriver> deriver;
  };

  std::vector<DeriverSpec> derivers_;

  void AddDeriver(std::string name, std::shared_ptr<data_derivers::BaseDeriver> deriver);
  void AddDeriver(std::vector<std::string> names,
                  std::shared_ptr<data_derivers::BaseDeriver> deriver);

  virtual std::shared_ptr<arrow::Table> AddDerivedColumns(
      const std::shared_ptr<arrow::Table>& table) const;

  std::shared_ptr<arrow::Table> LoadAndConcatenateTables(
      const std::vector<std::string>& parquet_paths,
      bool add_derived) const;

  std::vector<std::string> input_columns_;
  std::vector<std::string> target_columns_;

  std::string JoinNames(const std::vector<std::string>& names,
                        const std::string& sep = ", ") const;

  std::vector<std::string> MissingColumns(const arrow::Table& table,
                                          const std::vector<std::string>& required) const;

  void ValidateColumns(const arrow::Table& table,
                       const std::vector<std::string>& required,
                       const std::vector<std::string>& optional,
                       bool require_single_chunk,
                       const std::string& context) const;

  std::vector<std::string> MergeColumns(const std::vector<std::string>& left,
                                        const std::vector<std::string>& right) const;

  using ColumnMap = std::unordered_map<std::string, std::shared_ptr<arrow::ChunkedArray>>;

  ColumnMap BindColumns(const arrow::Table& table,
                        const std::vector<std::string>& names,
                        bool require_all,
                        bool require_single_chunk,
                        const std::string& context) const;
};

}  // namespace pioneerml::dataloaders
