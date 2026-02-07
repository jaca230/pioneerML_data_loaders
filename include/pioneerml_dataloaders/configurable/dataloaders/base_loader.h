#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/buffer.h>

#include "pioneerml_dataloaders/batch/base_batch.h"
#include "pioneerml_dataloaders/configurable/data_derivers/base_deriver.h"
#include "pioneerml_dataloaders/utils/parquet/parquet_utils.h"

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
  virtual TrainingBundle LoadTraining(const std::shared_ptr<arrow::Table>& table) const = 0;

  // Load inputs only for inference.
  virtual InferenceBundle LoadInference(const std::shared_ptr<arrow::Table>& table) const = 0;

 protected:
  // Derived loaders can override if they need custom input/target splitting.
  virtual TrainingBundle SplitInputsTargets(std::unique_ptr<BaseBatch> batch) const = 0;
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

  std::shared_ptr<arrow::Table> PrepareTable(
      const std::shared_ptr<arrow::Table>& table,
      bool add_derived) const;

  std::vector<std::string> input_columns_;
  std::vector<std::string> target_columns_;

  using NumericAccessor = pioneerml::utils::parquet::NumericAccessor;

  NumericAccessor MakeNumericAccessor(const std::shared_ptr<arrow::Array>& arr,
                                      const std::string& context) const;
  std::shared_ptr<arrow::Buffer> AllocBuffer(int64_t bytes) const;
  std::shared_ptr<arrow::Array> MakeArray(std::shared_ptr<arrow::Buffer> buffer,
                                          const std::shared_ptr<arrow::DataType>& type,
                                          int64_t length) const;
  double ResolveCoordinateForView(const NumericAccessor& x_values,
                                  const NumericAccessor& y_values,
                                  int32_t view,
                                  int64_t idx) const;

  std::vector<int64_t> BuildOffsets(const std::vector<int64_t>& counts) const;
  void FillPointerArrayFromOffsets(const std::vector<int64_t>& offsets,
                                   int64_t* out_ptr) const;

  using ColumnMap = pioneerml::utils::parquet::ColumnMap;
};

}  // namespace pioneerml::dataloaders
