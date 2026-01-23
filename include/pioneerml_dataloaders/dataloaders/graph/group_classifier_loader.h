#pragma once

#include <memory>
#include <string>

#include "pioneerml_dataloaders/dataloaders/graph/graph_loader.h"
#include "pioneerml_dataloaders/batch/graph_batch.h"

namespace pioneerml::dataloaders::graph {

struct GroupClassifierConfig {
  double time_window_ns{1.0};
  bool compute_time_groups{true};
};

class GroupClassifierLoader : public GraphLoader {
 public:
  explicit GroupClassifierLoader(GroupClassifierConfig cfg = {});

  std::shared_ptr<arrow::Table> LoadTable(const std::string& parquet_path) const override;

 protected:
  std::unique_ptr<GraphBatch> BuildGraph(const arrow::Table& table) const override;

 private:
  GroupClassifierConfig cfg_;
};

}  // namespace pioneerml::dataloaders::graph
