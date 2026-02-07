# Graph Loader Schemas

This document defines the graph batch contract produced by each loader in
`pioneerml_dataloaders/configurable/dataloaders/graph`.

## GroupClassifierLoader

Code:
- `src/pioneerml_dataloaders/configurable/dataloaders/graph/group_classifier_loader.cpp`

Sample definition:
- One graph per `(event, time_group)`.

Inputs:
- `node_features` (`float32`, `[total_nodes, 4]`)
  - `[coord, z, edep, view]`
  - `coord` is `x` for view `0`, `y` for view `1`, with fallback to the other axis if null.
- `edge_index` (`int64`, `[2, total_edges]`) complete directed graph within each sample.
- `edge_attr` (`float32`, `[total_edges, 4]`)
  - `[dcoord, dz, dE, same_view]`.
- `time_group_ids` (`int64`, `[total_nodes]`) local group id for each node.
- `u` (`float32`, `[num_graphs]`) per-graph sum of node energy.
- `node_ptr` (`int64`, `[num_graphs + 1]`) prefix offsets into flattened nodes.
- `edge_ptr` (`int64`, `[num_graphs + 1]`) prefix offsets into flattened edges.
- `group_ptr` (`int64`, `[num_graphs + 1]`) identity prefix for graph-level grouping.
- `graph_event_ids` (`int64`, `[num_graphs]`) source event index.
- `graph_group_ids` (`int64`, `[num_graphs]`) source time-group id within event.

Targets:
- `y` (`float32`, `[num_graphs, 3]`)
  - `[pion_in_group, muon_in_group, mip_in_group]`.

## GroupClassifierEventLoader

Code:
- `src/pioneerml_dataloaders/configurable/dataloaders/graph/group_classifier_event_loader.cpp`

Sample definition:
- One graph per event (all event hits in one graph).

Inputs:
- `node_features` (`float32`, `[total_nodes, 4]`)
  - `[coord, z, edep, view]`.
- `edge_index` (`int64`, `[2, total_edges]`) complete directed graph per event.
- `edge_attr` (`float32`, `[total_edges, 4]`)
  - `[dcoord, dz, dE, same_view]`.
- `time_group_ids` (`int64`, `[total_nodes]`) original time-group id for each node.
- `u` (`float32`, `[num_graphs]`) per-event sum of node energy.
- `node_ptr`, `edge_ptr`, `group_ptr` as prefix arrays for flattened storage.

Targets:
- `y` (`float32`, `[total_groups, 3]`)
  - flattened per-event time-group labels:
  - `[pion_in_group, muon_in_group, mip_in_group]`.

## GroupSplitterLoader

Code:
- `src/pioneerml_dataloaders/configurable/dataloaders/graph/group_splitter_loader.cpp`

Sample definition:
- One graph per `(event, time_group)`.

Inputs:
- `node_features` (`float32`, `[total_nodes, 4]`)
  - `[coord, z, edep, view]`.
- `edge_index` (`int64`, `[2, total_edges]`) complete directed graph within each sample.
- `edge_attr` (`float32`, `[total_edges, 4]`)
  - `[dcoord, dz, dE, same_view]`.
- `u` (`float32`, `[num_graphs]`) per-graph sum of node energy.
- `group_probs` (`float32`, `[num_graphs, 3]`)
  - class probabilities injected from classifier predictions when available:
  - `[pred_pion, pred_muon, pred_mip]`.
- `node_ptr`, `edge_ptr` prefix arrays.
- `graph_event_ids`, `graph_group_ids` graph identity columns.

Targets:
- `y_node` (`float32`, `[total_nodes, 3]`)
  - one-hot per-node class from `hits_pdg_id` mapping:
  - pion: `211`, muon: `-13`, mip: `11/-11`.

## Notes

- All loaders flatten variable-size graphs into contiguous arrays with pointer
  columns (`node_ptr`, `edge_ptr`) for efficient zero-copy transfer.
- Inference mode strips targets from output batches.
