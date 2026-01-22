# pioneerml_dataloaders

Cross-language data loader library for PIONEER ML parquet datasets.

Goals
- Single source of truth (JSON specs) for feature derivation, padding/masks, and graph construction.
- C++ first (Arrow/Parquet + libtorch-friendly tensors) with a thin Python wrapper (pybind11).
- Model-specific input contracts expressed as schemas/specs shared by both C++ and Python.

Scope (initial)
1. Group classifier
   - Inputs: hits_x/hits_y/hits_z/hits_edep/hits_strip_type/hits_pdg_id (+ time if needed)
   - Derived: time-group labels (window_ns=1), dense class labels (pion/muon/mip), padding/mask.
   - Outputs: contiguous tensors for node features, masks, labels, edge indices/attrs.
2. Extend to splitter/endpoint/pion-stop with additional specs as needed.

Architecture
- `/cpp`: Arrow/Parquet reader + processors producing contiguous CPU buffers; expose a C++ API returning structs of tensors.
- `/python`: pybind11 bindings exposing the same API; optional pure-Python parity tests.
- `/specs`: JSON/YAML or header-only descriptions of column requirements, derived fields, padding sizes.

Next steps
- Implement C++ Arrow reader that column-projects, adds time-group labels, pads to max_hits, and builds edge_index/edge_attr for group classifier.
- Add pybind11 bindings and a minimal Python wheel build.
- Add golden tests comparing Python and C++ outputs on sample parquet rows.
