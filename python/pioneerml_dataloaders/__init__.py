"""
Python bindings shim for pioneerml_dataloaders.

Actual implementation will be provided via pybind11 once the C++ core is ready.
"""

from .specs import GROUP_CLASSIFIER_SPEC

__all__ = ["GROUP_CLASSIFIER_SPEC"]
