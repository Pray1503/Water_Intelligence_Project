"""Stage 10 - Model Training Pipeline (Layers 1-3 only).

Layer 1: Dataset loading      -> dataset_bundle.py
Layer 2: Input validation     -> validation.py
Layer 3: Model registry       -> model_registry.py

Layers 4-7 are out of scope for this module.
"""

from src.stage10.dataset_bundle import DatasetBundle, load_dataset_bundle
from src.stage10.model_registry import (
    ModelConfig,
    build_model_registry,
    load_model_configs_from_dict,
)
from src.stage10.validation import (
    ValidationConfig,
    validate_dataset_bundle,
)

__all__ = [
    "DatasetBundle",
    "load_dataset_bundle",
    "ValidationConfig",
    "validate_dataset_bundle",
    "ModelConfig",
    "build_model_registry",
    "load_model_configs_from_dict",
]
