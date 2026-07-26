"""Core package for BiasAuditFW schema 2.0."""

from .config import DatasetConfig
from .contracts import FACE_COLUMNS, IMAGE_COLUMNS, SCHEMA_VERSION, validate_contract
from .ingestion import DatasetDiscovery, discover_dataset
from .pipeline import export_results, process_dataset

__all__ = [
    "DatasetConfig",
    "DatasetDiscovery",
    "FACE_COLUMNS",
    "IMAGE_COLUMNS",
    "SCHEMA_VERSION",
    "discover_dataset",
    "export_results",
    "process_dataset",
    "validate_contract",
]
