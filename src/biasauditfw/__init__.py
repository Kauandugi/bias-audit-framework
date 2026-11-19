"""Core package for BiasAuditFW schema 2.0."""

from .config import DatasetConfig
from .contracts import FACE_COLUMNS, IMAGE_COLUMNS, SCHEMA_VERSION, validate_contract
from .ingestion import DatasetDiscovery, discover_dataset
from .manifest import build_path_manifest, build_sidecar_manifest
from .pipeline import export_results, process_dataset, select_smoke_inventory

__all__ = [
    "DatasetConfig",
    "DatasetDiscovery",
    "FACE_COLUMNS",
    "IMAGE_COLUMNS",
    "SCHEMA_VERSION",
    "build_path_manifest",
    "build_sidecar_manifest",
    "discover_dataset",
    "export_results",
    "process_dataset",
    "select_smoke_inventory",
    "validate_contract",
]
