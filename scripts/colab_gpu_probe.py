"""Minimal accelerator probe executed by the Google Colab CLI."""

from __future__ import annotations

import json
import platform

import torch


def main() -> None:
    """Print machine-readable CUDA metadata and fail when no GPU is available."""
    cuda_available = torch.cuda.is_available()
    payload = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }
    print(json.dumps(payload, sort_keys=True))
    if not cuda_available:
        raise RuntimeError("The Colab runtime does not expose a CUDA GPU.")


if __name__ == "__main__":
    main()
