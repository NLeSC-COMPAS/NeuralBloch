"""Helpers that the tests share."""

import os
from pathlib import Path
import pytest


# Not torch.cuda or libcuda: tests/conftest.py imports this module before
# torch, and they must not load before juliacall.
# CUDA_VISIBLE_DEVICES="" or "-1" hides the GPUs, so the GPU tests skip.
HAS_GPU = any(Path("/proc/driver/nvidia/gpus").glob("*")) and os.environ.get(
    "CUDA_VISIBLE_DEVICES"
) not in ("", "-1")

# The Julia simulations need a GPU, so the tests that use Julia need has_gpu
has_gpu = pytest.mark.skipif(not HAS_GPU, reason="needs an NVIDIA GPU")
