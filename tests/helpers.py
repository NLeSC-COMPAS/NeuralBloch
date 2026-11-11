"""Helpers that the tests share."""

from pathlib import Path
import pytest


# Not torch.cuda or libcuda: tests/conftest.py imports this module before
# torch, and they must not load before juliacall
HAS_GPU = any(Path("/proc/driver/nvidia/gpus").glob("*"))

# The Julia simulations need a GPU, so the tests that use Julia need has_gpu
has_gpu = pytest.mark.skipif(not HAS_GPU, reason="needs an NVIDIA GPU")
