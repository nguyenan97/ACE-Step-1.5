"""Tests for the ACESTEP_DTYPE CUDA dtype override."""

import os
import unittest
from unittest.mock import patch

import torch

from acestep.core.generation.handler.init_service_orchestrator import resolve_cuda_dtype_override
from acestep.core.generation.handler.memory_utils import MemoryUtilsMixin


class _Host(MemoryUtilsMixin):
    def __init__(self, dtype):
        self.device = "cuda"
        self.dtype = dtype


class ResolveCudaDtypeOverrideTests(unittest.TestCase):
    def test_returns_float32_when_set(self):
        with patch.dict(os.environ, {"ACESTEP_DTYPE": "float32"}):
            self.assertIs(resolve_cuda_dtype_override(), torch.float32)

    def test_returns_none_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_cuda_dtype_override())

    def test_unknown_value_is_ignored(self):
        with patch.dict(os.environ, {"ACESTEP_DTYPE": "int4"}):
            self.assertIsNone(resolve_cuda_dtype_override())


class VaeDtypeOverrideTests(unittest.TestCase):
    @patch("acestep.core.generation.handler.memory_utils.is_rocm_available", return_value=False)
    @patch("acestep.core.generation.handler.memory_utils.cuda_supports_bfloat16", return_value=False)
    def test_vae_follows_override(self, _bf16, _rocm):
        with patch.dict(os.environ, {"ACESTEP_DTYPE": "float32"}):
            self.assertIs(_Host(torch.float32)._get_vae_dtype("cuda"), torch.float32)

    @patch("acestep.core.generation.handler.memory_utils.is_rocm_available", return_value=False)
    @patch("acestep.core.generation.handler.memory_utils.cuda_supports_bfloat16", return_value=False)
    def test_vae_default_pre_ampere_is_float16(self, _bf16, _rocm):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIs(_Host(torch.float16)._get_vae_dtype("cuda"), torch.float16)


if __name__ == "__main__":
    unittest.main()
