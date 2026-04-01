from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class StabilityVAE(nn.Module):
    """Repo-facing alias for the backend-native JAX StabilityVAE encoder.

    This symbol exists so OmegaConf configs and notebooks can refer to
    ``stage1.StabilityVAE`` consistently. The actual implementation used by the
    JAX adapter lives in the vendored ``diffuse_nnx`` backend.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.init_args = args
        self.init_kwargs = kwargs

    def _raise_jax_only(self) -> None:
        raise NotImplementedError(
            "stage1.StabilityVAE is a JAX-only repo-facing alias. "
            "Use the src_jax entrypoints so the adapter can instantiate the "
            "backend-native StabilityVAE encoder."
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        del x
        self._raise_jax_only()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        del z
        self._raise_jax_only()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        del x
        self._raise_jax_only()
