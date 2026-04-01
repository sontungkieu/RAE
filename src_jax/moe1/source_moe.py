from __future__ import annotations

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class SourceMoEOutput:
    mu: jnp.ndarray
    logvar: jnp.ndarray
    alpha: jnp.ndarray
    logits: jnp.ndarray
    expert_mu: jnp.ndarray
    expert_logvar: jnp.ndarray


def _same_padding_conv(
    in_channels: int,
    out_channels: int,
    *,
    rngs: nnx.Rngs,
    dtype: jnp.dtype,
    kernel_init=None,
    bias_init=None,
) -> nnx.Conv:
    kwargs = {
        "kernel_init": kernel_init,
        "bias_init": bias_init,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return nnx.Conv(
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="SAME",
        dtype=dtype,
        rngs=rngs,
        **kwargs,
    )


class ConditionProjector(nnx.Module):
    def __init__(
        self,
        *,
        num_modes: int,
        condition_dim: int,
        hidden_channels: int,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        key = rngs.params()
        self.mode_embeddings = nnx.Param(jax.random.normal(key, (num_modes, condition_dim)) * 0.02)
        self.linear1 = nnx.Linear(condition_dim, hidden_channels, dtype=dtype, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_channels, hidden_channels, dtype=dtype, rngs=rngs)

    def __call__(self, condition_weights: jnp.ndarray) -> jnp.ndarray:
        condition_embedding = condition_weights @ self.mode_embeddings.value
        hidden = nnx.silu(self.linear1(condition_embedding))
        hidden = self.linear2(hidden)
        return hidden[:, None, None, :]


class Router(nnx.Module):
    def __init__(
        self,
        *,
        hidden_channels: int,
        num_modes: int,
        temperature: float,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.temperature = max(float(temperature), 1e-6)
        self.conv = _same_padding_conv(hidden_channels, hidden_channels, rngs=rngs, dtype=dtype)
        self.linear = nnx.Linear(hidden_channels, num_modes, dtype=dtype, rngs=rngs)

    def __call__(self, features: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        hidden = nnx.silu(self.conv(features))
        hidden = jnp.mean(hidden, axis=(1, 2))
        logits = self.linear(hidden)
        alpha = jax.nn.softmax(logits / self.temperature, axis=-1)
        return alpha, logits


class SourceExpert(nnx.Module):
    def __init__(
        self,
        *,
        hidden_channels: int,
        out_channels: int,
        logvar_min: float,
        logvar_max: float,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        zero_init = jax.nn.initializers.zeros
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.conv = _same_padding_conv(hidden_channels, hidden_channels, rngs=rngs, dtype=dtype)
        self.mu_head = _same_padding_conv(
            hidden_channels,
            out_channels,
            rngs=rngs,
            dtype=dtype,
            kernel_init=zero_init,
            bias_init=zero_init,
        )
        self.logvar_head = _same_padding_conv(
            hidden_channels,
            out_channels,
            rngs=rngs,
            dtype=dtype,
            kernel_init=zero_init,
            bias_init=zero_init,
        )

    def __call__(self, features: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        hidden = nnx.silu(self.conv(features))
        mu = self.mu_head(hidden)
        logvar = jnp.clip(self.logvar_head(hidden), self.logvar_min, self.logvar_max)
        return mu, logvar


class SourceMoE(nnx.Module):
    def __init__(
        self,
        *,
        num_modes: int,
        in_channels: int,
        condition_dim: int,
        hidden_channels: int,
        router_temperature: float,
        soft_moe: bool,
        logvar_min: float,
        logvar_max: float,
        var_floor: float,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.num_modes = int(num_modes)
        self.in_channels = int(in_channels)
        self.soft_moe = bool(soft_moe)
        self.var_floor = float(var_floor)
        self.cond_projector = ConditionProjector(
            num_modes=num_modes,
            condition_dim=condition_dim,
            hidden_channels=hidden_channels,
            rngs=rngs,
            dtype=dtype,
        )
        self.shared_conv1 = _same_padding_conv(in_channels, hidden_channels, rngs=rngs, dtype=dtype)
        self.shared_conv2 = _same_padding_conv(hidden_channels, hidden_channels, rngs=rngs, dtype=dtype)
        self.router = Router(
            hidden_channels=hidden_channels,
            num_modes=num_modes,
            temperature=router_temperature,
            rngs=rngs,
            dtype=dtype,
        )
        self.experts = [
            SourceExpert(
                hidden_channels=hidden_channels,
                out_channels=in_channels,
                logvar_min=logvar_min,
                logvar_max=logvar_max,
                rngs=rngs,
                dtype=dtype,
            )
            for _ in range(num_modes)
        ]

    def _resolve_alpha(self, alpha: jnp.ndarray) -> jnp.ndarray:
        if self.soft_moe:
            return alpha
        alpha_hard = jax.nn.one_hot(jnp.argmax(alpha, axis=-1), self.num_modes, dtype=alpha.dtype)
        return alpha + jax.lax.stop_gradient(alpha_hard - alpha)

    def __call__(self, z: jnp.ndarray, condition_weights: jnp.ndarray) -> SourceMoEOutput:
        features = nnx.silu(self.shared_conv1(z))
        features = nnx.silu(self.shared_conv2(features))
        condition_bias = self.cond_projector(condition_weights)
        conditioned = features + condition_bias

        alpha_soft, logits = self.router(conditioned)
        alpha = self._resolve_alpha(alpha_soft)

        mu_list: list[jnp.ndarray] = []
        logvar_list: list[jnp.ndarray] = []
        for expert in self.experts:
            expert_mu, expert_logvar = expert(conditioned)
            mu_list.append(expert_mu)
            logvar_list.append(expert_logvar)

        expert_mu = jnp.stack(mu_list, axis=1)
        expert_logvar = jnp.stack(logvar_list, axis=1)
        expert_var = jnp.exp(expert_logvar)

        alpha_full = alpha[:, :, None, None, None]
        mixture_mu = jnp.sum(alpha_full * expert_mu, axis=1)
        second_moment = jnp.sum(alpha_full * (expert_var + jnp.square(expert_mu)), axis=1)
        mixture_var = jnp.maximum(second_moment - jnp.square(mixture_mu), self.var_floor)
        mixture_logvar = jnp.log(mixture_var)

        return SourceMoEOutput(
            mu=mixture_mu,
            logvar=mixture_logvar,
            alpha=alpha,
            logits=logits,
            expert_mu=expert_mu,
            expert_logvar=expert_logvar,
        )
