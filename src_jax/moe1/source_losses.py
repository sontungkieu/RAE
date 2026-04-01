from __future__ import annotations

try:
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - optional in non-JAX environments
    import numpy as jnp


def balance_loss(alpha: jnp.ndarray) -> jnp.ndarray:
    mean_alpha = jnp.mean(alpha, axis=0)
    uniform = jnp.full_like(mean_alpha, 1.0 / mean_alpha.shape[0])
    return jnp.sum(jnp.square(mean_alpha - uniform))


def entropy_loss(alpha: jnp.ndarray, *, eps: float = 1e-8) -> jnp.ndarray:
    safe_alpha = jnp.clip(alpha, eps, 1.0)
    return -jnp.mean(jnp.sum(safe_alpha * jnp.log(safe_alpha), axis=-1))


def var_only_kld_loss(logvar: jnp.ndarray, *, target_variance: float = 1.0) -> jnp.ndarray:
    var = jnp.exp(logvar)
    target_var = jnp.asarray(target_variance, dtype=var.dtype)
    ratio = var / jnp.maximum(target_var, 1e-8)
    return 0.5 * jnp.mean(ratio - 1.0 - jnp.log(jnp.maximum(ratio, 1e-8)))


def summarize_router(alpha: jnp.ndarray) -> dict[str, jnp.ndarray]:
    safe_alpha = jnp.clip(alpha, 1e-8, 1.0)
    entropy = -jnp.mean(jnp.sum(safe_alpha * jnp.log(safe_alpha), axis=-1))
    router_max = jnp.mean(jnp.max(alpha, axis=-1))
    active_modes = jnp.sum(jnp.mean(alpha, axis=0) > 0.01)
    return {
        "source_router_entropy": entropy,
        "source_router_max": router_max,
        "source_active_modes": active_modes.astype(jnp.float32),
    }
