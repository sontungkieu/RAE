from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - optional in non-JAX environments
    jnp = np


LOG_2PI = float(np.log(2.0 * np.pi))


@dataclasses.dataclass(frozen=True)
class GMMArtifact:
    log_pi: np.ndarray
    mu: np.ndarray
    var: np.ndarray
    latent_mean: np.ndarray
    latent_std: np.ndarray
    standardize_eps: float
    latent_shape: tuple[int, ...]
    layout: str
    sample_posterior: bool
    latent_semantics: str
    vae_scale_factor: float
    count: int
    num_modes: int
    active_modes: int
    train_nll: float
    active_mode_fraction_threshold: float
    final_counts: np.ndarray
    n_iter: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_pi": np.asarray(self.log_pi, dtype=np.float32),
            "mu": np.asarray(self.mu, dtype=np.float32),
            "var": np.asarray(self.var, dtype=np.float32),
            "latent_mean": np.asarray(self.latent_mean, dtype=np.float32),
            "latent_std": np.asarray(self.latent_std, dtype=np.float32),
            "standardize_eps": np.asarray(self.standardize_eps, dtype=np.float32),
            "latent_shape": np.asarray(self.latent_shape, dtype=np.int32),
            "layout": np.asarray(self.layout),
            "sample_posterior": np.asarray(self.sample_posterior),
            "latent_semantics": np.asarray(self.latent_semantics),
            "vae_scale_factor": np.asarray(self.vae_scale_factor, dtype=np.float32),
            "count": np.asarray(self.count, dtype=np.int64),
            "num_modes": np.asarray(self.num_modes, dtype=np.int32),
            "active_modes": np.asarray(self.active_modes, dtype=np.int32),
            "train_nll": np.asarray(self.train_nll, dtype=np.float32),
            "active_mode_fraction_threshold": np.asarray(
                self.active_mode_fraction_threshold,
                dtype=np.float32,
            ),
            "final_counts": np.asarray(self.final_counts, dtype=np.float32),
            "n_iter": np.asarray(self.n_iter, dtype=np.int32),
        }


def flatten_latents_nhwc(latents: np.ndarray | jnp.ndarray) -> np.ndarray | jnp.ndarray:
    return latents.reshape((latents.shape[0], -1))


def compute_standardization_stats(
    latents_flat: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    mean = latents_flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    var = latents_flat.var(axis=0, dtype=np.float64).astype(np.float32)
    std = np.sqrt(np.maximum(var, eps)).astype(np.float32)
    return mean, std


def standardize_latents(
    latents_flat: np.ndarray | jnp.ndarray,
    mean: np.ndarray | jnp.ndarray,
    std: np.ndarray | jnp.ndarray,
    eps: float,
) -> np.ndarray | jnp.ndarray:
    return (latents_flat - mean) / (std + eps)


def _logsumexp_np(values: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - max_values)
    summed = np.sum(shifted, axis=axis, keepdims=True)
    output = max_values + np.log(np.maximum(summed, 1e-12))
    if keepdims:
        return output
    return np.squeeze(output, axis=axis)


def diag_gmm_log_prob_np(
    latents_std: np.ndarray,
    log_pi: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
) -> np.ndarray:
    safe_var = np.maximum(var, 1e-12)
    diff = latents_std[:, None, :] - mu[None, :, :]
    quad = np.sum(np.square(diff) / safe_var[None, :, :], axis=-1)
    log_det = np.sum(np.log(safe_var), axis=-1)
    dim = latents_std.shape[-1]
    return log_pi[None, :] - 0.5 * (dim * LOG_2PI + log_det[None, :] + quad)


def diag_gmm_log_prob_jax(
    latents_std: jnp.ndarray,
    log_pi: jnp.ndarray,
    mu: jnp.ndarray,
    var: jnp.ndarray,
) -> jnp.ndarray:
    safe_var = jnp.maximum(var, 1e-12)
    diff = latents_std[:, None, :] - mu[None, :, :]
    quad = jnp.sum(jnp.square(diff) / safe_var[None, :, :], axis=-1)
    log_det = jnp.sum(jnp.log(safe_var), axis=-1)
    dim = latents_std.shape[-1]
    return log_pi[None, :] - 0.5 * (dim * LOG_2PI + log_det[None, :] + quad)


def posterior_from_stats(
    latents_flat: np.ndarray | jnp.ndarray,
    *,
    latent_mean: np.ndarray | jnp.ndarray,
    latent_std: np.ndarray | jnp.ndarray,
    standardize_eps: float,
    log_pi: np.ndarray | jnp.ndarray,
    mu: np.ndarray | jnp.ndarray,
    var: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    latents_std = standardize_latents(
        jnp.asarray(latents_flat, dtype=jnp.float32),
        jnp.asarray(latent_mean, dtype=jnp.float32),
        jnp.asarray(latent_std, dtype=jnp.float32),
        standardize_eps,
    )
    log_prob = diag_gmm_log_prob_jax(
        latents_std,
        jnp.asarray(log_pi, dtype=jnp.float32),
        jnp.asarray(mu, dtype=jnp.float32),
        jnp.asarray(var, dtype=jnp.float32),
    )
    log_norm = jax_logsumexp(log_prob, axis=-1, keepdims=True)
    return jnp.exp(log_prob - log_norm)


def jax_logsumexp(values: jnp.ndarray, axis: int = -1, keepdims: bool = False) -> jnp.ndarray:
    max_values = jnp.max(values, axis=axis, keepdims=True)
    shifted = jnp.exp(values - max_values)
    summed = jnp.sum(shifted, axis=axis, keepdims=True)
    output = max_values + jnp.log(jnp.maximum(summed, 1e-12))
    if keepdims:
        return output
    return jnp.squeeze(output, axis=axis)


def compute_active_modes(
    counts: np.ndarray,
    total_count: int,
    *,
    fraction_threshold: float,
) -> int:
    if total_count <= 0:
        return 0
    return int(np.sum((counts / float(total_count)) > fraction_threshold))


def _chunk_em_stats(
    latents_std: np.ndarray,
    log_pi: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    num_modes, dim = mu.shape
    counts = np.zeros((num_modes,), dtype=np.float64)
    sum_x = np.zeros((num_modes, dim), dtype=np.float64)
    sum_x2 = np.zeros((num_modes, dim), dtype=np.float64)
    total_nll = 0.0

    for start in range(0, latents_std.shape[0], chunk_size):
        chunk = latents_std[start : start + chunk_size]
        log_prob = diag_gmm_log_prob_np(chunk, log_pi, mu, var)
        log_norm = _logsumexp_np(log_prob, axis=-1, keepdims=True)
        resp = np.exp(log_prob - log_norm)
        counts += resp.sum(axis=0, dtype=np.float64)
        sum_x += resp.T @ chunk
        sum_x2 += resp.T @ np.square(chunk)
        total_nll += float((-log_norm.squeeze(-1)).sum(dtype=np.float64))

    return counts, sum_x, sum_x2, total_nll / float(latents_std.shape[0])


def _compute_train_nll(
    latents_std: np.ndarray,
    log_pi: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    *,
    chunk_size: int,
) -> float:
    total_nll = 0.0
    for start in range(0, latents_std.shape[0], chunk_size):
        chunk = latents_std[start : start + chunk_size]
        log_prob = diag_gmm_log_prob_np(chunk, log_pi, mu, var)
        log_norm = _logsumexp_np(log_prob, axis=-1, keepdims=False)
        total_nll += float((-log_norm).sum(dtype=np.float64))
    return total_nll / float(latents_std.shape[0])


def _sample_data_rows(
    latents_std: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = rng.integers(0, latents_std.shape[0], size=count)
    return latents_std[indices].copy()


def kmeanspp_init(
    latents_std: np.ndarray,
    *,
    num_modes: int,
    rng: np.random.Generator,
    chunk_size: int,
) -> np.ndarray:
    num_samples, dim = latents_std.shape
    if num_samples < num_modes:
        raise ValueError(
            f"Need at least as many samples as modes. Got {num_samples} samples for {num_modes} modes."
        )

    centers = np.zeros((num_modes, dim), dtype=np.float32)
    first_idx = int(rng.integers(0, num_samples))
    centers[0] = latents_std[first_idx]
    min_distances = np.full((num_samples,), np.inf, dtype=np.float64)

    for center_idx in range(1, num_modes):
        previous_center = centers[center_idx - 1][None, :]
        for start in range(0, num_samples, chunk_size):
            chunk = latents_std[start : start + chunk_size]
            distances = np.sum(np.square(chunk - previous_center), axis=-1, dtype=np.float64)
            min_distances[start : start + chunk.shape[0]] = np.minimum(
                min_distances[start : start + chunk.shape[0]],
                distances,
            )

        total_distance = float(np.sum(min_distances))
        if not np.isfinite(total_distance) or total_distance <= 0.0:
            fallback = _sample_data_rows(latents_std, num_modes - center_idx, rng)
            centers[center_idx:] = fallback
            break

        probs = min_distances / total_distance
        next_idx = int(rng.choice(num_samples, p=probs))
        centers[center_idx] = latents_std[next_idx]

    return centers


def fit_diag_gmm(
    latents_flat: np.ndarray,
    *,
    num_modes: int,
    seed: int,
    weight_prior: float = 1e-2,
    var_floor: float = 1e-5,
    em_iters: int = 100,
    em_tol: float = 1e-4,
    em_restarts: int = 3,
    dead_count_threshold: float = 1.0,
    active_mode_fraction_threshold: float = 0.01,
    standardize_eps: float = 1e-6,
    chunk_size: int = 256,
) -> GMMArtifact:
    if latents_flat.ndim != 2:
        raise ValueError(f"Expected 2D latent array, got shape {latents_flat.shape}.")
    if latents_flat.shape[0] == 0:
        raise ValueError("Cannot fit GMM on an empty latent array.")

    latents_flat = np.asarray(latents_flat, dtype=np.float32)
    latent_mean, latent_std = compute_standardization_stats(latents_flat, eps=standardize_eps)
    latents_std = np.asarray(
        standardize_latents(latents_flat, latent_mean, latent_std, standardize_eps),
        dtype=np.float32,
    )

    global_var = np.maximum(latents_std.var(axis=0, dtype=np.float64).astype(np.float32), var_floor)
    best_artifact: GMMArtifact | None = None
    best_nll = float("inf")

    for restart_idx in range(em_restarts):
        rng = np.random.default_rng(seed + restart_idx)
        mu = kmeanspp_init(latents_std, num_modes=num_modes, rng=rng, chunk_size=chunk_size)
        var = np.broadcast_to(global_var[None, :], (num_modes, global_var.shape[0])).copy()
        log_pi = np.full((num_modes,), -np.log(float(num_modes)), dtype=np.float32)
        previous_nll: float | None = None
        counts = np.zeros((num_modes,), dtype=np.float64)
        n_iter = 0

        for iteration_idx in range(em_iters):
            counts, sum_x, sum_x2, train_nll = _chunk_em_stats(
                latents_std,
                log_pi,
                mu,
                var,
                chunk_size=chunk_size,
            )
            safe_counts = np.maximum(counts, 1e-12)
            mu_new = (sum_x / safe_counts[:, None]).astype(np.float32)
            second_moment = (sum_x2 / safe_counts[:, None]).astype(np.float32)
            var_new = np.maximum(second_moment - np.square(mu_new), var_floor).astype(np.float32)

            dead_mask = counts < dead_count_threshold
            if np.any(dead_mask):
                dead_count = int(np.sum(dead_mask))
                mu_new[dead_mask] = _sample_data_rows(latents_std, dead_count, rng)
                var_new[dead_mask] = global_var

            counts_with_prior = counts + weight_prior
            pi_new = counts_with_prior / np.sum(counts_with_prior)
            log_pi_new = np.log(np.maximum(pi_new, 1e-12)).astype(np.float32)

            mu = mu_new
            var = var_new
            log_pi = log_pi_new
            n_iter = iteration_idx + 1

            if previous_nll is not None:
                relative_delta = abs(train_nll - previous_nll) / max(abs(previous_nll), 1e-12)
                if relative_delta < em_tol:
                    previous_nll = train_nll
                    break
            previous_nll = train_nll

        final_nll = _compute_train_nll(latents_std, log_pi, mu, var, chunk_size=chunk_size)
        if final_nll < best_nll:
            final_counts, _, _, _ = _chunk_em_stats(
                latents_std,
                log_pi,
                mu,
                var,
                chunk_size=chunk_size,
            )
            best_nll = final_nll
            best_artifact = GMMArtifact(
                log_pi=log_pi.astype(np.float32),
                mu=mu.astype(np.float32),
                var=var.astype(np.float32),
                latent_mean=latent_mean.astype(np.float32),
                latent_std=latent_std.astype(np.float32),
                standardize_eps=float(standardize_eps),
                latent_shape=(latents_flat.shape[1],),
                layout="FLAT",
                sample_posterior=True,
                latent_semantics="stabilityvae_scaled_output",
                vae_scale_factor=0.18215,
                count=int(latents_flat.shape[0]),
                num_modes=int(num_modes),
                active_modes=compute_active_modes(
                    final_counts,
                    latents_flat.shape[0],
                    fraction_threshold=active_mode_fraction_threshold,
                ),
                train_nll=float(final_nll),
                active_mode_fraction_threshold=float(active_mode_fraction_threshold),
                final_counts=final_counts.astype(np.float32),
                n_iter=int(n_iter),
            )

    if best_artifact is None:
        raise RuntimeError("Failed to fit diagonal GMM.")
    return best_artifact


def save_gmm_artifact(path: str | Path, artifact: GMMArtifact) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **artifact.to_dict())
    return destination


def load_gmm_artifact(path: str | Path) -> GMMArtifact:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        return GMMArtifact(
            log_pi=np.asarray(payload["log_pi"], dtype=np.float32),
            mu=np.asarray(payload["mu"], dtype=np.float32),
            var=np.asarray(payload["var"], dtype=np.float32),
            latent_mean=np.asarray(payload["latent_mean"], dtype=np.float32),
            latent_std=np.asarray(payload["latent_std"], dtype=np.float32),
            standardize_eps=float(np.asarray(payload["standardize_eps"]).item()),
            latent_shape=tuple(int(v) for v in np.asarray(payload["latent_shape"]).tolist()),
            layout=str(np.asarray(payload["layout"]).item()),
            sample_posterior=bool(np.asarray(payload["sample_posterior"]).item()),
            latent_semantics=str(np.asarray(payload["latent_semantics"]).item()),
            vae_scale_factor=float(np.asarray(payload["vae_scale_factor"]).item()),
            count=int(np.asarray(payload["count"]).item()),
            num_modes=int(np.asarray(payload["num_modes"]).item()),
            active_modes=int(np.asarray(payload["active_modes"]).item()),
            train_nll=float(np.asarray(payload["train_nll"]).item()),
            active_mode_fraction_threshold=float(
                np.asarray(payload["active_mode_fraction_threshold"]).item()
            ),
            final_counts=np.asarray(payload["final_counts"], dtype=np.float32),
            n_iter=int(np.asarray(payload["n_iter"]).item()),
        )
