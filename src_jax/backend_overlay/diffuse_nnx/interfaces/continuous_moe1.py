from __future__ import annotations

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp

from interfaces.continuous import SiTInterface, TrainingTimeDistType
from moe1.gmm_utils import load_gmm_artifact, posterior_from_stats
from moe1.source_losses import balance_loss, entropy_loss, summarize_router, var_only_kld_loss
from moe1.source_moe import SourceMoE


class Buffer(nnx.Variable):
    """Non-trainable container for fixed GMM statistics."""


class SiTGMMMoe1Interface(SiTInterface):
    def __init__(
        self,
        network: nnx.Module,
        train_time_dist_type: str | TrainingTimeDistType,
        t_mu: float = 0.0,
        t_sigma: float = 1.0,
        n_mu: float = 0.0,
        n_sigma: float = 1.0,
        x_sigma: float = 0.5,
        t_shift_base: int = 4096,
        source: dict[str, Any] | None = None,
    ):
        super().__init__(
            network=network,
            train_time_dist_type=train_time_dist_type,
            t_mu=t_mu,
            t_sigma=t_sigma,
            n_mu=n_mu,
            n_sigma=n_sigma,
            x_sigma=x_sigma,
            t_shift_base=t_shift_base,
        )
        if not source or not bool(source.get("enabled", False)):
            raise ValueError("sit_gmm_moe1 requires interface.source.enabled=true.")

        self.source_cfg = dict(source)
        self.num_modes = int(self.source_cfg["num_modes"])
        self.posterior_eps = float(self.source_cfg.get("posterior_eps", 1e-6))
        self.balance_loss_weight = float(self.source_cfg.get("balance_loss_weight", 0.1))
        self.entropy_loss_weight = float(self.source_cfg.get("entropy_loss_weight", 0.01))
        self.var_kl_loss_weight = float(self.source_cfg.get("var_kl_loss_weight", 1.0))
        self.target_variance = float(self.source_cfg.get("target_variance", 1.0))
        self.source_seed = int(self.source_cfg.get("source_seed", 17))

        artifact = load_gmm_artifact(self.source_cfg["gmm_stats_path"])
        if artifact.num_modes != self.num_modes:
            raise ValueError(
                f"GMM artifact num_modes={artifact.num_modes} does not match source.num_modes={self.num_modes}."
            )
        self.gmm_log_pi = Buffer(jnp.asarray(artifact.log_pi, dtype=jnp.float32))
        self.gmm_mu = Buffer(jnp.asarray(artifact.mu, dtype=jnp.float32))
        self.gmm_var = Buffer(jnp.asarray(artifact.var, dtype=jnp.float32))
        self.gmm_latent_mean = Buffer(jnp.asarray(artifact.latent_mean, dtype=jnp.float32))
        self.gmm_latent_std = Buffer(jnp.asarray(artifact.latent_std, dtype=jnp.float32))
        self.gmm_standardize_eps = float(artifact.standardize_eps)

        self.source_rngs = nnx.Rngs(
            self.source_seed,
            params=self.source_seed,
            noise=self.source_seed + 1,
            mode=self.source_seed + 2,
        )
        default_hidden_channels = max(128, min(512, int(network.in_channels) // 3))
        self.source_moe = SourceMoE(
            num_modes=self.num_modes,
            in_channels=int(network.in_channels),
            condition_dim=int(self.source_cfg.get("condition_dim", 16)),
            hidden_channels=int(self.source_cfg.get("hidden_channels", default_hidden_channels)),
            router_temperature=float(self.source_cfg.get("router_temperature", 2.0)),
            soft_moe=bool(self.source_cfg.get("soft_moe", True)),
            logvar_min=float(self.source_cfg.get("logvar_min", -8.0)),
            logvar_max=float(self.source_cfg.get("logvar_max", 4.0)),
            var_floor=float(self.source_cfg.get("var_floor", 1e-5)),
            rngs=self.source_rngs,
        )

    def _gmm_arrays(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Read fixed GMM stats through `.value` so JAX sees plain arrays under jit/pjit.
        return (
            self.gmm_log_pi.value,
            self.gmm_mu.value,
            self.gmm_var.value,
            self.gmm_latent_mean.value,
            self.gmm_latent_std.value,
        )

    def _posterior(self, x_data: jnp.ndarray) -> jnp.ndarray:
        gmm_log_pi, gmm_mu, gmm_var, gmm_latent_mean, gmm_latent_std = self._gmm_arrays()
        x_flat = x_data.reshape((x_data.shape[0], -1))
        posterior = posterior_from_stats(
            x_flat,
            latent_mean=gmm_latent_mean,
            latent_std=gmm_latent_std,
            standardize_eps=self.gmm_standardize_eps,
            log_pi=gmm_log_pi,
            mu=gmm_mu,
            var=gmm_var,
        )
        return jax.lax.stop_gradient(posterior)

    def _sample_modes(self, probs: jnp.ndarray) -> jnp.ndarray:
        logits = jnp.log(jnp.clip(probs, 1e-12, 1.0))
        indices = jax.random.categorical(self.source_rngs.mode(), logits, axis=-1)
        return jax.nn.one_hot(indices, self.num_modes, dtype=jnp.float32)

    def _sample_source_from_condition(
        self,
        condition_weights: jnp.ndarray,
        shape: tuple[int, ...],
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        z = jax.random.normal(self.source_rngs.noise(), shape, dtype=jnp.float32)
        moe_out = self.source_moe(z, condition_weights)
        eps = jax.random.normal(self.source_rngs.noise(), moe_out.mu.shape, dtype=moe_out.mu.dtype)
        source = moe_out.mu + jnp.exp(0.5 * moe_out.logvar) * eps
        source_metrics = summarize_router(moe_out.alpha)
        source_metrics["source_condition_max"] = jnp.mean(jnp.max(condition_weights, axis=-1))
        source_metrics["source_condition_entropy"] = entropy_loss(condition_weights)
        source_metrics["source_logvar_mean"] = jnp.mean(moe_out.logvar)
        source_metrics["source_var_mean"] = jnp.mean(jnp.exp(moe_out.logvar))
        return source, {
            "alpha": moe_out.alpha,
            "logits": moe_out.logits,
            "logvar": moe_out.logvar,
            "metrics": source_metrics,
        }

    def sample_source_prior(self, shape: tuple[int, ...]) -> jnp.ndarray:
        gmm_log_pi, *_rest = self._gmm_arrays()
        pi = jnp.broadcast_to(jnp.exp(gmm_log_pi)[None, :], (shape[0], self.num_modes))
        condition_weights = self._sample_modes(pi)
        source, _payload = self._sample_source_from_condition(condition_weights, shape)
        return source

    def loss(self, x: jnp.ndarray, *args, return_aux: bool = False, **kwargs) -> jnp.ndarray:
        t = self.sample_t((x.shape[0],))
        t = self.t_shift(t, jnp.sqrt(jnp.prod(jnp.asarray(x.shape[1:])) / self.t_shift_base))

        posterior = self._posterior(x)
        condition_weights = self._sample_modes(posterior)
        source, source_payload = self._sample_source_from_condition(condition_weights, x.shape)

        x_t = self.sample_x_t(x, source, t)
        target = self.target(x, source, t)
        net_out, features = self.network(
            (self.bcast_right(self.c_in(t), x_t) * x_t),
            t,
            *args,
            **kwargs,
        )

        loss_fm = self.mean_flat(jnp.square(net_out - target))
        loss_balance = balance_loss(source_payload["alpha"])
        loss_entropy = entropy_loss(source_payload["alpha"])
        loss_var = var_only_kld_loss(
            source_payload["logvar"],
            target_variance=self.target_variance,
        )
        total_loss = (
            loss_fm
            + self.balance_loss_weight * loss_balance
            - self.entropy_loss_weight * loss_entropy
            + self.var_kl_loss_weight * loss_var
        )

        aux_metrics = {
            "loss_balance": jnp.broadcast_to(loss_balance, loss_fm.shape),
            "loss_entropy": jnp.broadcast_to(loss_entropy, loss_fm.shape),
            "loss_var": jnp.broadcast_to(loss_var, loss_fm.shape),
            "loss_fm": loss_fm,
        }
        aux_metrics.update(source_payload["metrics"])
        gmm_log_pi, *_rest = self._gmm_arrays()
        aux_metrics["source_prior_max"] = jnp.max(jnp.exp(gmm_log_pi))

        loss_dict = {
            "loss": total_loss,
            "loss_fm": loss_fm,
            "loss_balance": aux_metrics["loss_balance"],
            "loss_entropy": aux_metrics["loss_entropy"],
            "loss_var": aux_metrics["loss_var"],
            "source_router_entropy": aux_metrics["source_router_entropy"],
            "source_router_max": aux_metrics["source_router_max"],
            "source_active_modes": aux_metrics["source_active_modes"],
            "source_condition_max": aux_metrics["source_condition_max"],
            "source_condition_entropy": aux_metrics["source_condition_entropy"],
            "source_prior_max": aux_metrics["source_prior_max"],
            "source_logvar_mean": aux_metrics["source_logvar_mean"],
            "source_var_mean": aux_metrics["source_var_mean"],
        }

        if return_aux:
            return total_loss, net_out, {
                "intermediate_features": features,
                "loss_dict": loss_dict,
            }
        return loss_dict
