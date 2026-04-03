from .gmm_utils import (
    GMMArtifact,
    block_average_pool_nhwc,
    choose_gmm_feature_extractor,
    compute_active_modes,
    compute_standardization_stats,
    extract_gmm_features,
    fit_diag_gmm,
    flatten_latents_nhwc,
    load_gmm_artifact,
    posterior_from_stats,
    project_channels_nhwc,
    pyramid_16k_latents_nhwc,
    save_gmm_artifact,
    standardize_latents,
)
from .source_losses import balance_loss, entropy_loss, summarize_router, var_only_kld_loss

try:
    from .source_moe import SourceMoE
except ModuleNotFoundError:  # pragma: no cover - optional in non-JAX envs
    SourceMoE = None

__all__ = [
    "GMMArtifact",
    "SourceMoE",
    "balance_loss",
    "block_average_pool_nhwc",
    "choose_gmm_feature_extractor",
    "compute_active_modes",
    "compute_standardization_stats",
    "entropy_loss",
    "extract_gmm_features",
    "fit_diag_gmm",
    "flatten_latents_nhwc",
    "load_gmm_artifact",
    "posterior_from_stats",
    "project_channels_nhwc",
    "pyramid_16k_latents_nhwc",
    "save_gmm_artifact",
    "standardize_latents",
    "summarize_router",
    "var_only_kld_loss",
]
