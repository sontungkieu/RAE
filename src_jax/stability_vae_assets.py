from __future__ import annotations

import pickle
import tempfile
import warnings
from pathlib import Path
from typing import Any


STABILITY_VAE_REPO_ID = "stabilityai/sd-vae-ft-mse"
STABILITY_VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
STABILITY_VAE_DIFFUSERS_REQUIREMENT = "diffusers==0.37.1"


def _load_flax_autoencoder_kl():
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Flax classes are deprecated.*",
                category=FutureWarning,
            )
            from diffusers import FlaxAutoencoderKL
    except ImportError as exc:
        raise ImportError(
            "diffusers is required to materialize the default StabilityVAE checkpoint. "
            "Install it with `uv sync` or "
            f"`uv pip install {STABILITY_VAE_DIFFUSERS_REQUIREMENT}`, "
            "or point `stage_1.params.pretrained_path` at an existing `vae_trial1.pkl`."
        ) from exc
    return FlaxAutoencoderKL


def _set_diffusers_verbosity_error() -> object | None:
    try:
        from diffusers.utils import logging as diffusers_logging
    except ImportError:
        return None

    previous = diffusers_logging.get_verbosity()
    diffusers_logging.set_verbosity_error()
    return previous


def _restore_diffusers_verbosity(previous: object | None) -> None:
    if previous is None:
        return
    from diffusers.utils import logging as diffusers_logging

    diffusers_logging.set_verbosity(previous)


def _to_builtin_tree(node: Any) -> Any:
    if hasattr(node, "items"):
        return {key: _to_builtin_tree(value) for key, value in node.items()}
    return node


def ensure_stability_vae_checkpoint(destination: str | Path) -> Path:
    output_path = Path(destination).expanduser().resolve()
    if output_path.exists():
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    FlaxAutoencoderKL = _load_flax_autoencoder_kl()

    import jax.numpy as jnp

    previous_diffusers_verbosity = _set_diffusers_verbosity_error()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Flax classes are deprecated.*",
                category=FutureWarning,
            )
            _model, params = FlaxAutoencoderKL.from_pretrained(
                STABILITY_VAE_REPO_ID,
                revision=STABILITY_VAE_REVISION,
                from_pt=True,
                dtype=jnp.float32,
            )
    finally:
        _restore_diffusers_verbosity(previous_diffusers_verbosity)

    payload = _to_builtin_tree(params)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path.replace(output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return output_path
