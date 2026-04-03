from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

try:
    from .config_adapter import cfg_to_dict, load_repo_config
    from .moe1.gmm_utils import (
        choose_gmm_feature_extractor,
        extract_gmm_features,
        fit_diag_gmm,
        save_gmm_artifact,
    )
    from .stage1_runtime import _iter_batches, _load_batch, _load_stage1_encoder, list_image_files
except ImportError:
    from config_adapter import cfg_to_dict, load_repo_config
    from moe1.gmm_utils import (
        choose_gmm_feature_extractor,
        extract_gmm_features,
        fit_diag_gmm,
        save_gmm_artifact,
    )
    from stage1_runtime import _iter_batches, _load_batch, _load_stage1_encoder, list_image_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a diagonal GMM artifact for the moe1 learned source distribution."
    )
    parser.add_argument("--config", required=True, help="Path to the repo YAML config.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory used to fit the source GMM.")
    parser.add_argument("--output", type=Path, required=True, help="Destination .npz artifact.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0, help="Threaded image loading workers.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of images.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--backend-dir", default=None)
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--weight-prior", type=float, default=1e-2)
    parser.add_argument("--var-floor", type=float, default=1e-5)
    parser.add_argument("--em-iters", type=int, default=100)
    parser.add_argument("--em-tol", type=float, default=1e-4)
    parser.add_argument("--em-restarts", type=int, default=3)
    parser.add_argument("--dead-count-threshold", type=float, default=1.0)
    parser.add_argument("--active-mode-fraction-threshold", type=float, default=0.01)
    parser.add_argument("--standardize-eps", type=float, default=1e-6)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--feature-extractor",
        choices=["auto", "flatten", "spatial_mean"],
        default="auto",
        help=(
            "How to convert NHWC latents into GMM features. "
            "'auto' picks spatial_mean for very large RAE latents to avoid OOM."
        ),
    )
    parser.add_argument("--set", dest="set_values", action="append", default=[], help="OmegaConf override.")
    return parser


def _infer_artifact_metadata(args: argparse.Namespace) -> tuple[str, float]:
    repo_cfg, _config_path = load_repo_config(args.config, args.set_values)
    stage1_cfg = cfg_to_dict(repo_cfg.get("stage_1"))
    stage1_target = str(stage1_cfg.get("target", "stage1.RAE")).strip().lower()
    if "stabilityvae" in stage1_target or "stability_vae" in stage1_target:
        return "stabilityvae_scaled_output", 0.18215
    return "rae_encoded_output", 1.0


def run_build_source_gmm(args: argparse.Namespace) -> Path:
    import jax.numpy as jnp

    encoder = _load_stage1_encoder(args)
    latent_semantics, vae_scale_factor = _infer_artifact_metadata(args)
    image_paths = list_image_files(Path(args.input).expanduser().resolve())
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images selected from: {args.input}")

    feature_batches: list[np.ndarray] = []
    latent_shape: tuple[int, ...] | None = None
    feature_extractor: str | None = None
    total = len(image_paths)
    processed = 0

    for batch_idx, batch_paths in enumerate(_iter_batches(image_paths, args.batch_size), start=1):
        batch = _load_batch(batch_paths, args.num_workers)
        latents = np.asarray(encoder.encode(jnp.asarray(batch)), dtype=np.float32)
        if latent_shape is None:
            latent_shape = tuple(int(dim) for dim in latents.shape[1:])
            feature_extractor = choose_gmm_feature_extractor(
                latent_shape,
                requested=args.feature_extractor,
            )
            feature_shape = tuple(int(dim) for dim in extract_gmm_features(latents[:1], feature_extractor=feature_extractor).shape[1:])
            print(
                "[source-gmm] using feature_extractor="
                f"{feature_extractor} for latent_shape={latent_shape} -> feature_shape={feature_shape}"
            )
        feature_batches.append(
            np.asarray(
                extract_gmm_features(latents, feature_extractor=feature_extractor or "flatten"),
                dtype=np.float32,
            )
        )
        processed += latents.shape[0]
        if args.log_every > 0 and (batch_idx == 1 or batch_idx % args.log_every == 0 or processed == total):
            print(f"[source-gmm] processed {processed}/{total} images")

    if latent_shape is None or feature_extractor is None:
        raise RuntimeError("Failed to infer latent shape while building source GMM.")

    latents_flat = np.concatenate(feature_batches, axis=0)
    artifact = fit_diag_gmm(
        latents_flat,
        num_modes=args.num_modes,
        seed=args.seed,
        weight_prior=args.weight_prior,
        var_floor=args.var_floor,
        em_iters=args.em_iters,
        em_tol=args.em_tol,
        em_restarts=args.em_restarts,
        dead_count_threshold=args.dead_count_threshold,
        active_mode_fraction_threshold=args.active_mode_fraction_threshold,
        standardize_eps=args.standardize_eps,
        chunk_size=args.chunk_size,
        latent_semantics=latent_semantics,
        vae_scale_factor=vae_scale_factor,
        feature_extractor=feature_extractor,
    )
    artifact = dataclasses.replace(
        artifact,
        latent_shape=latent_shape,
        layout="NHWC",
        feature_extractor=feature_extractor,
    )
    return save_gmm_artifact(args.output, artifact)


def main() -> None:
    args = build_parser().parse_args()
    output_path = run_build_source_gmm(args)
    print(f"Saved source GMM artifact to {output_path}")


if __name__ == "__main__":
    main()
