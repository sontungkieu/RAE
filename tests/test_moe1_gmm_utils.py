from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from src_jax.moe1.gmm_utils import (
        choose_gmm_feature_extractor,
        compute_active_modes,
        extract_gmm_features,
        fit_diag_gmm,
        flatten_latents_nhwc,
        load_gmm_artifact,
        posterior_from_stats,
        save_gmm_artifact,
    )
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    choose_gmm_feature_extractor = None
    compute_active_modes = None
    extract_gmm_features = None
    fit_diag_gmm = None
    flatten_latents_nhwc = None
    load_gmm_artifact = None
    posterior_from_stats = None
    save_gmm_artifact = None
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"Missing optional dependency: {_IMPORT_ERROR}")
class Moe1GmmUtilsTests(unittest.TestCase):
    def test_flatten_latents_nhwc_preserves_batch(self) -> None:
        latents = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        flattened = flatten_latents_nhwc(latents)
        self.assertEqual(flattened.shape, (2, 60))
        np.testing.assert_allclose(flattened[0], latents[0].reshape(-1))

    def test_extract_gmm_features_supports_spatial_mean(self) -> None:
        latents = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        pooled = extract_gmm_features(latents, feature_extractor="spatial_mean")
        self.assertEqual(pooled.shape, (2, 5))
        np.testing.assert_allclose(pooled[0], latents[0].mean(axis=(0, 1)))

    def test_choose_gmm_feature_extractor_prefers_spatial_mean_for_large_rae_latents(self) -> None:
        self.assertEqual(
            choose_gmm_feature_extractor((16, 16, 768), requested="auto"),
            "spatial_mean",
        )
        self.assertEqual(
            choose_gmm_feature_extractor((32, 32, 4), requested="auto"),
            "flatten",
        )

    def test_fit_diag_gmm_round_trip_and_posterior_normalization(self) -> None:
        rng = np.random.default_rng(0)
        cluster_a = rng.normal(loc=-1.5, scale=0.2, size=(32, 8)).astype(np.float32)
        cluster_b = rng.normal(loc=1.5, scale=0.2, size=(32, 8)).astype(np.float32)
        latents = np.concatenate([cluster_a, cluster_b], axis=0)

        artifact = fit_diag_gmm(
            latents,
            num_modes=2,
            seed=3,
            em_iters=40,
            em_restarts=2,
            chunk_size=16,
            latent_semantics="rae_encoded_output",
            vae_scale_factor=1.0,
        )
        posterior = posterior_from_stats(
            latents[:6],
            latent_mean=artifact.latent_mean,
            latent_std=artifact.latent_std,
            standardize_eps=artifact.standardize_eps,
            log_pi=artifact.log_pi,
            mu=artifact.mu,
            var=artifact.var,
        )

        self.assertEqual(artifact.num_modes, 2)
        self.assertGreaterEqual(artifact.active_modes, 1)
        np.testing.assert_allclose(np.asarray(posterior).sum(axis=-1), 1.0, atol=1e-5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "source_stats.npz"
            save_gmm_artifact(path, artifact)
            loaded = load_gmm_artifact(path)

        np.testing.assert_allclose(loaded.log_pi, artifact.log_pi)
        np.testing.assert_allclose(loaded.mu, artifact.mu)
        np.testing.assert_allclose(loaded.var, artifact.var)
        self.assertEqual(loaded.active_modes, artifact.active_modes)
        self.assertEqual(loaded.latent_semantics, "rae_encoded_output")
        self.assertEqual(loaded.vae_scale_factor, 1.0)
        self.assertEqual(loaded.feature_extractor, "flatten")

    def test_compute_active_modes_uses_fraction_threshold(self) -> None:
        counts = np.asarray([50.0, 25.0, 20.0, 5.0], dtype=np.float32)
        self.assertEqual(
            compute_active_modes(counts, 100, fraction_threshold=0.1),
            3,
        )


if __name__ == "__main__":
    unittest.main()
