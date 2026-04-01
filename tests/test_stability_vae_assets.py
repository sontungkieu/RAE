from __future__ import annotations

import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src_jax.stability_vae_assets import (
    STABILITY_VAE_REPO_ID,
    STABILITY_VAE_REVISION,
    ensure_stability_vae_checkpoint,
)


class StabilityVAEAssetTests(unittest.TestCase):
    def test_materialize_checkpoint_from_diffusers_params(self) -> None:
        class FakeFlaxAutoencoderKL:
            call_args = None

            @classmethod
            def from_pretrained(cls, repo_id, revision, from_pt, dtype):
                cls.call_args = (repo_id, revision, from_pt, str(dtype))
                return object(), {"encoder": {"weight": 1}, "decoder": {"weight": 2}}

        fake_diffusers = types.SimpleNamespace(FlaxAutoencoderKL=FakeFlaxAutoencoderKL)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "vae_trial1.pkl"
            with mock.patch.dict(sys.modules, {"diffusers": fake_diffusers}):
                resolved = ensure_stability_vae_checkpoint(output_path)

            self.assertEqual(resolved, output_path.resolve())
            self.assertTrue(resolved.exists())
            with resolved.open("rb") as handle:
                payload = pickle.load(handle)

            self.assertEqual(payload, {"encoder": {"weight": 1}, "decoder": {"weight": 2}})
            self.assertIsNotNone(FakeFlaxAutoencoderKL.call_args)
            self.assertEqual(FakeFlaxAutoencoderKL.call_args[0], STABILITY_VAE_REPO_ID)
            self.assertEqual(FakeFlaxAutoencoderKL.call_args[1], STABILITY_VAE_REVISION)
            self.assertTrue(FakeFlaxAutoencoderKL.call_args[2])


if __name__ == "__main__":
    unittest.main()
