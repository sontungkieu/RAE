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
    _rename_backend_vae_keys,
    ensure_stability_vae_checkpoint,
)


class StabilityVAEAssetTests(unittest.TestCase):
    def test_rename_backend_vae_keys_matches_backend_names(self) -> None:
        payload = {
            "encoder": {
                "down_blocks_0": {
                    "downsamplers_0": {
                        "conv": {
                            "kernel": 1,
                        }
                    }
                }
            },
            "decoder": {
                "up_blocks_0": {
                    "upsamplers_0": {
                        "conv": {
                            "bias": 2,
                        }
                    }
                }
            },
        }

        renamed = _rename_backend_vae_keys(payload)

        self.assertIn("downsample", renamed["encoder"]["down_blocks_0"])
        self.assertNotIn("downsamplers_0", renamed["encoder"]["down_blocks_0"])
        self.assertIn("upsample", renamed["decoder"]["up_blocks_0"])
        self.assertNotIn("upsamplers_0", renamed["decoder"]["up_blocks_0"])
        self.assertEqual(renamed["encoder"]["down_blocks_0"]["downsample"]["conv"]["kernel"], 1)
        self.assertEqual(renamed["decoder"]["up_blocks_0"]["upsample"]["conv"]["bias"], 2)

    def test_materialize_checkpoint_from_diffusers_params(self) -> None:
        class FakeFlaxAutoencoderKL:
            call_args = None

            @classmethod
            def from_pretrained(cls, repo_id, revision, from_pt, dtype):
                cls.call_args = (repo_id, revision, from_pt, str(dtype))
                return object(), {
                    "encoder": {
                        "down_blocks_0": {
                            "downsamplers_0": {
                                "conv": {"weight": 1},
                            }
                        }
                    },
                    "decoder": {
                        "up_blocks_0": {
                            "upsamplers_0": {
                                "conv": {"weight": 2},
                            }
                        }
                    },
                }

        fake_diffusers = types.SimpleNamespace(FlaxAutoencoderKL=FakeFlaxAutoencoderKL)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "vae_trial1.pkl"
            with mock.patch.dict(sys.modules, {"diffusers": fake_diffusers}):
                resolved = ensure_stability_vae_checkpoint(output_path)

            self.assertEqual(resolved, output_path.resolve())
            self.assertTrue(resolved.exists())
            with resolved.open("rb") as handle:
                payload = pickle.load(handle)

            self.assertEqual(
                payload,
                {
                    "encoder": {
                        "down_blocks_0": {
                            "downsample": {
                                "conv": {"weight": 1},
                            }
                        }
                    },
                    "decoder": {
                        "up_blocks_0": {
                            "upsample": {
                                "conv": {"weight": 2},
                            }
                        }
                    },
                },
            )
            self.assertIsNotNone(FakeFlaxAutoencoderKL.call_args)
            self.assertEqual(FakeFlaxAutoencoderKL.call_args[0], STABILITY_VAE_REPO_ID)
            self.assertEqual(FakeFlaxAutoencoderKL.call_args[1], STABILITY_VAE_REVISION)
            self.assertTrue(FakeFlaxAutoencoderKL.call_args[2])


if __name__ == "__main__":
    unittest.main()
