from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src_jax.vendor import _apply_backend_compat_patches, _sync_backend_overlay


class JaxVendorPatchTests(unittest.TestCase):
    def test_backend_patches_preserve_model_initialized_ema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            backend_dir = Path(tmp_dir)
            (backend_dir / "networks" / "encoders").mkdir(parents=True, exist_ok=True)
            (backend_dir / "utils").mkdir(parents=True, exist_ok=True)

            (backend_dir / "networks" / "encoders" / "dino.py").write_text(
                "from transformers import FlaxDinov2Model, AutoImageProcessor\n",
                encoding="utf-8",
            )
            (backend_dir / "networks" / "encoders" / "dino_w_register.py").write_text(
                "from transformers import Dinov2WithRegistersModel\n",
                encoding="utf-8",
            )
            (backend_dir / "networks" / "encoders" / "utils.py").write_text(
                "\"\"\"File containing utility functions for the encoder.\"\"\"\n\n"
                "# built-in libs\n"
                "import math\n\n"
                "# external libs\n"
                "from google.cloud import storage\n\n"
                "def download_blob(bucket_name, source_blob_name, destination_file_name):\n"
                "    \"\"\"Downloads a blob from the bucket.\"\"\"\n"
                "    storage_client = storage.Client()\n"
                "    bucket = storage_client.bucket(bucket_name)\n"
                "    blob = bucket.blob(source_blob_name)\n"
                "    blob.download_to_filename(destination_file_name)\n",
                encoding="utf-8",
            )
            (backend_dir / "utils" / "ema.py").write_text(
                "class EMA:\n"
                "    def __init__(self, net, decay):\n"
                "        self.ema = copy.deepcopy(net)\n"
                "        ema_state = jax.tree.map(lambda x: jnp.zeros_like(x), nnx.state(net, nnx.Param))\n"
                "        nnx.update(self.ema, ema_state)\n"
                "        self.ema.eval()\n"
                "        self.decay = decay\n",
                encoding="utf-8",
            )
            (backend_dir / "networks" / "encoders" / "sd_vae.py").write_text(
                "class StabilityVAE:\n"
                "    def initialize(self):\n"
                "        ckpt_path = os.path.join(Path(__file__).parent, self.pretrained_path)\n"
                "        if not os.path.exists(ckpt_path):\n"
                "            utils.download_blob('will-data', 'stats/vae_trial1.pkl', ckpt_path)\n"
                "            \n"
                "        with open(ckpt_path, 'rb') as f:\n"
                "            params = pickle.load(f)\n"
                "        return params\n",
                encoding="utf-8",
            )

            _apply_backend_compat_patches(backend_dir)

            ema_text = (backend_dir / "utils" / "ema.py").read_text(encoding="utf-8")
            self.assertIn("self.ema = copy.deepcopy(net)", ema_text)
            self.assertIn("self.ema.eval()", ema_text)
            self.assertNotIn("jnp.zeros_like", ema_text)
            self.assertNotIn("nnx.update(self.ema, ema_state)", ema_text)

            sd_vae_text = (backend_dir / "networks" / "encoders" / "sd_vae.py").read_text(encoding="utf-8")
            self.assertIn("ensure_stability_vae_checkpoint", sd_vae_text)
            self.assertIn("StabilityVAE checkpoint not found", sd_vae_text)
            self.assertNotIn("utils.download_blob('will-data'", sd_vae_text)

    def test_backend_overlay_sync_copies_files_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            backend_dir = Path(tmp_dir) / "backend"
            overlay_dir = Path(tmp_dir) / "overlay"
            moe1_dir = Path(tmp_dir) / "moe1"
            backend_dir.mkdir(parents=True, exist_ok=True)
            (overlay_dir / "interfaces").mkdir(parents=True, exist_ok=True)
            moe1_dir.mkdir(parents=True, exist_ok=True)

            (overlay_dir / "interfaces" / "continuous_moe1.py").write_text(
                "class Dummy:\n    pass\n",
                encoding="utf-8",
            )
            (moe1_dir / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")

            with mock.patch("src_jax.vendor.OVERLAY_SOURCE_DIR", overlay_dir), mock.patch(
                "src_jax.vendor.MOE1_SOURCE_DIR",
                moe1_dir,
            ):
                _sync_backend_overlay(backend_dir)
                _sync_backend_overlay(backend_dir)

            self.assertTrue((backend_dir / "interfaces" / "continuous_moe1.py").exists())
            self.assertTrue((backend_dir / "moe1" / "__init__.py").exists())
            manifest_path = backend_dir / ".rae_jax_overlay_manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest_text = manifest_path.read_text(encoding="utf-8")
            self.assertIn("overlay_hash", manifest_text)


if __name__ == "__main__":
    unittest.main()
