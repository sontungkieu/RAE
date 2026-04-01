from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from src_jax.config_adapter import build_backend_config_dict, load_repo_config, maybe_convert_fid_reference
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    build_backend_config_dict = None
    load_repo_config = None
    maybe_convert_fid_reference = None
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"Missing optional dependency: {_IMPORT_ERROR}")
class JaxAdapterTests(unittest.TestCase):
    def _write_temp_config(self, text: str, *, filename: str = "config.yaml") -> str:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config_path = Path(temp_dir.name) / filename
        config_path.write_text(text, encoding="utf-8")
        return str(config_path)

    def test_build_backend_config_maps_sitdh_to_lightning_ddt(self) -> None:
        repo_cfg, config_path = load_repo_config("configs/stage2/training/ImageNet256/SiTDH-XL_DINOv2-B.yaml")
        backend_cfg = build_backend_config_dict(
            repo_cfg,
            config_path=config_path,
            mode="train",
            data_path="/tmp/imagenet",
            precision="bf16",
            seed=7,
            num_train_samples=1281167,
            enable_eval=False,
        )

        self.assertEqual(backend_cfg["network_class"], "lightning_ddt")
        self.assertEqual(backend_cfg["network"]["num_encoder_blocks"], 28)
        self.assertEqual(backend_cfg["network"]["num_decoder_blocks"], 2)
        self.assertEqual(backend_cfg["network"]["encoder_hidden_size"], 1152)
        self.assertEqual(backend_cfg["network"]["decoder_hidden_size"], 2048)
        self.assertEqual(backend_cfg["network"]["encoder_num_heads"], 16)
        self.assertEqual(backend_cfg["network"]["decoder_num_heads"], 16)
        self.assertEqual(backend_cfg["interface_class"], "sit")
        self.assertEqual(backend_cfg["dtype"], "bfloat16")
        self.assertEqual(backend_cfg["data"]["data_dir"], "/tmp/imagenet")

    def test_build_backend_config_maps_sit_to_lightning_dit(self) -> None:
        config_path = self._write_temp_config(
            """
stage_1:
  target: stage1.StabilityVAE
  params:
    sample_size: 256
    latent_channels: 4
    downsample_factor: 8
stage_2:
  target: stage2.models.SiT.SiT
  params:
    input_size: 32
    patch_size: 2
    in_channels: 4
    hidden_size: 768
    depth: 12
    num_heads: 12
    mlp_ratio: 4.0
    class_dropout_prob: 0.0
    num_classes: 1
transport:
  params:
    time_dist_type: uniform
sampler:
  params:
    sampling_method: euler
misc:
  latent_size: [4, 32, 32]
  num_classes: 1
"""
        )
        repo_cfg, resolved_config_path = load_repo_config(config_path)
        backend_cfg = build_backend_config_dict(
            repo_cfg,
            config_path=resolved_config_path,
            mode="train",
            data_path="/tmp/celebahq256",
            precision="bf16",
            seed=7,
            num_train_samples=30_000,
            enable_eval=False,
        )

        self.assertEqual(backend_cfg["encoder_class"], "StabilityVAE")
        self.assertEqual(backend_cfg["network_class"], "lightning_dit")
        self.assertEqual(backend_cfg["network"]["input_size"], 32)
        self.assertEqual(backend_cfg["network"]["patch_size"], 2)
        self.assertEqual(backend_cfg["network"]["in_channels"], 4)
        self.assertEqual(backend_cfg["network"]["hidden_size"], 768)
        self.assertEqual(backend_cfg["network"]["depth"], 12)
        self.assertEqual(backend_cfg["network"]["num_heads"], 12)
        self.assertEqual(backend_cfg["interface_class"], "sit")

    def test_build_backend_config_maps_source_block_to_moe1_interface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            gmm_stats_path = Path(tmp_dir) / "source_stats.npz"
            np.savez(
                gmm_stats_path,
                log_pi=np.zeros((4,), dtype=np.float32),
                mu=np.zeros((4, 8), dtype=np.float32),
                var=np.ones((4, 8), dtype=np.float32),
                latent_mean=np.zeros((8,), dtype=np.float32),
                latent_std=np.ones((8,), dtype=np.float32),
                standardize_eps=np.float32(1e-6),
                latent_shape=np.asarray([2, 2, 2], dtype=np.int32),
                layout=np.asarray("NHWC"),
                sample_posterior=np.asarray(True),
                latent_semantics=np.asarray("stabilityvae_scaled_output"),
                vae_scale_factor=np.float32(0.18215),
                count=np.int64(32),
                num_modes=np.int32(4),
                active_modes=np.int32(2),
                train_nll=np.float32(1.23),
                active_mode_fraction_threshold=np.float32(0.01),
                final_counts=np.asarray([12, 10, 6, 4], dtype=np.float32),
                n_iter=np.int32(7),
            )
            config_path = self._write_temp_config(
                f"""
stage_1:
  target: stage1.StabilityVAE
  params:
    sample_size: 256
    latent_channels: 4
    downsample_factor: 8
stage_2:
  target: stage2.models.SiT.SiT
  params:
    input_size: 32
    patch_size: 2
    in_channels: 4
    hidden_size: 768
    depth: 12
    num_heads: 12
    num_classes: 1
transport:
  params:
    time_dist_type: uniform
sampler:
  params:
    sampling_method: euler
misc:
  latent_size: [4, 32, 32]
  num_classes: 1
source:
  enabled: true
  kind: gmm_moe1
  gmm_stats_path: {gmm_stats_path.as_posix()}
  num_modes: 4
  condition_dim: 32
  hidden_channels: 96
"""
            )
            repo_cfg, resolved_config_path = load_repo_config(config_path)
            backend_cfg = build_backend_config_dict(
                repo_cfg,
                config_path=resolved_config_path,
                mode="train",
                data_path="/tmp/celebahq256",
                precision="bf16",
                seed=7,
                num_train_samples=30_000,
                enable_eval=False,
            )

            self.assertEqual(backend_cfg["interface_class"], "sit_gmm_moe1")
            self.assertIn("source", backend_cfg["interface"])
            self.assertEqual(backend_cfg["interface"]["source"]["gmm_stats_path"], str(gmm_stats_path))
            self.assertEqual(backend_cfg["interface"]["source"]["num_modes"], 4)
            self.assertEqual(backend_cfg["interface"]["source"]["condition_dim"], 32)
            self.assertEqual(backend_cfg["interface"]["source"]["hidden_channels"], 96)

    def test_npz_fid_reference_is_converted_to_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_path = Path(tmp_dir) / "ref_stats.npz"
            np.savez(ref_path, mu=np.zeros(2048, dtype=np.float64), sigma=np.eye(2048, dtype=np.float64))

            out_path = Path(maybe_convert_fid_reference(str(ref_path)))
            self.assertTrue(out_path.exists())
            self.assertEqual(out_path.suffix, ".pkl")

    def test_eval_block_maps_validation_and_fid_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_path = Path(tmp_dir) / "ref_stats.npz"
            np.savez(ref_path, mu=np.zeros(2048, dtype=np.float64), sigma=np.eye(2048, dtype=np.float64))

            repo_cfg, config_path = load_repo_config(
                "configs/stage2/training/ImageNet256/SiTDH-S_DINOv2-B.yaml",
                overrides=[
                    "eval.data_path=/tmp/imagenet_val",
                    "eval.eval_every=5000",
                    f"eval.fid_ref={ref_path}",
                    "eval.fid_every=25000",
                    "eval.fid_num_samples=4096",
                    "eval.batch_size=8",
                    "eval.num_workers=2",
                    "eval.max_batches=16",
                    "eval.eval_model=true",
                ],
            )
            backend_cfg = build_backend_config_dict(
                repo_cfg,
                config_path=config_path,
                mode="train",
                data_path="/tmp/imagenet_train",
                precision="bf16",
                seed=7,
                num_train_samples=1281167,
                enable_eval=True,
            )

            self.assertTrue(backend_cfg["eval"]["on"])
            self.assertTrue(backend_cfg["eval"]["loss_on"])
            self.assertEqual(backend_cfg["eval"]["data_dir"], "/tmp/imagenet_val")
            self.assertEqual(backend_cfg["eval"]["loss_every_steps"], 5000)
            self.assertEqual(backend_cfg["eval"]["loss_batch_size"], 8)
            self.assertEqual(backend_cfg["eval"]["num_workers"], 2)
            self.assertEqual(backend_cfg["eval"]["max_batches"], 16)
            self.assertTrue(backend_cfg["eval"]["eval_model"])
            self.assertTrue(backend_cfg["eval"]["fid_on"])
            self.assertTrue(str(backend_cfg["data"]["stat_dir"]).endswith(".pkl"))

    def test_infer_network_class_accepts_sitdh_target(self) -> None:
        repo_cfg, config_path = load_repo_config(
            "configs/stage2/training/ImageNet256/SiTDH-S_DINOv2-B.yaml",
        )
        backend_cfg = build_backend_config_dict(
            repo_cfg,
            config_path=config_path,
            mode="train",
            data_path="/tmp/imagenet",
            precision="bf16",
            seed=7,
            num_train_samples=1281167,
            enable_eval=False,
        )

        self.assertEqual(backend_cfg["network_class"], "lightning_ddt")
        self.assertEqual(backend_cfg["network"]["num_encoder_blocks"], 12)
        self.assertEqual(backend_cfg["network"]["num_decoder_blocks"], 2)
        self.assertEqual(backend_cfg["network"]["encoder_hidden_size"], 384)
        self.assertEqual(backend_cfg["network"]["decoder_hidden_size"], 2048)
        self.assertEqual(backend_cfg["network"]["encoder_num_heads"], 6)
        self.assertEqual(backend_cfg["network"]["decoder_num_heads"], 16)

    def test_stage1_stabilityvae_infers_latent_geometry_without_stage2(self) -> None:
        config_path = self._write_temp_config(
            """
stage_1:
  target: stage1.StabilityVAE
  params:
    sample_size: 256
    latent_channels: 4
    downsample_factor: 8
    raw_mean: [0.865, -0.278, 0.216, 0.374]
    raw_std: [4.86, 5.32, 3.94, 3.99]
    final_mean: 0.0
    final_std: 0.5
transport:
  params:
    time_dist_type: uniform
sampler:
  params:
    sampling_method: euler
"""
        )
        repo_cfg, resolved_config_path = load_repo_config(config_path)
        backend_cfg = build_backend_config_dict(
            repo_cfg,
            config_path=resolved_config_path,
            mode="sample",
            data_path="/tmp/celebahq256",
            precision="bf16",
            seed=7,
            num_train_samples=30_000,
            enable_eval=False,
            require_stage2=False,
        )

        self.assertEqual(backend_cfg["encoder_class"], "StabilityVAE")
        self.assertEqual(backend_cfg["data"]["image_size"], 256)
        self.assertEqual(backend_cfg["encoder"]["latent_channels"], 4)
        self.assertEqual(backend_cfg["encoder"]["downsample_factor"], 8)
        self.assertEqual(backend_cfg["network"]["input_size"], 32)
        self.assertEqual(backend_cfg["network"]["in_channels"], 4)
        self.assertEqual(backend_cfg["sampler"]["sampling_time_kwargs"]["t_shift_cur"], 4096)
        self.assertNotIn("stats_path", backend_cfg["encoder"])
        self.assertNotIn("pretrained_path", backend_cfg["encoder"])
        self.assertNotIn("pretrained_model_name_or_path", backend_cfg["encoder"])

    def test_stage1_stabilityvae_forwards_optional_pretrained_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_path = Path(tmp_dir) / "vae_trial1.pkl"
            weights_path.write_bytes(b"stub")
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                f"""
stage_1:
  target: stage1.StabilityVAE
  params:
    sample_size: 256
    latent_channels: 4
    downsample_factor: 8
    pretrained_path: {weights_path.name}
""".strip()
                + "\n",
                encoding="utf-8",
            )
            repo_cfg, resolved_config_path = load_repo_config(str(config_path))
            backend_cfg = build_backend_config_dict(
                repo_cfg,
                config_path=resolved_config_path,
                mode="sample",
                data_path="/tmp/celebahq256",
                precision="bf16",
                seed=7,
                num_train_samples=30_000,
                enable_eval=False,
                require_stage2=False,
            )

            self.assertEqual(backend_cfg["encoder"]["pretrained_path"], str(weights_path.resolve()))

    def test_train_data_dir_normalizes_split_path_back_to_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "celebahq256_imgfolder"
            for split_name in ("train", "val", "test"):
                (root / split_name / "face").mkdir(parents=True, exist_ok=True)

            repo_cfg, config_path = load_repo_config(
                "configs/stage2/training/ImageNet256/SiTDH-S_DINOv2-B.yaml",
            )
            backend_cfg = build_backend_config_dict(
                repo_cfg,
                config_path=config_path,
                mode="train",
                data_path=str(root / "train"),
                precision="bf16",
                seed=7,
                num_train_samples=1281167,
                enable_eval=False,
            )

            self.assertEqual(backend_cfg["data"]["data_dir"], str(root.resolve()))

    def test_training_diagnostics_flags_are_forwarded(self) -> None:
        repo_cfg, config_path = load_repo_config(
            "configs/stage2/training/ImageNet256/SiTDH-S_DINOv2-B.yaml",
            overrides=[
                "training.log_rae_latent_stats=true",
                "training.log_activation_stats=true",
            ],
        )
        backend_cfg = build_backend_config_dict(
            repo_cfg,
            config_path=config_path,
            mode="train",
            data_path="/tmp/imagenet",
            precision="bf16",
            seed=7,
            num_train_samples=1281167,
            enable_eval=False,
        )

        self.assertTrue(backend_cfg["diagnostics"]["log_rae_latent_stats"])
        self.assertTrue(backend_cfg["diagnostics"]["log_activation_stats"])

    def test_prefetch_factors_are_forwarded(self) -> None:
        repo_cfg, config_path = load_repo_config(
            "configs/stage2/training/ImageNet256/SiTDH-S_DINOv2-B.yaml",
            overrides=[
                "training.prefetch_factor=8",
                "eval.data_path=/tmp/imagenet_val",
                "eval.eval_every=5000",
                "eval.prefetch_factor=3",
            ],
        )
        backend_cfg = build_backend_config_dict(
            repo_cfg,
            config_path=config_path,
            mode="train",
            data_path="/tmp/imagenet_train",
            precision="bf16",
            seed=7,
            num_train_samples=1281167,
            enable_eval=True,
        )

        self.assertEqual(backend_cfg["data"]["prefetch_factor"], 8)
        self.assertEqual(backend_cfg["eval"]["prefetch_factor"], 3)

    def test_random_flip_flags_are_forwarded(self) -> None:
        config_path = self._write_temp_config(
            """
stage_1:
  target: stage1.StabilityVAE
  params:
    sample_size: 256
    latent_channels: 4
    downsample_factor: 8
stage_2:
  target: stage2.models.SiT.SiT
  params:
    input_size: 32
    patch_size: 2
    in_channels: 4
misc:
  latent_size: [4, 32, 32]
training:
  random_flip: true
eval:
  data_path: /tmp/celebahq256/val
  eval_every: 5000
  random_flip: false
"""
        )
        repo_cfg, resolved_config_path = load_repo_config(config_path)
        backend_cfg = build_backend_config_dict(
            repo_cfg,
            config_path=resolved_config_path,
            mode="train",
            data_path="/tmp/celebahq256",
            precision="bf16",
            seed=7,
            num_train_samples=30_000,
            enable_eval=True,
        )

        self.assertTrue(backend_cfg["data"]["random_flip"])
        self.assertFalse(backend_cfg["eval"]["random_flip"])

    def test_celebahq_configs_default_training_random_flip_to_true(self) -> None:
        config_path = self._write_temp_config(
            """
stage_1:
  target: stage1.StabilityVAE
  params:
    sample_size: 256
    latent_channels: 4
    downsample_factor: 8
stage_2:
  target: stage2.models.SiT.SiT
  params:
    input_size: 32
    patch_size: 2
    in_channels: 4
misc:
  latent_size: [4, 32, 32]
""",
            filename="CelebAHQ256_SiT-B_StabilityVAE.yaml",
        )
        repo_cfg, resolved_config_path = load_repo_config(config_path)
        backend_cfg = build_backend_config_dict(
            repo_cfg,
            config_path=resolved_config_path,
            mode="train",
            data_path="/tmp/celebahq256_imgfolder",
            precision="bf16",
            seed=7,
            num_train_samples=30_000,
            enable_eval=False,
        )

        self.assertTrue(backend_cfg["data"]["random_flip"])


if __name__ == "__main__":
    unittest.main()
