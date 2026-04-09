from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class GenerateAblationNotebooksTests(unittest.TestCase):
    def test_generator_writes_selected_notebooks_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "generated"
            subprocess.run(
                [
                    "python3",
                    "scripts/generate_ablation_notebooks.py",
                    "--spec",
                    "configs/ablation/celeba_sitdh_moe1_pyr16k.yaml",
                    "--output-dir",
                    str(output_dir),
                    "--only",
                    "01,04",
                    "--overwrite",
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            notebook_paths = sorted(output_dir.glob("*.ipynb"))
            self.assertEqual(
                [path.name for path in notebook_paths],
                [
                    "01-moe1-pyr16k-m4-tau2-vk1-bl01-ent001-tv1-cd16-hc256.ipynb",
                    "04-moe1-pyr16k-m6-tau4-vk01-bl01-ent001-tv1-cd24-hc256.ipynb",
                ],
            )

            manifest_path = output_dir / "manifest.csv"
            self.assertTrue(manifest_path.exists())
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["index"] for row in rows], ["01", "04"])
            self.assertEqual(rows[1]["gmm_feature_extractor"], "pyramid_16k")
            self.assertEqual(rows[1]["condition_dim"], "24")
            self.assertEqual(rows[1]["hidden_channels"], "256")
            self.assertEqual(rows[1]["var_kl_loss_weight"], "0.1")

            notebook_payload = json.loads(notebook_paths[1].read_text(encoding="utf-8"))
            cell_sources = ["".join(cell.get("source", [])) for cell in notebook_payload["cells"]]
            title_cell = cell_sources[0]
            setup_cell = next(source for source in cell_sources if 'stage2_results_dir = Path(' in source)
            config_cell = next(source for source in cell_sources if "stage2_cfg_text = textwrap.dedent" in source)
            gmm_cell = next(source for source in cell_sources if "build_source_gmm.py" in source)
            train_cell = next(source for source in cell_sources if "src_jax/train.py" in source)

            self.assertEqual(title_cell, "# 04-moe1-pyr16k-m6-tau4-vk01-bl01-ent001-tv1-cd24-hc256\n")
            self.assertIn('/kaggle/working/celeba256_source_gmm_pyr16k_04_moe1_pyr16k_m6_tau4_vk01_bl01_ent001_tv1_cd24_hc256.npz', setup_cell)
            self.assertIn('repo_root / "configs" / "ablation" / "generated" / "celeba_sitdh_moe1_pyr16k" / "04-moe1-pyr16k-m6-tau4-vk01-bl01-ent001-tv1-cd24-hc256.yaml"', config_cell)
            self.assertIn('celeba_val_path = (celeba_root / "val").as_posix()', config_cell)
            self.assertIn("hidden_channels: 256", config_cell)
            self.assertIn("condition_dim: 24", config_cell)
            self.assertIn("router_temperature: 2.0", config_cell)
            self.assertIn("data_path: '{celeba_val_path}'", config_cell)
            self.assertNotIn("\\'val\\'", config_cell)
            self.assertIn("normalization_stat_path: '{latent_stats_path.as_posix()}'", config_cell)
            self.assertIn("--feature-extractor pyramid_16k", gmm_cell)
            self.assertIn("--storage-dtype float16", gmm_cell)
            self.assertIn("--output /kaggle/working/celeba256_source_gmm_pyr16k_04_moe1_pyr16k_m6_tau4_vk01_bl01_ent001_tv1_cd24_hc256.npz", gmm_cell)
            self.assertIn('run_slug="04-moe1-pyr16k-m6-tau4-vk01-bl01-ent001-tv1-cd24-hc256"', train_cell)
            self.assertIn('--wandb-group "${wandb_group}"', train_cell)
            self.assertIn('--wandb-tags "${wandb_tags}"', train_cell)
            self.assertIn("gmmfeat:pyramid_16k", train_cell)
            self.assertIn("slug:moe1-pyr16k-m6-tau4-vk01-bl01-ent001-tv1-cd24-hc256", train_cell)
            self.assertIn('secrets.get_secret("WANDB_Tung")', "".join(cell_sources))
            self.assertNotIn('secrets.get_secret("WANDB2")', "".join(cell_sources))


if __name__ == "__main__":
    unittest.main()
