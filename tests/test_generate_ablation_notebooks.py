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
                    "configs/ablation/celeba_vae_moe1.yaml",
                    "--output-dir",
                    str(output_dir),
                    "--only",
                    "01,03",
                    "--overwrite",
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            notebook_paths = sorted(output_dir.glob("*.ipynb"))
            self.assertEqual([path.name for path in notebook_paths], [
                "01-moe1-m4-tau2-vk1-bl01-ent001-tv1-cd16-hc64.ipynb",
                "03-moe1-m6-tau4-vk01-bl01-ent001-tv1-cd16-hc96.ipynb",
            ])

            manifest_path = output_dir / "manifest.csv"
            self.assertTrue(manifest_path.exists())
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["index"] for row in rows], ["01", "03"])
            self.assertEqual(rows[1]["hidden_channels"], "96")
            self.assertEqual(rows[1]["var_kl_loss_weight"], "0.1")
            self.assertEqual(rows[1]["target_variance"], "1")

            notebook_payload = json.loads(notebook_paths[1].read_text(encoding="utf-8"))
            cell_sources = ["".join(cell.get("source", [])) for cell in notebook_payload["cells"]]
            title_cell = cell_sources[0]
            build_cell = next(source for source in cell_sources if "build_source_gmm.py" in source)
            train_cell = next(source for source in cell_sources if "src_jax/train.py" in source)
            config_cell = next(source for source in cell_sources if "stage2_cfg_text = textwrap.dedent" in source)
            view_cell = next(source for source in cell_sources if "PYVIEW" in source)

            self.assertTrue(config_cell.startswith("%%bash\n"))
            self.assertIn('repo_root / "configs" / "ablation" / "generated" / "celeba_vae_moe1" / "03-moe1-m6-tau4-vk01-bl01-ent001-tv1-cd16-hc96.yaml"', view_cell)
            self.assertIn("--output /kaggle/working/celeba256_source_gmm_03_moe1_m6_tau4_vk01_bl01_ent001_tv1_cd16_hc96.npz", build_cell)
            self.assertIn("--num-modes 6", build_cell)
            self.assertIn("# 03-moe1-m6-tau4-vk01-bl01-ent001-tv1-cd16-hc96", title_cell)
            self.assertIn('run_slug="03-moe1-m6-tau4-vk01-bl01-ent001-tv1-cd16-hc96"', train_cell)
            self.assertIn('--wandb-group "${wandb_group}"', train_cell)
            self.assertIn('--wandb-tags "${wandb_tags}"', train_cell)
            self.assertIn('slug:moe1-m6-tau4-vk01-bl01-ent001-tv1-cd16-hc96', train_cell)
            self.assertIn('target_var:1', train_cell)
            self.assertIn("hidden_channels: 96", config_cell)
            self.assertIn("router_temperature: 4.0", config_cell)
            self.assertIn('get_secret("WANDB_Tung")', "".join(cell_sources))
            self.assertNotIn("HF_TOK_WRITE_KAGGLE", "".join(cell_sources))
            self.assertNotIn('os.environ["HF_TOKEN"]', "".join(cell_sources))


if __name__ == "__main__":
    unittest.main()
