from __future__ import annotations

import json
from pathlib import Path
import textwrap


REPO_URL = "https://github.com/sontungkieu/RAE"


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.rstrip() + "\n",
    }


def notebook_metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10",
        },
    }


def _asset_download_cell(*, include_decoder: bool, robust_uv: bool) -> str:
    uv_init = 'import subprocess\n'
    decoder_block = ""
    if include_decoder:
        decoder_block = textwrap.dedent(
            """

            subprocess.run(
                [
                    "uv",
                    "run",
                    "hf",
                    "download",
                    "nyu-visionx/RAE-collections",
                    "decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
                    "--local-dir",
                    "models",
                ],
                check=True,
                cwd=repo_root,
            )
            """
        ).rstrip()
    return textwrap.dedent(
        """
        __UV_INIT__

        subprocess.run(
            [
                "uv",
                "run",
                "hf",
                "download",
                "nyu-visionx/RAE-collections",
                "discs/dino_vit_small_patch8_224.pth",
                "--local-dir",
                "models",
            ],
            check=True,
            cwd=repo_root,
        )__DECODER_BLOCK__

        print("Model assets are ready under", repo_root / "models")
        """
    ).replace("__UV_INIT__", uv_init).replace("__DECODER_BLOCK__", decoder_block).strip()


def _dataset_prep_cell(*, robust_uv: bool) -> str:
    uv_init = ""
    return textwrap.dedent(
        """
        import csv
        import os
        import subprocess
        from PIL import Image

        __UV_INIT__

        def center_crop_resize_256(src_path: Path, dst_path: Path) -> None:
            with Image.open(src_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                crop = min(width, height)
                left = (width - crop) // 2
                top = (height - crop) // 2
                image = image.crop((left, top, left + crop, top + crop))
                image = image.resize((256, 256), Image.BICUBIC)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(dst_path, quality=95)


        if prepare_dataset and dataset_name == "celebahq":
            if not face_root.exists():
                subprocess.run(
                    [
                        "uv",
                        "run",
                        "python",
                        "src_jax/export_celebahq_hf.py",
                        "--dataset",
                        celeb_hq_dataset_id,
                        "--output",
                        face_root.as_posix(),
                    ],
                    check=True,
                    cwd=repo_root,
                )
            else:
                print("CelebA-HQ ImageFolder already exists:", face_root)
        elif prepare_dataset and dataset_name == "celeba":
            if not celeba_src_dir.exists():
                raise FileNotFoundError(f"Missing CelebA source directory: {celeba_src_dir}")
            if not celeba_split_csv.exists():
                raise FileNotFoundError(f"Missing CelebA split csv: {celeba_split_csv}")
            if not face_root.exists():
                split_map = {"0": "train", "1": "val", "2": "test"}
                created = 0
                with celeba_split_csv.open("r", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        split = split_map[row["partition"]]
                        src = celeba_src_dir / row["image_id"]
                        dst = face_root / split / "face" / row["image_id"]
                        center_crop_resize_256(src, dst)
                        created += 1
                        if created % 10000 == 0:
                            print(f"prepared {created} CelebA images...")
                print(f"Prepared {created} CelebA images under {face_root}")
            else:
                print("CelebA ImageFolder already exists:", face_root)
        else:
            print("Skipping dataset preparation; expecting ImageFolder roots to already exist.")

        print("train root:", train_root)
        print("val root:", val_root)
        print("train images:", sum(1 for _ in train_root.rglob('*.jpg')) + sum(1 for _ in train_root.rglob('*.png')))
        print("val images:", sum(1 for _ in val_root.rglob('*.jpg')) + sum(1 for _ in val_root.rglob('*.png')))
        """
    ).replace("__UV_INIT__", uv_init).strip()


def _config_cell(*, mode: str) -> str:
    if mode == "scratch":
        title = "TuneDinoV2 Stage 1 from scratch"
        pretrained_decoder = "null"
        stage1_ckpt = "null"
        train_encoder = "false"
        lr = "2.0e-4"
        encoder_lr = "2.0e-5"
        epochs = 20
        batch_size = 8
        group = "stage1-scratch"
        tags = "stage1,mode:scratch,encoder:frozen,decoder:random,project:TuneDinoV2"
        exp_prefix = "TuneDinoV2-stage1-scratch"
    else:
        title = "TuneDinoV2 Stage 1 finetune DINOv2"
        pretrained_decoder = "'models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt'"
        stage1_ckpt = "{stage1_ckpt_yaml}"
        train_encoder = "true"
        lr = "1.0e-4"
        encoder_lr = "1.0e-5"
        epochs = 12
        batch_size = 4
        group = "stage1-finetune-dino"
        tags = "stage1,mode:finetune,encoder:trainable,decoder:imagenet_init,project:TuneDinoV2"
        exp_prefix = "TuneDinoV2-stage1-finetune-dino"

    base = f"""
    import json
    import textwrap
    from datetime import datetime
    from pathlib import Path

    repo_root = Path("/kaggle/working/RAE")
    dataset_name = "celebahq"  # change to "celeba" if needed
    prepare_dataset = True
    celeb_hq_dataset_id = "eurecom-ds/celeba-hq-256"
    celeba_src_dir = Path("/kaggle/input/datasets/jessicali9530/celeba-dataset/img_align_celeba/img_align_celeba")
    celeba_split_csv = Path("/kaggle/input/datasets/jessicali9530/celeba-dataset/list_eval_partition.csv")
    face_root = Path("/kaggle/working/celebahq256_imgfolder" if dataset_name == "celebahq" else "/kaggle/working/celeba256_imgfolder")
    train_root = face_root / "train"
    val_root = face_root / "val"
    results_dir = Path("/kaggle/working/results_stage1")
    generated_cfg_dir = repo_root / "configs" / "stage1" / "training" / "generated"
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)
    config_path = generated_cfg_dir / "{mode}_tunedinov2_stage1.yaml"
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_prefix = "{exp_prefix}"
    wandb_group = "{group}"
    wandb_tags = "{tags}"
    stage1_ckpt_path = None  # set to an existing Stage 1 checkpoint to continue from your best run
    grad_accum_steps = 8      # accumulation on top of the per-GPU micro batch
    fid_ref_path = None       # optional reconstruction FID reference stats (.npz or .pkl)
    fid_every = 1             # epoch cadence when fid_ref_path is set
    fid_batch_size = 64
    fid_device = "auto"
    fid_num_threads = None

    stage1_ckpt_yaml = "null" if stage1_ckpt_path is None else f"'{{stage1_ckpt_path}}'"
    fid_ref_yaml = "null" if fid_ref_path is None else f"'{{fid_ref_path}}'"
    fid_device_yaml = f"'{{fid_device}}'"
    fid_num_threads_yaml = "null" if fid_num_threads is None else str(fid_num_threads)
    config_text = textwrap.dedent(
        f\"\"\"
        stage_1:
          target: stage1.RAE
          ckpt: {stage1_ckpt}
          params:
            encoder_cls: 'Dinov2withNorm'
            encoder_config_path: 'facebook/dinov2-with-registers-base'
            encoder_input_size: 224
            encoder_params:
              dinov2_path: 'facebook/dinov2-with-registers-base'
              normalize: true
            decoder_config_path: 'configs/decoder/ViTXL'
            pretrained_decoder_path: {pretrained_decoder}
            noise_tau: 0.0
            reshape_to_2d: true

        training:
          global_seed: 0
          epochs: {epochs}
          batch_size: {batch_size}
          grad_accum_steps: {{grad_accum_steps}}
          num_workers: 4
          image_size: 256
          precision: fp16
          log_every: 20
          image_log_every: 100
          eval_every: 1
          save_every: 1
          recon_weight: 1.0
          train_encoder: {train_encoder}
          encoder_lr: {encoder_lr}
          random_flip: true
          clip_grad: 1.0
          num_visuals: 8
          optimizer:
            lr: {lr}
            betas: [0.9, 0.95]
            weight_decay: 0.0
          scheduler:
            type: cosine
            warmup_epochs: 1
            decay_end_epoch: {epochs}
            base_lr: {lr}
            final_lr: 2.0e-5

        gan:
          disc:
            arch:
              dino_ckpt_path: 'models/discs/dino_vit_small_patch8_224.pth'
              ks: 9
              norm_type: 'bn'
              using_spec_norm: true
              recipe: 'S_8'
            optimizer:
              lr: 2.0e-4
              betas: [0.5, 0.9]
              weight_decay: 0.0
            scheduler:
              type: cosine
              warmup_epochs: 1
              decay_end_epoch: {epochs}
              base_lr: 2.0e-4
              final_lr: 2.0e-5
            augment:
              prob: 1.0
              cutout: 0.0
          loss:
            disc_loss: hinge
            gen_loss: vanilla
            disc_weight: 0.75
            perceptual_weight: 1.0
            disc_start: 1
            disc_upd_start: 1
            lpips_start: 0
            max_d_weight: 10000.0
            disc_updates: 1

        data:
          train_path: '{{train_root.as_posix()}}'

        eval:
          data_path: '{{val_root.as_posix()}}'
          batch_size: {batch_size}
          num_workers: 2
          max_batches: 50
          fid_ref: {{fid_ref_yaml}}
          fid_every: {{fid_every}}
          fid_batch_size: {{fid_batch_size}}
          fid_device: {{fid_device_yaml}}
          fid_num_threads: {{fid_num_threads_yaml}}
        \"\"\"
    ).strip() + "\\n"
    config_path.write_text(config_text, encoding="utf-8")

    run_info = {{
        "title": "{title}",
        "dataset_name": dataset_name,
        "train_root": train_root.as_posix(),
        "val_root": val_root.as_posix(),
        "config_path": config_path.as_posix(),
        "results_dir": results_dir.as_posix(),
        "wandb_group": wandb_group,
        "wandb_tags": wandb_tags,
        "exp_name": f"{{exp_prefix}}-{{dataset_name}}-{{run_timestamp}}",
    }}
    print(json.dumps(run_info, indent=2))
    print(config_text)
    """
    return textwrap.dedent(base).strip()


def _train_cell(*, robust_uv: bool) -> str:
    uv_init = ""
    return textwrap.dedent(
        """
        import json
        import os
        import subprocess
        import torch

        __UV_INIT__
        env = os.environ.copy()
        env["WANDB_ENTITY"] = WANDB_ENTITY
        env["PROJECT"] = PROJECT
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("OMP_NUM_THREADS", "1")

        visible_gpus = torch.cuda.device_count()
        launcher = ["uv", "run"]
        if visible_gpus > 1:
            launcher.extend(
                [
                    "torchrun",
                    "--standalone",
                    f"--nproc_per_node={visible_gpus}",
                ]
            )
        else:
            launcher.extend(["python"])

        cmd = launcher + [
            "src/train_stage1_rae.py",
            "--config",
            config_path.as_posix(),
            "--results-dir",
            results_dir.as_posix(),
            "--exp-name",
            run_info["exp_name"],
            "--wandb",
            "--wandb-entity",
            WANDB_ENTITY,
            "--wandb-project",
            PROJECT,
            "--wandb-group",
            wandb_group,
            "--wandb-tags",
            wandb_tags,
        ]
        print("Launching Stage 1 trainer with", visible_gpus, "visible GPU(s)")
        print("Command:", cmd)
        subprocess.run(cmd, check=True, cwd=repo_root, env=env)

        workdir = results_dir / run_info["exp_name"]
        run_info["workdir"] = workdir.as_posix()
        run_info["last_ckpt"] = (workdir / "checkpoints" / "last.pt").as_posix()
        run_info["best_ckpt"] = (workdir / "checkpoints" / "best.pt").as_posix()
        summary_path = Path("/kaggle/working") / f"{run_info['exp_name']}_summary.json"
        summary_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
        print("Saved run summary to", summary_path)
        print(json.dumps(run_info, indent=2))
        """
    ).replace("__UV_INIT__", uv_init).strip()


def _bootstrap_cell(*, robust_uv: bool) -> str:
    return textwrap.dedent(
        f"""
        %cd /kaggle/working
        !rm -rf RAE
        !git clone {REPO_URL}
        %cd /kaggle/working/RAE

        import os
        import subprocess
        from pathlib import Path

        subprocess.run(["bash", "-lc", "curl -LsSf https://astral.sh/uv/install.sh | sh"], check=True)
        os.environ["PATH"] = f"{{Path.home() / '.local/bin'}}:{{os.environ.get('PATH', '')}}"
        subprocess.run(["bash", "-lc", "uv --version"], check=True)
        print("Using uv from PATH:", os.environ["PATH"].split(":")[0])
        """
    ).strip()


def _sync_cell(*, robust_uv: bool) -> str:
    return textwrap.dedent(
        """
        import os
        import subprocess

        os.environ["UV_PROJECT_ENVIRONMENT"] = "/tmp/.venv"
        os.environ["UV_CACHE_DIR"] = "/tmp/uv-cache"
        subprocess.run(["uv", "sync", "-q"], check=True, cwd="/kaggle/working/RAE")
        subprocess.run(["nvidia-smi"], check=False)
        print("Repo dependencies are synced into /tmp/.venv")
        """
    ).strip()


def _post_train_cell() -> str:
    return textwrap.dedent(
        """
        import json
        from pathlib import Path

        summary_candidates = sorted(Path("/kaggle/working").glob("TuneDinoV2-stage1-*_summary.json"))
        if not summary_candidates:
            raise FileNotFoundError("No run summary json found under /kaggle/working")

        summary_path = summary_candidates[-1]
        run_info = json.loads(summary_path.read_text(encoding="utf-8"))
        workdir = Path(run_info["workdir"])

        print("Summary file:", summary_path)
        print(json.dumps(run_info, indent=2))
        print("\\nCheckpoint directory:")
        for path in sorted((workdir / "checkpoints").glob("*")):
            print("-", path.name)
        """
    ).strip()


def build_notebook(*, mode: str) -> dict:
    include_decoder = mode != "scratch"
    robust_uv = True
    if mode == "scratch":
        title = "# TuneDinoV2 Stage 1 Kaggle Notebook (From Scratch)"
        description = """
Notebook này chạy Stage 1 trên Kaggle GPU theo hướng train decoder từ đầu:

- dùng encoder DINOv2-with-registers làm backbone Stage 1;
- không nạp `pretrained_decoder_path`, để decoder khởi tạo ngẫu nhiên;
- giữ encoder frozen;
- tự dò binary `uv`, tự chuyển sang `torchrun` khi session Kaggle có nhiều GPU, và hỗ trợ `grad_accum_steps`;
- có thể bật reconstruction FID bằng `fid_ref_path`;
- log đầy đủ `L1`, `LPIPS`, `PSNR`, GAN losses, latent stats, learning rate, grad norm, reconstruction FID, ảnh reconstruction lên W&B project `TuneDinoV2`.

Notebook hỗ trợ cả `CelebA` và `CelebA-HQ` bằng biến `dataset_name` ở cell cấu hình.
"""
    else:
        title = "# TuneDinoV2 Stage 1 Kaggle Notebook (Finetune DINOv2)"
        description = """
Notebook này chạy Stage 1 trên Kaggle GPU theo hướng finetune DINOv2:

- nạp decoder ImageNet DINOv2 làm khởi tạo;
- cho phép gắn thêm `stage1_ckpt_path` nếu bạn đã có checkpoint Stage 1 cũ;
- mở train encoder DINOv2 với `encoder_lr` nhỏ hơn decoder;
- tự dò binary `uv`, tự chuyển sang `torchrun` khi session Kaggle có nhiều GPU, và hỗ trợ `grad_accum_steps`;
- có thể bật reconstruction FID bằng `fid_ref_path`;
- log đầy đủ `L1`, `LPIPS`, `PSNR`, GAN losses, latent stats, learning rate, grad norm, reconstruction FID, ảnh reconstruction lên W&B project `TuneDinoV2`.

Notebook hỗ trợ cả `CelebA` và `CelebA-HQ` bằng biến `dataset_name` ở cell cấu hình.
"""

    cells = [
        md_cell(f"{title}\n\n{textwrap.dedent(description).strip()}"),
        code_cell(_bootstrap_cell(robust_uv=robust_uv)),
        code_cell(_sync_cell(robust_uv=robust_uv)),
        code_cell(
            textwrap.dedent(
                """
                import os
                import pathlib

                PROJECT = "TuneDinoV2"
                WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "TungBangDSLab")

                try:
                    from kaggle_secrets import UserSecretsClient

                    secrets = UserSecretsClient()
                    wandb_token = None
                    for secret_name in ("WANDB_Tung", "WANDB2", "WANDB_KEY"):
                        try:
                            wandb_token = secrets.get_secret(secret_name)
                            print(f"Loaded W&B secret from {secret_name}")
                            break
                        except Exception:
                            continue
                    if wandb_token is None:
                        raise RuntimeError("No W&B Kaggle secret found")

                    os.environ["WANDB_API_KEY"] = wandb_token
                    os.environ["WANDB_KEY"] = wandb_token
                    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
                    os.environ["PROJECT"] = PROJECT
                    netrc = pathlib.Path.home() / ".netrc"
                    netrc.write_text(f"machine api.wandb.ai login user password {wandb_token}\\n", encoding="utf-8")
                    os.chmod(netrc, 0o600)
                except Exception as exc:
                    print(f"Skipping Kaggle W&B bootstrap: {exc}")

                print("W&B entity:", WANDB_ENTITY)
                print("W&B project:", PROJECT)
                """
            ).strip()
        ),
        code_cell(_config_cell(mode=mode)),
        code_cell(_asset_download_cell(include_decoder=include_decoder, robust_uv=robust_uv)),
        code_cell(_dataset_prep_cell(robust_uv=robust_uv)),
        code_cell(_train_cell(robust_uv=robust_uv)),
        code_cell(_post_train_cell()),
    ]
    return {
        "cells": cells,
        "metadata": notebook_metadata(),
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    outputs = {
        "scratch": repo_root / "tunedinov2-stage1-scratch-kaggle.ipynb",
        "finetune": repo_root / "tunedinov2-stage1-finetune-dinov2-kaggle.ipynb",
    }
    for mode, output_path in outputs.items():
        notebook = build_notebook(mode=mode)
        output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
