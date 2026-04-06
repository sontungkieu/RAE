# Workflows

## 1. Environment Setup

The repository supports two practical runtime stacks:

- `torch_xla` for the existing TPU branch under `src/`
- `jax` / `flax.nnx` for the compatibility layer under `src_jax/`

Baseline XLA dependencies:

Baseline install flow:

```bash
conda create -n rae python=3.10 -y
conda activate rae
pip install uv
uv pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 torchvision==0.20.1 -f https://storage.googleapis.com/libtpu-releases/index.html
uv pip install timm==0.9.16 accelerate==0.23.0 torchdiffeq==0.2.5 wandb scipy torch-fidelity
uv pip install "numpy<2" "transformers==4.57.1" einops
```

JAX / NNX additions for `src_jax/`:

```bash
uv pip install "jax[cuda12]==0.5.1" flax==0.10.4 optax==0.2.4 orbax-checkpoint==0.11.16
uv pip install ml-collections clu absl-py etils datasets diffusers huggingface_hub tensorflow-datasets
```

The first JAX run automatically bootstraps `diffuse_nnx` into
`~/.cache/rae_jax/diffuse_nnx` and pins it to commit
`023afd23c7b62a8cdb00e840b36a4ab8fc970bba`.
The Kaggle JAX notebooks do not replay those package lists manually anymore:
they set `UV_PROJECT_ENVIRONMENT=/tmp/.venv`, `UV_CACHE_DIR=/tmp/uv-cache`, and
run `uv sync -q` against the repo `pyproject.toml` before the package-backed
cells.
This repo patches the pinned `diffuse_nnx` checkout to import the Dinov2 models
from `transformers` subpackages, so use `transformers==4.57.1` on this path.
The same patch also lazy-loads `google-cloud-storage` for the remaining
GCS-backed encoder assets, while the public `stage1.StabilityVAE` path now
materializes its default `vae_trial1.pkl` locally from
`stabilityai/sd-vae-ft-mse` instead of depending on the old backend bucket.
The bootstrap patch also fixes the backend EMA initialization so the EMA starts
from a copy of the live model instead of an all-zero parameter tree. Existing
Orbax checkpoints keep the EMA state they already saved, so use a fresh run if
you need the corrected EMA trajectory end to end.
The adapter also derives the Stage-1 latent downsample factor from
`misc.latent_size`, so backend preview sampling and FID stay in latent space
instead of allocating image-resolution latent noise.

## 2. Prepare Models and Data

### Model Weights

Download weights into `models/`:

```bash
hf download nyu-visionx/RAE-collections --local-dir models
```

### Dataset Format

All current dataset-consuming scripts still assume an `ImageFolder` layout:

```text
dataset_root/
  class_a/
    0001.png
  class_b/
    0002.png
```

For unlabeled one-class datasets, you still need one subdirectory, for example
after preparing CelebA into `ImageFolder`:

```text
celeba256_imgfolder/
  train/
    face/
      ...
  val/
    face/
      ...
```

## 3. Reconstruct Images with Stage 1

### Single Image

```bash
python src/stage1_sample.py \
  --config <config> \
  --image assets/pixabay_cat.png \
  --output recon.png
```

Use this when you want to verify the Stage 1 checkpoint, latent shape, or input
preprocessing.

### Distributed Reconstruction

```bash
python src/stage1_sample_ddp.py \
  --config <config> \
  --data-path <imagefolder_root> \
  --sample-dir recon_samples \
  --image-size 256 \
  --per-proc-batch-size 4 \
  --save-npz
```

Outputs:

- PNG reconstructions
- optional `.npz` archive for later scoring

## 4. Train Stage 2 on TPU

Main entrypoint:

```bash
python src/train.py \
  --config configs/stage2/training/ImageNet256/SiTDH-XL_DINOv2-B.yaml \
  --data-path <imagenet_train_split> \
  --results-dir results \
  --image-size 256 \
  --precision bf16
```

What happens inside the loop:

1. images are center-cropped and converted to tensors
2. the RAE encodes them to latent tensors
3. the transport loss is computed in latent space
4. the optimizer and scheduler step on TPU
5. the EMA model is updated
6. logging, checkpointing, preview sampling, validation, and FID run on their
   configured cadences

### Important Training Outputs

- `results/<exp>/log.txt`
- `results/<exp>/checkpoints/*.pt`
- `results/<exp>/fid_eval/step_*/` when FID is enabled

### Resume Training

```bash
python src/train.py \
  --config <config> \
  --data-path <train_split> \
  --results-dir results \
  --image-size 256 \
  --precision bf16 \
  --ckpt results/<exp>/checkpoints/0050000.pt
```

The loop restores:

- model
- EMA
- optimizer
- scheduler
- `epoch`
- `train_steps`

Cadence checks such as `log_every`, `eval_every`, `fid_every`, `sample_every`,
and `ckpt_every` continue from the restored step count.

## 4b. Train Stage 2 with JAX / NNX

Main JAX entrypoint:

```bash
python3 src_jax/train.py \
  --config configs/stage2/training/ImageNet256/SiTDH-XL_DINOv2-B.yaml \
  --data-path <imagenet_train_root> \
  --results-dir results_jax \
  --precision bf16 \
  --wandb
```

On Kaggle TPU, start from `--set training.num_workers=1` and then raise
`--set training.prefetch_factor=<n>` before jumping straight to a large worker
pool. This keeps the backend PyTorch loader compatible with
`persistent_workers=True` after JAX has already initialized multithreaded
runtime state, while still letting the host queue several ready batches ahead
of the TPU. The adapter also disables the backend TensorBoard summary writer on
Kaggle and keeps metric logging on stdout plus wandb.

Useful additions:

- `--set training.global_batch_size=256`: override YAML values from the CLI
- `--set training.prefetch_factor=4`: deepen the host-side train prefetch queue
- `--set eval.prefetch_factor=4`: do the same for the validation loader
- `--wandb-run-id <existing_run_id>`: bind a legacy JAX resume workdir to the exact historical W&B run once, then persist that binding inside the workdir
- `--hf-repo-id <user>/<repo>`: upload the finished workdir to Hugging Face
- `--workdir <path>`: force an explicit output directory instead of letting the
  adapter derive one from `--results-dir`

Resume / initialize behavior:

- If `stage_2.ckpt` points to a PyTorch `.pt`, the adapter ports those weights
  into the NNX model for initialization.
- If `stage_2.ckpt` points to an Orbax directory from a previous JAX run, the
  adapter restores that JAX checkpoint instead.
- When `--wandb` is enabled, a fresh JAX workdir writes `wandb_run.json` and
  starts with a fresh W&B run ID plus `resume="never"`.
- Later resumes of the same JAX workdir read `wandb_run.json`, keep that exact
  W&B run ID, and auto-rewind W&B history to the latest `checkpoint_*` step via
  `resume_from`, so post-checkpoint logs from a crashed session do not survive
  into the resumed history.
- If a legacy Orbax workdir already has checkpoints but predates
  `wandb_run.json`, pass `--wandb-run-id <existing_run_id>` once so the
  adapter can persist the exact historical run binding before training
  continues.

## 5. Enable wandb Logging

Set:

```bash
export ENTITY=<wandb_entity>
export PROJECT=<wandb_project>
export WANDB_KEY=<wandb_api_key>
```

Then add `--wandb` to `src/train.py`.

The same environment variables are also honored by `src_jax/train.py`. The JAX
adapter bridges the legacy `ENTITY`, `PROJECT`, and `WANDB_KEY` names into the
`WANDB_*` names expected by the NNX backend.
On the JAX path, `wandb_run.json` inside the workdir is now the source of truth
for exact resume behavior. A later launch must match that stored binding, and
the adapter rewinds the W&B run to the latest checkpoint step before logging
continues.

Current Stage 2 namespaces:

- `train/*`
- `eval/*`
- `checkpoint/*`
- `network_samples`
- `ema_network_samples`
- `sample/duration_sec`

If you enable `training.log_rae_latent_stats: true`, the JAX path also logs
`train_rae_latent_rms` and `train_rae_latent_var`.
If you enable `training.log_activation_stats: true`, it additionally logs
`train_sitdh_output_rms`, `train_sitdh_output_var`, and one RMS/variance pair
per encoder/decoder block such as `train_sitdh_act_enc_00_rms` and
`train_sitdh_act_dec_01_var`.
If you enable `source.enabled: true`, the same JAX loop also logs
`train_loss_fm`, `train_loss_balance`, `train_loss_entropy`, `train_loss_var`,
`train_source_router_entropy`, `train_source_router_max`, and
`train_source_active_modes`, `train_source_logvar_mean`, and
`train_source_var_mean`.

The current VAE `moe1` defaults are aligned with the public
`shortcut-models@moe1` branch for the source-side recipe:
`condition_dim=16`, `hidden_channels=64`, `router_temperature=2.0`,
`balance_loss_weight=0.1`, `entropy_loss_weight=1.0e-2`, and
`var_kl_loss_weight=1.0`.

Only the master rank initializes and logs to wandb.

## 6. Validation Loss During Training

Add an `eval` block:

```yaml
eval:
  data_path: data/imagenet/val/
  eval_every: 5000
  batch_size: 128
  num_workers: 4
  max_batches: 32
  eval_model: false
```

Behavior:

- runs a deterministic center-crop validation loader
- evaluates EMA by default
- optionally evaluates the non-EMA model as well
- logs `eval/ema_loss`, `eval/ema_batches`, `eval/ema_samples`, and
  `eval/ema_duration_sec`
- logs the matching `eval/model_*` metrics when `eval_model: true`

## 7. FID During Training

Train-time FID is optional and uses two execution domains:

- TPU: latent sampling and image decoding
- host CPU or GPU: Inception feature extraction

Config:

```yaml
eval:
  fid_ref: /path/to/reference_stats.pkl
  fid_every: 25000
  fid_num_samples: 4096
  fid_per_proc_batch_size: 4
  fid_batch_size: 128
  fid_device: cpu
  fid_num_threads: 96
  fid_label_sampling: random
  fid_eval_model: false
```

### Meaning of the Main Knobs

- `fid_every`: FID cadence in optimizer steps
- `fid_num_samples`: sample budget per FID measurement
- `fid_per_proc_batch_size`: generation batch per TPU core
- `fid_batch_size`: host-side Inception batch size
- `fid_device`: `cpu`, `cuda`, or `auto`
- `fid_num_threads`: only matters on CPU, maps to `torch.set_num_threads`
- `fid_label_sampling`: `equal` or `random`
- `fid_eval_model`: also score the non-EMA model

Behavior on the JAX path:

- the default `FID-4K (cfg=...)` series measures EMA
- `fid_eval_model: true` adds `FID-4K/model (cfg=...)` for the live model
- `network_samples` and `ema_network_samples` therefore no longer refer to the
  same weights once EMA starts diverging from the online model

### Recommended CPU-Only Starting Point

For a TPU VM with a strong CPU host and no GPU:

```yaml
eval:
  fid_ref: /path/to/reference_stats.pkl
  fid_every: 25000
  fid_num_samples: 4096
  fid_per_proc_batch_size: 4
  fid_batch_size: 128
  fid_device: cpu
  fid_num_threads: 96
```

### Cost vs Stability

- `fid_num_samples: 1024` is fast, but noisy
- `fid_num_samples: 4096` is a practical online-monitoring compromise
- `fid_num_samples: 50000` is the closest to standard FID-50k, but expensive

Do not compare FID across runs unless the sample count and reference stats are
the same.

## 8. Sample Stage 2 Outputs

### Quick Single-Process Sample

```bash
python src/sample.py \
  --config <sample_config> \
  --seed 42 \
  --class-labels 207,360 \
  --output sample.png
```

### Distributed Sampling

```bash
python src/sample_ddp.py \
  --config <sample_config> \
  --sample-dir samples \
  --per-proc-batch-size 4 \
  --num-fid-samples 50000 \
  --precision bf16 \
  --label-sampling equal
```

### JAX Single-Run Sampling

```bash
python3 src_jax/sample.py \
  --config configs/stage2/sampling/ImageNet256/SiTDHXL-DINOv2-B_AG.yaml \
  --class-labels 207,360 \
  --output sample_jax.png
```

Notes:

- On this branch, the checked-in JAX sampling configs are templates; set
  `stage_2.ckpt` and `guidance.guidance_model.ckpt` to SiTDH-compatible weights
  before running sampling.
- `guidance.method=cfg` uses the same model for conditional and unconditional
  passes.
- `guidance.method=autoguidance` loads `guidance.guidance_model` as the guide
  network.

### JAX Distributed Sampling

```bash
python3 src_jax/sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/SiTDHXL-DINOv2-B.yaml \
  --sample-dir samples_jax \
  --num-samples 50000 \
  --label-sampling equal \
  --fid-ref /path/to/reference_stats.npz
```

Outputs:

- per-image PNG files under `samples_jax/`
- optional per-rank `.npz` archives when `--save-npz` is enabled
- optional printed FID score when `--fid-ref` is supplied

### Stage 1 Reconstruction on the JAX Path

```bash
python3 src_jax/stage1_sample.py \
  --config configs/stage1/pretrained/DINOv2-B_512.yaml \
  --image assets/pixabay_cat.png \
  --output recon_jax.png
```

This is intentionally scoped to inference and checkpoint verification. The
adversarial Stage 1 training loop has not been ported to JAX in this branch.

### Stage 1 Latent Stats on the JAX Path

For a new dataset, first bootstrap normalization with an identity stats file,
then run the JAX stats pass:

```bash
python3 src_jax/build_stage1_stats.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --input /path/to/train_imagefolder \
  --output /path/to/stage1_stat.pt \
  --batch-size 16 \
  --set stage_1.params.normalization_stat_path=/path/to/bootstrap_identity_stat.pt
```

The output `stat.pt` matches the original repo format (`mean` / `var` tensors),
so it can be consumed by both `src/` and `src_jax/`.
These Stage 1-only JAX utilities can run from a YAML that only defines
`stage_1`; they do not require `stage_2.target`.

### JAX Folder Reconstruction

```bash
python3 src_jax/reconstruct_folder.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --input /path/to/val_imagefolder \
  --output-dir recon_jax_dir \
  --batch-size 8
```

This is the simplest way to export a validation reconstruction set before
building FID references or comparing Stage 1 decoder changes.

### Kaggle CelebA Notebook

This branch now keeps one public CelebA Kaggle notebook flow:

- `StabilityVAE + SiT-B`: the single-tower VAE baseline for this branch

The public notebook path now reads the Kaggle dataset
`jessicali9530/celeba-dataset`, center-crops each image, resizes to `256x256`,
and materializes the result into `/kaggle/working/celeba256_imgfolder` in the
same `ImageFolder` layout used by the current JAX runtime. The older
[`src_jax/export_celebahq_hf.py`](../src_jax/export_celebahq_hf.py) and
[`src_jax/export_celebahq_tfds.py`](../src_jax/export_celebahq_tfds.py) helpers
are still available if you specifically want a manual CelebA-HQ export route.

If you want the VAE learned-source flow on the same dataset, start from
[../vaes-jax-celeba-kaggle-moe1.ipynb](../vaes-jax-celeba-kaggle-moe1.ipynb).
That notebook keeps the same `uv sync` Kaggle flow, but prepares
`celeba256_imgfolder`, switches Stage 1 to `stage1.StabilityVAE` and Stage 2
to single-tower `stage2.models.SiT.SiT`, then builds an offline diagonal GMM
artifact and injects a `source:` block into the generated Stage 2 config. The
generated artifacts are:

- `configs/stage1/pretrained/CelebA256_StabilityVAE_jax.yaml`
- `configs/stage2/training/CelebA256_SiT-B_StabilityVAE_moe1_jax.yaml`
- `/kaggle/working/celeba256_source_gmm.npz`

Key properties of this VAE notebook flow:

- no `hf download nyu-visionx/RAE-collections` step
- no bootstrap identity-stat file
- no required `src_jax/build_stage1_stats.py` pass before Stage 2 training
- one required `src_jax/build_source_gmm.py` pass before Stage 2 training
- `training.random_flip=true` is enabled in the generated Stage 2 config
- `training.log_rae_latent_stats=true` and `training.log_activation_stats=true`
  are enabled in the generated Stage 2 config
- Stage 2 uses the DiT-B-style `SiT-B` shape from the `shortcut-models`
  CelebA example (`hidden_size=768`, `depth=12`, `num_heads=12`,
  `patch_size=2`) while keeping this repo's `sit` flow-matching objective
- sampling, preview images, and online FID start from the learned source prior
  instead of a pure Gaussian latent
- the final train cell points `--data-path` at
  `/kaggle/working/celeba256_imgfolder/train` while the generated config keeps
  `eval.data_path` on `/kaggle/working/celeba256_imgfolder/val`

For Kaggle `TPU v5e-8`, use
[../vaes-jax-celeba-kaggle-tpuv5e8-sitb-moe1.ipynb](../vaes-jax-celeba-kaggle-tpuv5e8-sitb-moe1.ipynb).
That notebook keeps the same TPU/JAX workarounds, writes the Stage 2 config as
`CelebA256_SiT-B_StabilityVAE_moe1_jax_tpuv5e8.yaml`, keeps
`--data-path /kaggle/working/celeba256_imgfolder`, and bakes the same
host-side loader defaults into both the generated YAML and CLI overrides. The
TPU variants set `training.num_workers=16`, `eval.num_workers=16`,
`training.prefetch_factor=4`, and `eval.prefetch_factor=4` so the host can
queue batches more aggressively without editing the cached backend checkout by
hand. Both public VAE notebooks also write their generated
Stage 2 configs with `training.log_rae_latent_stats=true` and
`training.log_activation_stats=true` so latent/VAE diagnostics, SiT activation
RMS/variance, and source metrics are available by default.

If you already have an Orbax run directory for that VAE flow and want to
continue training from its latest checkpoint, use
[../vaes-jax-celeba-kaggle-tpuv5e8-sitb-moe1-resume.ipynb](../vaes-jax-celeba-kaggle-tpuv5e8-sitb-moe1-resume.ipynb).
That notebook mirrors the resume-only Kaggle TPU pattern used by the DH
notebook, but searches for the newest
`CelebA256_SiT-B_StabilityVAE_moe1_jax_tpuv5e8-*` run directory instead. Its
resume train cell also re-applies `training.log_rae_latent_stats=true` and
`training.log_activation_stats=true` from the CLI so resumed runs keep the same
diagnostics enabled by default. If the target workdir was created before
`wandb_run.json` existed, pass `--wandb-run-id <existing_run_id>` once so the
resumed Kaggle session binds to the exact old W&B run instead of aborting.

The same branch also keeps the parallel CelebA-HQ `moe1` notebooks:
[../vaes-jax-celebahq-kaggle-moe1.ipynb](../vaes-jax-celebahq-kaggle-moe1.ipynb),
[../vaes-jax-celebahq-kaggle-tpuv5e8-sitb-moe1.ipynb](../vaes-jax-celebahq-kaggle-tpuv5e8-sitb-moe1.ipynb),
and
[../vaes-jax-celebahq-kaggle-tpuv5e8-sitb-moe1-resume.ipynb](../vaes-jax-celebahq-kaggle-tpuv5e8-sitb-moe1-resume.ipynb).
They keep the same `StabilityVAE + SiT-B + moe1` recipe but export
`/kaggle/working/celebahq256_imgfolder`, build `celebahq256_source_gmm.npz`,
and resume the timestamped `CelebAHQ256_SiT-B_StabilityVAE_moe1_jax_tpuv5e8-*`
workdirs instead of the CelebA ones.

## 9. Upload a JAX Run to Hugging Face

Standalone upload:

```bash
python3 src_jax/push_hf.py \
  --path results_jax/<run_name> \
  --repo-id <hf_user_or_org>/<repo_name>
```

Or upload automatically at the end of training:

```bash
python3 src_jax/train.py \
  --config <config> \
  --data-path <train_root> \
  --hf-repo-id <hf_user_or_org>/<repo_name>
```

This produces:

- PNG files
- a packed `.npz` archive

### Distributed Sampling with Immediate FID

```bash
python src/sample_ddp.py \
  --config <sample_config> \
  --sample-dir samples \
  --per-proc-batch-size 4 \
  --num-fid-samples 4096 \
  --precision bf16 \
  --label-sampling random \
  --fid-ref /path/to/reference_stats.npz \
  --fid-device cpu \
  --fid-batch-size 128 \
  --fid-num-threads 96
```

## 9. Build FID Reference Statistics

From an image folder:

```bash
python src/build_fid_stats.py \
  --input /path/to/reference_images \
  --output /path/to/reference_stats.npz \
  --device cpu \
  --batch-size 64 \
  --num-workers 32 \
  --num-threads 96
```

For JAX online FID, build the reference with the backend-native detector:

```bash
python3 src_jax/build_fid_stats.py \
  --input /path/to/reference_images \
  --output /path/to/reference_stats.pkl \
  --batch-size 64 \
  --num-workers 8
```

Trên host JAX như Kaggle TPU, nên giữ `--num-workers` ở mức vừa phải (`0-8`
thường là đủ). Script JAX này dùng `spawn` khi `num_workers > 0`, nên tránh
được cảnh báo `os.fork()` thường gặp với JAX đa luồng.

From an existing `.npz` or `.npy` archive:

```bash
python src/build_fid_stats.py \
  --input /path/to/reference_images.npz \
  --output /path/to/reference_stats.npz \
  --device cpu \
  --batch-size 64 \
  --num-threads 96
```

## 10. Evaluate an Existing Archive

```bash
python src/evaluate_fid.py \
  --samples /path/to/samples.npz \
  --ref /path/to/reference_stats.npz \
  --device cpu \
  --batch-size 128 \
  --num-threads 96 \
  --output-json /path/to/samples.fid.json
```

`src_jax/build_fid_stats.py` only accepts an `ImageFolder`, because it must run
the same Flax Inception detector used by the backend online FID path.

## 11. Common Operational Notes

- `src/train.py` expects the Stage 2 training config to use `training.global_batch_size`,
  not `batch_size`.
- `src/train.py` reads `full_cfg.get("eval")` directly, so the eval block does
  not need to be part of `parse_configs(...)`.
- `src/sample.py` and `src/sample_ddp.py` only support the manual ODE path on
  this branch.
- Stage 1 reconstruction scripts are for inference and inspection; this branch
  does not include a local Stage 1 trainer entrypoint.
- For CPU-only FID, increase `fid_batch_size` cautiously. Throughput improves
  until memory bandwidth becomes the bottleneck.
