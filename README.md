## Diffusion Transformers with Representation Autoencoders (RAE)<br><sub>Official PyTorch Implementation</sub>

### [Paper](https://arxiv.org/abs/2510.11690) | [Project Page](https://rae-dit.github.io/) 


This repository contains **PyTorch/GPU** and **TorchXLA/TPU** implementations of our paper: 
Diffusion Transformers with Representation Autoencoders. For JAX/TPU implementation, please refer to [diffuse_nnx](https://github.com/willisma/diffuse_nnx)

> [**Diffusion Transformers with Representation Autoencoders**](https://arxiv.org/abs/2510.11690)<br>
> [Boyang Zheng](https://bytetriper.github.io/), [Nanye Ma](https://willisma.github.io), [Shengbang Tong](https://tsb0601.github.io/),  [Saining Xie](https://www.sainingxie.com)
> <br>New York University<br>


We present Representation Autoencoders (RAE), a class of autoencoders that utilize  pretrained, frozen representation encoders such as [DINOv2](https://arxiv.org/abs/2304.07193) and [SigLIP2](https://arxiv.org/abs/2502.14786) as encoders with trained ViT decoders. RAE can be used in a two-stage training pipeline for high-fidelity image synthesis, where a Stage 2 diffusion model is trained on the latent space of a pretrained RAE to generate images.

This branch contains:

TorchXLA/TPU:
* A TPU implementation of RAE and pretrained weights.
* Sampling of RAE and SiT<sup>DH</sup> on TPU.

JAX/NNX:
* A lightweight JAX/NNX compatibility layer under `src_jax/`.
* Stage 2 train/sample entrypoints that reuse the existing YAML schema.
* Public CelebA workflows centered on backend-native `StabilityVAE + SiT-B`.
* Weights & Biases logging, Hugging Face upload, and optional FID scoring without duplicating the full PyTorch codebase.

## Documentation

Use the docs folder as the detailed guide for this branch:

- [docs/README.md](docs/README.md): documentation index
- [docs/architecture.md](docs/architecture.md): architecture and code map
- [docs/workflows.md](docs/workflows.md): practical runbooks for XLA and JAX/NNX training, sampling, and FID
- [docs/config-reference.md](docs/config-reference.md): YAML schema reference
- [pdf/main.pdf](pdf/main.pdf): detailed Vietnamese PDF for architecture, workflow, config, and operations
- [vaes-jax-celeba-kaggle-tpuv5e8-sitb.ipynb](vaes-jax-celeba-kaggle-tpuv5e8-sitb.ipynb): `TPU v5e-8` notebook for `StabilityVAE + SiT-B`, using `stage1.StabilityVAE`, `stage2.models.SiT.SiT`, default `num_workers=16`, `prefetch_factor=(4,4)`, enabled latent/activation diagnostics, and the single-tower `SiT-B` CelebA recipe (`hidden_size=768`, `depth=12`, `num_heads=12`)
- [vaes-jax-celeba-kaggle-tpuv5e8-sitb-resume.ipynb](vaes-jax-celeba-kaggle-tpuv5e8-sitb-resume.ipynb): resume-only `TPU v5e-8` notebook for the timestamped `CelebA256_SiT-B_StabilityVAE_jax_tpuv5e8-*` Orbax runs; it now clones the newest `checkpoint_*` into a fresh timestamped workdir, seeds a fresh `wandb_run.json`, and resumes with the original experiment name but a new W&B run ID
- [vaes-jax-celebahq-kaggle-tpuv5e8-sitb.ipynb](vaes-jax-celebahq-kaggle-tpuv5e8-sitb.ipynb): `TPU v5e-8` notebook for the `StabilityVAE + SiT-B` CelebA-HQ flow, using the same single-tower `SiT-B` recipe while pointing data/export steps at the CelebA-HQ 256 dataset
- [vaes-jax-celebahq-kaggle-tpuv5e8-sitb-resume.ipynb](vaes-jax-celebahq-kaggle-tpuv5e8-sitb-resume.ipynb): resume-only `TPU v5e-8` notebook for the timestamped `CelebAHQ256_SiT-B_StabilityVAE_jax_tpuv5e8-*` Orbax runs, using the same fresh-workdir and fresh-W&B-run resume flow as the CelebA notebook

## Environment

### Dependency Setup
1. Create environment and install via `uv`:
   ```bash
   conda create -n rae python=3.10 -y
   conda activate rae
   pip install uv
   
   # Install PyTorch 2.2.0 with CUDA 12.1
   uv pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 torchvision==0.20.1 -f https://storage.googleapis.com/libtpu-releases/index.html
   
   # Install other dependencies
   uv pip install timm==0.9.16 accelerate==0.23.0 torchdiffeq==0.2.5 wandb scipy torch-fidelity
   uv pip install "numpy<2" "transformers==4.57.1" einops
   ```

2. If you want to use the JAX/NNX adapter in `src_jax/`, also install:
   ```bash
   uv pip install "jax[cuda12]==0.5.1" flax==0.10.4 optax==0.2.4 orbax-checkpoint==0.11.16
   uv pip install ml-collections clu absl-py etils datasets diffusers huggingface_hub tensorflow-datasets
   ```

   Notes:
   - On TPU, replace `jax[cuda12]` with the TPU wheel flow you already use in your environment.
   - If Kaggle TPU rejects `jaxlib/xla_extension.so` with `cannot enable executable stack`, run `uv run python scripts/clear_elf_execstack.py --package jaxlib` once inside the same environment.
   - The JAX Kaggle notebooks now set `UV_PROJECT_ENVIRONMENT=/tmp/.venv`, `UV_CACHE_DIR=/tmp/uv-cache`, and use `uv sync -q` against the repo `pyproject.toml` before running the package-backed cells.
   - This repo patches the pinned `diffuse_nnx` checkout to import Dinov2 models from `transformers` subpackages, and the supported version for that path is `transformers==4.57.1`.
   - The same backend patch still lazy-loads `google-cloud-storage` for the remaining GCS-backed encoder assets, but the public `stage1.StabilityVAE` path now materializes its default `vae_trial1.pkl` locally from `stabilityai/sd-vae-ft-mse` instead of depending on the old backend bucket.
   - The backend bootstrap patch also fixes `diffuse_nnx` EMA initialization so the EMA starts from a copy of the current model instead of an all-zero parameter tree. Old Orbax checkpoints keep the EMA state they already saved, so start a fresh run if you need the corrected EMA trajectory.
   - The JAX adapter derives the Stage-1 latent `downsample_factor` and `latent_channels` from `misc.latent_size`, so backend preview sampling and FID operate at latent resolution instead of accidentally allocating image-resolution latent noise.
   - `src_jax/` pins `diffuse_nnx` at commit `023afd23c7b62a8cdb00e840b36a4ab8fc970bba` and bootstraps it into `~/.cache/rae_jax/diffuse_nnx` on first run.

## Data & Model Preparation

### Download Pre-trained Models

The upstream asset collection still contains the original RAE decoders, legacy DiT<sup>DH</sup> checkpoints, and latent-normalization stats. The default `StabilityVAE + SiT-B` CelebA workflow on this branch does not require an external RAE decoder download; on the first `StabilityVAE` run, the repo materializes the backend-compatible `vae_trial1.pkl` locally from `stabilityai/sd-vae-ft-mse`. If you already have a compatible pickle and want to skip that download, point `stage_1.params.pretrained_path` at it. If you still want to run a manual `RAE + SiTDH` experiment through YAML and the existing entrypoints, you can download the shared RAE assets with:


```bash

cd RAE
pip install huggingface_hub
hf download nyu-visionx/RAE-collections \
  --local-dir models 
```


To download specific models, run:
```bash
hf download nyu-visionx/RAE-collections \
  <remote_model_path> \
  --local-dir models 
```

### Prepare Dataset

1. Download ImageNet-1k.
2. Point Stage 1 and Stage 2 scripts to the training split via `--data-path`.


## Config-based Initialization

All training and sampling entrypoints are driven by OmegaConf YAML files. A
single config describes the Stage 1 autoencoder, the Stage 2 diffusion model,
and the solver used during training or inference. A minimal example looks like:

```yaml
stage_1:
   target: stage1.RAE  # or stage1.StabilityVAE on the JAX VAE path
   params: { ... }
   ckpt: <path_to_ckpt>  

stage_2:
   target: stage2.models.SiT.SiTDH  # or stage2.models.SiT.SiT for single-tower SiT
   params: { ... }
   ckpt: <path_to_ckpt>  

transport:
   params:
      path_type: Linear
      prediction: velocity
      ...
sampler:
   mode: ODE
   params:
      num_steps: 50
      ...
guidance:
   method: cfg/autoguidance
   scale: 1.0
   ...
misc:
   latent_size: [768, 16, 16]
   num_classes: 1000
training:
   ...

# Optional online validation during Stage 2 training.
eval:
   ...
```

- `stage_1` instantiates either the RAE encoder/decoder pair or the backend-native
  `StabilityVAE` on the JAX path. For Stage 1 training you can point to an
  existing RAE checkpoint via `stage_1.ckpt` or start from
  `pretrained_decoder_path`; the `StabilityVAE` path does not use those RAE
  decoder fields.
- `stage_2` defines the diffusion transformer. During sampling you must provide
  `ckpt`; during training you typically omit it so weights initialise randomly.
  `SiTDH` selects the DH/two-tower backbone, while `SiT` selects the single-tower
  `LightningDiT`-style backbone used by the new CelebA VAE notebooks.
- `transport`, `sampler`, and `guidance` select the forward/backward SDE/ODE
  integrator and optional classifier-free or autoguidance schedule.
- `misc` collects shapes, class counts, and scaling constants used by both
  stages.
- `training` contains defaults that the training scripts consume (epochs,
  learning rate, EMA decay, gradient accumulation, etc.).
- `eval` is optional. On the XLA path it controls TPU-native validation loss and
  optional host-side FID. On the JAX path it now also drives held-out
  validation loss plus optional online FID.

Stage 1 training configs additionally include a top-level `gan` block that
configures the discriminator architecture and the LPIPS/GAN loss schedule.


### Provided Configs:

#### Stage1

We release decoders for DINOv2-B, SigLIP-B, MAE-B, at `configs/stage1/pretrained/`.

There is also a training script for training a ViT-XL decoder on DINOv2-B: `configs/stage1/training/DINOv2-B_decXL.yaml`

#### Stage2

This branch still provides checked-in SiTDH config templates for both training and sampling at `configs/stage2/`.
The new CelebA `StabilityVAE + SiT-B` flow writes its JAX configs at notebook runtime as `CelebA256_StabilityVAE_*.yaml` and `CelebA256_SiT-B_StabilityVAE_*.yaml`.
The checked-in SiTDH sampling configs intentionally leave `stage_2.ckpt` and `guidance.guidance_model.ckpt` as `null` until you point them at SiTDH-compatible weights.

## Stage 1: Representation Autoencoder

### Sampling/Reconstruction

Use `src/stage1_sample.py` to encode/decode a single image:

```bash
python src/stage1_sample.py \
  --config <config> \
  --image assets/pixabay_cat.png \
```

For batched reconstructions and `.npz` export, run the XLA native DDP variant:

```bash
  python src/stage1_sample_ddp.py \
  --config <config> \
  --data-path <imagenet_val_split> \
  --sample-dir recon_samples \
  --image-size 256
```

The script writes per-image PNGs as well as a packed `.npz` suitable for FID.

## Stage 2: Latent Diffusion Transformer


For sampling, XLA branch only supports a manually implemented Euler sampler as `torchdiffeq` is not compatible with TPU.

### Training

Train Stage 2 on TPU with:

```bash
python src/train.py \
  --config configs/stage2/training/ImageNet256/SiTDH-XL_DINOv2-B.yaml \
  --data-path <imagenet_train_split> \
  --results-dir results \
  --image-size 256 \
  --precision bf16
```

To enable Weights & Biases logging, set:

```bash
export ENTITY=<wandb_entity>
export PROJECT=<wandb_project>
export WANDB_KEY=<wandb_api_key>
```

and add `--wandb` to the training command.

On the JAX path, each workdir now persists a `wandb_run.json` binding file.
Fresh workdirs create a fresh W&B run ID and start with `resume="never"`.
Later resumes of the same workdir reuse that exact run ID and automatically
rewind W&B history to the latest `checkpoint_*` step with `resume_from`, so the
run continues from the checkpoint instead of keeping stale post-checkpoint
history such as `120k -> 150k` after a crash. If you resume an older Orbax
workdir that predates `wandb_run.json`, pass `--wandb-run-id <existing_run_id>`
once so the adapter can bind that legacy workdir to the exact historical W&B
run before continuing. When you pass `--workdir` without `--exp-name`, the JAX
adapter now infers the resume experiment name from `wandb_run.json` when
present, otherwise from the workdir basename, so resume launches do not need a
second manual `--exp-name` override just to satisfy the strict W&B binding.
If the account or workspace does not have W&B rewind enabled yet, the adapter
now catches that private-preview error and falls back to plain
`id=<run_id>, resume="must"` so resume launches continue instead of crashing.
When a resumed training run restores an Orbax checkpoint, the runtime now
deletes the on-disk `checkpoint_*` directory immediately after the state is
loaded. Later saves also clear any older `checkpoint_*` directories before
writing the next checkpoint, so these JAX workdirs never need space for two
Orbax checkpoints at once.
This cleanup now normalizes both `Path` and string-style workdir values from
the backend trainer, so resume runs on the DH and VAE branches do not crash
while pruning restored checkpoints.

Stage 2 training now logs the following namespaces:

- `train/*`: loss, learning rate, optimizer steps/sec, images/sec, epoch, and gradient norm when clipping is enabled.
- `train/*` on the JAX path can additionally log latent and activation diagnostics when you enable `training.log_rae_latent_stats=true` and `training.log_activation_stats=true`, for example `train_rae_latent_rms`, `train_rae_latent_var`, `train_sitdh_output_rms`, `train_sitdh_output_var`, `train_sitdh_act_enc_00_rms`, and `train_sitdh_act_dec_01_var`.
- `eval/*`: periodic validation loss on a held-out `ImageFolder` split
  (`eval/ema_loss` by default, plus `eval/model_loss` when enabled), together
  with duration/batch counters for each validation pass.
- `checkpoint/*`: checkpoint save step.
- `network_samples`: current online-model preview images logged at the training step.
- `ema_network_samples`: EMA preview images logged at the training step.

To enable online validation loss, add an optional `eval` block to the Stage 2 training config:

```yaml
eval:
  data_path: data/imagenet/val/
  eval_every: 5000
  batch_size: 128        # per TPU core; defaults to the train micro batch size
  num_workers: 4         # defaults to training.num_workers
  max_batches: 32        # optional cap per rank to limit eval cost
  eval_model: false      # set true to also evaluate the non-EMA model
  fid_ref: data/imagenet/VIRTUAL_imagenet256_labeled.npz
  fid_every: 25000       # defaults to eval_every when omitted
  fid_num_samples: 4096  # trade off speed vs stability (e.g. 1024 / 4096 / 50000)
  fid_per_proc_batch_size: 4
  fid_batch_size: 64     # Inception batch size on the host CPU/GPU
  fid_device: cpu
  fid_num_threads: 96    # optional torch CPU thread count for host-side FID
  fid_label_sampling: equal
  fid_eval_model: false  # set true to also compute FID for the non-EMA model
```

This XLA branch runs validation loss on TPU inside the training loop. Optional FID
evaluation is also available at eval checkpoints by sampling on TPU and computing
Inception features on the host CPU or GPU.

On the JAX path, the default `FID-4K (cfg=...)` series follows the EMA model.
Setting `fid_eval_model: true` adds separate `FID-4K/model (cfg=...)` metrics
for the live online model without changing the default EMA series.

### JAX / NNX Compatibility Layer

This branch also ships a thin adapter in `src_jax/` that keeps the current
OmegaConf YAML files, but runs Stage 2 through a JAX/NNX backend instead of
duplicating the entire PyTorch codebase.

Main entrypoints:

```bash
python3 src_jax/train.py \
  --config configs/stage2/training/ImageNet256/SiTDH-XL_DINOv2-B.yaml \
  --data-path <imagenet_train_root> \
  --results-dir results_jax \
  --precision bf16 \
  --wandb \
  --set training.global_batch_size=256
```

On Kaggle TPU, generic ImageNet-scale runs can still start from
`--set training.num_workers=1` and then raise
`--set training.prefetch_factor=<n>` before jumping straight to a large worker
pool. This keeps the backend PyTorch loader compatible with
`persistent_workers=True` after JAX has already initialized multithreaded
runtime state, while still letting the host queue several ready batches ahead
of the TPU. The public CelebA VAE notebooks on this branch are the deliberate
exception: they pin both train and eval loaders to `num_workers=16` with
`prefetch_factor=4`. The adapter also disables the backend TensorBoard summary
writer on Kaggle and keeps metric logging on stdout plus wandb.

When `--wandb` is enabled on `src_jax/train.py`, the adapter also writes a
`wandb_run.json` file inside the selected workdir. That file is now the source
of truth for exact W&B resume behavior on later launches. When checkpoints
exist, later launches now auto-rewind the bound W&B run to the latest
checkpoint step before logging continues. If you point `--workdir` at a legacy
Orbax directory with checkpoints but no `wandb_run.json`, pass
`--wandb-run-id <existing_run_id>` once; otherwise the adapter aborts instead
of creating a fresh W&B run by accident.

```bash
python3 src_jax/sample.py \
  --config configs/stage2/sampling/ImageNet256/SiTDHXL-DINOv2-B_AG.yaml \
  --output sample_jax.png \
  --class-labels 207,360
```

```bash
python3 src_jax/sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/SiTDHXL-DINOv2-B.yaml \
  --sample-dir samples_jax \
  --num-samples 50000 \
  --label-sampling equal \
  --fid-ref /path/to/reference_stats.npz
```

```bash
python3 src_jax/stage1_sample.py \
  --config configs/stage1/pretrained/DINOv2-B_512.yaml \
  --image assets/pixabay_cat.png \
  --output recon_jax.png
```

```bash
python3 src_jax/build_stage1_stats.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --input /path/to/train_imagefolder \
  --output /path/to/stage1_stat.pt \
  --set stage_1.params.normalization_stat_path=/path/to/bootstrap_identity_stat.pt
```

```bash
python3 src_jax/reconstruct_folder.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --input /path/to/val_imagefolder \
  --output-dir recon_jax_dir \
  --batch-size 8
```

```bash
python3 src_jax/push_hf.py \
  --path results_jax/<run_name> \
  --repo-id <hf_user_or_org>/<repo_name>
```

Key behavior:

- `src_jax/` accepts the same top-level YAML blocks: `stage_1`, `stage_2`, `transport`, `sampler`, `guidance`, `misc`, `training`, and `eval`.
- `stage1.StabilityVAE` now maps to the backend-native `StabilityVAE` encoder, so the JAX VAE flow can reconstruct and sample without `pretrained_decoder_path` or `normalization_stat_path`.
- when `stage_1.params.pretrained_path` is unset on that path, the first run materializes the default backend-compatible `vae_trial1.pkl` from `stabilityai/sd-vae-ft-mse` into the cached backend checkout; set the field if you want to reuse an existing local pickle instead.
- `stage2.models.SiT.SiT` now maps to the backend single-tower `lightning_dit`, while `stage2.models.SiT.SiTDH` still maps to the DH/two-tower `lightning_ddt`.
- `stage_2.ckpt` compatibility now follows the selected Stage 2 target: single-tower `SiT` checkpoints must match the `lightning_dit` shape, while `SiTDH` checkpoints must match the DH/two-tower `lightning_ddt` shape.
- `--set key=value` applies OmegaConf CLI overrides without adding a second config format.
- `src_jax/build_stage1_stats.py` writes a PyTorch-compatible `stat.pt` file, so the same Stage 1 normalization stats can be reused by both the original repo code and the JAX adapter.
- the Stage 1-only JAX utilities (`src_jax/stage1_sample.py`, `src_jax/build_stage1_stats.py`, and `src_jax/reconstruct_folder.py`) can run with a YAML that only defines `stage_1`; they do not require `stage_2.target`.
- the shipped TPU notebooks on this branch now default `export PROJECT="moe-diffusion"`; the existing `run_name` strings already encode the dataset, `StabilityVAE`, backbone, and TPU pipeline, so no extra suffix was added to the experiment name.
- `src_jax/export_celebahq_hf.py` and `src_jax/export_celebahq_tfds.py` remain available if you specifically want a CelebA-HQ `ImageFolder` export, but the public VAE notebooks on this branch now mirror `jax-sit-dh` and build `celeba256_imgfolder` directly from Kaggle's `jessicali9530/celeba-dataset`.
- for dataset-specific Stage 1 stats, start from a bootstrap identity stats file (`mean=0`, `var=1`) and override `stage_1.params.normalization_stat_path` during the stats pass.
- `training.random_flip` and `eval.random_flip` now control the raw-image transform on the JAX Stage 2 path; the public CelebA notebooks on this branch enable training-time horizontal flips while keeping eval/FID unflipped.
- `ENTITY` / `PROJECT` / `WANDB_KEY` are bridged to the `WANDB_*` variables expected by the JAX backend.
- `--hf-repo-id` on `src_jax/train.py` uploads the finished workdir directly to Hugging Face.
- `src_jax/build_fid_stats.py` builds backend-native `fid_ref` files with the same Flax Inception detector used by JAX online FID.
- `training.log_rae_latent_stats=true` keeps the existing `train_rae_latent_*` metric names, but now covers whichever Stage 1 latent tensor the JAX path actually feeds into Stage 2, including `StabilityVAE`.
- `training.log_activation_stats=true` logs RMS and variance for the selected Stage 2 backbone: `train_sitdh_*` for DH/two-tower runs and `train_sit_*` for single-tower `SiT` runs.
- `training.prefetch_factor` and `eval.prefetch_factor` now forward directly into the host-side PyTorch `DataLoader` used by the JAX Stage 2 path, so you can deepen the per-worker prefetch queue without editing the cached backend checkout by hand.
- `src_jax/train.py` now binds each JAX workdir to a persisted `wandb_run.json`; resume reuses that exact W&B run ID and auto-rewinds the W&B history to the latest Orbax checkpoint step, while legacy workdirs without metadata require a one-time `--wandb-run-id <existing_run_id>`.
- the branch now keeps the public CelebA notebook flow below, with dataset preparation aligned to `jax-sit-dh`.
- `vaes-jax-celeba-kaggle-tpuv5e8-sitb.ipynb` is the `TPU v5e-8` sibling for that VAE flow, writing `CelebA256_SiT-B_StabilityVAE_jax_tpuv5e8.yaml`, enabling `training.random_flip=true`, `training.log_rae_latent_stats=true`, and `training.log_activation_stats=true`, defaulting both train and eval to `num_workers=16` with `prefetch_factor=4`, and keeping the DiT-B-style `SiT-B` shape (`hidden_size=768`, `depth=12`, `num_heads=12`, `patch_size=2`) referenced from the `shortcut-models` CelebA example while still training with this repo's flow-matching `sit` objective.
- `vaes-jax-celeba-kaggle-tpuv5e8-sitb-resume.ipynb` now copies the latest `checkpoint_*` from `CelebA256_SiT-B_StabilityVAE_jax_tpuv5e8-*` into a fresh timestamped resume workdir, seeds a fresh `wandb_run.json` with a new W&B run ID, keeps the original experiment name for readability, and re-applies the same `num_workers=16`, `prefetch_factor=4`, latent RMS/variance, and activation RMS/variance settings from the CLI during resume.
- The shipped VAE CelebA TPU notebooks materialize `/kaggle/working/celeba256_imgfolder`, keep `eval.data_path` on the `val` split, use `/kaggle/working/celeba256_imgfolder` as the train path, and keep both train and eval host loaders at `num_workers=16` with `prefetch_factor=4` while leaving latent/output/activation diagnostics enabled by default.
- The same branch now also keeps parallel CelebA-HQ notebooks: `vaes-jax-celebahq-kaggle*.ipynb` export `/kaggle/working/celebahq256_imgfolder`, keep the same `StabilityVAE + SiT-B` architecture, use `CelebAHQ256_SiT-B_StabilityVAE_jax_tpuv5e8-*` as the TPU resume pattern, and pin the same Stage 2 host-loader/diagnostic defaults (`training.num_workers=16`, `training.prefetch_factor=4`, `eval.prefetch_factor=4`, `training.log_rae_latent_stats=true`, `training.log_activation_stats=true`) so both datasets live on one branch without replacing each other.

Current limitation:

- The JAX path intentionally focuses on Stage 2 training/sampling and Stage 1 reconstruction. A dedicated JAX port of the adversarial Stage 1 decoder training loop is not shipped yet, because porting LPIPS + GAN + discriminator into JAX would make the branch substantially larger and harder to maintain.

### Sampling

`src/sample.py` uses the same config schema to draw a small batch of images on a
single device and saves them to `sample.png`:

```bash
python src/sample.py \
  --config <sample_config> \
  --seed 42
```


### Distributed sampling for evaluation

`src/sample_ddp.py` parallelises sampling across TPU cores, producing PNGs and an
FID-ready `.npz`:

```bash
python src/sample_ddp.py \
  --config <sample_config> \
  --sample-dir samples \
  --precision bf16 \
  --label-sampling equal
```
`--label-sampling {equal,random}`: `equal` uses exactly 50 images per class for FID-50k; `random` uniformly samples labels. Using `equal` brings consistently lower FID than `random` by around 0.1. We use `equal` by default.

Autoguidance and classifier-free guidance are controlled via the config’s
`guidance` block.

To compute FID immediately after sampling on the same TPU VM, point `sample_ddp.py`
to reference statistics:

```bash
python src/sample_ddp.py \
  --config <sample_config> \
  --sample-dir samples \
  --precision bf16 \
  --label-sampling equal \
  --fid-ref /path/to/VIRTUAL_imagenet256_labeled.npz \
  --fid-device cpu
```

This still samples on TPU, but the Inception feature extraction for FID runs on
the host CPU or GPU (`--fid-device auto|cpu|cuda`). You can also set
`--fid-num-threads` to control CPU parallelism explicitly. Rank 0 writes a
`<sample_dir>.fid.json` file with the result.

## Evaluation

### Online evaluation during training (TPU + host CPU/GPU)

The `eval` block above runs held-out validation loss inside `src/train.py` and
sends the resulting scalars to wandb when `--wandb` is enabled. If `fid_ref` is
set, the same block can also trigger periodic FID evaluation. Sampling still
runs on TPU; the Inception feature extraction runs on the host CPU or GPU.

For the JAX adapter, prefer building `fid_ref` with the backend-native detector
so the online FID path and reference statistics use the same Flax Inception
implementation:

```bash
python3 src_jax/build_fid_stats.py \
  --input /path/to/reference_images \
  --output /path/to/reference_stats.pkl \
  --batch-size 64 \
  --num-workers 8
```

The JAX adapter accepts either a backend-native `.pkl`/`.pickle` file from
`src_jax/build_fid_stats.py` or an older `.npz` file. When given `.npz`, the
adapter converts it to the backend pickle format on first use.

### Local FID from generated `.npz`

For custom datasets, first build reference statistics once:

```bash
python src/build_fid_stats.py \
  --input /path/to/reference_images \
  --output /path/to/reference_stats.npz \
  --device cpu \
  --num-workers 32 \
  --num-threads 96
```

`--input` accepts either an image folder (recursively scanned) or an existing
`.npy`/`.npz` image archive. This path remains the right choice for the
PyTorch/XLA utilities and for offline `torch-fidelity` scoring. For JAX online
FID, prefer `src_jax/build_fid_stats.py`.

If you already have a generated `.npz`, you can score it directly with:

```bash
python src/evaluate_fid.py \
  --samples /path/to/samples.npz \
  --ref /path/to/VIRTUAL_imagenet256_labeled.npz \
  --device cpu \
  --num-threads 96 \
  --output-json /path/to/samples.fid.json
```

This path uses `torch-fidelity`'s InceptionV3-compatible feature extractor and
works on CPU or CUDA. It avoids moving samples to another machine, but it is not
the ADM TensorFlow evaluator.

### ADM Suite FID setup (only available on GPU)

Use GPU with the ADM evaluation suite to score generated samples. You need to port the npz file generated from TPU to a GPU machine.

1. Clone the repo:

   ```bash
   git clone https://github.com/openai/guided-diffusion.git
   cd guided-diffusion/evaluation
   ```

2. Create an environment and install dependencies:

   ```bash
   conda create -n adm-fid python=3.10
   conda activate adm-fid
   pip install 'tensorflow[and-cuda]'==2.19 scipy requests tqdm
   ```

3. Download ImageNet statistics (256×256 shown here):

   ```bash
   wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
   ```

4. Evaluate:

   ```bash
   python evaluator.py VIRTUAL_imagenet256_labeled.npz /path/to/samples.npz
   ```
