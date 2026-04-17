# Documentation

This folder is the detailed documentation entrypoint for the XLA branch of RAE
and the lightweight JAX/NNX adapter shipped in `src_jax/`.

## Documents

- [Architecture](./architecture.md): system overview, module boundaries, and code map.
- [Workflows](./workflows.md): practical runbooks for setup, Stage 2 training, sampling, FID, wandb, and Hugging Face upload.
- [Config Reference](./config-reference.md): YAML schema and which scripts consume each block.
- [PDF Manual](../pdf/main.pdf): detailed Vietnamese PDF for architecture, workflows, config, and operations.
- [Kaggle TPU Notebook (SiTDH-B moe1)](../raes-jax-celeba-kaggle-tpuv5e8-sitdh-b-moe1.ipynb): sibling `TPU v5e-8` notebook for the DH/two-tower `SiTDH-B + moe1` setup, keeping the same Kaggle/JAX flow but now building an explicit `pyramid_16k` source artifact as `celeba256_source_gmm_pyr16k.npz` before Stage 2 training.
- [Kaggle TPU Notebook (SiTDH-B moe1 Resume)](../raes-jax-celeba-kaggle-tpuv5e8-sitdh-b-moe1-resume.ipynb): minimal `TPU v5e-8` resume-only notebook for the `SiTDH-B + moe1` runs, checking both the restored Orbax workdir and the persisted `celeba256_source_gmm_pyr16k.npz` artifact, then cloning the latest checkpoint into a fresh timestamped resume workdir with a fresh `wandb_run.json` so the resumed job keeps the original experiment name but logs to a new W&B run.
- [Kaggle TPU Notebook (CelebA-HQ SiTDH-B moe1)](../raes-jax-celebahq-kaggle-tpuv5e8-sitdh-b-moe1.ipynb): CelebA-HQ TPU `v5e-8` notebook for the `SiTDH-B + moe1` recipe, also building `celebahq256_source_gmm_pyr16k.npz` explicitly via `pyramid_16k`.
- [Kaggle TPU Notebook (CelebA-HQ SiTDH-B moe1 Resume)](../raes-jax-celebahq-kaggle-tpuv5e8-sitdh-b-moe1-resume.ipynb): resume-only CelebA-HQ TPU notebook for the `SiTDH-B + moe1` runs, cloning the latest checkpoint into a fresh timestamped resume workdir with a fresh `wandb_run.json` while keeping the original experiment name and checking the persisted `celebahq256_source_gmm_pyr16k.npz` artifact.
- [Kaggle GPU Notebook (TuneDinoV2 Scratch)](../tunedinov2-stage1-scratch-kaggle.ipynb): Stage 1 decoder-from-scratch workflow with CelebA / CelebA-HQ `ImageFolder` preparation and full W&B logging to `TuneDinoV2`.
- [Kaggle GPU Notebook (TuneDinoV2 Finetune DINOv2)](../tunedinov2-stage1-finetune-dinov2-kaggle.ipynb): Stage 1 finetuning workflow that initializes from the ImageNet decoder or an existing Stage 1 checkpoint, unfreezes DINOv2, and logs the full run to `TuneDinoV2`.

## Recommended Reading Order

1. Read [Architecture](./architecture.md) to understand the Stage 1 and Stage 2 split.
2. Read [Workflows](./workflows.md) before running experiments.
3. Use [Config Reference](./config-reference.md) while editing YAML files.

## Branch Scope

This branch is centered on two runtime paths:

- `src/`: the existing `torch_xla` TPU path for Stage 2 training, Stage 2 sampling, Stage 1 reconstruction, host-side FID utilities, and now a local PyTorch/CUDA Stage 1 trainer at `src/train_stage1_rae.py`
- `src_jax/`: a thin JAX/NNX adapter that keeps the current YAML schema, logs to wandb, can upload workdirs to Hugging Face, and supports Stage 2 train/sample plus Stage 1 reconstruction, folder reconstruction, and latent-stat building without forking the whole codebase

Limitations:

- `src/` still does not ship a dedicated Stage 1 training entrypoint on the XLA / TPU side; the new Stage 1 trainer is local PyTorch/CUDA only.
- `src_jax/` intentionally does not port the full adversarial Stage 1 training loop yet.
