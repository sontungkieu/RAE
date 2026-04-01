# Documentation

This folder is the detailed documentation entrypoint for the XLA branch of RAE
and the lightweight JAX/NNX adapter shipped in `src_jax/`.

## Documents

- [Architecture](./architecture.md): system overview, module boundaries, and code map.
- [Workflows](./workflows.md): practical runbooks for setup, Stage 2 training, sampling, FID, wandb, and Hugging Face upload.
- [Config Reference](./config-reference.md): YAML schema and which scripts consume each block.
- [PDF Manual](../pdf/main.pdf): detailed Vietnamese PDF for architecture, workflows, config, and operations.
- [Kaggle Notebook (StabilityVAE + SiT-B + moe1)](../vaes-jax-celebahq-kaggle-moe1.ipynb): end-to-end CelebA-HQ notebook for the backend-native VAE flow with an offline `GMM + SourceMoE` source build step.
- [Kaggle TPU Notebook (StabilityVAE + SiT-B + moe1)](../vaes-jax-celebahq-kaggle-tpuv5e8-sitb-moe1.ipynb): `TPU v5e-8` notebook that uses `stage1.StabilityVAE`, single-tower `stage2.models.SiT.SiT`, `training.random_flip=true`, and a generated Stage 2 config with `source.enabled=true`.
- [Kaggle TPU Notebook (StabilityVAE + SiT-B + moe1 Resume)](../vaes-jax-celebahq-kaggle-tpuv5e8-sitb-moe1-resume.ipynb): resume-only `TPU v5e-8` notebook for the newest `CelebAHQ256_SiT-B_StabilityVAE_moe1_jax_tpuv5e8-*` workdir.

## Recommended Reading Order

1. Read [Architecture](./architecture.md) to understand the Stage 1 and Stage 2 split.
2. Read [Workflows](./workflows.md) before running experiments.
3. Use [Config Reference](./config-reference.md) while editing YAML files.

## Branch Scope

This branch is centered on two runtime paths:

- `src/`: the existing `torch_xla` TPU path for Stage 2 training, Stage 2 sampling, Stage 1 reconstruction, and host-side FID utilities
- `src_jax/`: a thin JAX/NNX adapter that keeps the current YAML schema, logs to wandb, can upload workdirs to Hugging Face, and now exposes the public `StabilityVAE + SiT-B + moe1` CelebA-HQ notebook flow without forking the whole codebase

Limitations:

- `src/` still does not ship a dedicated Stage 1 training entrypoint on the XLA side.
- `src_jax/` intentionally does not port the full adversarial Stage 1 training loop yet.
