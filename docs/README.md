# Documentation

This folder is the detailed documentation entrypoint for the XLA branch of RAE
and the lightweight JAX/NNX adapter shipped in `src_jax/`.

## Documents

- [Architecture](./architecture.md): system overview, module boundaries, and code map.
- [Workflows](./workflows.md): practical runbooks for setup, Stage 2 training, sampling, FID, wandb, and Hugging Face upload.
- [Config Reference](./config-reference.md): YAML schema and which scripts consume each block.
- [PDF Manual](../pdf/main.pdf): detailed Vietnamese PDF for architecture, workflows, config, and operations.
- [Kaggle Notebook (StabilityVAE + SiT-B)](../vaes-jax-celeba-kaggle.ipynb): end-to-end CelebA notebook for the backend-native VAE flow, matching the CelebA `ImageFolder` preparation used on `jax-sit-dh`.
- [Kaggle TPU Notebook (StabilityVAE + SiT-B)](../vaes-jax-celeba-kaggle-tpuv5e8-sitb.ipynb): `TPU v5e-8` notebook that uses `stage1.StabilityVAE`, single-tower `stage2.models.SiT.SiT`, default `num_workers=16`, `prefetch_factor=(4,4)`, `training.random_flip=true`, and default latent/activation diagnostics on the CelebA `SiT-B` recipe.
- [Kaggle TPU Notebook (StabilityVAE + SiT-B Resume)](../vaes-jax-celeba-kaggle-tpuv5e8-sitb-resume.ipynb): resume-only `TPU v5e-8` notebook for the newest `CelebA256_SiT-B_StabilityVAE_jax_tpuv5e8-*` workdir, reapplying the same `num_workers=16`, `prefetch_factor=(4,4)`, and diagnostics defaults.

## Recommended Reading Order

1. Read [Architecture](./architecture.md) to understand the Stage 1 and Stage 2 split.
2. Read [Workflows](./workflows.md) before running experiments.
3. Use [Config Reference](./config-reference.md) while editing YAML files.

## Branch Scope

This branch is centered on two runtime paths:

- `src/`: the existing `torch_xla` TPU path for Stage 2 training, Stage 2 sampling, Stage 1 reconstruction, and host-side FID utilities
- `src_jax/`: a thin JAX/NNX adapter that keeps the current YAML schema, logs to wandb, can upload workdirs to Hugging Face, and now exposes the public `StabilityVAE + SiT-B` CelebA notebook flow without forking the whole codebase

Limitations:

- `src/` still does not ship a dedicated Stage 1 training entrypoint on the XLA side.
- `src_jax/` intentionally does not port the full adversarial Stage 1 training loop yet.
