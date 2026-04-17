from __future__ import annotations

import argparse
import csv
import json
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_CELL_ANCHOR = 'stage2_results_dir = Path("/kaggle/working/results_jax_tpu")'
CONFIG_CELL_ANCHOR = "stage2_cfg_text = textwrap.dedent("
GMM_CELL_ANCHOR = "uv run python src_jax/build_source_gmm.py"
TRAIN_CELL_ANCHOR = 'run_name="CelebA256_SiTDH-B_DINOv2-B_moe1_jax_tpuv5e8-${timestamp}"'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate numbered CelebA SiTDH-B + moe1 Kaggle notebooks for pyramid_16k ablations.")
    parser.add_argument("--spec", required=True, help="Path to the ablation YAML spec.")
    parser.add_argument("--output-dir", default=None, help="Optional output-dir override for generated notebooks.")
    parser.add_argument("--only", default=None, help="Comma-separated run indexes or slugs to generate.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated notebooks and manifest.")
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not resolve to a mapping.")
    return payload


def _repo_relative(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _find_cell_index(nb: dict[str, Any], needle: str) -> int:
    for idx, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))
        if needle in source:
            return idx
    raise ValueError(f"Could not find notebook cell containing anchor: {needle!r}")


def _split_cell_source(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _apply_ablation_secret_policy(nb: dict[str, Any]) -> None:
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        updated = source.replace('secrets.get_secret("WANDB2")', 'secrets.get_secret("WANDB_Tung")')
        updated = updated.replace("secrets.get_secret('WANDB2')", "secrets.get_secret('WANDB_Tung')")
        if updated != source:
            cell["source"] = _split_cell_source(updated)


def _apply_title_cell(nb: dict[str, Any], title: str) -> None:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        if source.startswith("# "):
            cell["source"] = _split_cell_source(f"# {title}\n")
            return


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _format_float(value: float) -> str:
    if value.is_integer():
        return f"{value:.1f}"
    rendered = format(value, ".15g")
    return rendered.replace("e-0", "e-").replace("e+0", "e+")


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _render_mapping(name: str, payload: dict[str, Any], *, indent: int = 0) -> str:
    lines: list[str] = []

    def render_dict(mapping: dict[str, Any], depth: int) -> None:
        pad = " " * depth
        for key, value in mapping.items():
            if isinstance(value, dict):
                lines.append(f"{pad}{key}:")
                render_dict(value, depth + 2)
            else:
                lines.append(f"{pad}{key}: {_format_scalar(value)}")

    lines.append(" " * indent + f"{name}:")
    render_dict(payload, indent + 2)
    return "\n".join(lines)


def _path_expr_from_repo_root(relative_path: str | Path) -> str:
    path = Path(relative_path)
    expr = "repo_root"
    for part in path.parts:
        expr += f' / "{part}"'
    return expr


def _slug_to_safe_token(slug: str) -> str:
    return slug.replace("-", "_")


def _compact_slug_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        rendered = format(value, ".15g")
    else:
        rendered = str(value)
    return rendered.replace(".", "").replace("-", "m")


def _compact_metric_token(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, ".6g")
    return str(value)


def _build_slug(source_cfg: dict[str, Any]) -> str:
    token_specs = [
        ("m", "num_modes"),
        ("tau", "router_temperature"),
        ("vk", "var_kl_loss_weight"),
        ("bl", "balance_loss_weight"),
        ("ent", "entropy_loss_weight"),
        ("tv", "target_variance"),
        ("cd", "condition_dim"),
        ("hc", "hidden_channels"),
    ]
    return "-".join(["moe1", "pyr16k", *(f"{prefix}{_compact_slug_value(source_cfg[key])}" for prefix, key in token_specs)])


def _build_auto_tags(study_name: str, index: str, source_cfg: dict[str, Any], *, slug: str | None = None) -> list[str]:
    resolved_slug = slug or _build_slug(source_cfg)
    return [
        f"study:{study_name}",
        "dataset:celeba256",
        "model:sitdh-b",
        "stage1:dinov2-b",
        "source:moe1",
        "gmmfeat:pyramid_16k",
        f"idx:{index}",
        f"slug:{resolved_slug}",
        f"modes:{_compact_metric_token(source_cfg['num_modes'])}",
        f"tau:{_compact_metric_token(source_cfg['router_temperature'])}",
        f"var_kl:{_compact_metric_token(source_cfg['var_kl_loss_weight'])}",
        f"balance:{_compact_metric_token(source_cfg['balance_loss_weight'])}",
        f"entropy:{_compact_metric_token(source_cfg['entropy_loss_weight'])}",
        f"target_var:{_compact_metric_token(source_cfg['target_variance'])}",
        f"cond_dim:{_compact_metric_token(source_cfg['condition_dim'])}",
        f"hidden:{_compact_metric_token(source_cfg['hidden_channels'])}",
    ]


def _render_setup_cell(context: dict[str, Any]) -> str:
    stage1_cfg_expr = _path_expr_from_repo_root(context["stage1_cfg_relpath"])
    stage2_cfg_expr = _path_expr_from_repo_root(context["stage2_cfg_relpath"])
    return textwrap.dedent(
        f"""\
        from pathlib import Path

        repo_root = Path("/kaggle/working/RAE")
        celeba_src_dir = Path("/kaggle/input/datasets/jessicali9530/celeba-dataset/img_align_celeba/img_align_celeba")
        celeba_split_csv = Path("/kaggle/input/datasets/jessicali9530/celeba-dataset/list_eval_partition.csv")
        celeba_root = Path("/kaggle/working/celeba256_imgfolder")
        stage1_cfg_path = {stage1_cfg_expr}
        stage2_cfg_path = {stage2_cfg_expr}
        bootstrap_stats_path = Path("/kaggle/working/bootstrap_identity_stat.pt")
        latent_stats_path = Path("/kaggle/working/celeba256_stage1_latent_stat_tpu.pt")
        fid_stats_path = Path("{context['fid_stats_path']}")
        source_gmm_path = Path("{context['source_gmm_path']}")
        stage1_single_recon_path = Path("/kaggle/working/celeba256_stage1_single_recon_tpu.png")
        stage1_recon_dir = Path("/kaggle/working/celeba256_stage1_recon_val_tpu")
        stage2_results_dir = Path("{context['results_dir']}")

        image_size = 256
        stats_batch_size = 64
        stats_num_workers = 16
        recon_batch_size = 16
        recon_num_workers = 16
        fid_num_workers = 16
        recon_limit = 2048  # bỏ limit nếu muốn reconstruct toàn bộ val split

        print("repo_root:", repo_root)
        print("celeba_root:", celeba_root)
        print("stage1_cfg_path:", stage1_cfg_path)
        print("stage2_cfg_path:", stage2_cfg_path)
        print("source_gmm_path:", source_gmm_path)
        print("fid_stats_path:", fid_stats_path)
        print("stage2_results_dir:", stage2_results_dir)
        """
    )


def _render_config_cell(context: dict[str, Any]) -> str:
    stage1_cfg_expr = _path_expr_from_repo_root(context["stage1_cfg_relpath"])
    stage2_cfg_expr = _path_expr_from_repo_root(context["stage2_cfg_relpath"])
    stage1_payload = {
        "stage_1": {
            "target": "stage1.RAE",
            "params": {
                "encoder_cls": "Dinov2withNorm",
                "encoder_config_path": "facebook/dinov2-with-registers-base",
                "encoder_input_size": 224,
                "encoder_params": {
                    "dinov2_path": "facebook/dinov2-with-registers-base",
                    "normalize": True,
                },
                "decoder_config_path": "configs/decoder/ViTXL",
                "pretrained_decoder_path": "models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
                "noise_tau": 0.0,
                "reshape_to_2d": True,
                "normalization_stat_path": "{latent_stats_path.as_posix()}",
            },
        }
    }
    stage2_payload = {
        "stage_1": {
            "target": "stage1.RAE",
            "ckpt": None,
            "params": deepcopy(stage1_payload["stage_1"]["params"]),
        },
        "stage_2": {
            "target": "stage2.models.SiT.SiTDH",
            "ckpt": None,
            "params": {
                "input_size": 16,
                "patch_size": 1,
                "in_channels": 768,
                "hidden_size": [768, 2048],
                "depth": [12, 2],
                "num_heads": [12, 16],
                "mlp_ratio": 4.0,
                "class_dropout_prob": 0.0,
                "num_classes": 1,
                "use_qknorm": False,
                "use_swiglu": True,
                "use_rope": True,
                "use_rmsnorm": True,
                "use_pos_embed": True,
                "wo_shift": False,
            },
        },
        "transport": {
            "params": {
                "path_type": "Linear",
                "prediction": "velocity",
                "loss_weight": None,
                "time_dist_type": "uniform",
            }
        },
        "sampler": {
            "mode": "ODE",
            "params": {
                "sampling_method": "euler",
                "num_steps": 50,
                "atol": 1.0e-6,
                "rtol": 1.0e-3,
                "reverse": False,
            },
        },
        "guidance": {
            "method": "cfg",
            "scale": 1.0,
            "t_min": 0.0,
            "t_max": 1.0,
        },
        "misc": {
            "latent_size": [768, 16, 16],
            "num_classes": 1,
            "time_dist_shift_dim": 196608,
            "time_dist_shift_base": 4096,
        },
        "eval": {
            "data_path": "{celeba_val_path}",
            "eval_every": 5000,
            "batch_size": 4,
            "num_workers": context["eval_cfg"]["num_workers"],
            "max_batches": 32,
            "eval_model": False,
            "fid_ref": "{fid_stats_path.as_posix()}",
            "fid_every": context["eval_cfg"]["fid_every"],
            "fid_num_samples": context["eval_cfg"]["fid_num_samples"],
            "fid_per_proc_batch_size": 4,
            "fid_batch_size": 128,
            "prefetch_factor": context["eval_cfg"]["prefetch_factor"],
        },
        "source": deepcopy(context["source_cfg"]),
        "training": {
            "global_seed": 0,
            "epochs": 200,
            "global_batch_size": context["train_cfg"]["global_batch_size"],
            "grad_accum_steps": 1,
            "ema_decay": 0.9995,
            "num_workers": context["train_cfg"]["num_workers"],
            "prefetch_factor": context["train_cfg"]["prefetch_factor"],
            "log_every": 10,
            "ckpt_every": context["train_cfg"]["ckpt_every"],
            "sample_every": 5000,
            "base_lr": 1.0e-4,
            "final_lr": 1.0e-5,
            "beta": [0.9, 0.95],
            "wd": 0.0,
            "schedule_type": "linear",
            "decay_start_epoch": 150,
            "decay_end_epoch": 200,
            "clip_grad": 1.0,
            "log_rae_latent_stats": context["train_cfg"]["log_rae_latent_stats"],
            "log_activation_stats": context["train_cfg"]["log_activation_stats"],
        },
    }
    stage2_payload["source"]["gmm_stats_path"] = "{source_gmm_path.as_posix()}"

    stage1_yaml = _render_mapping("stage_1", stage1_payload["stage_1"], indent=0)
    stage2_sections = [_render_mapping(key, value, indent=0) for key, value in stage2_payload.items()]
    stage2_yaml = "\n\n".join(stage2_sections)

    return """%%bash
set -euo pipefail

cd /kaggle/working/RAE

uv run python - <<'PYCFG'
from pathlib import Path
import textwrap

repo_root = Path("/kaggle/working/RAE")
celeba_root = Path("/kaggle/working/celeba256_imgfolder")
celeba_val_path = (celeba_root / "val").as_posix()
stage1_cfg_path = {stage1_cfg_expr}
stage2_cfg_path = {stage2_cfg_expr}
bootstrap_stats_path = Path("/kaggle/working/bootstrap_identity_stat.pt")
latent_stats_path = Path("/kaggle/working/celeba256_stage1_latent_stat_tpu.pt")
fid_stats_path = Path("{fid_stats_path}")
source_gmm_path = Path("{source_gmm_path}")

import textwrap

import torch

bootstrap_stats_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {{
        "mean": torch.zeros((1, 1, 1), dtype=torch.float32),
        "var": torch.ones((1, 1, 1), dtype=torch.float32),
        "count": 0,
    }},
    bootstrap_stats_path,
)

stage1_cfg_text = textwrap.dedent(
    f\"\"\"
{stage1_yaml}
    \"\"\"
).strip() + "\\n"

stage2_cfg_text = textwrap.dedent(
    f\"\"\"
{stage2_yaml}
    \"\"\"
).strip() + "\\n"

stage1_cfg_path.parent.mkdir(parents=True, exist_ok=True)
stage2_cfg_path.parent.mkdir(parents=True, exist_ok=True)
stage1_cfg_path.write_text(stage1_cfg_text, encoding="utf-8")
stage2_cfg_path.write_text(stage2_cfg_text, encoding="utf-8")

print(f"Wrote {{stage1_cfg_path}}")
print(f"Wrote {{stage2_cfg_path}}")
print(f"Bootstrap stats path: {{bootstrap_stats_path}}")
print(f"Stage 2 source artifact path: {{source_gmm_path}}")
PYCFG
""".format(
        stage1_cfg_expr=stage1_cfg_expr,
        stage2_cfg_expr=stage2_cfg_expr,
        fid_stats_path=context["fid_stats_path"],
        source_gmm_path=context["source_gmm_path"],
        stage1_yaml=stage1_yaml,
        stage2_yaml=stage2_yaml,
    )


def _render_gmm_cell(context: dict[str, Any]) -> str:
    gmm_cfg = context["gmm_build_cfg"]
    return textwrap.dedent(
        f"""\
        %%bash
        set -euo pipefail

        cd /kaggle/working/RAE

        uv run python scripts/clear_elf_execstack.py --package jaxlib --quiet-unchanged

        uv run python src_jax/build_source_gmm.py \\
          --config {context['stage1_cfg_relpath'].as_posix()} \\
          --input /kaggle/working/celeba256_imgfolder/train \\
          --output {context['source_gmm_path']} \\
          --batch-size {gmm_cfg['batch_size']} \\
          --num-workers {gmm_cfg['num_workers']} \\
          --num-modes {context['source_cfg']['num_modes']} \\
          --chunk-size {gmm_cfg['chunk_size']} \\
          --feature-extractor {gmm_cfg['feature_extractor']} \\
          --storage-dtype {gmm_cfg['storage_dtype']} \\
          --compute-dtype {gmm_cfg['compute_dtype']}
        """
    )


def _render_train_cell(context: dict[str, Any]) -> str:
    tags_csv = ",".join(context["wandb_tags"])
    train_cfg = context["train_cfg"]
    eval_cfg = context["eval_cfg"]
    return textwrap.dedent(
        f"""\
        %%bash
        set -euo pipefail

        cd /kaggle/working/RAE

        uv run python scripts/clear_elf_execstack.py --package jaxlib --quiet-unchanged

        timestamp="$(TZ=Asia/Bangkok date +%Y%m%d-%H%M%S)"
        run_slug="{context['run_prefix']}"
        run_name="${{run_slug}}-${{timestamp}}"
        wandb_group="{context['wandb_group']}"
        wandb_tags="{tags_csv}"

        export ENTITY="{context['wandb_entity']}"
        export PROJECT="{context['wandb_project']}"
        export RAE_JAX_REBUILD_BACKEND=1

        uv run python src_jax/train.py \\
          --config {context['stage2_cfg_relpath'].as_posix()} \\
          --data-path /kaggle/working/celeba256_imgfolder \\
          --results-dir {context['results_dir']} \\
          --precision bf16 \\
          --exp-name "${{run_name}}" \\
          --wandb \\
          --wandb-entity {context['wandb_entity']} \\
          --wandb-project "${{PROJECT}}" \\
          --wandb-group "${{wandb_group}}" \\
          --wandb-tags "${{wandb_tags}}" \\
          --set training.global_batch_size={train_cfg['global_batch_size']} \\
          --set training.num_workers={train_cfg['num_workers']} \\
          --set training.prefetch_factor={train_cfg['prefetch_factor']} \\
          --set training.ckpt_every={train_cfg['ckpt_every']} \\
          --set training.log_rae_latent_stats={str(train_cfg['log_rae_latent_stats']).lower()} \\
          --set training.log_activation_stats={str(train_cfg['log_activation_stats']).lower()} \\
          --set eval.num_workers={eval_cfg['num_workers']} \\
          --set eval.prefetch_factor={eval_cfg['prefetch_factor']} \\
          --set eval.fid_ref={context['fid_stats_path']} \\
          --set eval.fid_every={eval_cfg['fid_every']} \\
          --set eval.fid_num_samples={eval_cfg['fid_num_samples']}
        """
    )


def _prepare_context(spec: dict[str, Any], run_spec: dict[str, Any]) -> dict[str, Any]:
    shared = spec["shared_defaults"]
    source_cfg = _deep_merge(shared["source"], run_spec.get("source_overrides", {}))
    train_cfg = _deep_merge(shared["train"], run_spec.get("train_overrides", {}))
    eval_cfg = _deep_merge(shared["eval"], run_spec.get("eval_overrides", {}))
    gmm_build_cfg = deepcopy(shared["gmm_build"])
    index = str(run_spec["index"])
    slug = str(run_spec.get("slug") or _build_slug(source_cfg))
    run_prefix = f"{index}-{slug}"
    safe_token = _slug_to_safe_token(run_prefix)
    stage2_cfg_relpath = Path(spec["paths"]["generated_config_dir"]) / f"{run_prefix}.yaml"
    source_gmm_path = f"{spec['paths']['source_gmm_prefix']}_{safe_token}.npz"
    auto_tags = _build_auto_tags(spec["study_name"], index, source_cfg, slug=slug)
    extra_tags = [str(tag) for tag in run_spec.get("extra_tags", [])]
    wandb_tags = auto_tags + [tag for tag in extra_tags if tag not in auto_tags]
    return {
        "study_name": spec["study_name"],
        "index": index,
        "slug": slug,
        "run_prefix": run_prefix,
        "notebook_name": f"{run_prefix}.ipynb",
        "stage1_cfg_relpath": Path(spec["paths"]["stage1_cfg_path"]),
        "stage2_cfg_relpath": stage2_cfg_relpath,
        "source_gmm_path": source_gmm_path,
        "fid_stats_path": spec["paths"]["fid_stats_path"],
        "results_dir": spec["paths"]["results_dir"],
        "source_cfg": source_cfg,
        "train_cfg": train_cfg,
        "eval_cfg": eval_cfg,
        "gmm_build_cfg": gmm_build_cfg,
        "wandb_entity": spec["wandb"]["entity"],
        "wandb_project": spec["wandb"]["project"],
        "wandb_group": spec["wandb"]["group"],
        "wandb_tags": wandb_tags,
    }


def _select_runs(runs: list[dict[str, Any]], only: str | None) -> list[dict[str, Any]]:
    if not only:
        return runs
    requested = {token.strip() for token in only.split(",") if token.strip()}
    selected = [run for run in runs if str(run["index"]) in requested or str(run.get("slug", "")) in requested]
    if not selected:
        raise ValueError(f"No runs matched --only={only!r}.")
    return selected


def _write_manifest(manifest_path: Path, contexts: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "filename",
        "slug",
        "gmm_path",
        "gmm_feature_extractor",
        "num_modes",
        "router_temperature",
        "var_kl_loss_weight",
        "balance_loss_weight",
        "entropy_loss_weight",
        "target_variance",
        "condition_dim",
        "hidden_channels",
        "wandb_group",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for context in contexts:
            writer.writerow(
                {
                    "index": context["index"],
                    "filename": context["notebook_name"],
                    "slug": context["slug"],
                    "gmm_path": context["source_gmm_path"],
                    "gmm_feature_extractor": context["gmm_build_cfg"]["feature_extractor"],
                    "num_modes": context["source_cfg"]["num_modes"],
                    "router_temperature": _compact_metric_token(context["source_cfg"]["router_temperature"]),
                    "var_kl_loss_weight": _compact_metric_token(context["source_cfg"]["var_kl_loss_weight"]),
                    "balance_loss_weight": _compact_metric_token(context["source_cfg"]["balance_loss_weight"]),
                    "entropy_loss_weight": _compact_metric_token(context["source_cfg"]["entropy_loss_weight"]),
                    "target_variance": _compact_metric_token(context["source_cfg"]["target_variance"]),
                    "condition_dim": context["source_cfg"]["condition_dim"],
                    "hidden_channels": context["source_cfg"]["hidden_channels"],
                    "wandb_group": context["wandb_group"],
                }
            )


def main() -> None:
    args = build_parser().parse_args()
    spec_path = _repo_relative(args.spec)
    spec = _load_yaml(spec_path)
    template_path = _repo_relative(spec["template_notebook"])
    output_dir = _repo_relative(args.output_dir or spec["output_dir"])
    runs = _select_runs(spec["runs"], args.only)

    template_nb = json.loads(template_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts: list[dict[str, Any]] = []

    for run_spec in runs:
        context = _prepare_context(spec, run_spec)
        contexts.append(context)
        notebook_path = output_dir / context["notebook_name"]
        if notebook_path.exists() and not args.overwrite:
            raise FileExistsError(f"{notebook_path} already exists. Pass --overwrite to replace it.")

        notebook_payload = deepcopy(template_nb)
        _apply_title_cell(notebook_payload, context["run_prefix"])
        _apply_ablation_secret_policy(notebook_payload)

        setup_idx = _find_cell_index(notebook_payload, SETUP_CELL_ANCHOR)
        config_idx = _find_cell_index(notebook_payload, CONFIG_CELL_ANCHOR)
        gmm_idx = _find_cell_index(notebook_payload, GMM_CELL_ANCHOR)
        train_idx = _find_cell_index(notebook_payload, TRAIN_CELL_ANCHOR)

        notebook_payload["cells"][setup_idx]["source"] = _split_cell_source(_render_setup_cell(context))
        notebook_payload["cells"][config_idx]["source"] = _split_cell_source(_render_config_cell(context))
        notebook_payload["cells"][gmm_idx]["source"] = _split_cell_source(_render_gmm_cell(context))
        notebook_payload["cells"][train_idx]["source"] = _split_cell_source(_render_train_cell(context))

        notebook_path.write_text(json.dumps(notebook_payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    _write_manifest(output_dir / "manifest.csv", contexts)


if __name__ == "__main__":
    main()
