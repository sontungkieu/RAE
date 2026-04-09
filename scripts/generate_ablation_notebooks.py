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
TRAIN_CELL_ANCHOR = 'run_name="CelebA256_SiT-B_StabilityVAE_moe1_jax_tpuv5e8-${timestamp}"'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate numbered ablation notebooks from the CelebA VAE+moe1 TPU template.")
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


def _render_mapping(name: str, payload: dict[str, Any], *, indent: int = 4) -> str:
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


def _compact_metric_token(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, ".6g")
    return str(value)


def _build_auto_tags(study_name: str, index: str, source_cfg: dict[str, Any], source_defaults: dict[str, Any]) -> list[str]:
    tags = [
        f"idx:{index}",
        f"study:{study_name}",
        f"modes:{source_cfg['num_modes']}",
        f"tau:{_compact_metric_token(source_cfg['router_temperature'])}",
        f"vk:{_compact_metric_token(source_cfg['var_kl_loss_weight'])}",
    ]
    optional_fields = {
        "hidden_channels": "hc",
        "condition_dim": "cd",
        "balance_loss_weight": "bal",
        "entropy_loss_weight": "ent",
    }
    for field, tag_key in optional_fields.items():
        if source_cfg.get(field) != source_defaults.get(field):
            tags.append(f"{tag_key}:{_compact_metric_token(source_cfg[field])}")
    return tags


def _render_setup_cell(context: dict[str, Any]) -> str:
    stage2_cfg_expr = _path_expr_from_repo_root(context["stage2_cfg_relpath"])
    stage1_cfg_expr = _path_expr_from_repo_root(context["stage1_cfg_relpath"])
    return textwrap.dedent(
        f"""\
        from pathlib import Path

        repo_root = Path("/kaggle/working/RAE")
        celeba_src_dir = Path("/kaggle/input/datasets/jessicali9530/celeba-dataset/img_align_celeba/img_align_celeba")
        celeba_split_csv = Path("/kaggle/input/datasets/jessicali9530/celeba-dataset/list_eval_partition.csv")
        celeba_root = Path("/kaggle/working/celeba256_imgfolder")
        stage1_cfg_path = {stage1_cfg_expr}
        stage2_cfg_path = {stage2_cfg_expr}
        source_gmm_path = Path("{context['source_gmm_path']}")
        fid_stats_path = Path("{context['fid_stats_path']}")
        stage1_single_recon_path = Path("/kaggle/working/celeba256_vae_stage1_single_recon_tpu.png")
        stage1_recon_dir = Path("/kaggle/working/celeba256_vae_stage1_recon_val_tpu")
        stage2_results_dir = Path("{context['results_dir']}")

        image_size = 256
        recon_batch_size = 16
        recon_num_workers = 16
        fid_num_workers = 32
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
            "target": "stage1.StabilityVAE",
            "params": {
                "sample_size": 256,
                "latent_channels": 4,
                "downsample_factor": 8,
                "raw_mean": [0.865, -0.278, 0.216, 0.374],
                "raw_std": [4.86, 5.32, 3.94, 3.99],
                "final_mean": 0.0,
                "final_std": 0.5,
            },
        }
    }
    stage2_payload = {
        "stage_1": {
            "target": "stage1.StabilityVAE",
            "ckpt": None,
            "params": {
                "sample_size": 256,
                "latent_channels": 4,
                "downsample_factor": 8,
                "raw_mean": [0.865, -0.278, 0.216, 0.374],
                "raw_std": [4.86, 5.32, 3.94, 3.99],
                "final_mean": 0.0,
                "final_std": 0.5,
            },
        },
        "stage_2": {
            "target": "stage2.models.SiT.SiT",
            "ckpt": None,
            "params": {
                "input_size": 32,
                "patch_size": 2,
                "in_channels": 4,
                "hidden_size": 768,
                "depth": 12,
                "num_heads": 12,
                "mlp_ratio": 4.0,
                "class_dropout_prob": 0.0,
                "num_classes": 1,
                "use_qknorm": False,
                "use_swiglu": True,
                "use_rope": True,
                "use_rmsnorm": True,
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
            "latent_size": [4, 32, 32],
            "num_classes": 1,
            "time_dist_shift_dim": 4096,
            "time_dist_shift_base": 4096,
        },
        "source": deepcopy(context["source_cfg"]),
        "eval": deepcopy(context["eval_cfg"]),
        "training": deepcopy(context["train_cfg"]),
    }
    stage2_payload["source"]["gmm_stats_path"] = "{source_gmm_path.as_posix()}"
    stage2_payload["eval"]["data_path"] = '{(celeba_root / "val").as_posix()}'
    stage2_payload["eval"]["fid_ref"] = "{fid_stats_path.as_posix()}"

    stage1_yaml = _render_mapping("stage_1", stage1_payload["stage_1"], indent=4)
    stage2_sections = [
        _render_mapping(key, value, indent=4)
        for key, value in stage2_payload.items()
    ]
    stage2_yaml = "\n\n".join(stage2_sections)

    return textwrap.dedent(
        f"""\
        %%bash
        set -euo pipefail

        cd /kaggle/working/RAE

        uv run python - <<'PYCFG'
        from pathlib import Path
        import textwrap

        repo_root = Path("/kaggle/working/RAE")
        celeba_root = Path("/kaggle/working/celeba256_imgfolder")
        stage1_cfg_path = {stage1_cfg_expr}
        stage2_cfg_path = {stage2_cfg_expr}
        source_gmm_path = Path("{context['source_gmm_path']}")
        fid_stats_path = Path("{context['fid_stats_path']}")

        stage1_cfg_text = textwrap.dedent(
            \"\"\"
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
        print(f"Stage 2 source artifact path: {{source_gmm_path}}")
        PYCFG
        """
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
          --precision {gmm_cfg['precision']} \\
          --num-modes {context['source_cfg']['num_modes']} \\
          --em-iters {context['source_cfg']['em_iters']} \\
          --em-restarts {context['source_cfg']['em_restarts']}
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
    slug = str(run_spec["slug"])
    index = str(run_spec["index"])
    run_prefix = f"{index}-{slug}"
    safe_token = _slug_to_safe_token(run_prefix)
    stage2_cfg_relpath = Path(spec["paths"]["generated_config_dir"]) / f"{run_prefix}.yaml"
    source_gmm_path = f"{spec['paths']['source_gmm_prefix']}_{safe_token}.npz"
    source_defaults = shared["source"]
    auto_tags = _build_auto_tags(spec["study_name"], index, source_cfg, source_defaults)
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
    selected = [run for run in runs if str(run["index"]) in requested or str(run["slug"]) in requested]
    if not selected:
        raise ValueError(f"No runs matched --only={only!r}.")
    return selected


def _validate_spec(spec: dict[str, Any]) -> None:
    required_keys = {"study_name", "template_notebook", "output_dir", "wandb", "paths", "shared_defaults", "runs"}
    missing = sorted(required_keys - spec.keys())
    if missing:
        raise KeyError(f"Spec is missing required keys: {', '.join(missing)}")


def _write_manifest(rows: list[dict[str, Any]], manifest_path: Path) -> None:
    fieldnames = [
        "index",
        "filename",
        "slug",
        "run_name_prefix",
        "wandb_group",
        "wandb_tags",
        "stage2_cfg_path",
        "gmm_path",
        "results_dir",
        "num_modes",
        "router_temperature",
        "var_kl_loss_weight",
        "hidden_channels",
        "condition_dim",
        "balance_loss_weight",
        "entropy_loss_weight",
        "train_overrides",
        "eval_overrides",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_notebooks(args: argparse.Namespace) -> list[Path]:
    spec_path = _repo_relative(args.spec)
    spec = _load_yaml(spec_path)
    _validate_spec(spec)

    template_path = _repo_relative(spec["template_notebook"])
    output_dir = _repo_relative(args.output_dir or spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_runs = _select_runs(list(spec["runs"]), args.only)
    notebook_template = json.loads(template_path.read_text(encoding="utf-8"))

    setup_idx = _find_cell_index(notebook_template, SETUP_CELL_ANCHOR)
    config_idx = _find_cell_index(notebook_template, CONFIG_CELL_ANCHOR)
    gmm_idx = _find_cell_index(notebook_template, GMM_CELL_ANCHOR)
    train_idx = _find_cell_index(notebook_template, TRAIN_CELL_ANCHOR)

    generated_paths: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []

    for run_spec in selected_runs:
        context = _prepare_context(spec, run_spec)
        notebook = deepcopy(notebook_template)
        notebook["cells"][setup_idx]["source"] = _split_cell_source(_render_setup_cell(context))
        notebook["cells"][config_idx]["source"] = _split_cell_source(_render_config_cell(context))
        notebook["cells"][gmm_idx]["source"] = _split_cell_source(_render_gmm_cell(context))
        notebook["cells"][train_idx]["source"] = _split_cell_source(_render_train_cell(context))

        output_path = output_dir / context["notebook_name"]
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Re-run with --overwrite to replace it.")
        output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        generated_paths.append(output_path)

        source_cfg = context["source_cfg"]
        manifest_rows.append(
            {
                "index": context["index"],
                "filename": context["notebook_name"],
                "slug": context["slug"],
                "run_name_prefix": context["run_prefix"],
                "wandb_group": context["wandb_group"],
                "wandb_tags": ",".join(context["wandb_tags"]),
                "stage2_cfg_path": context["stage2_cfg_relpath"].as_posix(),
                "gmm_path": context["source_gmm_path"],
                "results_dir": context["results_dir"],
                "num_modes": source_cfg["num_modes"],
                "router_temperature": _compact_metric_token(source_cfg["router_temperature"]),
                "var_kl_loss_weight": _compact_metric_token(source_cfg["var_kl_loss_weight"]),
                "hidden_channels": source_cfg["hidden_channels"],
                "condition_dim": source_cfg["condition_dim"],
                "balance_loss_weight": _compact_metric_token(source_cfg["balance_loss_weight"]),
                "entropy_loss_weight": _compact_metric_token(source_cfg["entropy_loss_weight"]),
                "train_overrides": json.dumps(run_spec.get("train_overrides", {}), sort_keys=True),
                "eval_overrides": json.dumps(run_spec.get("eval_overrides", {}), sort_keys=True),
            }
        )

    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} already exists. Re-run with --overwrite to replace it.")
    _write_manifest(manifest_rows, manifest_path)
    return generated_paths


def main() -> None:
    args = build_parser().parse_args()
    generated = generate_notebooks(args)
    for path in generated:
        try:
            print(path.relative_to(REPO_ROOT))
        except ValueError:
            print(path)


if __name__ == "__main__":
    main()
