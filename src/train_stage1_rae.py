from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import torch
import torch.distributed as dist
from torch import nn
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import wandb

from disc import LPIPS, build_discriminator, hinge_d_loss, vanilla_d_loss, vanilla_g_loss
from utils.fid_utils import compute_fid_with_metadata, write_fid_result
from utils.model_utils import instantiate_from_config
from utils.optim_utils import build_scheduler


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass
class RunState:
    epoch: int = 0
    global_step: int = 0
    best_val_lpips: float | None = None


@dataclass
class DistState:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_master: bool


class ImageFileDataset(torch.utils.data.Dataset[tuple[torch.Tensor, int]]):
    """Dataset wrapper for a flat folder of images when ImageFolder is inconvenient."""

    def __init__(self, root: Path, transform: transforms.Compose):
        self.root = root
        self.transform = transform
        self.paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Stage 1 RAE decoder with optional DINOv2 encoder finetuning.")
    parser.add_argument("--config", required=True, help="Path to the YAML config containing stage_1/training/gan blocks.")
    parser.add_argument("--results-dir", default="results_stage1", help="Directory where checkpoints and samples are stored.")
    parser.add_argument("--exp-name", default=None, help="Optional explicit experiment name.")
    parser.add_argument("--resume", default=None, help="Resume a previous Stage 1 training checkpoint created by this script.")
    parser.add_argument("--train-data-path", default=None, help="Override the training ImageFolder path.")
    parser.add_argument("--val-data-path", default=None, help="Override the validation ImageFolder path.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-entity", default=None, help="Override WANDB entity.")
    parser.add_argument("--wandb-project", default="TuneDinoV2", help="W&B project name.")
    parser.add_argument("--wandb-group", default=None, help="Optional W&B group.")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated W&B tags.")
    parser.add_argument("--set", dest="set_values", action="append", default=[], help="OmegaConf override, e.g. training.epochs=10")
    return parser.parse_args()


def setup_distributed() -> DistState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available()

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if use_cuda else "gloo"
        timeout_seconds = int(os.environ.get("RAE_DDP_TIMEOUT_SECONDS", "7200"))
        if use_cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))

    if use_cuda:
        device = torch.device("cuda", local_rank if world_size > 1 else 0)
    else:
        device = torch.device("cpu")

    return DistState(
        enabled=world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_master=rank == 0,
    )


def barrier_if_needed(dist_state: DistState) -> None:
    if dist_state.enabled:
        if dist_state.device.type == "cuda":
            dist.barrier(device_ids=[dist_state.local_rank])
        else:
            dist.barrier()


def cleanup_distributed(dist_state: DistState) -> None:
    if dist_state.enabled and dist.is_initialized():
        if dist_state.device.type == "cuda":
            dist.barrier(device_ids=[dist_state.local_rank])
        else:
            dist.barrier()
        dist.destroy_process_group()


def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def distributed_mean_dict(metrics: dict[str, float], *, device: torch.device, enabled: bool, world_size: int) -> dict[str, float]:
    if not enabled or not metrics:
        return metrics
    keys = sorted(metrics.keys())
    payload = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    payload /= world_size
    return {key: float(value) for key, value in zip(keys, payload.tolist())}


def sync_context(module: nn.Module, should_sync: bool):
    if should_sync or not isinstance(module, DDP):
        return nullcontext()
    return module.no_sync()


def seed_everything(seed: int, *, rank: int = 0) -> None:
    final_seed = seed + rank
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(final_seed)


def resolve_repo_path(config_path: Path, value: Any) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.as_posix()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate.as_posix()
    config_candidate = (config_path.parent / path).resolve()
    if config_candidate.exists():
        return config_candidate.as_posix()
    return str(value)


def parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def load_config(config_path: str, set_values: list[str]) -> tuple[DictConfig, Path]:
    path = Path(config_path).expanduser().resolve()
    cfg = OmegaConf.load(path)
    if set_values:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(set_values))
    return cfg, path


def ensure_data_path(path: str | None, *, label: str) -> Path:
    if not path:
        raise ValueError(f"Missing {label} path.")
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} path not found: {resolved}")
    return resolved


def build_image_dataset(root: Path, transform: transforms.Compose) -> torch.utils.data.Dataset[tuple[torch.Tensor, int]]:
    try:
        dataset = ImageFolder(root.as_posix(), transform=transform)
        if len(dataset) > 0:
            return dataset
    except Exception:
        pass
    return ImageFileDataset(root=root, transform=transform)


def build_transforms(image_size: int, random_flip: bool) -> tuple[transforms.Compose, transforms.Compose]:
    interpolation = transforms.InterpolationMode.BICUBIC
    common = [
        transforms.Resize(image_size, interpolation=interpolation),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    train_ops: list[Any] = [
        transforms.Resize(image_size, interpolation=interpolation),
        transforms.CenterCrop(image_size),
    ]
    if random_flip:
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops.append(transforms.ToTensor())
    return transforms.Compose(train_ops), transforms.Compose(common)


def module_grad_norm(module: nn.Module) -> float:
    sq_norm = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        norm = float(param.grad.detach().float().norm(2).item())
        sq_norm += norm * norm
    return sq_norm ** 0.5


def module_param_norm(module: nn.Module) -> float:
    sq_norm = 0.0
    for param in module.parameters():
        norm = float(param.detach().float().norm(2).item())
        sq_norm += norm * norm
    return sq_norm ** 0.5


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(-10.0 * math.log10(mse))


def make_preview(real: torch.Tensor, recon: torch.Tensor, max_items: int = 8) -> np.ndarray:
    count = max(1, min(max_items, real.shape[0]))
    stacked = torch.cat([real[:count], recon[:count]], dim=0)
    grid = make_grid(stacked, nrow=count, normalize=True, value_range=(0, 1))
    return grid.mul(255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()


def save_preview(path: Path, preview: np.ndarray | None) -> None:
    if preview is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(path)


def recent_mean(history: deque[float]) -> float:
    return float(sum(history) / len(history)) if history else 0.0


def extract_images(batch: Any) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def select_precision(training_cfg: dict[str, Any], device: torch.device) -> tuple[str, Any, GradScaler, GradScaler]:
    precision = str(training_cfg.get("precision", "fp32")).lower()
    if device.type != "cuda":
        return "fp32", nullcontext, GradScaler("cpu", enabled=False), GradScaler("cpu", enabled=False)

    if precision == "fp16":
        autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        return precision, autocast_ctx, GradScaler("cuda", enabled=True), GradScaler("cuda", enabled=True)
    if precision == "bf16":
        autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return precision, autocast_ctx, GradScaler("cuda", enabled=False), GradScaler("cuda", enabled=False)
    return "fp32", nullcontext, GradScaler("cuda", enabled=False), GradScaler("cuda", enabled=False)


def calculate_adaptive_weight(
    recon_objective: torch.Tensor,
    adversarial_objective: torch.Tensor,
    last_layer: torch.Tensor,
    disc_weight: float,
    max_d_weight: float,
) -> torch.Tensor:
    if disc_weight <= 0:
        return torch.zeros((), device=last_layer.device, dtype=last_layer.dtype)

    recon_grads = torch.autograd.grad(
        recon_objective,
        last_layer,
        retain_graph=True,
        allow_unused=True,
    )[0]
    adv_grads = torch.autograd.grad(
        adversarial_objective,
        last_layer,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if recon_grads is None or adv_grads is None:
        return torch.as_tensor(disc_weight, device=last_layer.device, dtype=last_layer.dtype)

    recon_norm = recon_grads.float().norm(2)
    adv_norm = adv_grads.float().norm(2)
    if not torch.isfinite(recon_norm) or not torch.isfinite(adv_norm) or adv_norm.item() <= 0:
        return torch.as_tensor(disc_weight, device=last_layer.device, dtype=last_layer.dtype)

    weight = torch.clamp(recon_norm / (adv_norm + 1e-4), 0.0, max_d_weight)
    return (weight * disc_weight).to(device=last_layer.device, dtype=last_layer.dtype).detach()


def build_stage1_optimizer(
    rae: nn.Module,
    training_cfg: dict[str, Any],
    train_encoder: bool,
) -> torch.optim.Optimizer:
    opt_cfg: dict[str, Any] = dict(training_cfg.get("optimizer", {}))
    decoder_lr = float(opt_cfg.get("lr", training_cfg.get("base_lr", 2e-4)))
    encoder_lr = float(training_cfg.get("encoder_lr", opt_cfg.get("encoder_lr", decoder_lr)))
    betas = opt_cfg.get("betas", opt_cfg.get("beta", (0.9, 0.95)))
    weight_decay = float(opt_cfg.get("weight_decay", opt_cfg.get("wd", 0.0)))
    eps = float(opt_cfg.get("eps", 1e-8))

    param_groups: list[dict[str, Any]] = [
        {"params": [param for param in rae.decoder.parameters() if param.requires_grad], "lr": decoder_lr},
    ]
    if train_encoder:
        encoder_params = [param for param in rae.encoder.parameters() if param.requires_grad]
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": encoder_lr})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=decoder_lr,
        betas=tuple(float(v) for v in betas),
        weight_decay=weight_decay,
        eps=eps,
    )
    training_cfg.setdefault("base_lr", decoder_lr)
    training_cfg.setdefault("final_lr", float(training_cfg.get("final_lr", decoder_lr)))
    return optimizer


def prepare_wandb(
    *,
    enabled: bool,
    entity: str | None,
    project: str,
    group: str | None,
    tags: str,
    exp_name: str,
    cli_args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    if not enabled:
        return
    api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY")
    if api_key:
        wandb.login(key=api_key)
    run_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    wandb.init(
        entity=entity,
        project=project,
        group=group,
        tags=run_tags or None,
        name=exp_name,
        config={
            "cli": vars(cli_args),
            "config": config,
        },
    )
    wandb.define_metric("train/step")
    wandb.define_metric("*", step_metric="train/step")


def save_checkpoint(
    path: Path,
    *,
    rae: nn.Module,
    disc: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    sched_g: Any,
    sched_d: Any,
    g_scaler: GradScaler,
    d_scaler: GradScaler,
    run_state: RunState,
    config: dict[str, Any],
) -> None:
    payload = {
        "model": rae.state_dict(),
        "discriminator": disc.state_dict(),
        "optimizer_g": opt_g.state_dict(),
        "optimizer_d": opt_d.state_dict(),
        "scheduler_g": sched_g.state_dict(),
        "scheduler_d": sched_d.state_dict(),
        "scaler_g": g_scaler.state_dict(),
        "scaler_d": d_scaler.state_dict(),
        "epoch": run_state.epoch,
        "global_step": run_state.global_step,
        "best_val_lpips": run_state.best_val_lpips,
        "config": config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def restore_checkpoint(
    path: Path,
    *,
    rae: nn.Module,
    disc: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    sched_g: Any,
    sched_d: Any,
    g_scaler: GradScaler,
    d_scaler: GradScaler,
) -> RunState:
    state = torch.load(path, map_location="cpu")
    rae.load_state_dict(state["model"], strict=True)
    disc.load_state_dict(state["discriminator"], strict=True)
    opt_g.load_state_dict(state["optimizer_g"])
    opt_d.load_state_dict(state["optimizer_d"])
    sched_g.load_state_dict(state["scheduler_g"])
    sched_d.load_state_dict(state["scheduler_d"])
    if state.get("scaler_g"):
        g_scaler.load_state_dict(state["scaler_g"])
    if state.get("scaler_d"):
        d_scaler.load_state_dict(state["scaler_d"])
    return RunState(
        epoch=int(state.get("epoch", 0)),
        global_step=int(state.get("global_step", 0)),
        best_val_lpips=state.get("best_val_lpips"),
    )


@torch.no_grad()
def evaluate(
    *,
    rae: nn.Module,
    loader: DataLoader,
    lpips_model: LPIPS,
    device: torch.device,
    max_batches: int | None,
    image_log_count: int,
    autocast_ctx: Any,
) -> tuple[dict[str, float], np.ndarray | None]:
    rae.eval()
    losses_l1: list[float] = []
    losses_lpips: list[float] = []
    psnrs: list[float] = []
    latent_rms: list[float] = []
    latent_var: list[float] = []
    preview: np.ndarray | None = None

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = extract_images(batch).to(device, non_blocking=True)
        with autocast_ctx():
            latents, recon = rae(images, return_latents=True)
        recon = recon.clamp(0, 1)
        real_pm1 = images * 2 - 1
        recon_pm1 = recon * 2 - 1
        losses_l1.append(float(torch.mean(torch.abs(recon - images)).item()))
        losses_lpips.append(float(lpips_model(recon_pm1.float(), real_pm1.float(), reduction="mean").item()))
        psnrs.append(compute_psnr(recon, images))
        latent_rms.append(float(torch.sqrt(torch.mean(latents.float() ** 2)).item()))
        latent_var.append(float(torch.var(latents.float(), unbiased=False).item()))
        if preview is None:
            preview = make_preview(images.detach().cpu(), recon.detach().cpu(), max_items=image_log_count)

    return {
        "val/l1": float(np.mean(losses_l1)) if losses_l1 else 0.0,
        "val/lpips": float(np.mean(losses_lpips)) if losses_lpips else 0.0,
        "val/psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "val/latent_rms": float(np.mean(latent_rms)) if latent_rms else 0.0,
        "val/latent_var": float(np.mean(latent_var)) if latent_var else 0.0,
    }, preview


@torch.no_grad()
def run_reconstruction_fid(
    *,
    rae: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    autocast_ctx: Any,
    fid_ref: str,
    fid_batch_size: int,
    fid_device: str,
    fid_num_threads: int | None,
    workdir: Path,
    global_step: int,
) -> dict[str, float]:
    rae.eval()
    fid_dir = workdir / "fid_eval" / f"step_{global_step:07d}"
    fid_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = extract_images(batch).to(device, non_blocking=True)
        with autocast_ctx():
            _, recon = rae(images, return_latents=True)
        recon = recon.clamp(0, 1)
        arr = recon.mul(255).round().clamp(0, 255).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        shard_path = fid_dir / f"recon_batch_{batch_idx:05d}.npz"
        np.savez(shard_path, arr_0=arr)
        shard_paths.append(shard_path)

    if not shard_paths:
        raise RuntimeError("FID evaluation did not produce any reconstructed samples.")

    fid_value, actual_num_samples = compute_fid_with_metadata(
        shard_paths,
        reference_path=fid_ref,
        batch_size=fid_batch_size,
        device=fid_device,
        num_threads=fid_num_threads,
    )
    write_fid_result(
        fid_dir / "reconstruction.fid.json",
        fid=fid_value,
        sample_path=fid_dir,
        reference_path=fid_ref,
        batch_size=fid_batch_size,
        device=fid_device,
        num_samples=actual_num_samples,
        num_threads=fid_num_threads,
    )
    for shard_path in shard_paths:
        if shard_path.exists():
            shard_path.unlink()
    return {
        "val/fid": float(fid_value),
        "val/fid_num_samples": float(actual_num_samples),
    }


def main() -> None:
    args = parse_args()
    dist_state = setup_distributed()
    wandb_enabled = args.wandb and dist_state.is_master

    try:
        full_cfg, config_path = load_config(args.config, args.set_values)
        cfg_dict = OmegaConf.to_container(full_cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise TypeError("The root config must resolve to a mapping.")

        stage1_cfg = deepcopy(cfg_dict.get("stage_1"))
        if not stage1_cfg:
            raise ValueError("Config must define a stage_1 block.")
        training_cfg: dict[str, Any] = deepcopy(cfg_dict.get("training", {}))
        gan_cfg: dict[str, Any] = deepcopy(cfg_dict.get("gan", {}))
        data_cfg: dict[str, Any] = deepcopy(cfg_dict.get("data", {}))
        eval_cfg: dict[str, Any] = deepcopy(cfg_dict.get("eval", {}))

        seed_everything(int(training_cfg.get("global_seed", 0)), rank=dist_state.rank)

        precision, autocast_ctx, g_scaler, d_scaler = select_precision(training_cfg, dist_state.device)
        image_size = int(training_cfg.get("image_size", 256))
        random_flip = bool(training_cfg.get("random_flip", True))
        train_tf, val_tf = build_transforms(image_size=image_size, random_flip=random_flip)

        if "disc" in gan_cfg and isinstance(gan_cfg["disc"], dict):
            arch_cfg = gan_cfg["disc"].get("arch", {})
            if isinstance(arch_cfg, dict) and arch_cfg.get("dino_ckpt_path") is not None:
                arch_cfg["dino_ckpt_path"] = resolve_repo_path(config_path, arch_cfg.get("dino_ckpt_path"))

        if data_cfg.get("train_path") is not None:
            data_cfg["train_path"] = resolve_repo_path(config_path, data_cfg.get("train_path"))
        if data_cfg.get("val_path") is not None:
            data_cfg["val_path"] = resolve_repo_path(config_path, data_cfg.get("val_path"))
        if eval_cfg.get("data_path") is not None:
            eval_cfg["data_path"] = resolve_repo_path(config_path, eval_cfg.get("data_path"))
        if eval_cfg.get("fid_ref") is not None:
            eval_cfg["fid_ref"] = resolve_repo_path(config_path, eval_cfg.get("fid_ref"))

        train_root = ensure_data_path(
            args.train_data_path or data_cfg.get("train_path"),
            label="training data",
        )
        val_root_value = args.val_data_path or eval_cfg.get("data_path") or data_cfg.get("val_path")
        val_root = ensure_data_path(val_root_value, label="validation data") if val_root_value else None

        train_set = build_image_dataset(train_root, train_tf)
        val_set = build_image_dataset(val_root, val_tf) if val_root is not None else None

        batch_size = int(training_cfg.get("batch_size", 16))
        num_workers = int(training_cfg.get("num_workers", 4))
        grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
        if batch_size <= 0:
            raise ValueError("training.batch_size must be > 0.")
        if grad_accum_steps <= 0:
            raise ValueError("training.grad_accum_steps must be > 0.")

        train_sampler = (
            DistributedSampler(
                train_set,
                num_replicas=dist_state.world_size,
                rank=dist_state.rank,
                shuffle=True,
                drop_last=False,
            )
            if dist_state.enabled
            else None
        )
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=dist_state.device.type == "cuda",
            drop_last=False,
        )

        val_loader = None
        if val_set is not None and dist_state.is_master:
            val_loader = DataLoader(
                val_set,
                batch_size=int(eval_cfg.get("batch_size", batch_size)),
                shuffle=False,
                num_workers=int(eval_cfg.get("num_workers", num_workers)),
                pin_memory=dist_state.device.type == "cuda",
                drop_last=False,
            )

        stage1_params: dict[str, Any] = dict(stage1_cfg.get("params", {}))
        for key in ("decoder_config_path", "pretrained_decoder_path", "normalization_stat_path"):
            resolved = resolve_repo_path(config_path, stage1_params.get(key))
            if resolved is not None:
                stage1_params[key] = resolved
        if stage1_cfg.get("ckpt") is not None:
            stage1_cfg["ckpt"] = resolve_repo_path(config_path, stage1_cfg.get("ckpt"))
        if "encoder_params" in stage1_params and isinstance(stage1_params["encoder_params"], dict):
            dinov2_path = resolve_repo_path(config_path, stage1_params["encoder_params"].get("dinov2_path"))
            if dinov2_path is not None:
                stage1_params["encoder_params"]["dinov2_path"] = dinov2_path
        stage1_cfg["params"] = stage1_params

        train_encoder = bool(training_cfg.get("train_encoder", False))
        if train_encoder and stage1_params.get("noise_tau", 0.0):
            raise ValueError("Finetuning the encoder expects stage_1.params.noise_tau=0.0 for stable recon training.")

        epochs = int(training_cfg.get("epochs", 16))
        log_every = int(training_cfg.get("log_every", 50))
        image_log_every = int(training_cfg.get("image_log_every", 200))
        eval_every = int(training_cfg.get("eval_every", 1))
        save_every = int(training_cfg.get("save_every", 1))
        eval_max_batches = parse_optional_int(eval_cfg.get("max_batches"))
        clip_grad = float(training_cfg.get("clip_grad", 1.0))
        recon_weight = float(training_cfg.get("recon_weight", 1.0))
        perceptual_weight = float(gan_cfg.get("loss", {}).get("perceptual_weight", 1.0))
        disc_weight = float(gan_cfg.get("loss", {}).get("disc_weight", 1.0))
        max_d_weight = float(gan_cfg.get("loss", {}).get("max_d_weight", 1.0e4))
        lpips_start = int(gan_cfg.get("loss", {}).get("lpips_start", 0))
        disc_start = int(gan_cfg.get("loss", {}).get("disc_start", 0))
        disc_upd_start = int(gan_cfg.get("loss", {}).get("disc_upd_start", disc_start))
        disc_updates = int(gan_cfg.get("loss", {}).get("disc_updates", 1))
        visual_count = int(training_cfg.get("num_visuals", 8))
        if grad_accum_steps > 1 and disc_updates != 1:
            raise ValueError("training.grad_accum_steps > 1 currently requires gan.loss.disc_updates=1.")

        fid_ref = eval_cfg.get("fid_ref")
        fid_enabled = fid_ref is not None
        fid_every = int(eval_cfg.get("fid_every", eval_every))
        fid_batch_size = int(eval_cfg.get("fid_batch_size", 64))
        fid_device = str(eval_cfg.get("fid_device", "auto"))
        fid_num_threads = parse_optional_int(eval_cfg.get("fid_num_threads"))
        if fid_enabled and val_loader is None:
            raise ValueError("eval.fid_ref requires validation data via eval.data_path or data.val_path.")

        cfg_dict["stage_1"] = deepcopy(stage1_cfg)
        cfg_dict["gan"] = deepcopy(gan_cfg)
        cfg_dict["data"] = deepcopy(data_cfg)
        cfg_dict["eval"] = deepcopy(eval_cfg)

        rae_module = instantiate_from_config(stage1_cfg).to(dist_state.device)
        for param in rae_module.encoder.parameters():
            param.requires_grad_(train_encoder)
        rae_module.encoder.train(mode=train_encoder)
        rae_module.decoder.train()

        lpips_model = LPIPS().eval().to(dist_state.device)
        disc_module, augment = build_discriminator(gan_cfg.get("disc", {}), device=dist_state.device)

        opt_g = build_stage1_optimizer(rae_module, training_cfg, train_encoder=train_encoder)
        opt_d = torch.optim.AdamW(
            [param for param in disc_module.parameters() if param.requires_grad],
            lr=float(gan_cfg.get("disc", {}).get("optimizer", {}).get("lr", 2e-4)),
            betas=tuple(float(v) for v in gan_cfg.get("disc", {}).get("optimizer", {}).get("betas", (0.5, 0.9))),
            weight_decay=float(gan_cfg.get("disc", {}).get("optimizer", {}).get("weight_decay", 0.0)),
            eps=float(gan_cfg.get("disc", {}).get("optimizer", {}).get("eps", 1e-8)),
        )

        steps_per_epoch = max(math.ceil(len(train_loader) / grad_accum_steps), 1)
        sched_g, _ = build_scheduler(opt_g, steps_per_epoch, training_cfg)
        disc_sched_cfg = {
            "scheduler": deepcopy(gan_cfg.get("disc", {}).get("scheduler", {})),
            "base_lr": float(gan_cfg.get("disc", {}).get("optimizer", {}).get("lr", 2e-4)),
            "final_lr": float(
                gan_cfg.get("disc", {}).get("scheduler", {}).get(
                    "final_lr",
                    gan_cfg.get("disc", {}).get("optimizer", {}).get("lr", 2e-4),
                )
            ),
        }
        sched_d, _ = build_scheduler(opt_d, steps_per_epoch, disc_sched_cfg)

        if args.resume:
            restore_state = restore_checkpoint(
                Path(args.resume).expanduser().resolve(),
                rae=rae_module,
                disc=disc_module,
                opt_g=opt_g,
                opt_d=opt_d,
                sched_g=sched_g,
                sched_d=sched_d,
                g_scaler=g_scaler,
                d_scaler=d_scaler,
            )
        else:
            restore_state = RunState()

        if dist_state.enabled:
            ddp_kwargs: dict[str, Any] = {
                "device_ids": [dist_state.local_rank] if dist_state.device.type == "cuda" else None,
                "output_device": dist_state.local_rank if dist_state.device.type == "cuda" else None,
                "broadcast_buffers": False,
                "find_unused_parameters": False,
            }
            rae: nn.Module = DDP(rae_module, **ddp_kwargs)
            disc: nn.Module = DDP(disc_module, **ddp_kwargs)
        else:
            rae = rae_module
            disc = disc_module

        disc_loss_name = str(gan_cfg.get("loss", {}).get("disc_loss", "hinge")).lower()
        if disc_loss_name == "hinge":
            disc_loss_fn = hinge_d_loss
        elif disc_loss_name == "vanilla":
            disc_loss_fn = vanilla_d_loss
        else:
            raise ValueError(f"Unsupported gan.loss.disc_loss '{disc_loss_name}'.")

        gen_loss_name = str(gan_cfg.get("loss", {}).get("gen_loss", "vanilla")).lower()
        if gen_loss_name != "vanilla":
            raise ValueError("Only gan.loss.gen_loss=vanilla is currently supported.")

        results_dir = Path(args.results_dir).expanduser().resolve()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = args.exp_name or f"stage1-rae-{timestamp}"
        workdir = results_dir / exp_name
        ckpt_dir = workdir / "checkpoints"
        sample_dir = workdir / "samples"
        if dist_state.is_master:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            sample_dir.mkdir(parents=True, exist_ok=True)
            (workdir / "config.resolved.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
        barrier_if_needed(dist_state)

        prepare_wandb(
            enabled=wandb_enabled,
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            tags=args.wandb_tags,
            exp_name=exp_name,
            cli_args=args,
            config=cfg_dict,
        )

        if dist_state.is_master:
            effective_batch_size = batch_size * dist_state.world_size * grad_accum_steps
            print(
                f"Stage 1 trainer using device={dist_state.device}, "
                f"world_size={dist_state.world_size}, "
                f"micro_batch_size={batch_size}, "
                f"grad_accum_steps={grad_accum_steps}, "
                f"effective_batch_size={effective_batch_size}."
            )

        run_state = restore_state
        recon_history: deque[float] = deque(maxlen=log_every)
        lpips_history: deque[float] = deque(maxlen=log_every)
        adv_history: deque[float] = deque(maxlen=log_every)
        disc_history: deque[float] = deque(maxlen=log_every)
        psnr_history: deque[float] = deque(maxlen=log_every)

        for epoch in range(run_state.epoch, epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            rae.train()
            unwrap_module(rae).encoder.train(mode=train_encoder)
            disc.train()
            preview_for_log: np.ndarray | None = None
            opt_g.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)

            accum_counter = 0
            window_recon = 0.0
            window_lpips = 0.0
            window_adv = 0.0
            window_disc = 0.0
            window_psnr = 0.0
            window_latent_rms = 0.0
            window_latent_var = 0.0
            window_logits_real = 0.0
            window_logits_fake = 0.0
            window_adaptive_weight = 0.0
            window_gen_total = 0.0

            for batch_idx, batch in enumerate(train_loader):
                images = extract_images(batch).to(dist_state.device, non_blocking=True)
                last_batch = batch_idx == len(train_loader) - 1
                should_sync = ((accum_counter + 1) % grad_accum_steps == 0) or last_batch

                with sync_context(rae, should_sync):
                    with autocast_ctx():
                        latents, recon = rae(images, return_latents=True)
                    recon = recon.clamp(0, 1)
                    preview_for_log = make_preview(images.detach().cpu(), recon.detach().cpu(), max_items=visual_count)

                    recon_loss = torch.mean(torch.abs(recon - images))
                    real_pm1 = images * 2 - 1
                    recon_pm1 = recon * 2 - 1
                    use_lpips = epoch >= lpips_start
                    lpips_loss = (
                        lpips_model(recon_pm1.float(), real_pm1.float(), reduction="mean")
                        if use_lpips
                        else torch.zeros((), device=dist_state.device)
                    )
                    g_adv_raw = torch.zeros((), device=dist_state.device)
                    adaptive_disc_weight = torch.zeros((), device=dist_state.device)
                    if epoch >= disc_start:
                        with autocast_ctx():
                            logits_fake, _ = disc(augment.aug(recon_pm1), None)
                        g_adv_raw = vanilla_g_loss(logits_fake.float())
                        adaptive_disc_weight = calculate_adaptive_weight(
                            recon_weight * recon_loss + perceptual_weight * lpips_loss,
                            g_adv_raw,
                            unwrap_module(rae).decoder.decoder_pred.weight,
                            disc_weight=disc_weight,
                            max_d_weight=max_d_weight,
                        )

                    gen_total = recon_weight * recon_loss + perceptual_weight * lpips_loss + adaptive_disc_weight * g_adv_raw
                    if precision == "fp16":
                        g_scaler.scale(gen_total / grad_accum_steps).backward()
                    else:
                        (gen_total / grad_accum_steps).backward()

                disc_value = torch.zeros((), device=dist_state.device)
                logits_real_mean = torch.zeros((), device=dist_state.device)
                logits_fake_mean = torch.zeros((), device=dist_state.device)
                if epoch >= disc_upd_start:
                    with sync_context(disc, should_sync):
                        with autocast_ctx():
                            fake_aug = augment.aug(recon_pm1.detach())
                            real_aug = augment.aug(real_pm1)
                            logits_fake_d, logits_real_d = disc(fake_aug, real_aug)
                            disc_value = disc_loss_fn(logits_real_d.float(), logits_fake_d.float())
                        logits_real_mean = logits_real_d.float().mean()
                        logits_fake_mean = logits_fake_d.float().mean()
                        if precision == "fp16":
                            d_scaler.scale(disc_value / grad_accum_steps).backward()
                        else:
                            (disc_value / grad_accum_steps).backward()

                accum_counter += 1
                window_recon += float(recon_loss.item())
                window_lpips += float(lpips_loss.item())
                window_adv += float(g_adv_raw.item())
                window_disc += float(disc_value.item())
                window_psnr += compute_psnr(recon.detach(), images.detach())
                window_latent_rms += float(torch.sqrt(torch.mean(latents.detach().float() ** 2)).item())
                window_latent_var += float(torch.var(latents.detach().float(), unbiased=False).item())
                window_logits_real += float(logits_real_mean.item())
                window_logits_fake += float(logits_fake_mean.item())
                window_adaptive_weight += float(adaptive_disc_weight.item())
                window_gen_total += float(gen_total.item())

                if not should_sync:
                    continue

                if precision == "fp16":
                    g_scaler.unscale_(opt_g)
                decoder_grad_norm = module_grad_norm(unwrap_module(rae).decoder)
                encoder_grad_norm = module_grad_norm(unwrap_module(rae).encoder) if train_encoder else 0.0
                if clip_grad > 0:
                    clip_grad_norm_([p for p in unwrap_module(rae).parameters() if p.requires_grad], clip_grad)
                if precision == "fp16":
                    g_scaler.step(opt_g)
                    g_scaler.update()
                else:
                    opt_g.step()
                sched_g.step()
                decoder_param_norm = module_param_norm(unwrap_module(rae).decoder)
                encoder_param_norm = module_param_norm(unwrap_module(rae).encoder) if train_encoder else 0.0
                opt_g.zero_grad(set_to_none=True)

                discriminator_grad_norm = 0.0
                discriminator_param_norm = module_param_norm(unwrap_module(disc))
                if epoch >= disc_upd_start:
                    if precision == "fp16":
                        d_scaler.unscale_(opt_d)
                    discriminator_grad_norm = module_grad_norm(unwrap_module(disc))
                    if clip_grad > 0:
                        clip_grad_norm_(unwrap_module(disc).parameters(), clip_grad)
                    if precision == "fp16":
                        d_scaler.step(opt_d)
                        d_scaler.update()
                    else:
                        opt_d.step()
                    sched_d.step()
                    discriminator_param_norm = module_param_norm(unwrap_module(disc))
                    opt_d.zero_grad(set_to_none=True)

                run_state.global_step += 1
                step = run_state.global_step
                denom = float(accum_counter)
                reduced_step_metrics = distributed_mean_dict(
                    {
                        "train/l1_step": window_recon / denom,
                        "train/lpips_step": window_lpips / denom,
                        "train/g_adv_step": window_adv / denom,
                        "train/d_loss_step": window_disc / denom,
                        "train/psnr_step": window_psnr / denom,
                        "train/latent_rms_step": window_latent_rms / denom,
                        "train/latent_var_step": window_latent_var / denom,
                        "train/logits_real_step": window_logits_real / denom,
                        "train/logits_fake_step": window_logits_fake / denom,
                        "train/adaptive_disc_weight_step": window_adaptive_weight / denom,
                        "train/gen_total_step": window_gen_total / denom,
                        "train/decoder_grad_norm_step": decoder_grad_norm,
                        "train/decoder_param_norm_step": decoder_param_norm,
                        "train/discriminator_grad_norm_step": discriminator_grad_norm,
                        "train/discriminator_param_norm_step": discriminator_param_norm,
                        "train/encoder_grad_norm_step": encoder_grad_norm,
                        "train/encoder_param_norm_step": encoder_param_norm,
                    },
                    device=dist_state.device,
                    enabled=dist_state.enabled,
                    world_size=dist_state.world_size,
                )

                if dist_state.is_master:
                    recon_history.append(reduced_step_metrics["train/l1_step"])
                    lpips_history.append(reduced_step_metrics["train/lpips_step"])
                    adv_history.append(reduced_step_metrics["train/g_adv_step"])
                    disc_history.append(reduced_step_metrics["train/d_loss_step"])
                    psnr_history.append(reduced_step_metrics["train/psnr_step"])

                    should_log = step == 1 or (log_every > 0 and step % log_every == 0)
                    if should_log:
                        train_metrics = {
                            "train/step": step,
                            "train/epoch": epoch,
                            "train/l1": recent_mean(recon_history),
                            "train/lpips": recent_mean(lpips_history),
                            "train/g_adv": recent_mean(adv_history),
                            "train/d_loss": recent_mean(disc_history),
                            "train/psnr": recent_mean(psnr_history),
                            "train/lr_decoder": float(opt_g.param_groups[0]["lr"]),
                            "train/latent_rms": reduced_step_metrics["train/latent_rms_step"],
                            "train/latent_var": reduced_step_metrics["train/latent_var_step"],
                            "train/decoder_grad_norm": reduced_step_metrics["train/decoder_grad_norm_step"],
                            "train/decoder_param_norm": reduced_step_metrics["train/decoder_param_norm_step"],
                            "train/discriminator_grad_norm": reduced_step_metrics["train/discriminator_grad_norm_step"],
                            "train/discriminator_param_norm": reduced_step_metrics["train/discriminator_param_norm_step"],
                            "train/logits_real": reduced_step_metrics["train/logits_real_step"],
                            "train/logits_fake": reduced_step_metrics["train/logits_fake_step"],
                            "train/adaptive_disc_weight": reduced_step_metrics["train/adaptive_disc_weight_step"],
                            "train/gen_total": reduced_step_metrics["train/gen_total_step"],
                            "train/precision": {"fp32": 32, "bf16": 16, "fp16": 15}[precision],
                            "train/world_size": float(dist_state.world_size),
                            "train/micro_batch_size": float(batch_size),
                            "train/grad_accum_steps": float(grad_accum_steps),
                            "train/effective_batch_size": float(batch_size * dist_state.world_size * grad_accum_steps),
                        }
                        if train_encoder:
                            train_metrics["train/encoder_lr"] = float(opt_g.param_groups[-1]["lr"])
                            train_metrics["train/encoder_grad_norm"] = reduced_step_metrics["train/encoder_grad_norm_step"]
                            train_metrics["train/encoder_param_norm"] = reduced_step_metrics["train/encoder_param_norm_step"]
                        else:
                            train_metrics["train/encoder_grad_norm"] = 0.0
                        print(
                            f"[epoch {epoch:03d} step {step:07d}] "
                            f"l1={train_metrics['train/l1']:.4f} "
                            f"lpips={train_metrics['train/lpips']:.4f} "
                            f"g_adv={train_metrics['train/g_adv']:.4f} "
                            f"d={train_metrics['train/d_loss']:.4f} "
                            f"psnr={train_metrics['train/psnr']:.2f}"
                        )
                        if wandb_enabled:
                            wandb.log(train_metrics, step=step)

                    if image_log_every > 0 and step % image_log_every == 0 and preview_for_log is not None:
                        save_preview(sample_dir / f"train_step_{step:07d}.png", preview_for_log)
                    if wandb_enabled and image_log_every > 0 and step % image_log_every == 0 and preview_for_log is not None:
                        wandb.log(
                            {
                                "train/step": step,
                                "images/train_recon": wandb.Image(preview_for_log),
                            },
                            step=step,
                        )

                accum_counter = 0
                window_recon = 0.0
                window_lpips = 0.0
                window_adv = 0.0
                window_disc = 0.0
                window_psnr = 0.0
                window_latent_rms = 0.0
                window_latent_var = 0.0
                window_logits_real = 0.0
                window_logits_fake = 0.0
                window_adaptive_weight = 0.0
                window_gen_total = 0.0

            if val_loader is not None and eval_every > 0 and (epoch + 1) % eval_every == 0:
                barrier_if_needed(dist_state)
                if dist_state.is_master:
                    val_metrics, val_preview = evaluate(
                        rae=rae_module,
                        loader=val_loader,
                        lpips_model=lpips_model,
                        device=dist_state.device,
                        max_batches=eval_max_batches,
                        image_log_count=visual_count,
                        autocast_ctx=autocast_ctx,
                    )
                    val_metrics["train/step"] = run_state.global_step
                    print(
                        f"[val epoch {epoch:03d}] "
                        f"l1={val_metrics['val/l1']:.4f} "
                        f"lpips={val_metrics['val/lpips']:.4f} "
                        f"psnr={val_metrics['val/psnr']:.2f}"
                    )
                    best_so_far = run_state.best_val_lpips
                    current_lpips = val_metrics["val/lpips"]
                    improved = best_so_far is None or current_lpips < best_so_far
                    if improved:
                        run_state.best_val_lpips = current_lpips
                        save_checkpoint(
                            ckpt_dir / "best.pt",
                            rae=rae_module,
                            disc=disc_module,
                            opt_g=opt_g,
                            opt_d=opt_d,
                            sched_g=sched_g,
                            sched_d=sched_d,
                            g_scaler=g_scaler,
                            d_scaler=d_scaler,
                            run_state=run_state,
                            config=cfg_dict,
                        )
                    if val_preview is not None:
                        save_preview(sample_dir / f"val_epoch_{epoch + 1:03d}.png", val_preview)
                    if wandb_enabled:
                        log_payload: dict[str, Any] = dict(val_metrics)
                        if val_preview is not None:
                            log_payload["images/val_recon"] = wandb.Image(val_preview)
                        wandb.log(log_payload, step=run_state.global_step)
                barrier_if_needed(dist_state)

            if fid_enabled and val_loader is not None and fid_every > 0 and (epoch + 1) % fid_every == 0:
                barrier_if_needed(dist_state)
                if dist_state.is_master:
                    fid_started_at = datetime.now()
                    fid_metrics = run_reconstruction_fid(
                        rae=rae_module,
                        loader=val_loader,
                        device=dist_state.device,
                        max_batches=eval_max_batches,
                        autocast_ctx=autocast_ctx,
                        fid_ref=str(fid_ref),
                        fid_batch_size=fid_batch_size,
                        fid_device=fid_device,
                        fid_num_threads=fid_num_threads,
                        workdir=workdir,
                        global_step=run_state.global_step,
                    )
                    fid_metrics["train/step"] = run_state.global_step
                    fid_metrics["val/fid_duration_sec"] = float((datetime.now() - fid_started_at).total_seconds())
                    print(
                        f"[fid epoch {epoch:03d}] "
                        f"fid={fid_metrics['val/fid']:.4f} "
                        f"num_samples={int(fid_metrics['val/fid_num_samples'])}"
                    )
                    if wandb_enabled:
                        wandb.log(fid_metrics, step=run_state.global_step)
                barrier_if_needed(dist_state)

            run_state.epoch = epoch + 1
            if save_every > 0 and (epoch + 1) % save_every == 0:
                barrier_if_needed(dist_state)
                if dist_state.is_master:
                    save_checkpoint(
                        ckpt_dir / f"epoch_{epoch + 1:03d}.pt",
                        rae=rae_module,
                        disc=disc_module,
                        opt_g=opt_g,
                        opt_d=opt_d,
                        sched_g=sched_g,
                        sched_d=sched_d,
                        g_scaler=g_scaler,
                        d_scaler=d_scaler,
                        run_state=run_state,
                        config=cfg_dict,
                    )
                    save_checkpoint(
                        ckpt_dir / "last.pt",
                        rae=rae_module,
                        disc=disc_module,
                        opt_g=opt_g,
                        opt_d=opt_d,
                        sched_g=sched_g,
                        sched_d=sched_d,
                        g_scaler=g_scaler,
                        d_scaler=d_scaler,
                        run_state=run_state,
                        config=cfg_dict,
                    )
                barrier_if_needed(dist_state)
    finally:
        if wandb_enabled:
            wandb.finish()
        cleanup_distributed(dist_state)


if __name__ == "__main__":
    main()
