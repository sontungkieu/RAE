from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path


BACKEND_REPO_URL = "https://github.com/willisma/diffuse_nnx"
BACKEND_COMMIT = "023afd23c7b62a8cdb00e840b36a4ab8fc970bba"
BACKEND_ENV_VAR = "RAE_JAX_BACKEND_DIR"
DEFAULT_BACKEND_DIR = Path.home() / ".cache" / "rae_jax" / "diffuse_nnx"
OVERLAY_MANIFEST_NAME = ".rae_jax_overlay_manifest.json"
OVERLAY_LOCK_NAME = ".rae_jax_overlay.lock"
FORCE_REBUILD_ENV_VAR = "RAE_JAX_REBUILD_BACKEND"
SOURCE_DIR = Path(__file__).resolve().parent
OVERLAY_SOURCE_DIR = SOURCE_DIR / "backend_overlay" / "diffuse_nnx"
MOE1_SOURCE_DIR = SOURCE_DIR / "moe1"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def _git_head(path: Path) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return output.strip()


def _patch_backend_file(path: Path, old: str, new: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Backend compatibility patch target not found: {path}")
    content = path.read_text(encoding="utf-8")
    if new in content:
        return
    if old not in content:
        raise RuntimeError(f"Unexpected backend source layout while patching {path}")
    path.write_text(content.replace(old, new, 1), encoding="utf-8")


def _apply_backend_compat_patches(backend_dir: Path) -> None:
    dino_path = backend_dir / "networks" / "encoders" / "dino.py"
    _patch_backend_file(
        dino_path,
        "from transformers import FlaxDinov2Model, AutoImageProcessor\n",
        textwrap.dedent(
            """\
            from transformers import AutoImageProcessor
            from transformers.models.dinov2.modeling_flax_dinov2 import FlaxDinov2Model
            """
        ),
    )

    dino_w_register_path = backend_dir / "networks" / "encoders" / "dino_w_register.py"
    _patch_backend_file(
        dino_w_register_path,
        "from transformers import Dinov2WithRegistersModel\n",
        "from transformers.models.dinov2_with_registers import Dinov2WithRegistersModel\n",
    )

    encoder_utils_path = backend_dir / "networks" / "encoders" / "utils.py"
    _patch_backend_file(
        encoder_utils_path,
        textwrap.dedent(
            """\
            \"\"\"File containing utility functions for the encoder.\"\"\"

            # built-in libs
            import math

            # external libs
            from google.cloud import storage

            def download_blob(bucket_name, source_blob_name, destination_file_name):
                \"\"\"Downloads a blob from the bucket.\"\"\"
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(source_blob_name)
                blob.download_to_filename(destination_file_name)
            """
        ),
        textwrap.dedent(
            """\
            \"\"\"File containing utility functions for the encoder.\"\"\"

            # built-in libs
            import math


            def _load_storage():
                try:
                    from google.cloud import storage
                except ImportError as exc:
                    raise ImportError(
                        \"google-cloud-storage is only required when diffuse_nnx needs to download \"
                        \"encoder assets from GCS. Install it with `uv pip install google-cloud-storage` \"
                        \"or provide the expected local checkpoint files.\"
                    ) from exc
                return storage


            def download_blob(bucket_name, source_blob_name, destination_file_name):
                \"\"\"Downloads a blob from the bucket.\"\"\"
                storage = _load_storage()
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(source_blob_name)
                blob.download_to_filename(destination_file_name)
            """
        ),
    )

    sd_vae_path = backend_dir / "networks" / "encoders" / "sd_vae.py"
    _patch_backend_file(
        sd_vae_path,
        (
            "    def initialize(self):\n"
            "        ckpt_path = os.path.join(Path(__file__).parent, self.pretrained_path)\n"
            "        if not os.path.exists(ckpt_path):\n"
            "            utils.download_blob('will-data', 'stats/vae_trial1.pkl', ckpt_path)\n"
            "            \n"
            "        with open(ckpt_path, 'rb') as f:\n"
            "            params = pickle.load(f)\n"
            "        return params\n"
        ),
        (
            "    def initialize(self):\n"
            "        ckpt_path = Path(self.pretrained_path)\n"
            "        if not ckpt_path.is_absolute():\n"
            "            ckpt_path = Path(__file__).parent / ckpt_path\n"
            "\n"
            "        if not ckpt_path.exists():\n"
            "            if ckpt_path.name != 'vae_trial1.pkl':\n"
            "                raise FileNotFoundError(f'StabilityVAE checkpoint not found: {ckpt_path}')\n"
            "            try:\n"
            "                from src_jax.stability_vae_assets import ensure_stability_vae_checkpoint\n"
            "            except ImportError:\n"
            "                from stability_vae_assets import ensure_stability_vae_checkpoint\n"
            "            ensure_stability_vae_checkpoint(ckpt_path)\n"
            "\n"
            "        with open(ckpt_path, 'rb') as f:\n"
            "            params = pickle.load(f)\n"
            "        return params\n"
        ),
    )

    ema_path = backend_dir / "utils" / "ema.py"
    _patch_backend_file(
        ema_path,
        (
            "        self.ema = copy.deepcopy(net)\n"
            "        ema_state = jax.tree.map(lambda x: jnp.zeros_like(x), nnx.state(net, nnx.Param))\n"
            "        nnx.update(self.ema, ema_state)\n"
            "        self.ema.eval()\n"
        ),
        (
            "        self.ema = copy.deepcopy(net)\n"
            "        self.ema.eval()\n"
        ),
    )

    initialize_path = backend_dir / "utils" / "initialize.py"
    if initialize_path.exists():
        _patch_backend_file(
            initialize_path,
            "from interfaces import continuous, discrete, repa\n",
            "from interfaces import continuous, continuous_moe1, discrete, repa\n",
        )
        _patch_backend_file(
            initialize_path,
            (
                "INTERFACE_REGISTRY = {\n"
                "    'sit': continuous.SiTInterface,\n"
                "    'edm': continuous.EDMInterface,\n"
                "    'mean_flow': continuous.MeanFlowInterface,\n"
                "}\n"
            ),
            (
                "INTERFACE_REGISTRY = {\n"
                "    'sit': continuous.SiTInterface,\n"
                "    'sit_gmm_moe1': continuous_moe1.SiTGMMMoe1Interface,\n"
                "    'edm': continuous.EDMInterface,\n"
                "    'mean_flow': continuous.MeanFlowInterface,\n"
                "}\n"
            ),
        )


def _iter_overlay_files() -> list[tuple[Path, Path]]:
    mappings: list[tuple[Path, Path]] = []
    for base_source, relative_target in (
        (OVERLAY_SOURCE_DIR, Path(".")),
        (MOE1_SOURCE_DIR, Path("moe1")),
    ):
        if not base_source.exists():
            continue
        for source_path in sorted(path for path in base_source.rglob("*") if path.is_file()):
            mappings.append((source_path, relative_target / source_path.relative_to(base_source)))
    return mappings


def _compute_overlay_hash(files: list[tuple[Path, Path]]) -> str:
    digest = hashlib.sha256()
    for source_path, relative_target in files:
        digest.update(relative_target.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_overlay_manifest(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _acquire_overlay_lock(lock_path: Path, *, timeout_sec: float = 60.0) -> None:
    deadline = time.time() + timeout_sec
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for backend overlay lock: {lock_path}")
            time.sleep(0.1)
            continue
        else:
            os.close(fd)
            return


def _release_overlay_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def _sync_backend_overlay(backend_dir: Path) -> None:
    overlay_files = _iter_overlay_files()
    if not overlay_files:
        return

    overlay_hash = _compute_overlay_hash(overlay_files)
    manifest_path = backend_dir / OVERLAY_MANIFEST_NAME
    force_rebuild = os.environ.get(FORCE_REBUILD_ENV_VAR, "").strip() not in {"", "0", "false", "False"}
    manifest = _load_overlay_manifest(manifest_path)
    if not force_rebuild and manifest is not None and manifest.get("overlay_hash") == overlay_hash:
        return

    lock_path = backend_dir / OVERLAY_LOCK_NAME
    _acquire_overlay_lock(lock_path)
    try:
        manifest = _load_overlay_manifest(manifest_path)
        if not force_rebuild and manifest is not None and manifest.get("overlay_hash") == overlay_hash:
            return

        for source_path, relative_target in overlay_files:
            target_path = backend_dir / relative_target
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)

        manifest_payload = {
            "overlay_hash": overlay_hash,
            "backend_commit": BACKEND_COMMIT,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    finally:
        _release_overlay_lock(lock_path)


def resolve_backend_dir(explicit_dir: str | None = None) -> Path:
    raw = explicit_dir or os.environ.get(BACKEND_ENV_VAR)
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_BACKEND_DIR


def ensure_backend(explicit_dir: str | None = None) -> Path:
    backend_dir = resolve_backend_dir(explicit_dir)
    backend_dir.parent.mkdir(parents=True, exist_ok=True)

    if not (backend_dir / ".git").exists():
        _run(["git", "clone", "--depth", "1", BACKEND_REPO_URL, str(backend_dir)])

    current_head = _git_head(backend_dir)
    if current_head != BACKEND_COMMIT:
        _run(["git", "-C", str(backend_dir), "fetch", "--depth", "1", "origin", BACKEND_COMMIT])
        _run(["git", "-C", str(backend_dir), "checkout", BACKEND_COMMIT])

    _apply_backend_compat_patches(backend_dir)
    _sync_backend_overlay(backend_dir)

    return backend_dir


def activate_backend(explicit_dir: str | None = None) -> Path:
    backend_dir = ensure_backend(explicit_dir)
    backend_path = str(backend_dir)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    return backend_dir
