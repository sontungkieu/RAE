from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from src_jax.stage2_runtime import (
        _install_strict_wandb_initializer,
        _load_wandb_resume_metadata,
        _resolve_stage2_exp_name,
        _resolve_wandb_resume_binding,
        _wandb_resume_metadata_path,
    )
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _install_strict_wandb_initializer = None
    _load_wandb_resume_metadata = None
    _resolve_stage2_exp_name = None
    _resolve_wandb_resume_binding = None
    _wandb_resume_metadata_path = None
    _IMPORT_ERROR = exc


class _DummyConfig:
    def to_dict(self) -> dict[str, int]:
        return {"seed": 7}


class _FakeWandb:
    def __init__(self, generated_ids: list[str] | None = None) -> None:
        self._generated_ids = list(generated_ids or ["fresh123"])
        self.login_keys: list[str] = []
        self.calls: list[dict[str, object]] = []
        self.util = SimpleNamespace(generate_id=self._generate_id)

    def _generate_id(self) -> str:
        if not self._generated_ids:
            raise AssertionError("No generated W&B run ids left in test double.")
        return self._generated_ids.pop(0)

    def login(self, key: str) -> None:
        self.login_keys.append(key)

    def init(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        run_id = kwargs.get("id")
        if run_id is None and "resume_from" in kwargs:
            run_id = str(kwargs["resume_from"]).split("?", 1)[0]
        return SimpleNamespace(id=run_id)


class _FakeWandbUtils:
    def __init__(self, fake_wandb: _FakeWandb) -> None:
        self.wandb = fake_wandb

    @staticmethod
    def is_main_process() -> bool:
        return True


@unittest.skipIf(_IMPORT_ERROR is not None, f"Missing optional dependency: {_IMPORT_ERROR}")
class Stage2RuntimeWandbTests(unittest.TestCase):
    def test_resolve_stage2_exp_name_prefers_explicit_cli_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SimpleNamespace(exp_name="cli-exp", workdir=tmp_dir)
            self.assertEqual(_resolve_stage2_exp_name(args), "cli-exp")

    def test_resolve_stage2_exp_name_uses_saved_wandb_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            _wandb_resume_metadata_path(workdir).write_text(
                '{"entity": "entity", "project": "project", "exp_name": "saved-exp", "run_id": "resume123"}',
                encoding="utf-8",
            )
            args = SimpleNamespace(exp_name=None, workdir=str(workdir))
            self.assertEqual(_resolve_stage2_exp_name(args), "saved-exp")

    def test_resolve_stage2_exp_name_falls_back_to_workdir_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir) / "custom-run-name"
            workdir.mkdir()
            args = SimpleNamespace(exp_name=None, workdir=str(workdir))
            self.assertEqual(_resolve_stage2_exp_name(args), "custom-run-name")

    def test_new_workdir_generates_unique_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            metadata, init_kwargs = _resolve_wandb_resume_binding(
                workdir=workdir,
                entity="entity",
                project_name="project",
                exp_name="exp-train",
                explicit_run_id=None,
                wandb_utils=SimpleNamespace(wandb=_FakeWandb(["fresh123"])),
            )

            self.assertEqual(init_kwargs, {"id": "fresh123", "resume": "never"})
            self.assertEqual(metadata["run_id"], "fresh123")
            self.assertEqual(metadata["entity"], "entity")
            self.assertEqual(metadata["project"], "project")
            self.assertEqual(metadata["exp_name"], "exp-train")

    def test_legacy_resume_requires_explicit_run_id_when_metadata_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            (workdir / "checkpoint_000010").mkdir()

            with self.assertRaises(FileNotFoundError):
                _resolve_wandb_resume_binding(
                    workdir=workdir,
                    entity="entity",
                    project_name="project",
                    exp_name="exp-train",
                    explicit_run_id=None,
                    wandb_utils=SimpleNamespace(wandb=_FakeWandb(["unused123"])),
                )

    def test_legacy_resume_accepts_explicit_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            (workdir / "checkpoint_000010").mkdir()

            metadata, init_kwargs = _resolve_wandb_resume_binding(
                workdir=workdir,
                entity="entity",
                project_name="project",
                exp_name="exp-train",
                explicit_run_id="legacy123",
                wandb_utils=SimpleNamespace(wandb=_FakeWandb(["unused123"])),
            )

            self.assertEqual(init_kwargs, {"resume_from": "legacy123?_step=10"})
            self.assertEqual(metadata["run_id"], "legacy123")

    def test_stored_metadata_rewinds_to_latest_checkpoint_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            (workdir / "checkpoint_000010").mkdir()
            (workdir / "checkpoint_000120").mkdir()
            metadata_path = _wandb_resume_metadata_path(workdir)
            metadata_path.write_text(
                '{"entity": "entity", "project": "project-a", "exp_name": "exp-a", "run_id": "saved123"}',
                encoding="utf-8",
            )

            metadata, init_kwargs = _resolve_wandb_resume_binding(
                workdir=workdir,
                entity="entity",
                project_name="project-a",
                exp_name="exp-a",
                explicit_run_id=None,
                wandb_utils=SimpleNamespace(wandb=_FakeWandb(["unused123"])),
            )

            self.assertEqual(metadata["run_id"], "saved123")
            self.assertEqual(init_kwargs, {"resume_from": "saved123?_step=120"})

    def test_stored_metadata_rejects_project_or_exp_name_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            metadata_path = _wandb_resume_metadata_path(workdir)
            metadata_path.write_text(
                '{"entity": "entity", "project": "project-a", "exp_name": "exp-a", "run_id": "saved123"}',
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                _resolve_wandb_resume_binding(
                    workdir=workdir,
                    entity="entity",
                    project_name="project-b",
                    exp_name="exp-a",
                    explicit_run_id=None,
                    wandb_utils=SimpleNamespace(wandb=_FakeWandb(["unused123"])),
                )

            with self.assertRaises(ValueError):
                _resolve_wandb_resume_binding(
                    workdir=workdir,
                    entity="entity",
                    project_name="project-a",
                    exp_name="exp-b",
                    explicit_run_id=None,
                    wandb_utils=SimpleNamespace(wandb=_FakeWandb(["unused123"])),
                )

    def test_initializer_persists_metadata_and_rewinds_to_checkpoint_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            fake_wandb = _FakeWandb(["fresh123"])
            fake_utils = _FakeWandbUtils(fake_wandb)
            _install_strict_wandb_initializer(fake_utils, workdir=workdir, explicit_run_id=None)

            with patch.dict(
                "os.environ",
                {"WANDB_API_KEY": "token", "WANDB_ENTITY": "entity"},
                clear=False,
            ):
                fake_utils.initialize(_DummyConfig(), exp_name="exp-train", project_name="project")
                (workdir / "checkpoint_000120").mkdir()
                fake_utils.initialize(_DummyConfig(), exp_name="exp-train", project_name="project")

            metadata = _load_wandb_resume_metadata(_wandb_resume_metadata_path(workdir))
            self.assertIsNotNone(metadata)
            assert metadata is not None
            self.assertEqual(metadata["run_id"], "fresh123")
            self.assertEqual(metadata["entity"], "entity")
            self.assertEqual(metadata["project"], "project")
            self.assertEqual(metadata["exp_name"], "exp-train")

            self.assertEqual(fake_wandb.login_keys, ["token", "token"])
            self.assertEqual(fake_wandb.calls[0]["id"], "fresh123")
            self.assertEqual(fake_wandb.calls[0]["resume"], "never")
            self.assertEqual(fake_wandb.calls[1]["resume_from"], "fresh123?_step=120")

    def test_stored_metadata_without_checkpoint_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            metadata_path = _wandb_resume_metadata_path(workdir)
            metadata_path.write_text(
                '{"entity": "entity", "project": "project-a", "exp_name": "exp-a", "run_id": "saved123"}',
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                _resolve_wandb_resume_binding(
                    workdir=workdir,
                    entity="entity",
                    project_name="project-a",
                    exp_name="exp-a",
                    explicit_run_id=None,
                    wandb_utils=SimpleNamespace(wandb=_FakeWandb(["unused123"])),
                )


if __name__ == "__main__":
    unittest.main()
