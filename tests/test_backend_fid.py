from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from src_jax.backend_fid import RawImageDataset, _build_loader, list_image_files

    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    RawImageDataset = None
    list_image_files = None
    _build_loader = None
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"Missing optional dependency: {_IMPORT_ERROR}")
class BackendFidTests(unittest.TestCase):
    def test_dataset_returns_tensor_from_writable_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.png"
            Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(image_path)

            dataset = RawImageDataset([image_path], image_size=8)
            tensor, label = dataset[0]

            self.assertEqual(tuple(tensor.shape), (3, 8, 8))
            self.assertEqual(label, 0)

    def test_loader_uses_spawn_for_worker_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.png"
            Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(image_path)

            dataset = RawImageDataset([image_path], image_size=8)
            loader = _build_loader(dataset, batch_size=1, num_workers=2)

            self.assertEqual(loader.multiprocessing_context.get_start_method(), "spawn")

    def test_list_image_files_finds_uppercase_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.PNG"
            Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(image_path)

            found = list_image_files(tmp_dir)
            self.assertEqual(found, [image_path.resolve()])
