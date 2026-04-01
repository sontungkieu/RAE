from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    from src_jax.stage2_runtime import (
        _build_stage2_activation_names,
        _infer_stage2_metric_prefix,
        _resolve_prefetch_factor,
    )

    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"Missing optional dependency: {_IMPORT_ERROR}")
class Stage2RuntimeHelpersTests(unittest.TestCase):
    def test_resolve_prefetch_factor_respects_worker_count(self) -> None:
        self.assertIsNone(_resolve_prefetch_factor(None, num_workers=0))
        self.assertEqual(_resolve_prefetch_factor(None, num_workers=16), 2)
        self.assertEqual(_resolve_prefetch_factor(4, num_workers=16), 4)
        with self.assertRaises(ValueError):
            _resolve_prefetch_factor(0, num_workers=16)

    def test_activation_name_builder_matches_sitdh_layout(self) -> None:
        self.assertEqual(
            _build_stage2_activation_names({"num_encoder_blocks": 2, "num_decoder_blocks": 1}),
            [("enc", 0), ("enc", 1), ("dec", 0)],
        )
        self.assertEqual(
            _build_stage2_activation_names({"depth": 3}),
            [("blk", 0), ("blk", 1), ("blk", 2)],
        )

    def test_metric_prefix_matches_network_class(self) -> None:
        self.assertEqual(_infer_stage2_metric_prefix(SimpleNamespace(network_class="lightning_ddt")), "sitdh")
        self.assertEqual(_infer_stage2_metric_prefix(SimpleNamespace(network_class="lightning_dit")), "sit")


if __name__ == "__main__":
    unittest.main()
