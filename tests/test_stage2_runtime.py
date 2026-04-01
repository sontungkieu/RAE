from __future__ import annotations

import types
import unittest

try:
    from src_jax.stage2_runtime import _sample_initial_latents
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    _sample_initial_latents = None
    _IMPORT_ERROR = exc


class _ModelWithSource:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def sample_source_prior(self, shape: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        self.calls.append(shape)
        return ("source", shape)


class _RngFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return "rng-key"


@unittest.skipIf(_IMPORT_ERROR is not None, f"Missing dependency: {_IMPORT_ERROR}")
class Stage2RuntimeTests(unittest.TestCase):
    def test_sample_initial_latents_prefers_source_prior(self) -> None:
        model = _ModelWithSource()
        rngs = _RngFactory()

        class _RandomStub:
            def normal(self, *_args, **_kwargs):  # pragma: no cover - should never run
                raise AssertionError("jax.random.normal should not be used when source prior exists")

        jax = types.SimpleNamespace(random=_RandomStub())
        jnp = types.SimpleNamespace(float32="float32")

        result = _sample_initial_latents(
            model,
            batch_size=2,
            input_size=32,
            in_channels=4,
            rngs=rngs,
            jax=jax,
            jnp=jnp,
        )

        self.assertEqual(result, ("source", (2, 32, 32, 4)))
        self.assertEqual(model.calls, [(2, 32, 32, 4)])
        self.assertEqual(rngs.calls, 0)

    def test_sample_initial_latents_falls_back_to_gaussian_noise(self) -> None:
        rngs = _RngFactory()
        recorded: dict[str, object] = {}

        class _RandomStub:
            def normal(self, key: str, shape: tuple[int, ...], dtype: str) -> tuple[str, str, tuple[int, ...], str]:
                recorded["key"] = key
                recorded["shape"] = shape
                recorded["dtype"] = dtype
                return ("normal", key, shape, dtype)

        jax = types.SimpleNamespace(random=_RandomStub())
        jnp = types.SimpleNamespace(float32="float32")

        result = _sample_initial_latents(
            object(),
            batch_size=3,
            input_size=16,
            in_channels=8,
            rngs=rngs,
            jax=jax,
            jnp=jnp,
        )

        self.assertEqual(result, ("normal", "rng-key", (3, 16, 16, 8), "float32"))
        self.assertEqual(recorded["key"], "rng-key")
        self.assertEqual(recorded["shape"], (3, 16, 16, 8))
        self.assertEqual(recorded["dtype"], "float32")
        self.assertEqual(rngs.calls, 1)


if __name__ == "__main__":
    unittest.main()
