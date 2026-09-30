"""vmap semantics of pg_sample; needs no reference sampler, so it runs everywhere."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import pgjax  # noqa: E402


def _draw(key, n=4):
    return pgjax.pg_sample(jnp.ones(n), jnp.zeros(n), key)


def test_vmap_over_keys_gives_independent_streams():
    keys = jax.random.split(jax.random.key(0), 5)
    out = jax.vmap(_draw)(keys)
    assert out.shape == (5, 4)
    assert len({tuple(np.asarray(row)) for row in out}) == 5


@pytest.mark.parametrize("jit", [False, True])
def test_nested_vmap_matches_flat_vmap(jit):
    """vmap of vmap folds into one batched call and equals the flat batch."""
    keys = jax.random.split(jax.random.key(1), 6)
    nested = jax.vmap(jax.vmap(_draw))
    if jit:
        nested = jax.jit(nested)
    out = nested(keys.reshape(2, 3))
    flat = jax.vmap(_draw)(keys)
    assert out.shape == (2, 3, 4)
    np.testing.assert_array_equal(np.asarray(out).reshape(6, 4), np.asarray(flat))


def test_vmap_with_unbatched_arguments():
    """Only z batched: h and the key broadcast across the batch."""
    z = jnp.linspace(-2.0, 2.0, 3)[:, None] * jnp.ones((3, 4))
    out = jax.vmap(lambda zz: pgjax.pg_sample(jnp.ones(4), zz, jax.random.key(2)))(z)
    assert out.shape == (3, 4) and bool(jnp.all(out > 0))
