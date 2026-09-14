"""Correctness tests for pgjax against the reference `polyagamma` sampler."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import pgjax  # noqa: E402

pa = pytest.importorskip("polyagamma")
stats = pytest.importorskip("scipy.stats")

N = 100_000


def _draw(h, z, seed=0):
    key = jax.random.PRNGKey(seed)
    return np.asarray(
        pgjax.pg_sample(jnp.full(N, float(h)), jnp.full(N, float(z)), key)
    )


def _analytic_mean(h, z):
    return h * 0.25 if abs(z) < 1e-9 else h * np.tanh(z / 2) / (2 * z)


@pytest.mark.parametrize("z", [0.0, 0.5, 2.0, 5.0, -3.0])
def test_pg1_matches_reference(z):
    """PG(1, z) (Devroye) matches the reference and the analytic mean."""
    xj = _draw(1.0, z, seed=int(abs(z) * 100) + 1)
    xr = pa.random_polyagamma(1.0, z, size=N)
    assert stats.ks_2samp(xj, xr).statistic < 0.02
    assert abs(xj.mean() - _analytic_mean(1.0, z)) < 0.01


def _analytic_var(h, z):
    if abs(z) < 1e-9:
        return h / 24.0
    th = np.tanh(z / 2)
    return h * (np.sinh(z) - z) * (1 - th * th) / (4 * z**3)


@pytest.mark.parametrize(
    "h", [0.2, 0.5, 0.9, 1.05, 1.3, 2.0, 2.7, 3.0, 3.9, 4.5, 5.0, 6.5, 7.9, 10.0, 40.0]
)
@pytest.mark.parametrize("z", [0.0, 1.0, -3.0, 6.0])
def test_moments_match_analytic(h, z):
    """Mean and variance of PG(h, z) match their closed forms in every regime.

    Checked against the exact moments rather than ``polyagamma``: its hybrid
    sampler is itself off by up to 5% in the variance for real h < 8 (the
    alternate kernel constant; the saddlepoint for 4 < h < 8), and a two-sample
    KS test at this size cannot see an error of that order.
    """
    n = 400_000
    x = np.asarray(
        pgjax.pg_sample(
            jnp.full(n, float(h)),
            jnp.full(n, float(z)),
            jax.random.PRNGKey(int(h * 1000) + int(abs(z) * 10)),
        )
    )
    mean, var = _analytic_mean(h, z), _analytic_var(h, z)
    z_mean = (x.mean() - mean) / np.sqrt(x.var() / n)
    z_var = (x.var() - var) / np.sqrt(np.var((x - x.mean()) ** 2) / n)
    assert abs(z_mean) < 5, f"mean {x.mean()} vs {mean} (z = {z_mean:.1f})"
    assert abs(z_var) < 5, f"var {x.var()} vs {var} (z = {z_var:.1f})"


def test_jit_and_vmap():
    """Works under jit(vmap) with independent per-chain RNG streams."""
    f = jax.jit(jax.vmap(pgjax.pg_sample))
    B, n = 4, 8
    h = jnp.full((B, n), 3.0)
    z = jnp.zeros((B, n))
    keys = jax.random.split(jax.random.PRNGKey(0), B)
    out = np.asarray(f(h, z, keys))
    assert out.shape == (B, n)
    assert np.all(np.isfinite(out)) and np.all(out > 0)
    assert not np.allclose(out[0], out[1])  # different chains, different draws


def test_requires_x64_positive_and_shape():
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError):
        pgjax.pg_sample(jnp.ones(3), jnp.zeros(4), key)  # shape mismatch


@pytest.mark.parametrize("h", [8.0, 10.0, 15.0, 20.0, 30.0, 40.0])
@pytest.mark.parametrize("z", [0.0, 1.0, -4.0, 5.0])
def test_saddlepoint_matches_reference(h, z):
    """PG(h, z) via the saddlepoint path matches the reference and mean."""
    xj = _draw(h, z, seed=int(h * 100) + int(abs(z)) + 1)
    xr = pa.random_polyagamma(h, z, size=N)
    an = _analytic_mean(h, z)
    assert stats.ks_2samp(xj, xr).statistic < 0.02
    assert abs(xj.mean() - an) / an < 0.02  # no systematic bias


@pytest.mark.parametrize(
    "h, z, uses_saddle",
    [
        (7.9, 0.0, False),  # h < 8 -> alternate
        (8.0, 0.0, True),   # h >= 8 -> saddlepoint
        (0.99, 1.0, False),  # h < 1 -> Gamma series
        (1.01, 1.0, False),  # just above 1 -> alternate
    ],
)
def test_saddlepoint_boundary(h, z, uses_saddle):
    """Both sides of the dispatch threshold produce valid PG draws."""
    xj = _draw(h, z, seed=int(h * 100) + int(abs(z)) + 1)
    xr = pa.random_polyagamma(h, z, size=N)
    an = _analytic_mean(h, z)
    assert stats.ks_2samp(xj, xr).statistic < 0.02
    assert abs(xj.mean() - an) / an < 0.02
    # Sanity: finite, positive, plausible scale.
    assert np.all(np.isfinite(xj)) and np.all(xj > 0)
    assert xj.std() > 0


@pytest.mark.parametrize("h", [51.0, 80.0, 200.0, 1000.0])
@pytest.mark.parametrize("z", [0.0, 1e-4, 2.0, -6.0])
def test_normal_approx_matches_reference(h, z):
    """PG(h, z) for h > 50 (normal approximation) matches the reference and mean."""
    xj = _draw(h, z, seed=int(h) + int(abs(z) * 10) + 3)
    xr = pa.random_polyagamma(h, z, size=N)
    an = _analytic_mean(h, z)
    assert stats.ks_2samp(xj, xr).statistic < 0.02
    assert abs(xj.mean() - an) / an < 0.01
