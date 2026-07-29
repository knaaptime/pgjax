"""pgjax: on-device Pólya-Gamma sampling as an XLA FFI custom call.

Draw ``omega ~ PG(h, z)`` from inside ``@jax.jit`` / ``lax.scan`` with no host
round-trip — the native replacement for ``jax.pure_callback(random_polyagamma)``
that taxes every Gibbs sweep (and serializes under ``jax.pmap``).

Exact for the Bernoulli/logit augmentation (``h = 1``) via the Devroye method
(integer ``h`` = sum of Devroye draws), and for real-valued ``h`` — the
Negative-Binomial ``h = y + alpha`` — via the tail-corrected Gamma-sum
representation (validated bias-free vs ``polyagamma``; see the README).

Example::

    import jax, pgjax
    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def step(h, z, key):
        return pgjax.pg_sample(h, z, key)   # on-device PG draw
"""

import jax
import jax.numpy as jnp
import numpy as np

import pgjax_cpp as _cpp

__version__ = "0.1.0"
__all__ = ["pg_sample"]

jax.ffi.register_ffi_target(
    "pgjax_sample_f64", _cpp.pg_sample_f64_capsule(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "pgjax_sample_batched_f64", _cpp.pg_sample_batched_f64_capsule(), platform="cpu"
)


def _require_x64():
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            'pgjax requires 64-bit mode: jax.config.update("jax_enable_x64", True)'
        )


def _seed_from_key(key):
    """Derive an int64 RNG seed from a JAX PRNG key (per-key, reproducible)."""
    kd = jax.random.key_data(key).ravel().astype(jnp.int64)
    # Combine the key words for a bit more entropy; kept non-negative-ish (the
    # C++ RNG treats it as an unsigned 64-bit seed).
    seed = kd[0]
    if kd.shape[0] > 1:
        seed = seed ^ (kd[1] << jnp.int64(23))
    return seed


def _sample_batched(h, z, seed):
    call = jax.ffi.ffi_call(
        "pgjax_sample_batched_f64", jax.ShapeDtypeStruct(h.shape, h.dtype)
    )
    return call(h, z, seed)


@jax.custom_batching.custom_vmap
def _dispatch(h, z, seed):
    call = jax.ffi.ffi_call("pgjax_sample_f64", jax.ShapeDtypeStruct(h.shape, h.dtype))
    return call(h, z, seed)


@_dispatch.def_vmap
def _dispatch_vmap(axis_size, in_batched, h, z, seed):
    h_b, z_b, seed_b = in_batched
    if not h_b:
        h = jnp.broadcast_to(h, (axis_size,) + h.shape)
    if not z_b:
        z = jnp.broadcast_to(z, (axis_size,) + z.shape)
    if not seed_b:
        seed = jnp.broadcast_to(seed, (axis_size,))
    return _sample_batched(h, z, seed), True


def pg_sample(h, z, key):
    """Draw ``omega ~ PG(h, z)`` element-wise, on-device.

    Parameters
    ----------
    h : array_like
        Shape ``[n]`` float64. PG shape (h = 1 for Bernoulli/logit; positive
        integer for the exact sum-of-Devroye path).
    z : array_like
        Shape ``[n]`` float64. PG tilt (the linear predictor).
    key : jax.Array
        A JAX PRNG key. ``vmap`` over a batch of keys draws each element's
        batch with its own RNG stream.

    Returns
    -------
    omega : jax.Array
        Draws from ``PG(h, z)`` with the same shape as ``h``.
    """
    _require_x64()
    h = jnp.asarray(h, jnp.float64)
    z = jnp.asarray(z, jnp.float64)
    if h.shape != z.shape:
        raise ValueError(f"h and z must have the same shape, got {h.shape}, {z.shape}")
    seed = _seed_from_key(key)
    return _dispatch(h, z, seed)
