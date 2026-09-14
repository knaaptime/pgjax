# pgjax

On-device **Pólya-Gamma** sampling for JAX, as an XLA FFI custom call.

`pgjax.pg_sample(h, z, key)` draws `omega ~ PG(h, z)` from *inside* a JIT'd /
`lax.scan`-ed / `pmap`-ed program with **no host round-trip** — the native
replacement for `jax.pure_callback(random_polyagamma, ...)`, which pays a
device↔host round-trip on every Pólya-Gamma Gibbs sweep and serializes under
`jax.pmap`.

```python
import jax, pgjax
jax.config.update("jax_enable_x64", True)

@jax.jit
def omega_update(h, z, key):
    return pgjax.pg_sample(h, z, key)
```

## Method

`pg_sample` follows the hybrid dispatch of
[`zoj613/polyagamma`](https://github.com/zoj613/polyagamma) (`method=None`),
changed where that scheme is measurably biased:

- **`h > 50`**: a normal with the exact Pólya-Gamma mean and variance.
- **`8 <= h <= 50`**: the **saddlepoint** rejection sampler (Windle, Polson &
  Scott 2014, arXiv:1405.0506), O(1) per accepted draw.
- **`h = 1`** (Bernoulli/logit), **or integer `h <= 4` with `|z| <= 1`**: the
  exact **Devroye** sampler for `PG(1, z)` (Polson, Scott & Windle 2013), summed
  `h` times.
- **Otherwise, `1 < h < 8`** — e.g. Negative-Binomial `h = y + alpha`: the
  **alternate** sampler (Windle, Polson & Scott 2014, Section 4).
- **`h < 1`**: the Gamma-series representation — 20 terms plus a remainder
  matched in mean and variance — exact in both moments.

Each departure from `polyagamma` 2.0.2 removes a bias measured with 2M draws
against the exact mean and variance:

- **Alternate sampler, `1 < h < 4`, small `|z|`**: `polyagamma` runs +0.4–0.9% in
  the mean and +1–5% in the variance. Its right-hand envelope uses the constant
  `(π/2)^(h/2)`, where both its Gamma proposal and the density's tail carry
  `(π/2)^h`; pgjax uses the latter.
- **`h < 1`**: the alternate method assumes `h >= 1`; below it `polyagamma` runs
  −0.4–0.8% in the mean. pgjax uses the Gamma series.
- **`4 < h < 8` with `|z| <= 4`**: `polyagamma` uses the saddlepoint
  approximation, 1–2% low in the variance. pgjax uses the exact alternate
  sampler.

The saddlepoint and alternate samplers are ported from `polyagamma`
(BSD-3-Clause). Every regime is tested against the closed-form mean and
variance.

The RNG is a per-call `xoshiro256**` seeded (via `splitmix64`) from the JAX PRNG
key, so draws are reproducible and each call owns its stream — safe when several
threads call at once. Normals and exponentials come from ziggurat samplers.

## Status

- ✅ All `h > 0`: the logit (`h = 1`), Negative-Binomial (`h = y + alpha`) and
  large-shape regimes.
- Dispatch is automatic: the sampler is selected from `(h, z)` with no API
  change.

CPU-only (the sampler is cheap scalar work; the point is to keep it on-device
next to the rest of a JIT'd Gibbs sweep).
