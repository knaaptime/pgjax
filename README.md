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

- **Small integer `h`** (incl. Bernoulli/logit `h = 1`): the exact **Devroye**
  sampler for `PG(1, z)` (Polson, Scott & Windle 2013), summed `h` times.
- **Real `h`** (e.g. Negative-Binomial `h = y + alpha`): the exact infinite-sum
  (Gamma) representation truncated at `K` terms with a **moment-matched Gamma
  tail correction** for the remainder. The tail mean/variance have closed forms
  (`Σ 1/d_k = (π/2a)tanh(πa)`, and its `a`-derivative), so the correction cancels
  the truncation bias in the mean — it is precisely that bias (truncation with no
  tail term) that collapses `alpha` in NB models when omitted.
- **Large `h`**: the **saddlepoint** rejection sampler (Windle, Polson & Scott
  2014, arXiv:1405.0506). A two-piece bounding-kernel envelope proposes, and the
  saddlepoint density approximation accepts, giving **O(1)** per accepted draw
  versus the O(K) Gamma sum. Used automatically when `h >= 8` or
  (`h > 4` and `|z| <= 4`) — the hybrid dispatch thresholds validated in
  `zoj613/polyagamma`.

Validated against `random_polyagamma` over `h ∈ [0.5, 40] × z ∈ [-8, 8]`:
KS ≤ 0.009, relative mean error ≤ 0.34%. The saddlepoint path is validated to
KS < 0.02 and relative mean error < 2% over `h ∈ {8,…,40} × z ∈ {0,1,-4,5}`.

The RNG is a per-call `xoshiro256**` seeded (via `splitmix64`) from the JAX PRNG
key, so draws are reproducible and each call owns its stream — safe under the
concurrent per-device calls `jax.pmap` issues.

## Status

- ✅ Exact `PG(1, z)` / integer `h` (Devroye), real `h` (tail-corrected Gamma
  sum), and large `h` (saddlepoint) — covers the logit (`h=1`),
  Negative-Binomial (`h=y+alpha`), and large-shape regimes.
- Hybrid dispatch is automatic: the appropriate sampler is selected from
  `(h, z)` with no API change.

CPU-only (the sampler is cheap scalar work; the point is to keep it on-device
next to the rest of a JIT'd Gibbs sweep).
