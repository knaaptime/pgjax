"""Post-build checks on an installed pgjax wheel (cibuildwheel test step).

pgjax is self-contained C++, so its wheel should bundle no shared libraries at
all; a bundled OpenMP runtime in particular can abort a process that loads
another one.  Then check the sampler loads and draws from the right law.
"""

import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pgjax

root = pathlib.Path(pgjax.__file__).resolve().parent
bundled = [
    p.name
    for d in (root / ".dylibs", root.parent / "pgjax.libs")
    if d.is_dir()
    for p in d.iterdir()
]
if bundled:
    sys.exit(f"pgjax wheels should bundle no libraries, found: {bundled}")

# E[PG(h, z)] = h / (2 z) * tanh(z / 2), which is h / 4 at z = 0.
n = 200_000
for h, z in ((1.0, 0.0), (1.0, 2.0), (3.0, -1.5)):
    draws = np.asarray(
        pgjax.pg_sample(jnp.full(n, h), jnp.full(n, z), jax.random.key(0))
    )
    mean = h / 4 if z == 0 else h / (2 * z) * np.tanh(z / 2)
    se = draws.std() / np.sqrt(n)
    if abs(draws.mean() - mean) > 5 * se:
        sys.exit(f"PG({h}, {z}) mean {draws.mean():.5f} differs from {mean:.5f}")

# Nested vmap folds into one batched call.
keys = jax.random.split(jax.random.key(1), 6).reshape(2, 3)
out = jax.vmap(jax.vmap(lambda k: pgjax.pg_sample(jnp.ones(4), jnp.zeros(4), k)))(keys)
assert out.shape == (2, 3, 4) and bool(jnp.all(out > 0))
print(f"pgjax {pgjax.__version__}: bundles nothing; checks passed")
