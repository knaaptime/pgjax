# Installation

`pgjax` supports Python >= 3.9. We recommend using [miniforge] or [pixi].

## Installing a released version

`pgjax` is available on PyPI and can be installed with:

```bash
pip install pgjax
```

JAX is a core dependency and is always installed.

## Installing from source

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/knaaptime/pgjax.git
cd pgjax
conda env create -f environment.yml
conda activate pgjax
pip install -e . --no-deps
```

## Verifying the installation

```python
import jax
jax.config.update("jax_enable_x64", True)
import pgjax

key = jax.random.PRNGKey(0)
h = jax.numpy.ones(5)
z = jax.numpy.zeros(5)
print(pgjax.pg_sample(h, z, key))
```

!!! note
    `pgjax` requires 64-bit mode (`jax_enable_x64 = True`). A
    `RuntimeError` is raised if x64 is not enabled.

[miniforge]: https://github.com/conda-forge/miniforge
[pixi]: https://pixi.sh
