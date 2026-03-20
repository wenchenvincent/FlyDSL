# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL JAX integration.

Provides two levels of integration:

Level 1 — Eager mode (``from_jax``):
    Wrap JAX arrays as FlyDSL JitArguments so they can be passed directly to
    ``@flyc.jit`` functions.  Requires ``jax.block_until_ready()`` for
    synchronization.

Level 2 — ``jax.jit`` integration (``jax_kernel``):
    Register compiled FlyDSL kernels as JAX primitives via the XLA FFI so they
    compose with ``jax.jit``.

Usage (eager)::

    import jax.numpy as jnp
    import flydsl.compiler as flyc
    from flydsl.jax import from_jax

    a = jnp.ones(1024, dtype=jnp.float32)
    ta = from_jax(a)
    my_jit_func(ta, ...)

Usage (jax.jit)::

    from flydsl.jax import jax_kernel

    wrapped = jax_kernel(my_flyc_jit_func, grid=..., block=...)

    @jax.jit
    def f(a, b):
        return wrapped(a, b)
"""

from flydsl.jax.adapter import JaxTensorAdaptor, from_jax
from flydsl.jax.primitive import jax_kernel

__all__ = [
    "from_jax",
    "JaxTensorAdaptor",
    "jax_kernel",
]
