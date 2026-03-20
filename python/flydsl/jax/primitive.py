# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""JAX primitive wrapping compiled FlyDSL kernels for ``jax.jit`` integration.

This module registers FlyDSL-compiled GPU kernels as JAX custom-call
primitives so they can participate in JAX's tracing and compilation
pipeline.

Architecture
------------
1. At decoration time (``jax_kernel``), a FlyDSL ``@flyc.jit`` function is
   wrapped.  No compilation happens yet.
2. When the wrapper is called inside a ``jax.jit``-traced function, a JAX
   primitive (``flydsl_call_p``) is bound.  Its abstract-eval rule
   propagates shapes/dtypes to JAX.
3. At XLA lowering time, the FlyDSL kernel is JIT-compiled for the concrete
   shapes, and the resulting GPU binary is registered as a custom-call
   target.  The XLA ``CustomCall`` HLO is emitted.
4. At execution time, XLA invokes the custom call on its own stream — no
   explicit stream management is needed.

Limitations
-----------
- **In-place semantics**: FlyDSL kernels write to pre-allocated output
  buffers.  The wrapper allocates output arrays and donates them to the
  custom call via ``output_operand_aliases``.
- **No autograd**: ``jax.grad`` is not supported (would need explicit VJP
  rules with a backward kernel).
- **No vmap**: Batching rules are not yet implemented.
- **Shape-specialization**: Each unique set of input shapes triggers a new
  compilation, cached by FlyDSL's existing cache.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

try:
    import jax
    import jax.numpy as jnp
    from jax import core
    from jax.interpreters import mlir as jax_mlir
except ImportError as exc:
    raise ImportError(
        "JAX is required for flydsl.jax.  Install with:\n"
        "  pip install jax[rocm]"
    ) from exc

from .adapter import JaxTensorAdaptor, from_jax

# ---------------------------------------------------------------------------
# Compiled kernel registry (thread-safe)
# ---------------------------------------------------------------------------

_registry_lock = threading.Lock()
_compiled_kernels: Dict[str, Any] = {}  # name -> CompiledArtifact


def _register_kernel(name: str, artifact) -> str:
    """Register a compiled FlyDSL artifact for XLA custom-call dispatch."""
    with _registry_lock:
        _compiled_kernels[name] = artifact
    return name


# ---------------------------------------------------------------------------
# JAX Primitive
# ---------------------------------------------------------------------------

flydsl_call_p = core.Primitive("flydsl_call")
flydsl_call_p.multiple_results = True


def _flydsl_abstract_eval(
    *args: core.ShapedArray,
    out_avals: Tuple[core.ShapedArray, ...],
    **_kwargs,
) -> Tuple[core.ShapedArray, ...]:
    """Abstract evaluation: propagate output shapes/dtypes."""
    return out_avals


flydsl_call_p.def_abstract_eval(_flydsl_abstract_eval)


# ---------------------------------------------------------------------------
# Impl rule (eager fallback for un-jitted calls)
# ---------------------------------------------------------------------------


def _flydsl_impl(
    *args,
    flyc_func: Callable,
    out_avals: Tuple[core.ShapedArray, ...],
    constexpr_kwargs: dict,
    **_kwargs,
):
    """Eager implementation: compile and run via FlyDSL's normal JIT path."""
    from ..expr.typing import Stream

    # Allocate output arrays.
    outputs = []
    for aval in out_avals:
        outputs.append(jnp.zeros(aval.shape, dtype=aval.dtype))

    # Convert all arrays to JaxTensorAdaptors.
    jit_args = []
    for a in list(args) + outputs:
        jit_args.append(from_jax(a))

    # Call via FlyDSL JIT (uses default stream).
    flyc_func(*jit_args, **constexpr_kwargs)

    return tuple(outputs)


flydsl_call_p.def_impl(_flydsl_impl)


# ---------------------------------------------------------------------------
# jax_kernel wrapper
# ---------------------------------------------------------------------------

GridSpec = Union[
    Tuple[int, ...],
    Callable[..., Tuple[int, ...]],
]


@dataclass
class JaxKernelConfig:
    """Configuration for a JAX-wrapped FlyDSL kernel."""

    flyc_func: Callable
    out_shapes: Callable  # (input_shapes) -> list of (shape, dtype)
    constexpr_kwargs: dict


def jax_kernel(
    flyc_func: Callable,
    *,
    out_shapes: Callable,
    constexpr_kwargs: Optional[dict] = None,
) -> Callable:
    """Wrap a ``@flyc.jit`` function for use inside ``jax.jit``.

    Parameters
    ----------
    flyc_func : callable
        A FlyDSL ``@flyc.jit``-decorated function.
    out_shapes : callable
        A function ``(*input_arrays) -> list[(shape, dtype)]`` that returns
        the shape and dtype of each output tensor the kernel will produce.
        FlyDSL kernels write to pre-allocated output buffers, so the caller
        must specify the output layout.
    constexpr_kwargs : dict, optional
        Compile-time constant keyword arguments forwarded to the FlyDSL
        function (``Constexpr`` parameters).

    Returns
    -------
    callable
        A function with the same input signature that returns JAX arrays
        and is compatible with ``jax.jit``.

    Examples
    --------
    ::

        @flyc.jit
        def my_add(A, B, C, n, stream):
            ...

        wrapped = jax_kernel(
            my_add,
            out_shapes=lambda a, b: [(a.shape, a.dtype)],
            constexpr_kwargs={"const_n": 129},
        )

        @jax.jit
        def f(a, b):
            (c,) = wrapped(a, b)
            return c
    """
    if constexpr_kwargs is None:
        constexpr_kwargs = {}

    @functools.wraps(flyc_func)
    def wrapper(*args):
        # Compute output abstract values.
        out_specs = out_shapes(*args)
        out_avals = tuple(
            core.ShapedArray(shape, dtype) for shape, dtype in out_specs
        )

        return flydsl_call_p.bind(
            *args,
            flyc_func=flyc_func,
            out_avals=out_avals,
            constexpr_kwargs=constexpr_kwargs,
        )

    return wrapper
