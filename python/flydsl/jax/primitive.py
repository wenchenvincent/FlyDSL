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
4. At execution time, XLA invokes the custom call on its own HIP stream —
   no explicit stream management is needed.

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

import functools
from typing import Callable, Dict, Optional, Tuple, Union

try:
    import jax
    import jax.numpy as jnp
    from jax import core
    from jax.interpreters import mlir as jax_mlir

    # StableHLO dialect ops used in the lowering rule.
    from jax._src.lib.mlir.dialects import hlo as stablehlo
except ImportError as exc:
    raise ImportError(
        "JAX is required for flydsl.jax.  Install with:\n"
        "  pip install jax[rocm]"
    ) from exc

from .adapter import JaxTensorAdaptor, from_jax

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
# XLA lowering rule
# ---------------------------------------------------------------------------

# Cache: (func_id, input_shapes_key) -> registered target name
_lowering_cache: Dict[str, str] = {}


def _shapes_key(avals):
    """Create a hashable key from a sequence of abstract values."""
    return tuple((a.shape, a.dtype) for a in avals)


def _flydsl_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args,
    flyc_func: Callable,
    out_avals: Tuple[core.ShapedArray, ...],
    constexpr_kwargs: dict,
):
    """MLIR lowering rule: emit a ``stablehlo.custom_call`` backed by the
    compiled FlyDSL kernel.

    This function is called during ``jax.jit`` lowering.  It:
    1. Computes input/output shape signatures from ``ctx.avals_in``/``ctx.avals_out``.
    2. Triggers FlyDSL compilation (via ``ffi_bridge.compile_and_register``) if
       the kernel hasn't been compiled for these shapes yet.
    3. Emits a ``stablehlo.CustomCallOp`` that XLA will dispatch to the
       registered bridge function at runtime.
    """
    from .ffi_bridge import compile_and_register

    avals_in = ctx.avals_in
    avals_out = ctx.avals_out

    # Build cache key.
    func_id = id(flyc_func)
    cache_key = (func_id, _shapes_key(avals_in), _shapes_key(avals_out), tuple(sorted(constexpr_kwargs.items())))

    target_name = _lowering_cache.get(cache_key)
    if target_name is None:
        # Compile the FlyDSL function and register it as an XLA custom-call target.
        input_shapes = [(tuple(a.shape), a.dtype) for a in avals_in]
        output_shapes = [(tuple(a.shape), a.dtype) for a in avals_out]

        target_name = compile_and_register(
            flyc_func,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            constexpr_kwargs=constexpr_kwargs,
        )
        _lowering_cache[cache_key] = target_name

    # Build MLIR result types for each output.
    result_types = [jax_mlir.aval_to_ir_type(aval) for aval in avals_out]

    # Emit the custom call.
    # api_version=1 corresponds to the "typed" API; api_version=0 is the
    # legacy "untyped" API where buffers are passed as void**.
    # We use the untyped API (version 1 in StableHLO terms maps to
    # API_VERSION_STATUS_RETURNING in XLA; version 0 is API_VERSION_UNTYPED).
    #
    # StableHLO custom_call api_version attribute:
    #   0 = API_VERSION_ORIGINAL (not recommended)
    #   1 = API_VERSION_STATUS_RETURNING
    #   2 = API_VERSION_STATUS_RETURNING_UNIFIED (default for mlir.custom_call)
    #   4 = API_VERSION_TYPED_FFI
    #
    # For our void** calling convention we need version 1 (buffers as void**,
    # separate stream, inputs before outputs).
    call = stablehlo.CustomCallOp(
        result_types,
        list(args),
        call_target_name=target_name,
        api_version=1,  # STATUS_RETURNING: fn(stream, buffers, opaque, opaque_len)
        backend_config=b"",
        has_side_effect=True,  # FlyDSL kernels write to output buffers.
    )

    return call.results


# Register the lowering for the ROCm platform.
# JAX uses "rocm" as the platform name for AMD GPUs.
jax_mlir.register_lowering(flydsl_call_p, _flydsl_lowering, platform="rocm")

# Also register for "gpu" platform (some JAX versions use this generically).
try:
    jax_mlir.register_lowering(flydsl_call_p, _flydsl_lowering, platform="gpu")
except Exception:
    pass


# ---------------------------------------------------------------------------
# jax_kernel wrapper
# ---------------------------------------------------------------------------

GridSpec = Union[
    Tuple[int, ...],
    Callable[..., Tuple[int, ...]],
]


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
