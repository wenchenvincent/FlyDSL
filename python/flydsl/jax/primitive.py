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
  buffers.  The wrapper pre-allocates outputs and passes them as additional
  XLA buffers to the custom call.
- **No autograd**: ``jax.grad`` is not supported (would need explicit VJP
  rules with a backward kernel).
- **No vmap**: Batching rules are not yet implemented.
- **Shape-specialization**: Each unique set of input shapes triggers a new
  compilation, cached by FlyDSL's existing cache.
- **Scalar args baked at compile time**: Non-tensor runtime arguments (e.g.
  ``n: Int32``) are traced with their concrete values during compilation.
  Changing them requires recompilation (use ``Constexpr`` or pass via
  ``runtime_scalars``).
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
    runtime_scalars: dict,
    **_kwargs,
):
    """Eager implementation: compile and run via FlyDSL's normal JIT path.

    Note: the kernel executes asynchronously on the default HIP stream.
    Callers should use ``jax.block_until_ready()`` on the returned arrays.
    """
    # Allocate output arrays.
    outputs = []
    for aval in out_avals:
        outputs.append(jnp.zeros(aval.shape, dtype=aval.dtype))

    # Convert all JAX arrays to JaxTensorAdaptors.
    jit_args = []
    for a in list(args) + outputs:
        jit_args.append(from_jax(a))

    # Append runtime scalar values.
    call_args = list(jit_args)
    for _name, val in sorted(runtime_scalars.items()):
        call_args.append(val)

    # Call via FlyDSL JIT (uses default stream).
    flyc_func(*call_args, **constexpr_kwargs)

    return tuple(outputs)


flydsl_call_p.def_impl(_flydsl_impl)


# ---------------------------------------------------------------------------
# XLA lowering rule
# ---------------------------------------------------------------------------

# Cache: hashable key -> registered target name
_lowering_cache: Dict[tuple, str] = {}


def _shapes_key(avals):
    """Create a hashable key from a sequence of abstract values."""
    return tuple((a.shape, a.dtype) for a in avals)


def _flydsl_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args,
    flyc_func: Callable,
    out_avals: Tuple[core.ShapedArray, ...],
    constexpr_kwargs: dict,
    runtime_scalars: dict,
):
    """MLIR lowering rule: emit a ``stablehlo.custom_call`` backed by the
    compiled FlyDSL kernel.

    This function is called during ``jax.jit`` lowering.  It:
    1. Computes input/output shape signatures from ``ctx.avals_in``/``ctx.avals_out``.
    2. Triggers FlyDSL compilation (via ``ffi_bridge.compile_and_register``) if
       the kernel hasn't been compiled for these shapes yet.
    3. Emits a ``stablehlo.CustomCallOp`` that XLA will dispatch to the
       registered bridge function at runtime.

    Non-tensor arguments (``runtime_scalars``) are baked into the compiled
    kernel during tracing — the XLA custom call only receives tensor buffers.
    """
    from .ffi_bridge import compile_and_register

    avals_in = ctx.avals_in
    avals_out = ctx.avals_out

    # Build cache key.
    func_id = id(flyc_func)
    rt_key = tuple(sorted(runtime_scalars.items()))
    cache_key = (
        func_id,
        _shapes_key(avals_in),
        _shapes_key(avals_out),
        tuple(sorted(constexpr_kwargs.items())),
        rt_key,
    )

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
            runtime_scalars=runtime_scalars,
        )
        _lowering_cache[cache_key] = target_name

    # Build MLIR result types for each output.
    result_types = [jax_mlir.aval_to_ir_type(aval) for aval in avals_out]

    # Emit the custom call.
    #
    # StableHLO custom_call api_version attribute:
    #   0 = API_VERSION_ORIGINAL (not recommended)
    #   1 = API_VERSION_STATUS_RETURNING
    #   2 = API_VERSION_STATUS_RETURNING_UNIFIED
    #   4 = API_VERSION_TYPED_FFI
    #
    # We use api_version=1 (STATUS_RETURNING): the registered function
    # receives (stream, void** buffers, opaque, opaque_len).
    # This must match the api_version used in ffi_bridge._register_with_xla.
    call = stablehlo.CustomCallOp(
        result_types,
        list(args),
        call_target_name=target_name,
        api_version=1,
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


def jax_kernel(
    flyc_func: Callable,
    *,
    out_shapes: Callable,
    constexpr_kwargs: Optional[dict] = None,
    runtime_scalars: Optional[dict] = None,
) -> Callable:
    """Wrap a ``@flyc.jit`` function for use inside ``jax.jit``.

    Parameters
    ----------
    flyc_func : callable
        A FlyDSL ``@flyc.jit``-decorated function.
    out_shapes : callable
        A function ``(*wrapper_args) -> list[(shape, dtype)]`` that returns
        the shape and dtype of each output tensor the kernel will produce.
        FlyDSL kernels write to pre-allocated output buffers, so the caller
        must specify the output layout.
    constexpr_kwargs : dict, optional
        Compile-time constant keyword arguments forwarded to the FlyDSL
        function (``Constexpr`` parameters).
    runtime_scalars : dict, optional
        Non-tensor runtime arguments (e.g. ``{"n": 128}``).  These are
        passed to the FlyDSL function during compilation tracing but are
        NOT passed through the XLA custom call at runtime — they are baked
        into the compiled kernel.  Changing them requires recompilation.

    Returns
    -------
    callable
        A function that accepts only JAX array arguments and returns a
        tuple of JAX output arrays.  Compatible with ``jax.jit``.

    Examples
    --------
    ::

        @flyc.jit
        def my_add(A, B, C, n, const_n, stream):
            ...

        wrapped = jax_kernel(
            my_add,
            out_shapes=lambda a, b: [(a.shape, a.dtype)],
            constexpr_kwargs={"const_n": 129},
            runtime_scalars={"n": 128},
        )

        @jax.jit
        def f(a, b):
            (c,) = wrapped(a, b)
            return c
    """
    if constexpr_kwargs is None:
        constexpr_kwargs = {}
    if runtime_scalars is None:
        runtime_scalars = {}

    @functools.wraps(flyc_func)
    def wrapper(*args):
        # Compute output abstract values from the user-provided function.
        out_specs = out_shapes(*args)
        out_avals = tuple(
            core.ShapedArray(shape, dtype) for shape, dtype in out_specs
        )

        # Only JAX arrays are passed as primitive operands — scalars are
        # carried as primitive parameters and baked into the compiled kernel.
        return flydsl_call_p.bind(
            *args,
            flyc_func=flyc_func,
            out_avals=out_avals,
            constexpr_kwargs=constexpr_kwargs,
            runtime_scalars=runtime_scalars,
        )

    return wrapper
