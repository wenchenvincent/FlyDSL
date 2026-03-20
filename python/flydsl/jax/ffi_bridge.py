# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Bridge between FlyDSL compiled kernels and JAX's XLA custom-call interface.

This module compiles ``@flyc.jit`` functions via the normal FlyDSL MLIR
pipeline, then registers the resulting native function pointer as an XLA
custom-call target so that ``jax.jit`` can invoke it.

Calling convention translation
------------------------------
XLA GPU custom call (``api_version=0``)::

    void custom_call(hipStream_t stream,
                     void** buffers,       // buffers[0..n_in-1] = inputs,
                                           // buffers[n_in..n_in+n_out-1] = outputs
                     const char* opaque,
                     size_t opaque_len)

FlyDSL bare-pointer convention::

    void jit_func(void** ptrs)
    // ptrs[i] = address of a storage cell holding the i-th argument value
    //   - tensor args: storage cell contains the device data pointer
    //   - stream arg (last): storage cell contains the hipStream_t value

The bridge function created by ``_make_xla_bridge`` translates between
these two conventions using pre-allocated ctypes storage.
"""

from __future__ import annotations

import ctypes
import hashlib
import struct
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:
    raise ImportError(
        "JAX is required for flydsl.jax.  Install with:\n"
        "  pip install jax[rocm]"
    ) from exc

from ..compiler.jit_executor import CompiledArtifact
from ..compiler.jit_function import MlirCompiler
from ..expr.typing import Stream

# ---------------------------------------------------------------------------
# Thread-safe registry of compiled kernels
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_registered_targets: Dict[str, "_RegisteredTarget"] = {}


class _RegisteredTarget:
    """Bookkeeping for a registered XLA custom-call target."""

    __slots__ = ("name", "artifact", "func_exe", "n_buffers", "bridge_cfunc")

    def __init__(self, name: str, artifact: CompiledArtifact, n_buffers: int):
        self.name = name
        self.artifact = artifact
        self.func_exe = artifact._get_func_exe()
        self.n_buffers = n_buffers
        self.bridge_cfunc = _make_xla_bridge(self.func_exe, n_buffers)


# ---------------------------------------------------------------------------
# Calling-convention bridge (ctypes)
# ---------------------------------------------------------------------------

# XLA GPU custom call signature (api_version=0):
#   void fn(hipStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
_XLA_GPU_CALL_TYPE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,    # stream (hipStream_t)
    ctypes.c_void_p,    # buffers (void**)
    ctypes.c_char_p,    # opaque
    ctypes.c_size_t,    # opaque_len
)


def _make_xla_bridge(
    fly_func: ctypes.CFUNCTYPE,
    n_buffers: int,
) -> _XLA_GPU_CALL_TYPE:
    """Create a ctypes callback that bridges XLA's GPU custom-call convention
    to FlyDSL's bare-pointer convention.

    Parameters
    ----------
    fly_func : ctypes function
        The compiled FlyDSL function with signature ``void fn(void** ptrs)``.
    n_buffers : int
        Total number of buffer arguments (inputs + outputs).  The stream
        is passed separately by XLA and appended as the last FlyDSL arg.

    Returns
    -------
    ctypes callback
        A function matching XLA's GPU custom-call signature.
    """
    # FlyDSL expects n_buffers + 1 (stream) pointer-to-pointer slots.
    n_fly_args = n_buffers + 1

    # Pre-allocate per-call storage (thread-local for safety).
    tls = threading.local()

    def _init_storage():
        # Storage cells: one c_void_p per argument.
        storage = (ctypes.c_void_p * n_fly_args)()
        # Packed pointer array: each element points to the corresponding storage cell.
        packed = (ctypes.c_void_p * n_fly_args)()
        for i in range(n_fly_args):
            packed[i] = ctypes.addressof(storage) + i * ctypes.sizeof(ctypes.c_void_p)
        tls.storage = storage
        tls.packed = packed

    def bridge(stream, buffers_ptr, opaque, opaque_len):
        if not hasattr(tls, "storage"):
            _init_storage()

        storage = tls.storage
        packed = tls.packed

        # Unpack XLA's buffers array (void** → individual void*).
        # buffers_ptr points to a C array of void* pointers.
        xla_buffers = ctypes.cast(buffers_ptr, ctypes.POINTER(ctypes.c_void_p))
        for i in range(n_buffers):
            storage[i] = xla_buffers[i]

        # Stream goes in the last slot.
        storage[n_buffers] = stream

        # Call FlyDSL's compiled function.
        fly_func(packed)

    # Prevent the bridge from being garbage-collected.
    cfunc = _XLA_GPU_CALL_TYPE(bridge)
    # Store a reference to prevent GC of the closure.
    cfunc._bridge_closure = bridge
    cfunc._fly_func_ref = fly_func
    return cfunc


# ---------------------------------------------------------------------------
# Compilation + registration
# ---------------------------------------------------------------------------


def compile_and_register(
    flyc_func: Callable,
    *,
    input_shapes: List[Tuple[Tuple[int, ...], Any]],
    output_shapes: List[Tuple[Tuple[int, ...], Any]],
    constexpr_kwargs: Optional[dict] = None,
) -> str:
    """Compile a ``@flyc.jit`` function and register it as an XLA custom-call target.

    Parameters
    ----------
    flyc_func : callable
        A FlyDSL ``@flyc.jit``-decorated function.
    input_shapes : list of (shape, dtype)
        Shape and dtype of each input tensor.
    output_shapes : list of (shape, dtype)
        Shape and dtype of each output tensor.
    constexpr_kwargs : dict, optional
        Compile-time constant keyword arguments.

    Returns
    -------
    str
        The registered custom-call target name.
    """
    if constexpr_kwargs is None:
        constexpr_kwargs = {}

    # Build a unique name based on function + shapes.
    sig_parts = [flyc_func.func.__name__ if hasattr(flyc_func, "func") else str(flyc_func)]
    for shape, dtype in input_shapes:
        sig_parts.append(f"i{shape}:{dtype}")
    for shape, dtype in output_shapes:
        sig_parts.append(f"o{shape}:{dtype}")
    for k, v in sorted(constexpr_kwargs.items()):
        sig_parts.append(f"c{k}={v}")

    name_hash = hashlib.sha256("|".join(sig_parts).encode()).hexdigest()[:16]
    target_name = f"flydsl_{name_hash}"

    with _lock:
        if target_name in _registered_targets:
            return target_name

    # Create concrete JAX arrays to trigger FlyDSL compilation.
    all_arrays = []
    for shape, dtype in list(input_shapes) + list(output_shapes):
        all_arrays.append(jnp.zeros(shape, dtype=dtype))

    # Import here to avoid circular imports at module level.
    from .adapter import from_jax

    jit_args = [from_jax(a) for a in all_arrays]

    # Trigger compilation by calling the JitFunction.
    # We need to actually compile it, so we call the internal compile path.
    # The simplest way is to do a dry-run call which triggers JIT.
    flyc_func(*jit_args, **constexpr_kwargs)

    # Now retrieve the compiled artifact from the JitFunction's cache.
    jit_fn = flyc_func  # JitFunction wrapper
    if not hasattr(jit_fn, "_mem_cache") or not jit_fn._mem_cache:
        raise RuntimeError(
            "FlyDSL compilation did not produce a cached artifact.  "
            "Ensure the function is a @flyc.jit-decorated function."
        )

    # Get the most recently compiled artifact.
    artifact = next(reversed(jit_fn._mem_cache.values()))
    if not isinstance(artifact, CompiledArtifact):
        raise RuntimeError(f"Expected CompiledArtifact, got {type(artifact).__name__}")

    n_buffers = len(input_shapes) + len(output_shapes)
    target = _RegisteredTarget(target_name, artifact, n_buffers)

    # Register with JAX's XLA custom-call mechanism.
    _register_with_xla(target)

    with _lock:
        _registered_targets[target_name] = target

    return target_name


def _register_with_xla(target: _RegisteredTarget) -> None:
    """Register the bridge function as an XLA custom-call target.

    Tries ``jax.ffi.register_ffi_target`` (modern API) first, falling back
    to ``jax.lib.xla_client.register_custom_call_target`` (legacy API).
    """
    # Get the raw function pointer from the ctypes callback.
    cfunc_ptr = ctypes.cast(target.bridge_cfunc, ctypes.c_void_p).value

    # Create a PyCapsule wrapping the function pointer.
    # Try the modern jax.ffi API first.
    try:
        import jax.ffi

        capsule = jax.ffi.pycapsule(ctypes.cast(cfunc_ptr, ctypes.c_void_p))
        jax.ffi.register_ffi_target(
            target.name,
            capsule,
            platform="rocm",
            api_version=0,
        )
        return
    except (AttributeError, ImportError, TypeError):
        pass

    # Fallback: legacy xla_client API.
    try:
        from jax.lib import xla_client

        capsule = _make_pycapsule(target.name, cfunc_ptr)
        xla_client.register_custom_call_target(
            target.name,
            capsule,
            platform="ROCM",
            api_version=0,
        )
        return
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(
            "Could not register XLA custom-call target.  "
            "Requires jax.ffi.register_ffi_target (JAX >=0.4.31) or "
            "jax.lib.xla_client.register_custom_call_target."
        ) from exc


def _make_pycapsule(name: str, func_ptr: int):
    """Create a PyCapsule wrapping a function pointer (for legacy xla_client API)."""
    import ctypes as ct

    PyCapsule_New = ct.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ct.py_object
    PyCapsule_New.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p]
    return PyCapsule_New(func_ptr, b"xla._CUSTOM_CALL_TARGET", None)
