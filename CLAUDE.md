# FlyDSL Project Guide

FlyDSL (Flexible Layout Python DSL) — a Python DSL and MLIR-based compiler stack for authoring high-performance GPU kernels with explicit layouts and tiling on AMD GPUs (MI300X/MI350).

## Repository Layout

```
FlyDSL/
├── python/flydsl/              # Python DSL core
│   ├── expr/                   # DSL expression API (arith, vector, gpu, rocdl, buffer_ops)
│   │   ├── arith.py            # Arithmetic ops (constant, select, index_cast, etc.)
│   │   ├── vector.py           # Vector ops (extract, insert, load_op, store, broadcast)
│   │   ├── gpu.py              # GPU indexing (thread_idx, block_idx, barrier)
│   │   ├── rocdl.py            # AMD intrinsics (mfma, cvt_pk_fp8_f32, ds_bpermute)
│   │   ├── rocdl/              # ROCDL sub-package (universal ops)
│   │   ├── buffer_ops.py       # Buffer resource ops (buffer_load, buffer_store)
│   │   ├── typing.py           # Type system (T, Tensor, Stream, Constexpr, Int32)
│   │   ├── primitive.py        # Primitive value wrappers and operations
│   │   ├── numeric.py          # Numeric type support
│   │   ├── derived.py          # Derived operations (logical_divide, composition)
│   │   ├── meta.py             # Metaprogramming utilities
│   │   └── utils/              # Expression utility helpers
│   ├── compiler/               # JIT compilation
│   │   ├── kernel_function.py  # @flyc.kernel decorator
│   │   ├── jit_function.py     # @flyc.jit decorator and MlirCompiler
│   │   ├── ast_rewriter.py     # AST rewriting for kernel tracing
│   │   ├── jit_executor.py     # JitCFunction wrapper for MLIR ExecutionEngine
│   │   ├── jit_argument.py     # DLPack argument handling (from_dlpack)
│   │   └── protocol.py         # Type protocols (fly_construct, fly_types, fly_values)
│   ├── utils/                  # Utilities
│   │   ├── smem_allocator.py   # Shared memory (LDS) allocation (SmemAllocator, SmemPtr)
│   │   ├── env.py              # Environment variable helpers
│   │   └── logger.py           # Logging utilities
│   ├── runtime/                # Runtime utilities
│   │   └── device.py           # GPU arch detection (get_rocm_arch, is_rdna_arch)
│   ├── lang/ir/                # IR type definitions
│   └── _mlir/                  # Embedded MLIR Python bindings (auto-generated, NEVER edit)
├── kernels/                    # Production GPU kernels (importable as kernels.*)
│   ├── pa_decode_fp8.py        # Paged attention decode (FP8)
│   ├── preshuffle_gemm.py      # Preshuffle GEMM with ping-pong LDS
│   ├── blockscale_preshuffle_gemm.py  # Block-scale GEMM variant
│   ├── layernorm_kernel.py     # LayerNorm (two-pass)
│   ├── rmsnorm_kernel.py       # RMSNorm (LDS-cached 3-pass)
│   ├── softmax_kernel.py       # Online softmax
│   ├── moe_gemm_2stage.py      # MoE GEMM 2-stage
│   ├── moe_blockscale_2stage.py # MoE block-scale 2-stage
│   ├── layout_utils.py         # Layout coordinate helpers (crd2idx, idx2crd)
│   ├── mfma_epilogues.py       # MFMA accumulation epilogues (C-shuffle)
│   ├── mfma_preshuffle_pipeline.py  # MFMA preshuffle pipeline helpers
│   └── kernels_common.py       # Shared kernel utilities
├── include/flydsl/             # C++ Fly dialect headers
│   ├── Dialect/Fly/            # Fly dialect (IR/, Transforms/, Utils/)
│   │   ├── IR/                 # FlyDialect.td, FlyOps.td, FlyTypeDefs.td, FlyAttrDefs.td
│   │   ├── Transforms/         # Passes.td, LayoutLowering, IntTupleLowering, MemrefLowering
│   │   └── Utils/              # LayoutUtils.h, IntTupleUtils.h, NormalForm.h
│   ├── Dialect/FlyROCDL/       # FlyROCDL dialect (CopyAtom, MmaAtom for CDNA3/CDNA4)
│   │   └── IR/                 # Dialect.td, Atom.td, CopyAtom.td, MmaAtom.td
│   └── Conversion/             # FlyToROCDL conversion headers
├── lib/                        # C++ dialect implementation
│   ├── Dialect/Fly/            # Fly ops, types, transforms, utils
│   ├── Dialect/FlyROCDL/       # FlyROCDL ops (CDNA3/ subdirectory)
│   ├── Conversion/             # FlyToROCDL lowering
│   ├── Bindings/Python/        # Python bindings (FlyExtension.cpp, FlyROCDLExtension.cpp)
│   ├── CAPI/                   # C API bindings
│   └── Runtime/                # ROCm runtime wrappers
├── tools/fly-opt/              # MLIR optimizer tool (fly-opt)
├── tests/                      # All tests
│   ├── kernels/                # Kernel correctness tests
│   ├── pyir/                   # IR-level tests (layout algebra, static vs dynamic)
│   ├── mlir/                   # MLIR FileCheck tests (Conversion/, LayoutAlgebra/, Transforms/)
│   ├── unit/                   # Unit tests (streams, async, launch)
│   ├── python/                 # Python example tests
│   ├── conftest.py             # pytest fixtures (ctx, module, insert_point)
│   └── test_common.py          # Performance profiling (@perftest decorator)
├── examples/                   # Runnable examples
│   ├── 01-vectorAdd.py         # Basic vector addition with layout algebra
│   ├── 02-tiledCopy.py         # Tiled copy with partitioned tensors
│   └── 03-tiledMma.py          # Tiled matrix multiply accumulate
├── scripts/                    # Build & test scripts
│   ├── build_llvm.sh           # Build LLVM/MLIR from source (~30min)
│   ├── build.sh                # Build FlyDSL C++ + Python bindings (~5min)
│   ├── run_tests.sh            # Run all tests (pytest + FileCheck)
│   ├── run_benchmark.sh        # Performance benchmarking harness
│   ├── build_wheels.sh         # Build Python wheels
│   └── dumpir.sh               # Dump IR for debugging
├── docs/                       # Documentation (architecture, layout, kernel guides)
├── thirdparty/                 # Third-party deps (dlpack, tvm-ffi, llvm-hash.txt)
└── .github/workflows/          # CI/CD (ci.yaml, build-whl.yaml, test-whl.yaml, etc.)
```

## Build & Install

```bash
# Build LLVM/MLIR (one-time, ~30min)
bash scripts/build_llvm.sh -j64

# Build FlyDSL C++ + Python bindings (~5min)
bash scripts/build.sh -j64

# Install in dev mode
pip install -e .
```

**Prerequisites**: ROCm 6.x/7.x, cmake >=3.20, C++17 compiler, Python 3.10+.

## Running Tests

```bash
# All tests
PYTHONPATH=./ pytest tests/

# Full test suite (includes MLIR FileCheck tests)
bash scripts/run_tests.sh

# Specific kernel test
PYTHONPATH=./ python tests/kernels/test_pa.py --num_iters 50

# Specific test by name
PYTHONPATH=./ pytest tests/ -k "test_rmsnorm" -v

# Performance benchmarks
bash scripts/run_benchmark.sh

# Disable JIT cache during development
FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPATH=./ python tests/kernels/test_pa.py
```

## Code Style

- **Python**: black (line-length=120), ruff for linting (E/W/F/I rules). Config in `pyproject.toml`.
- **C++**: LLVM style (ColumnLimit=100, 2-space indent). Config in `.clang-format`.
- **Imports**: isort via ruff with `flydsl` as known first-party.
- **License header**: Required on all new source files. See `CONTRIBUTING.md` for format.

Before submitting:
```bash
ruff check python/ kernels/ tests/
ruff format --check python/ kernels/ tests/
```

## Key Concepts

### DSL Expression API (`python/flydsl/expr/`)

Kernels are written in Python using the FlyDSL expression API:
- `arith` — arithmetic ops (constant, select, index_cast, trunci, extsi, shli, xori, andi, etc.)
- `vector` — vector ops (extract, insert, load_op, store, broadcast, from_elements, bitcast)
- `gpu` — GPU indexing (thread_idx, block_idx, block_dim, grid_dim, barrier)
- `rocdl` — AMD-specific intrinsics (mfma, cvt_pk_fp8_f32, ds_bpermute); also `rocdl.universal`
- `buffer_ops` — buffer resource ops (create_buffer_resource, buffer_load, buffer_store)

The `arith` module provides `ArithValue` class for operator overloading (e.g., `c + 1`, `c * 2`).

### Compiler Decorators

- `@flyc.kernel` — declares a GPU kernel function (uses `kernel_function.py`)
- `@flyc.jit` — JIT-compiles a function via MLIR tracing (uses `jit_function.py`)
- `flyc.from_dlpack` — converts DLPack tensors for kernel arguments

### Kernel Authoring Pattern

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, gpu, rocdl, buffer_ops
from flydsl.expr.typing import T, Int32, Constexpr

@flyc.kernel
def my_kernel(input_ptr: fx.Tensor, output_ptr: fx.Tensor, N: Int32):
    tid = gpu.thread_idx.x + gpu.block_idx.x * arith.constant(256, type=T.i32)
    rsrc_in = buffer_ops.create_buffer_resource(input_ptr, max_size=True)
    val = buffer_ops.buffer_load(rsrc_in, tid, vec_width=1, dtype=T.f32)
    # ... compute ...
    rsrc_out = buffer_ops.create_buffer_resource(output_ptr, max_size=True)
    buffer_ops.buffer_store(result, rsrc_out, tid)
```

### SmemAllocator & SmemPtr

Shared memory (LDS) is managed via `SmemAllocator` and `SmemPtr`:
```python
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

allocator = SmemAllocator(None, arch=arch, global_sym_name="my_smem")
allocator.ptr = size_in_bytes
base = allocator.get_base()
lds_view = SmemPtr(base, offset, T.f32, shape=(N,)).get()  # returns memref for loads/stores
```

### scf.for Loops with Loop-Carried Values

FlyDSL supports `scf.for` loops via Python `range()` with `init=` keyword:
```python
loop_start = arith.index(0)
loop_stop = arith.index(N)
loop_step = arith.index(1)
for iv, state in range(loop_start, loop_stop, loop_step, init=[init_val1, init_val2]):
    # Use state[0], state[1] ...
    # Yield updated values:
    results = yield [new_val1, new_val2]
# After loop: results contains final values
```

Important: clear `SmemPtr._view_cache = None` after exiting scf.for to avoid MLIR dominance errors in epilogue code.

### Layout System

The Fly MLIR dialect provides layout algebra operations for explicit data layouts:
- Types: `!fly.int_tuple`, `!fly.layout`, `!fly.coord_tensor`, `!fly.memref`
- Operations: `make_shape`, `crd2idx`, `composition`, `divide`, `product`
- See `docs/layout_system_guide.md` and `docs/cute_layout_algebra_guide.md` for details
- `kernels/layout_utils.py` provides Python helpers: `crd2idx()`, `idx2crd()`, `get()`

### C++ Dialect Architecture

Two MLIR dialects:
- **Fly** (`include/flydsl/Dialect/Fly/`) — layout algebra types and operations
- **FlyROCDL** (`include/flydsl/Dialect/FlyROCDL/`) — hardware-specific copy and MMA atoms (CDNA3/CDNA4)

Lowering pipeline: Fly → FlyROCDL → ROCDL → LLVM IR → AMD GPU ISA

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `FLYDSL_RUNTIME_ENABLE_CACHE` | Set to `0` to disable JIT cache during development |
| `PYTHONPATH` | Set to `./` when running from repo root |
| `MLIR_PATH` | MLIR installation prefix (auto-detected by build.sh) |
| `FLY_BUILD_DIR` | Build directory (default: `build-fly/`) |
| `FLY_REBUILD` | Control auto-rebuild (`auto`/`0`/`1`) |
| `FLYDSL_GPU_ARCH` | Override GPU architecture detection (e.g., `gfx942`) |
| `FLYDSL_RUN_QUANT` | Enable quantization tests |
| `RUN_TESTS_FULL` | Run all tests including `large_shape` marked tests |

## Development Notes

- Always set `FLYDSL_RUNTIME_ENABLE_CACHE=0` when iterating on kernel code to bypass JIT cache
- `PYTHONPATH=./` is required when running from the repo root
- Kernel files in `kernels/` are importable as `from kernels.pa_decode_fp8 import ...`
- The `_mlir` package under `python/flydsl/_mlir/` is auto-generated during build — never edit it directly
- LLVM commit pinned in `thirdparty/llvm-hash.txt`
- Use `scripts/dumpir.sh` to dump MLIR IR for debugging
- The `fly-opt` tool in `tools/fly-opt/` runs MLIR passes standalone

## PR and Commit Conventions

- **PR title prefixes**: `[Bugfix]`, `[Feature]`, `[Kernel]`, `[DSL]`, `[Dialect]`, `[Perf]`, `[Doc]`, `[Test]`, `[CI]`, `[Misc]`
- **Commit messages**: imperative voice ("Fix this bug", not "Fixed the bug"), no trailing period on subject
- **Sign-off**: use `git commit -s` for DCO compliance
- **Performance PRs**: include benchmark results (hardware, baseline, optimized, improvement %)
- **License**: Apache-2.0. New source files need SPDX header (see `CONTRIBUTING.md`)

## Documentation

Detailed guides in `docs/`:
- `architecture_guide.md` — project structure, compilation pipeline, key abstractions
- `kernel_authoring_guide.md` — `@flyc.kernel`/`@flyc.jit` API, launch config, memory, sync
- `layout_system_guide.md` — layout algebra ops, coordinate mapping
- `cute_layout_algebra_guide.md` — CUTE-style layout algebra concepts
- `prebuilt_kernels_guide.md` — LayerNorm, RMSNorm, Softmax, GEMM builders and configs
- `testing_benchmarking_guide.md` — running tests, performance profiling, benchmark harness
