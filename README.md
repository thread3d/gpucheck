# gpucheck

A macOS multi-GPU diagnostic that **maps** each installed GPU's features,
**verifies** that it computes correctly, and **measures** what it can do — then
writes a report.

It was built for and validated on a multi-GPU macOS workstation with three
compute GPUs, and has no dependency on any particular configuration.

## Read this first: safety

This tool deliberately drives the GPUs hard. On a machine whose GPUs also drive
the displays, heavy compute can starve WindowServer and **hang the whole
system**. A GPU with a history of driver resets can be pushed into one.

That is not hypothetical: during development, an early full-intensity run hung
the development machine badly enough to need a power cycle. The system logs
afterwards showed a long history of GPU driver resets concentrated on the GPU
model driving the displays, plus WindowServer watchdog hangs.

The defaults are therefore conservative, and you should understand them before
running anything:

- **`--intensity safe` is the default.** Every correctness test and every
  benchmark still runs, but with reduced workloads and a cooldown between tests.
- **Sustained full-occupancy load and driving every GPU at once are off** unless
  you pass `--allow-heavy`. Those are the two highest-risk tests.
- **`--dry-run`** prints the selected GPUs and the exact workload sizes without
  touching the GPU.
- **`--skip-display-gpus`** is the lowest-risk way to exercise the machine.
- Every step is **journalled and fsync'd before the GPU is touched**, so if the
  system does hang, the journal names the test responsible and the one that never
  finished.
- Reports are **rewritten after each GPU completes** and carry a `partial` flag,
  so results survive a reboot.

The tool also reads the system's own GPU-reset and WindowServer-hang history and
reports it before doing any work. If you are on an unstable machine, start with:

```sh
./build.sh
./.build/gpucheck --device-name radeon --intensity safe --dry-run
```

## Build

No Xcode project and no SwiftPM resolution. Metal shaders are compiled at
runtime by the app, so the `metal` command-line tool is not required either —
only the Swift compiler from the Command Line Tools.

```sh
./build.sh          # produces .build/gpucheck
```

## Usage

```sh
./.build/gpucheck --list                              # list GPUs and exit
./.build/gpucheck --dry-run                           # show the plan, touch nothing
./.build/gpucheck                                     # safe defaults, all GPUs
./.build/gpucheck --device-name radeon --mode quick     # one GPU, shortest run
./.build/gpucheck --intensity full --allow-heavy      # full load (see safety)
```

### Options

| Option | Meaning |
|---|---|
| `--mode quick\|standard\|thorough` | How deep the run goes. `standard` is the default. |
| `--intensity safe\|normal\|full` | How hard any single workload may push the GPU. `safe` is the default. |
| `--allow-heavy` | Enables sustained-load and simultaneous all-GPU tests. |
| `--device N[,N...]` | Select by 1-based index. **Not stable across reboots.** |
| `--device-name TEXT[,TEXT...]` | Select by name substring. Stable across reboots. |
| `--device-id ID[,ID...]` | Select by Metal registry ID. Stable within a boot. |
| `--skip-display-gpus` | Exclude GPUs that drive a display. |
| `--out DIR` | Report directory (default `gpucheck-report`). |
| `--journal FILE` / `--no-journal` | Journal path / disable journalling. |
| `--bandwidth MB` | Memory-benchmark working set. |
| `--sustained SECONDS` | Sustained-load duration (needs `--allow-heavy`). |
| `--no-opencl`, `--no-render`, `--no-color` | Disable those parts. |
| `--list`, `--dry-run`, `-h` | Discovery and planning. |

Neither the Metal device order nor the registry IDs are guaranteed across a
reboot — device indices were observed to renumber after one. Use `--device-name`
for anything you want to reproduce.

## What it does

### 1. Feature mapping

Per GPU, from Metal: identity and location, memory limits (working set, max
buffer, threadgroup memory, heap alignment, sparse tile size), compute limits
(threadgroup size, argument-buffer tier, read-write texture tier, raster order
groups, sample counts, BC compression), supported GPU families, ray tracing and
function-pointer support, available performance counter sets, and an
**empirically probed pixel-format matrix** (each format is actually allocated as
a shader read/write target, a render target and a 4x MSAA target).

Per GPU, from OpenCL: a second, independent driver stack, which supplies the
**compute-unit count and clock that Metal never exposes** — the basis for the
theoretical-peak comparison — plus memory, work-group, atomics, fp64 and image
capabilities.

Shader groups are compiled on the GPU and reported individually, so a feature
the driver rejects is visible as such.

### 2. Functional verification

Every check compares GPU output against an independent **host-side CPU
reference**, so a PASS means the result was verified, not merely that nothing
crashed:

buffer round trips (bit-exact), SAXPY, threadgroup reduction with barriers,
atomic operations under contention, concurrent queue submission, FP16
(bit-exact), FP64, math-function accuracy versus libm, large-argument reduction,
fast-math delta, texture read / bilinear filtering / addressing modes, blit
paths, heap allocation, indirect command buffers, SIMD-group matrix, 64-bit
addressing across the 2 GiB byte boundary, GPU timestamp counters, an offscreen
render pass, and 4x MSAA with resolve.

### 3. Performance measurement

Memory copy / read / fill bandwidth, FP32 and FP16 FMA, FP64 FMA, INT32 ALU,
transcendental throughput, a threadgroup-size sweep, texture sampling rate,
dispatch latency (round-trip and batched), host↔device transfer, and — with
`--allow-heavy` — sustained-load stability and multi-GPU parallel scaling.

FP32 throughput is compared against a theoretical peak derived from the
OpenCL-reported compute units and clock. That peak assumes the maximum boost
clock, so the ratio is a **lower bound** on efficiency; the report says so.

## Output

Live progress on the console while running, then a summary plus three files in
the output directory:

- `gpucheck-report.json` — full structured results
- `gpucheck-report.md` — readable report with every feature and measurement
- `gpucheck-report.html` — self-contained, with cross-GPU bar charts
- `gpucheck-journal.log` — fsync'd step-by-step trace for crash diagnosis

## Design notes

- **Runtime shader compilation.** Embedding MSL as strings and compiling via
  `makeLibrary(source:)` removes the need for Xcode or the `metal` tool.
- **Isolated shader groups.** One unsupported construct must not disable
  everything. This mattered in practice: on a GPU that rejects `double` outright,
  sharing a group with the core kernels silently disabled 13 unrelated checks.
  FP64, the matrix coprocessor and the math library now each live in their own
  group.
- **Best-of-N with amortised dispatch.** Timings encode several dispatches per
  command buffer, because a single short dispatch is dominated by the
  commit/wait round trip (tens of microseconds) and would understate throughput.
- **Scale-based accuracy scoring.** A per-value ULP distance is ill-conditioned
  at a function's zeros: `sin` near ±π is ~1e-7, so a negligible absolute error
  reads as ~10⁹ ULP. Accuracy is scored in ULP of each function's own output
  scale, with large-argument behaviour measured separately against the bound
  implied by float's representation.

## Layout

```
build.sh                 build script
Sources/main.swift       orchestration and CLI
Sources/Inventory.swift  Metal feature map and pixel-format probe
Sources/OpenCLProbe.swift OpenCL inventory and cross-check benchmark
Sources/Shaders.swift    embedded MSL, grouped for isolation
Sources/MetalContext.swift pipeline cache and dispatch harness
Sources/Tests.swift      functional verification and workload configuration
Sources/Benchmarks.swift performance measurements and multi-GPU scaling
Sources/Report.swift     console, JSON, Markdown and HTML reports
Sources/Support.swift    data model, journal, environment risk assessment
```
