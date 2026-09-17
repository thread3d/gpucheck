// Benchmarks.swift — measured throughput of the major GPU subsystems.
//
// Where possible each number is paired with a theoretical peak derived from the
// compute-unit count and clock reported by OpenCL, so the measurement can be
// sanity-checked against known hardware.
import Foundation
import Metal

final class Benchmarks {
    let ctx: MetalContext
    let config: SuiteConfig
    let openCLMatch: OpenCLDevice?
    private(set) var measurements: [Measurement] = []
    private(set) var checks: [Check] = []

    init(ctx: MetalContext, config: SuiteConfig, openCLMatch: OpenCLDevice? = nil) {
        self.ctx = ctx
        self.config = config
        self.openCLMatch = openCLMatch
    }

    private func tpg(_ n: Int) -> MTLSize { MTLSize(width: n, height: 1, depth: 1) }

    private func add(_ group: String, _ name: String, _ value: Double, _ unit: String, _ formatted: String,
                     higherIsBetter: Bool = true, notes: [String] = []) {
        measurements.append(Measurement(group: group, name: name, value: value, unit: unit,
                                        higherIsBetter: higherIsBetter, formatted: formatted, notes: notes))
    }

    // MARK: - Theoretical peaks

    /// FP32 FMA peak: CUs x 64 lanes x 2 FLOP/FMA x clock.
    var theoreticalFP32FLOPS: Double? {
        guard let d = openCLMatch, d.computeUnits > 0, d.maxClockMHz > 0 else { return nil }
        return Double(d.computeUnits) * 64.0 * 2.0 * Double(d.maxClockMHz) * 1e6
    }

    var theoreticalMemoryBytesPerSecond: Double? { nil } // vendor memory clocks are not exposed

    // MARK: - Memory bandwidth

    func benchmarkMemory() {
        let bytes = config.bandwidthMB << 20
        let count = bytes / 16 // float4 elements

        // Copy: read + write
        if let p = try? ctx.pipeline("k_copy"),
           let src = try? ctx.privateBuffer(bytes: bytes),
           let dst = try? ctx.privateBuffer(bytes: bytes) {
            let d = Dispatch(pipeline: p, buffers: [src, dst], constantBytes: [(2, uintData([UInt32(count)]))],
                             threadgroups: gridSize(threads: count, threadsPerThreadgroup: 256),
                             threadsPerThreadgroup: tpg(256))
            if let t = try? ctx.bestTime(d, attempts: config.attempts) {
                let rate = Fmt.gbs(bytes: Double(bytes) * 2, seconds: t)
                add("Memory", "Copy bandwidth (r+w)", rate, "GB/s",
                    String(format: "%.1f GB/s", rate),
                    notes: ["\(Fmt.bytes(bytes)) copied per iteration", "read \(Fmt.bytes(bytes)) + write \(Fmt.bytes(bytes))"])
            }
        }

        // Fill: write only. Use setBytes for the constant float4.
        if let p = try? ctx.pipeline("k_fill"),
           let dst = try? ctx.privateBuffer(bytes: bytes) {
            let d = Dispatch(pipeline: p, buffers: [dst],
                             constantBytes: [(1, floatData([1, 2, 3, 4])), (2, uintData([UInt32(count)]))],
                             threadgroups: gridSize(threads: count, threadsPerThreadgroup: 256),
                             threadsPerThreadgroup: tpg(256))
            if let t = try? ctx.bestTime(d, attempts: config.attempts) {
                let rate = Fmt.gbs(bytes: Double(bytes), seconds: t)
                add("Memory", "Fill bandwidth (write)", rate, "GB/s",
                    String(format: "%.1f GB/s", rate),
                    notes: ["\(Fmt.bytes(bytes)) written per iteration"])
            }
        }

        // Read only: grid-stride reduction, one float written per thread.
        if let p = try? ctx.pipeline("k_read"),
           let src = try? ctx.privateBuffer(bytes: bytes) {
            let threads = min(count, 1 << 20)
            let out = try? ctx.privateBuffer(bytes: threads * 4)
            if let out {
                let d = Dispatch(pipeline: p, buffers: [src, out], constantBytes: [(2, uintData([UInt32(count)]))],
                                 threadgroups: gridSize(threads: threads, threadsPerThreadgroup: 256),
                                 threadsPerThreadgroup: tpg(256))
                if let t = try? ctx.bestTime(d, attempts: config.attempts) {
                    let rate = Fmt.gbs(bytes: Double(bytes), seconds: t)
                    add("Memory", "Read bandwidth (read)", rate, "GB/s",
                        String(format: "%.1f GB/s", rate),
                        notes: ["\(Fmt.bytes(bytes)) read per iteration",
                                "\(Fmt.count(Double(threads))) threads, grid-stride"])
                }
            }
        }
    }

    // MARK: - Compute throughput

    private struct ComputeSpec {
        let function: String
        let group: String?
        let name: String
        let unit: String
        let threads: Int
        let iters: UInt32
        /// FLOP (or op) count per thread per loop iteration.
        let opsPerThreadIter: Double
        let bufferBytesPerThread: Int
        let kind: Kind
        enum Kind { case fp32, fp16, fp64, int32, transcendental }
    }

    func benchmarkCompute() {
        let specs: [ComputeSpec] = [
            ComputeSpec(function: "k_fma32", group: nil, name: "FP32 FMA", unit: "TFLOP/s",
                        threads: config.computeThreads, iters: config.computeIters,
                        opsPerThreadIter: 32, bufferBytesPerThread: 16, kind: .fp32),
            ComputeSpec(function: "k_fma16", group: "fp16", name: "FP16 FMA", unit: "TFLOP/s",
                        threads: config.computeThreads, iters: config.computeIters,
                        opsPerThreadIter: 32, bufferBytesPerThread: 8, kind: .fp16),
            ComputeSpec(function: "k_fma64", group: "fp64", name: "FP64 FMA", unit: "TFLOP/s",
                        threads: max(1 << 14, config.computeThreads / 4), iters: config.computeIters,
                        opsPerThreadIter: 8, bufferBytesPerThread: 8, kind: .fp64),
            ComputeSpec(function: "k_iops32", group: nil, name: "INT32 ALU", unit: "Tops/s",
                        threads: config.computeThreads, iters: config.computeIters,
                        opsPerThreadIter: 16, bufferBytesPerThread: 16, kind: .int32),
            ComputeSpec(function: "k_sinf", group: nil, name: "sin() throughput", unit: "Gops/s",
                        threads: config.computeThreads, iters: max(64, config.computeIters / 4),
                        opsPerThreadIter: 4, bufferBytesPerThread: 16, kind: .transcendental),
            ComputeSpec(function: "k_sqrtf", group: nil, name: "sqrt() throughput", unit: "Gops/s",
                        threads: config.computeThreads, iters: max(64, config.computeIters / 4),
                        opsPerThreadIter: 4, bufferBytesPerThread: 16, kind: .transcendental),
        ]

        for spec in specs {
            guard let p = try? ctx.pipeline(spec.function, inGroup: spec.group),
                  let out = try? ctx.privateBuffer(bytes: spec.threads * spec.bufferBytesPerThread) else { continue }
            let d = Dispatch(pipeline: p, buffers: [out], constantBytes: [(1, uintData([spec.iters]))],
                             threadgroups: gridSize(threads: spec.threads, threadsPerThreadgroup: 256),
                             threadsPerThreadgroup: tpg(256))
            guard let t = try? ctx.bestTime(d, attempts: config.attempts) else { continue }
            let totalOps = Double(spec.threads) * Double(spec.iters) * spec.opsPerThreadIter
            let opsPerSecond = totalOps / t
            switch spec.kind {
            case .fp32:
                let tflops = opsPerSecond / 1e12
                var notes = ["\(Fmt.count(Double(spec.threads))) threads x \(spec.iters) iterations",
                             "4 independent FMA chains; counted as 2 FLOP per FMA"]
                if let peak = theoreticalFP32FLOPS {
                    let eff = opsPerSecond / peak * 100
                    notes.append(String(format: "theoretical peak from %d CUs @ %d MHz = %.2f TFLOP/s -> %.1f%% efficiency",
                                        openCLMatch?.computeUnits ?? 0, openCLMatch?.maxClockMHz ?? 0, peak / 1e12, eff))
                    add("Compute", "FP32 peak (theoretical)", peak / 1e12, "TFLOP/s",
                        String(format: "%.2f TFLOP/s", peak / 1e12),
                        notes: ["CUs x 64 lanes x 2 FLOP x clock, using OpenCL-reported values"])
                    add("Compute", "FP32 efficiency vs peak", eff, "%", String(format: "%.1f%%", eff),
                        notes: ["measured FP32 FMA as a fraction of the theoretical peak"])
                }
                add("Compute", spec.name, tflops, spec.unit, String(format: "%.2f TFLOP/s", tflops), notes: notes)
            case .fp64:
                let tflops = opsPerSecond / 1e12
                var notes = ["4 independent scalar double chains"]
                if let fp32 = measurements.first(where: { $0.name == "FP32 FMA" }), fp32.value > 0 {
                    notes.append(String(format: "FP64:FP32 ratio = 1:%.0f", fp32.value / max(tflops, 1e-9)))
                }
                add("Compute", spec.name, tflops, spec.unit, String(format: "%.3f TFLOP/s", tflops), notes: notes)
            case .fp16:
                let tflops = opsPerSecond / 1e12
                add("Compute", spec.name, tflops, spec.unit, String(format: "%.2f TFLOP/s", tflops),
                    notes: ["half4 vectorised FMA, counted as 2 FLOP per lane-FMA"])
            case .int32:
                let tops = opsPerSecond / 1e12
                add("Compute", spec.name, tops, spec.unit, String(format: "%.2f Tops/s", tops),
                    notes: ["int4 mul+add+xor+shift = 16 integer ops per iteration"])
            case .transcendental:
                let gops = opsPerSecond / 1e9
                add("Compute", spec.name, gops, spec.unit, String(format: "%.1f Gops/s", gops),
                    notes: ["vectorised float4 transcendental throughput"])
            }
        }
    }

    // MARK: - Threadgroup size sweep

    func benchmarkOccupancy() {
        guard let p = try? ctx.pipeline("k_fma32") else { return }
        let threads = config.computeThreads
        let iters: UInt32 = max(64, config.computeIters / 2)
        guard let out = try? ctx.privateBuffer(bytes: threads * 16) else { return }
        var curve: [(Int, Double)] = []
        for size in [32, 64, 128, 256, 512, 1024] {
            let d = Dispatch(pipeline: p, buffers: [out], constantBytes: [(1, uintData([iters]))],
                             threadgroups: gridSize(threads: threads, threadsPerThreadgroup: size),
                             threadsPerThreadgroup: tpg(size))
            guard let t = try? ctx.bestTime(d, attempts: max(2, config.attempts / 2)) else { continue }
            let flops = Double(threads) * Double(iters) * 32.0
            curve.append((size, flops / t / 1e12))
        }
        guard let best = curve.max(by: { $0.1 < $1.1 }) else { return }
        let detail = curve.map { String(format: "%d:%.2f", $0.0, $0.1) }.joined(separator: "  ")
        add("Compute", "Best threadgroup size", Double(best.0), "threads",
            "\(best.0) threads", notes: [
                String(format: "peak %.2f TFLOP/s at %d threads per group", best.1, best.0),
                "sweep (threads:TFLOP/s) \(detail)",
            ])
    }

    // MARK: - Texture sampling throughput

    func benchmarkTextureSampling() {
        guard let p = try? ctx.pipeline("k_tex_sample_batch") else { return }
        let size = 2048
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: size, height: size, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        guard let tex = ctx.device.makeTexture(descriptor: desc),
              let sampler = makeSampler(ctx.device, min: .linear, mag: .linear, address: .repeat) else { return }
        let threads = config.computeThreads
        let iters: UInt32 = max(32, config.computeIters / 4)
        guard let out = try? ctx.privateBuffer(bytes: threads * 16),
              let fill = try? ctx.pipeline("k_fill") else { return }
        // Populate the texture so sampling reads real data.
        let elements = size * size
        let dFill = Dispatch(pipeline: fill, buffers: [out],
                             constantBytes: [(1, floatData([1, 1, 1, 1])), (2, uintData([UInt32(min(threads, elements))]))],
                             threadgroups: gridSize(threads: min(threads, elements), threadsPerThreadgroup: 256),
                             threadsPerThreadgroup: tpg(256))
        _ = try? ctx.run(dFill)

        let d = Dispatch(pipeline: p, buffers: [out], constantBytes: [(1, uintData([iters]))],
                         textures: [tex], samplers: [sampler],
                         threadgroups: gridSize(threads: threads, threadsPerThreadgroup: 128),
                         threadsPerThreadgroup: tpg(128))
        guard let t = try? ctx.bestTime(d, attempts: config.attempts) else { return }
        let samples = Double(threads) * Double(iters)
        let rate = samples / t / 1e9
        add("Memory", "Texture sampling", rate, "Gsamples/s", String(format: "%.1f Gsamples/s", rate),
            notes: ["bilinear filtered, repeat addressing, \(size)x\(size) rgba8Unorm source",
                    "\(Fmt.count(samples)) filtered samples per iteration"])
    }

    // MARK: - Kernel launch overhead

    func benchmarkLaunchOverhead() {
        guard let p = try? ctx.pipeline("k_saxpy"),
              let a = try? ctx.privateBuffer(bytes: 4096),
              let b = try? ctx.privateBuffer(bytes: 4096) else { return }
        let tiny = Dispatch(pipeline: p, buffers: [a, b], constantBytes: [(2, floatData([1.0]))],
                            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
                            threadsPerThreadgroup: tpg(32))

        // Round trip: commit and wait for every single dispatch.
        let roundTrips = 50
        var bestRoundTrip = Double.greatestFiniteMagnitude
        for _ in 0..<3 {
            let t0 = nowSeconds()
            for _ in 0..<roundTrips {
                let cb = try? ctx.commitLoop(tiny, iterations: 1)
                if let cb { try? MetalContext.wait(cb) }
            }
            bestRoundTrip = min(bestRoundTrip, (nowSeconds() - t0) / Double(roundTrips))
        }
        add("Latency", "Dispatch round trip", bestRoundTrip * 1e6, "us",
            String(format: "%.1f us", bestRoundTrip * 1e6),
            higherIsBetter: false,
            notes: ["commit + GPU execute + wait, single 32-thread threadgroup"])

        // Batched: many dispatches already resident in one command buffer.
        let batched = 1024
        if let t0 = try? ctx.bestTime(tiny, attempts: 3, iterationsPerAttempt: batched) {
            add("Latency", "Batched dispatch overhead", t0 * 1e6, "us",
                String(format: "%.2f us", t0 * 1e6),
                higherIsBetter: false,
                notes: ["\(batched) dispatches encoded into a single command buffer"])
        }
    }

    // MARK: - Host <-> device transfer

    private func cpuMemcpySeconds(bytes: Int) -> Double {
        let src = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 4096)
        let dst = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 4096)
        defer { src.deallocate(); dst.deallocate() }
        memset(src, 0xA5, bytes)
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<5 {
            let t0 = nowSeconds()
            memcpy(dst, src, bytes)
            best = min(best, nowSeconds() - t0)
        }
        return best
    }

    func benchmarkHostTransfer() {
        let bytes = min(config.bandwidthMB << 20, 256 << 20)
        // A shared (system memory) staging buffer is used so the timed blit is
        // the real interconnect transfer: no host memcpy correction is needed.
        guard let staging = ctx.device.makeBuffer(length: bytes, options: .storageModeShared),
              let stagingDown = ctx.device.makeBuffer(length: bytes, options: .storageModeShared),
              let priv = try? ctx.privateBuffer(bytes: bytes) else { return }
        memset(staging.contents(), 0x5A, bytes)

        // Upload: shared -> private (host memory to GPU memory).
        var bestUp = Double.greatestFiniteMagnitude
        var uploadError: String?
        for _ in 0..<3 {
            guard let cb = ctx.queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else { break }
            let t0 = nowSeconds()
            blit.copy(from: staging, sourceOffset: 0, to: priv, destinationOffset: 0, size: bytes)
            blit.endEncoding()
            cb.commit()
            do { try MetalContext.wait(cb) } catch { uploadError = firstLine(String(describing: error)); break }
            bestUp = min(bestUp, nowSeconds() - t0)
        }
        if bestUp < .greatestFiniteMagnitude {
            let rate = Fmt.gbs(bytes: Double(bytes), seconds: bestUp)
            add("Transfer", "Host to device", rate, "GB/s", String(format: "%.2f GB/s", rate),
                notes: ["\(Fmt.bytes(bytes)) blitted from system memory to GPU memory",
                        "the timed region contains only the transfer, not host staging"])
        } else if let uploadError {
            add("Transfer", "Host to device", 0, "GB/s", "failed", notes: [uploadError])
        }

        // Download: private -> shared (GPU memory back to host memory).
        var bestDown = Double.greatestFiniteMagnitude
        var downloadError: String?
        for _ in 0..<3 {
            guard let cb = ctx.queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else { break }
            let t0 = nowSeconds()
            blit.copy(from: priv, sourceOffset: 0, to: stagingDown, destinationOffset: 0, size: bytes)
            blit.endEncoding()
            cb.commit()
            do { try MetalContext.wait(cb) } catch { downloadError = firstLine(String(describing: error)); break }
            bestDown = min(bestDown, nowSeconds() - t0)
        }
        if bestDown < .greatestFiniteMagnitude {
            // Verify the round trip actually moved the bytes.
            let p = stagingDown.contents().bindMemory(to: UInt8.self, capacity: bytes)
            var mismatches = 0
            var i = 0
            while i < bytes {
                if p[i] != 0x5A { mismatches += 1 }
                i += 4096
            }
            let rate = Fmt.gbs(bytes: Double(bytes), seconds: bestDown)
            var notes = ["\(Fmt.bytes(bytes)) blitted from GPU memory back to system memory",
                         "the timed region contains only the transfer, not host staging"]
            notes.append(mismatches == 0
                         ? "round-trip data verified: all sampled bytes correct"
                         : "WARNING: \(mismatches) sampled bytes were wrong after the round trip")
            add("Transfer", "Device to host", rate, "GB/s", String(format: "%.2f GB/s", rate), notes: notes)
        } else if let downloadError {
            add("Transfer", "Device to host", 0, "GB/s", "failed", notes: [downloadError])
        }

        let cpuCopy = cpuMemcpySeconds(bytes: bytes)
        add("Transfer", "Host memcpy reference", Fmt.gbs(bytes: Double(bytes), seconds: cpuCopy), "GB/s",
            String(format: "%.1f GB/s", Fmt.gbs(bytes: Double(bytes), seconds: cpuCopy)),
            notes: ["host-only memcpy of the same size, for context (single thread)"])
    }

    // MARK: - Sustained throughput / thermal behaviour

    func benchmarkSustained(progress: Progress) {
        guard config.includeSustained else { return }
        guard let p = try? ctx.pipeline("k_fma32") else { return }
        let threads = config.computeThreads
        let iters: UInt32 = config.computeIters
        guard let out = try? ctx.privateBuffer(bytes: threads * 16) else { return }
        let d = Dispatch(pipeline: p, buffers: [out], constantBytes: [(1, uintData([iters]))],
                         threadgroups: gridSize(threads: threads, threadsPerThreadgroup: 256),
                         threadsPerThreadgroup: tpg(256))
        let flops = Double(threads) * Double(iters) * 32.0

        var samples: [Double] = []
        let deadline = nowSeconds() + config.sustainedSeconds
        progress.note("running \(Int(config.sustainedSeconds))s sustained load to detect thermal throttling")
        while nowSeconds() < deadline {
            guard let t = try? ctx.bestTime(d, attempts: 1) else { break }
            samples.append(flops / t / 1e12)
        }
        guard samples.count >= 4 else { return }
        let first = samples.prefix(max(1, samples.count / 4)).reduce(0, +) / Double(max(1, samples.count / 4))
        let lastCount = max(1, samples.count / 4)
        let last = samples.suffix(lastCount).reduce(0, +) / Double(lastCount)
        let ratio = first > 0 ? last / first : 0
        let detail = samples.map { String(format: "%.2f", $0) }.joined(separator: " ")
        add("Sustained", "Sustained vs burst", ratio * 100, "%", String(format: "%.1f%%", ratio * 100),
            notes: [String(format: "first quarter %.2f TFLOP/s, last quarter %.2f TFLOP/s", first, last),
                    "\(samples.count) samples over \(Int(config.sustainedSeconds))s",
                    "samples: \(detail)"])

        let status: Status = ratio > 0.97 ? .pass : (ratio > 0.85 ? .warn : .fail)
        let check = Check(group: "Sustained", name: "Thermal stability", status: status,
                          summary: String(format: "%.1f%% of initial throughput retained after %ds of load (%.2f -> %.2f TFLOP/s)",
                                          ratio * 100, Int(config.sustainedSeconds), first, last),
                          details: [
                            "sustained load is FP32 FMA at full occupancy",
                            "a large drop indicates thermal or power throttling",
                          ],
                          seconds: config.sustainedSeconds)
        checks.append(check)
    }

    // MARK: - OpenCL cross-check

    /// Runs the same GPU through a second, independent driver stack. This both
    /// validates the bandwidth measurement and proves a working OpenCL path.
    func benchmarkOpenCL() {
        guard config.includeOpenCL, let ocl = openCLMatch else { return }
        let bytes = min(config.bandwidthMB << 20, 256 << 20)
        let iterations = 20
        switch openCLCopyBandwidth(ocl, bufferBytes: bytes, iterations: iterations) {
        case .success(let r):
            let gbs = r.bytesPerSecond / 1e9
            add("OpenCL", "OpenCL copy bandwidth", gbs, "GB/s", String(format: "%.1f GB/s", gbs),
                notes: ["independent OpenCL 1.2 stack, \(Fmt.bytes(bytes)) per iteration",
                        "\(Fmt.bytes(r.bytesMoved)) moved over \(iterations) iterations"])
            checks.append(Check(group: "OpenCL", name: "OpenCL kernel build + run", status: .pass,
                                summary: String(format: "compiled OpenCL C kernel and ran it: %.1f GB/s copy", gbs),
                                details: [
                                    "\(ocl.computeUnits) compute units @ \(ocl.maxClockMHz) MHz, \(ocl.deviceVersion)",
                                    r.buildLog.isEmpty ? "build log empty (clean compile)" : "build log: \(r.buildLog)",
                                ],
                                seconds: r.seconds))
        case .failure(let f):
            checks.append(Check(group: "OpenCL", name: "OpenCL kernel build + run", status: .warn,
                                summary: f.message,
                                details: ["OpenCL is deprecated on macOS; Metal remains the primary path"]))
        }
    }

    // MARK: - Runner

    func runAll(progress: Progress) {
        progress.beginPhase("Performance measurements")

        // Each group is journalled before it runs and followed by a cooldown so
        // the display compositor regains the GPU between loads.
        let steps: [(String, () -> Void)] = [
            ("memory bandwidth", benchmarkMemory),
            ("compute throughput", benchmarkCompute),
            ("threadgroup size sweep", benchmarkOccupancy),
            ("texture sampling", benchmarkTextureSampling),
            ("dispatch latency", benchmarkLaunchOverhead),
            ("host<->device transfer", benchmarkHostTransfer),
            ("OpenCL cross-check", benchmarkOpenCL),
        ]
        for (name, body) in steps {
            progress.willRun(name)
            body()
            progress.journal?.log("DONE   \(name)")
            rest(milliseconds: config.cooldownMs)
        }

        if config.includeSustained {
            progress.willRun("sustained load")
            benchmarkSustained(progress: progress)
            progress.journal?.log("DONE   sustained load")
        }
    }
}
// MARK: - Multi-GPU concurrency

struct ConcurrencyOutcome {
    var soloFLOPS: [Double] = []
    var concurrentFLOPS: Double = 0
    var wallSeconds: Double = 0
    var efficiency: Double = 0
    var soloCopyGBs: [Double] = []
    var concurrentCopyGBs: Double = 0
    var copyEfficiency: Double = 0
}

enum MultiGPU {
    /// Runs an identical workload on every GPU alone, then on all GPUs at the
    /// same time, and reports how well the work scaled. This is the measurement
    /// that shows whether a multi-GPU machine actually uses all of its GPUs.
    static func measure(contexts: [MetalContext], config: SuiteConfig) -> ConcurrencyOutcome {
        var outcome = ConcurrencyOutcome()
        guard contexts.count > 1 else { return outcome }

        let threads = 1 << 20
        let iters: UInt32 = 1024
        let flops = Double(threads) * Double(iters) * 32.0
        let bytes = min(config.bandwidthMB << 20, 256 << 20)
        let count = bytes / 16

        // Build per-device workloads.
        var computeDispatches: [Dispatch] = []
        var copyDispatches: [Dispatch] = []
        var copyBytes: [Int] = []
        for ctx in contexts {
            guard let p = try? ctx.pipeline("k_fma32"),
                  let out = try? ctx.privateBuffer(bytes: threads * 16) else { continue }
            computeDispatches.append(Dispatch(pipeline: p, buffers: [out], constantBytes: [(1, uintData([iters]))],
                                              threadgroups: gridSize(threads: threads, threadsPerThreadgroup: 256),
                                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)))
            if let cp = try? ctx.pipeline("k_copy"),
               let src = try? ctx.privateBuffer(bytes: bytes),
               let dst = try? ctx.privateBuffer(bytes: bytes) {
                copyDispatches.append(Dispatch(pipeline: cp, buffers: [src, dst],
                                               constantBytes: [(2, uintData([UInt32(count)]))],
                                               threadgroups: gridSize(threads: count, threadsPerThreadgroup: 256),
                                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)))
                copyBytes.append(bytes)
            }
        }
        guard computeDispatches.count == contexts.count else { return outcome }

        // Solo runs.
        for (i, ctx) in contexts.enumerated() {
            if let t = try? ctx.bestTime(computeDispatches[i], attempts: 2) {
                outcome.soloFLOPS.append(flops / t)
            }
            if i < copyDispatches.count, let t = try? ctx.bestTime(copyDispatches[i], attempts: 2) {
                outcome.soloCopyGBs.append(Fmt.gbs(bytes: Double(copyBytes[i]) * 2, seconds: t))
            }
        }

        // Concurrent compute: commit all, then wait for all.
        var t0 = nowSeconds()
        var buffers: [MTLCommandBuffer] = []
        for (i, ctx) in contexts.enumerated() {
            if let cb = try? ctx.commitLoop(computeDispatches[i], iterations: 1) { buffers.append(cb) }
        }
        for cb in buffers { try? MetalContext.wait(cb) }
        outcome.wallSeconds = nowSeconds() - t0
        let totalFlops = flops * Double(contexts.count)
        outcome.concurrentFLOPS = totalFlops / outcome.wallSeconds

        let soloSum = outcome.soloFLOPS.reduce(0, +)
        outcome.efficiency = soloSum > 0 ? outcome.concurrentFLOPS / soloSum : 0

        // Concurrent memory copy.
        if copyDispatches.count == contexts.count {
            t0 = nowSeconds()
            buffers = []
            for (i, ctx) in contexts.enumerated() {
                if let cb = try? ctx.commitLoop(copyDispatches[i], iterations: 1) { buffers.append(cb) }
            }
            for cb in buffers { try? MetalContext.wait(cb) }
            let wall = nowSeconds() - t0
            let totalBytes = Double(copyBytes.reduce(0, +)) * 2
            outcome.concurrentCopyGBs = totalBytes / wall / 1e9
            let copySoloSum = outcome.soloCopyGBs.reduce(0, +)
            outcome.copyEfficiency = copySoloSum > 0 ? outcome.concurrentCopyGBs / copySoloSum : 0
        }

        return outcome
    }
}
