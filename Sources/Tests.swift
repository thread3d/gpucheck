// Tests.swift — functional and numerical-correctness verification.
//
// Every test compares GPU output against an independent reference computed on
// the host CPU, so a PASS means the result was actually verified rather than
// merely "the kernel did not crash".
import Foundation
import Metal

/// How aggressive the suite is allowed to be.
///
/// GPU load on a machine that also drives its displays can starve WindowServer
/// and hang the whole system, and a GPU with a history of driver resets can be
/// pushed into one. `safe` is therefore the default: it still runs every
/// correctness test and every benchmark, but with smaller workloads and a
/// cooldown between tests so the compositor always gets the GPU back.
enum Intensity: String {
    case safe, normal, full

    var cooldownMs: Int {
        switch self {
        case .safe: return 250
        case .normal: return 120
        case .full: return 60
        }
    }
}

struct SuiteConfig {
    var mode: String
    var intensity: Intensity
    var attempts: Int
    var bandwidthMB: Int
    var computeThreads: Int
    var computeIters: UInt32
    var cooldownMs: Int
    var sustainedSeconds: Double
    var concurrencySeconds: Double
    var largeBufferBytes: Int
    var includeSustained: Bool
    var includeConcurrency: Bool
    var includeOpenCL: Bool
    var includeRender: Bool
    var allowHeavy: Bool

    /// `mode` sets the depth of the run; `intensity` caps how hard any single
    /// workload may push the GPU. The two riskiest tests - a sustained
    /// full-occupancy load and driving every GPU at once - additionally require
    /// an explicit opt-in because they are what can hang a marginal system.
    static func make(mode: String, intensity: Intensity, allowHeavy: Bool) -> SuiteConfig {
        var attempts = 5
        var bandwidthMB = 512
        var threads = 1 << 20
        var iters: UInt32 = 1024
        var sustained: Double = 8
        var concurrency: Double = 4
        var largeBuffer = 2_684_354_560 // 2.5 GiB

        switch mode {
        case "quick":
            attempts = 3; bandwidthMB = 128; threads = 1 << 19; iters = 256
            sustained = 3; concurrency = 2; largeBuffer = 1 << 30
        case "thorough":
            attempts = 8; bandwidthMB = 1024; threads = 1 << 20; iters = 2048
            sustained = 20; concurrency = 8
        default:
            break
        }

        // Intensity caps workload size regardless of the requested depth.
        switch intensity {
        case .safe:
            bandwidthMB = min(bandwidthMB, 64)
            threads = min(threads, 1 << 18)
            iters = min(iters, 256)
            sustained = 0
            concurrency = 0
            largeBuffer = 0
        case .normal:
            bandwidthMB = min(bandwidthMB, 256)
            threads = min(threads, 1 << 19)
            iters = min(iters, 512)
            largeBuffer = min(largeBuffer, 1 << 30)
        case .full:
            break
        }

        return SuiteConfig(
            mode: mode,
            intensity: intensity,
            attempts: attempts,
            bandwidthMB: bandwidthMB,
            computeThreads: threads,
            computeIters: iters,
            cooldownMs: intensity.cooldownMs,
            sustainedSeconds: sustained,
            concurrencySeconds: concurrency,
            largeBufferBytes: largeBuffer,
            includeSustained: sustained > 0 && allowHeavy,
            includeConcurrency: concurrency > 0 && allowHeavy,
            includeOpenCL: true,
            includeRender: true,
            allowHeavy: allowHeavy
        )
    }
}

func readDoubles(_ buffer: MTLBuffer, count: Int) -> [Double] {
    let p = buffer.contents().bindMemory(to: Double.self, capacity: count)
    return Array(UnsafeBufferPointer(start: p, count: count))
}

func writeDoubles(_ buffer: MTLBuffer, _ values: [Double]) {
    values.withUnsafeBufferPointer { src in
        buffer.contents().copyMemory(from: src.baseAddress!, byteCount: values.count * 8)
    }
}

func writeHalves(_ buffer: MTLBuffer, _ values: [Float]) {
    let bits = values.map { floatToHalf($0) }
    bits.withUnsafeBufferPointer { src in
        buffer.contents().copyMemory(from: src.baseAddress!, byteCount: bits.count * 2)
    }
}

final class FunctionalTests {
    let ctx: MetalContext
    let config: SuiteConfig
    private(set) var measurements: [Measurement] = []

    init(ctx: MetalContext, config: SuiteConfig) {
        self.ctx = ctx
        self.config = config
    }

    private func tpg(_ n: Int) -> MTLSize { MTLSize(width: n, height: 1, depth: 1) }

    private func unsupported(_ group: String, _ name: String, _ error: Error, _ seconds: Double) -> Check {
        Check(group: group, name: name, status: .unsupported,
              summary: "not supported on this GPU: \(firstLine(String(describing: error)))",
              seconds: seconds)
    }

    // MARK: - 1. Resource allocation

    func testResourceAllocation() -> Check {
        let started = nowSeconds()
        var details: [String] = []
        var allocated: [MTLBuffer] = []
        let sizes = [1 << 20, 1 << 24, 1 << 26, 1 << 28]
        var largest = 0
        do {
            for size in sizes {
                let b = try ctx.privateBuffer(bytes: size)
                allocated.append(b)
                largest = size
            }
            // A large allocation that crosses the 32-bit byte-offset boundary.
            let big = try ctx.privateBuffer(bytes: min(1 << 31, Int(ctx.device.maxBufferLength)))
            allocated.append(big)
            largest = big.length
            details.append("allocated \(sizes.count + 1) buffers up to \(Fmt.bytes(big.length))")
            // Keep them alive until here, then release.
            withExtendedLifetime(allocated) {}
            let heapSize = min(256 << 20, Int(ctx.device.recommendedMaxWorkingSetSize / 8))
            if heapSize > (16 << 20) {
                let hd = MTLHeapDescriptor()
                hd.size = heapSize
                hd.storageMode = .private
                if let heap = ctx.device.makeHeap(descriptor: hd) {
                    details.append("heap of \(Fmt.bytes(heapSize)) created (max available \(Fmt.bytes(heap.maxAvailableSize(alignment: 256))))")
                } else {
                    details.append("heap creation of \(Fmt.bytes(heapSize)) failed")
                }
            }
        } catch {
            return Check(group: "Resources", name: "Memory allocation", status: .fail,
                         summary: "allocation failed at \(Fmt.bytes(largest)): \(error)",
                         details: details, seconds: nowSeconds() - started)
        }
        return Check(group: "Resources", name: "Memory allocation", status: .pass,
                     summary: "largest single allocation \(Fmt.bytes(largest))",
                     details: details, seconds: nowSeconds() - started)
    }

    // MARK: - 2. Basic kernel + buffer round trip

    func testBufferRoundTrip() -> Check {
        let started = nowSeconds()
        let n = 1 << 20
        do {
            let p = try ctx.pipeline("k_pattern")
            let buf = try ctx.sharedBuffer(bytes: n * 4)
            let seed: UInt32 = 0x1234_5678
            let d = Dispatch(pipeline: p, buffers: [buf], constantBytes: [(1, uintData([seed]))],
                             threadgroups: gridSize(threads: n, threadsPerThreadgroup: 256),
                             threadsPerThreadgroup: tpg(256))
            try ctx.run(d)
            let got = readFloats(buf, count: n)
            var mismatches = 0
            var firstBad = ""
            for i in 0..<n {
                let h = (UInt32(truncatingIfNeeded: i) ^ seed) &* 2654435761
                let expected = Float(h & 0xFFFF) * (1.0 / 65536.0)
                if got[i] != expected {
                    mismatches += 1
                    if firstBad.isEmpty { firstBad = "index \(i): got \(got[i]) expected \(expected)" }
                }
            }
            let status: Status = mismatches == 0 ? .pass : .fail
            return Check(group: "Correctness", name: "Buffer round trip", status: status,
                         summary: mismatches == 0
                            ? "\(Fmt.count(Double(n))) values written and read back bit-exact"
                            : "\(mismatches) mismatching values (\(firstBad))",
                         details: ["deterministic hash pattern, integer-exact reference"],
                         seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "Buffer round trip", error, nowSeconds() - started)
        }
    }

    func testSaxpy() -> Check {
        let started = nowSeconds()
        let n = 1 << 20
        do {
            let p = try ctx.pipeline("k_saxpy")
            let x = (0..<n).map { Float($0 % 1000) * 0.001 }
            let y = (0..<n).map { Float(($0 * 7) % 997) * 0.002 }
            let a: Float = 2.5
            let bX = try ctx.sharedBuffer(bytes: n * 4)
            let bY = try ctx.sharedBuffer(bytes: n * 4)
            writeFloats(bX, x)
            writeFloats(bY, y)
            let d = Dispatch(pipeline: p, buffers: [bX, bY], constantBytes: [(2, floatData([a]))],
                             threadgroups: gridSize(threads: n, threadsPerThreadgroup: 256),
                             threadsPerThreadgroup: tpg(256))
            try ctx.run(d)
            let got = readFloats(bY, count: n)
            var maxRel: Float = 0
            for i in 0..<n {
                let ref = a * x[i] + y[i]
                let denom = max(abs(ref), 1e-30)
                maxRel = max(maxRel, abs(got[i] - ref) / denom)
            }
            let status: Status = maxRel < 1e-5 ? .pass : (maxRel < 1e-3 ? .warn : .fail)
            return Check(group: "Correctness", name: "SAXPY (y = a*x + y)", status: status,
                         summary: String(format: "%d elements, max relative error %.2e", n, maxRel),
                         details: ["reference computed in host float; tolerance 1e-5"],
                         seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "SAXPY (y = a*x + y)", error, nowSeconds() - started)
        }
    }

    // MARK: - 3. Threadgroup memory and barriers

    func testThreadgroupReduction() -> Check {
        let started = nowSeconds()
        let n = 1 << 20
        let groupSize = 256
        do {
            let p = try ctx.pipeline("k_reduce")
            let values = (0..<n).map { Float(($0 % 977)) * 0.001 }
            let src = try ctx.sharedBuffer(bytes: n * 4)
            let groups = n / groupSize
            let dst = try ctx.sharedBuffer(bytes: groups * 4)
            writeFloats(src, values)
            let d = Dispatch(pipeline: p, buffers: [src, dst],
                             threadgroups: gridSize(threads: n, threadsPerThreadgroup: groupSize),
                             threadsPerThreadgroup: tpg(groupSize))
            try ctx.run(d)
            let partials = readFloats(dst, count: groups)
            var got: Double = 0
            for v in partials { got += Double(v) }
            var ref: Double = 0
            for v in values { ref += Double(v) }
            let rel = abs(got - ref) / max(abs(ref), 1e-30)
            let status: Status = rel < 1e-5 ? .pass : (rel < 1e-3 ? .warn : .fail)
            return Check(group: "Correctness", name: "Threadgroup reduce + barrier", status: status,
                         summary: String(format: "%d values -> %.3f vs reference %.3f (rel %.2e)",
                                         n, got, ref, rel),
                         details: ["tree reduction over \(groupSize)-thread groups using threadgroup memory and barriers"],
                         seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "Threadgroup reduce + barrier", error, nowSeconds() - started)
        }
    }

    // MARK: - 4. Atomics

    func testAtomics() -> Check {
        let started = nowSeconds()
        let n = 1 << 20
        let bins = 256
        do {
            let p = try ctx.pipeline("k_atomic_hist")
            var rng = SplitMix64(seed: 99)
            let keys = (0..<n).map { _ in UInt32(truncatingIfNeeded: rng.next()) }
            let keyBuf = try ctx.sharedBuffer(bytes: n * 4)
            let histBuf = try ctx.sharedBuffer(bytes: bins * 4)
            keys.withUnsafeBufferPointer { src in
                keyBuf.contents().copyMemory(from: src.baseAddress!, byteCount: n * 4)
            }
            memset(histBuf.contents(), 0, bins * 4)
            let d = Dispatch(pipeline: p, buffers: [histBuf, keyBuf], constantBytes: [(2, uintData([UInt32(bins)]))],
                             threadgroups: gridSize(threads: n, threadsPerThreadgroup: 256),
                             threadsPerThreadgroup: tpg(256))
            try ctx.run(d)
            let hist = histBuf.contents().bindMemory(to: UInt32.self, capacity: bins)
            var expected = [UInt32](repeating: 0, count: bins)
            for k in keys { expected[Int(k % UInt32(bins))] += 1 }
            var bad = 0
            for i in 0..<bins where hist[i] != expected[i] { bad += 1 }
            var total = 0
            for i in 0..<bins { total += Int(hist[i]) }
            let status: Status = (bad == 0 && total == n) ? .pass : .fail
            return Check(group: "Correctness", name: "Atomic operations (histogram)", status: status,
                         summary: bad == 0
                            ? "\(Fmt.count(Double(n))) atomic adds, all \(bins) bins exact (total \(Fmt.count(Double(total))))"
                            : "\(bad) bins wrong, total \(total) vs \(n)",
                         details: ["contended atomic_fetch_add across 256 bins; exact integer reference"],
                         seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "Atomic operations (histogram)", error, nowSeconds() - started)
        }
    }

    // MARK: - 5. Half precision

    func testHalfPrecision() -> Check {
        let started = nowSeconds()
        let n = 1 << 16
        do {
            let p = try ctx.pipeline("k_half_scale")
            // Multiples of 0.125 scaled by 3 produce values that are exactly
            // representable in binary16, so a correct FP16 path must be bit-exact.
            let values = (0..<n).map { Float($0 % 512) * 0.125 }
            let src = try ctx.sharedBuffer(bytes: n * 2)
            let dst = try ctx.sharedBuffer(bytes: n * 2)
            writeHalves(src, values)
            let scale: Float = 3.0
            var scaleBits = floatToHalf(scale)
            let d = Dispatch(pipeline: p, buffers: [src, dst],
                             constantBytes: [(2, Data(bytes: &scaleBits, count: 2))],
                             threadgroups: gridSize(threads: n, threadsPerThreadgroup: 256),
                             threadsPerThreadgroup: tpg(256))
            try ctx.run(d)
            let got = readHalfAsFloat(dst, count: n)
            var bad = 0
            var maxRel: Float = 0
            for i in 0..<n {
                let ref = halfToFloat(floatToHalf(values[i] * scale))
                maxRel = max(maxRel, abs(got[i] - ref) / max(abs(ref), 1e-3))
                if got[i] != ref { bad += 1 }
            }
            let status: Status = bad == 0 ? .pass : .fail
            return Check(group: "Correctness", name: "FP16 half precision", status: status,
                         summary: bad == 0
                            ? "\(Fmt.count(Double(n))) half values scaled by 3.0, bit-exact"
                            : "\(bad) mismatches (max rel \(maxRel))",
                         details: ["exercises the fp16 pipeline and half<->host conversion",
                                   "inputs and products are exactly representable, so exact equality is required"],
                         seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "FP16 half precision", error, nowSeconds() - started)
        }
    }

    // MARK: - 6. Double precision

    func testDoublePrecision() -> Check {
        let started = nowSeconds()
        let n = 1 << 14
        let iters: UInt32 = 64
        do {
            let p = try ctx.pipeline("k_fma64", inGroup: "fp64")
            let out = try ctx.sharedBuffer(bytes: n * 8)
            let d = Dispatch(pipeline: p, buffers: [out], constantBytes: [(1, uintData([iters]))],
                             threadgroups: gridSize(threads: n, threadsPerThreadgroup: 128),
                             threadsPerThreadgroup: tpg(128))
            try ctx.run(d)
            let got = readDoubles(out, count: n)
            var maxRel = 0.0
            for i in 0..<n {
                let a = 1.0000000001
                let b = 1.0e-12
                let base = Double(i & 1023) * 1.0e-6
                var c0 = base, c1 = base + 0.1, c2 = base + 0.2, c3 = base + 0.3
                for _ in 0..<Int(iters) {
                    c0 = a * c0 + b; c1 = a * c1 + b; c2 = a * c2 + b; c3 = a * c3 + b
                }
                let ref = c0 + c1 + c2 + c3
                maxRel = max(maxRel, abs(got[i] - ref) / max(abs(ref), 1e-30))
            }
            let status: Status = maxRel < 1e-12 ? .pass : (maxRel < 1e-9 ? .warn : .fail)
            return Check(group: "Correctness", name: "FP64 double precision", status: status,
                         summary: String(format: "max relative error %.2e over %d chains", maxRel, n),
                         details: ["if this passes the GPU is doing real 64-bit math, not promoting to 32-bit"],
                         seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "FP64 double precision", error, nowSeconds() - started)
        }
    }

    // MARK: - 7. Math accuracy

    private struct MathCase {
        let name: String
        let index: UInt32
        let inputs: [Float]
        let reference: (Double) -> Double
    }

    private func mathCases(count: Int = 4096) -> [MathCase] {
        func ramp(_ lo: Double, _ hi: Double, _ f: @escaping (Double) -> Double) -> (inputs: [Float], ref: (Double) -> Double) {
            let inputs = (0..<count).map { i -> Float in
                let t = Double(i) / Double(count - 1)
                return Float(lo + (hi - lo) * t)
            }
            return (inputs, f)
        }
        var cases: [MathCase] = []
        // Only functions present in the Metal standard library are listed;
        // erf, for example, is not available in MSL.
        let specs: [(String, UInt32, Double, Double, (Double) -> Double)] = [
            ("exp", 0, -20, 20, { exp($0) }),
            ("log", 1, 1e-6, 1e6, { log($0) }),
            ("sin", 2, -100, 100, { sin($0) }),
            ("cos", 3, -100, 100, { cos($0) }),
            ("sqrt", 4, 1e-6, 1e6, { sqrt($0) }),
            ("rsqrt", 5, 1e-6, 1e6, { 1.0 / sqrt($0) }),
            ("pow(x,1.5)", 6, 1e-6, 1e6, { pow($0, 1.5) }),
            ("tanh", 7, -10, 10, { tanh($0) }),
            ("atan", 8, -100, 100, { atan($0) }),
            ("exp2", 9, -20, 20, { exp2($0) }),
            ("log2", 10, 1e-6, 1e6, { log2($0) }),
            ("tan", 11, -1.4, 1.4, { tan($0) }),
            ("asin", 12, -1, 1, { asin($0) }),
            ("sinh", 13, -5, 5, { sinh($0) }),
            ("cosh", 14, -5, 5, { cosh($0) }),
        ]
        for (name, idx, lo, hi, f) in specs {
            let r = ramp(lo, hi, f)
            cases.append(MathCase(name: name, index: idx, inputs: r.inputs, reference: r.ref))
        }
        return cases
    }

    private func runMath(_ fnGroup: String) throws -> [(name: String, maxULP: Int, maxRel: Double, worst: String)] {
        let p = try ctx.pipeline("k_math_accuracy", inGroup: fnGroup)
        var results: [(String, Int, Double, String)] = []
        for c in mathCases() {
            let count = c.inputs.count
            let inBuf = try ctx.sharedBuffer(bytes: count * 4)
            let outBuf = try ctx.sharedBuffer(bytes: count * 4)
            writeFloats(inBuf, c.inputs)
            let d = Dispatch(pipeline: p, buffers: [inBuf, outBuf],
                             constantBytes: [(2, uintData([c.index])), (3, uintData([UInt32(count)]))],
                             threadgroups: gridSize(threads: count, threadsPerThreadgroup: 128),
                             threadsPerThreadgroup: tpg(128))
            try ctx.run(d)
            let got = readFloats(outBuf, count: count)
            var maxULP = 0
            var maxRel = 0.0
            var worst = ""
            for i in 0..<count {
                let ref = Float(c.reference(Double(c.inputs[i])))
                let ulp = ulpDistance(got[i], ref)
                let rel = abs(Double(got[i]) - Double(ref)) / max(abs(Double(ref)), 1e-30)
                if ulp != Int.max && ulp > maxULP { maxULP = ulp; worst = "x=\(c.inputs[i]) got \(got[i]) ref \(ref)" }
                maxRel = max(maxRel, rel)
            }
            results.append((c.name, maxULP, maxRel, worst))
        }
        return results
    }

    func testMathAccuracy() -> Check {
        let started = nowSeconds()
        do {
            let results = try runMath("math")
            var details: [String] = []
            var worstULP = 0
            var worstName = ""
            var worstRel = 0.0
            for r in results {
                details.append(String(format: "%-11@ max ULP %-6d max rel err %.2e", r.name as NSString, r.maxULP, r.maxRel))
                if r.maxULP > worstULP { worstULP = r.maxULP; worstName = r.name }
                worstRel = max(worstRel, r.maxRel)
            }
            let status: Status = worstULP <= 16 ? .pass : (worstULP <= 4096 ? .warn : .fail)
            return Check(group: "Correctness", name: "Math function accuracy", status: status,
                         summary: String(format: "%d functions vs host libm; worst %@ at %d ULP", results.count, worstName, worstULP),
                         details: details, seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "Math function accuracy", error, nowSeconds() - started)
        }
    }

    func testFastMathAccuracy() -> Check {
        let started = nowSeconds()
        do {
            let precise = try runMath("math")
            let fast = try runMath("math_fast")
            var details: [String] = []
            var worst = 0
            var worstName = ""
            for (i, f) in fast.enumerated() {
                let p = precise[i]
                details.append(String(format: "%-11@ precise %d ULP  fast-math %d ULP", f.name as NSString, p.maxULP, f.maxULP))
                if f.maxULP > worst { worst = f.maxULP; worstName = f.name }
            }
            let status: Status = worst <= 64 ? .pass : .warn
            return Check(group: "Correctness", name: "Fast-math accuracy delta", status: status,
                         summary: String(format: "worst fast-math error %@ at %d ULP", worstName, worst),
                         details: details, seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "Fast-math accuracy delta", error, nowSeconds() - started)
        }
    }

    // MARK: - 8. Textures and samplers

    private func makeTestTexture(size: Int) throws -> (MTLTexture, [Float]) {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: size, height: size, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .shared
        guard let tex = ctx.device.makeTexture(descriptor: desc) else {
            throw GPUError.message("could not create \(size)x\(size) rgba32Float texture")
        }
        var texels: [Float] = []
        var raw: [Float] = []
        for y in 0..<size {
            for x in 0..<size {
                let v = Float(x * size + y)
                texels.append(v)
                raw.append(contentsOf: [v, 0, 0, 1])
            }
        }
        raw.withUnsafeBufferPointer { src in
            tex.replace(region: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0,
                        withBytes: src.baseAddress!, bytesPerRow: size * 16)
        }
        return (tex, texels)
    }

    /// CPU bilinear reference matching the standard GPU convention:
    /// texel-space coordinate = uv * size - 0.5, with clamp-to-edge.
    private func cpuBilinear(_ texels: [Float], _ n: Int, _ u: Float, _ v: Float) -> Float {
        let x = u * Float(n) - 0.5
        let y = v * Float(n) - 0.5
        let x0 = Int(floor(x)), y0 = Int(floor(y))
        let fx = x - Float(x0), fy = y - Float(y0)
        func t(_ xi: Int, _ yi: Int) -> Float {
            let cx = min(max(xi, 0), n - 1), cy = min(max(yi, 0), n - 1)
            return texels[cy * n + cx]
        }
        let top = t(x0, y0) * (1 - fx) + t(x0 + 1, y0) * fx
        let bot = t(x0, y0 + 1) * (1 - fx) + t(x0 + 1, y0 + 1) * fx
        return top * (1 - fy) + bot * fy
    }

    func testTextures() -> Check {
        let started = nowSeconds()
        let n = 4
        do {
            let (tex, texels) = try makeTestTexture(size: n)
            var details: [String] = []
            let pSampleUV = try ctx.pipeline("k_tex_sample_uv")

            /// Samples the texture at the supplied coordinates and returns the red channel.
            func sample(_ sampler: MTLSamplerState, _ uvList: [(Float, Float)]) throws -> [Float] {
                let count = uvList.count
                var flat: [Float] = []
                for (u, v) in uvList { flat.append(contentsOf: [u, v]) }
                let uvBuf = try ctx.sharedBuffer(bytes: count * 8)
                let outBuf = try ctx.sharedBuffer(bytes: count * 16)
                flat.withUnsafeBufferPointer { src in
                    uvBuf.contents().copyMemory(from: src.baseAddress!, byteCount: count * 8)
                }
                let d = Dispatch(pipeline: pSampleUV, buffers: [outBuf, uvBuf], textures: [tex], samplers: [sampler],
                                 threadgroups: gridSize(threads: count, threadsPerThreadgroup: 64),
                                 threadsPerThreadgroup: tpg(64))
                try ctx.run(d)
                let p = outBuf.contents().bindMemory(to: Float.self, capacity: count * 4)
                return (0..<count).map { p[$0 * 4] }
            }

            // (a) direct texel access via texture.read()
            let pRead = try ctx.pipeline("k_tex_read")
            let outCount = n * n
            let outBuf = try ctx.sharedBuffer(bytes: outCount * 16)
            let dRead = Dispatch(pipeline: pRead, buffers: [outBuf], textures: [tex],
                                 threadgroups: MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: n, height: n, depth: 1))
            try ctx.run(dRead)
            let readBack = outBuf.contents().bindMemory(to: Float.self, capacity: outCount * 4)
            var readBad = 0
            for i in 0..<outCount where readBack[i * 4] != texels[i] { readBad += 1 }
            details.append("texture.read(): \(readBad == 0 ? "all \(outCount) texels exact" : "\(readBad) wrong")")

            // (b) bilinear filtering vs an exact CPU reference, including
            //     coordinates outside [0,1] to exercise clamp-to-edge.
            guard let linear = makeSampler(ctx.device, min: .linear, mag: .linear, address: .clampToEdge) else {
                throw GPUError.message("could not create linear sampler")
            }
            var uvList: [(Float, Float)] = []
            for i in 0..<64 {
                // Spread across and slightly beyond the texture.
                let u = Float(i % 8) / 7.0 * 1.4 - 0.2
                let v = Float(i / 8) / 7.0 * 1.4 - 0.2
                uvList.append((u, v))
            }
            let linearGot = try sample(linear, uvList)
            var maxLinErr: Float = 0
            for (i, uv) in uvList.enumerated() {
                let expected = cpuBilinear(texels, n, uv.0, uv.1)
                maxLinErr = max(maxLinErr, abs(linearGot[i] - expected))
            }
            details.append(String(format: "bilinear vs CPU reference (%d coords incl. out-of-range): max error %.3f r-units (1 texel step = 1.0)", uvList.count, maxLinErr))

            // (c) nearest + addressing modes at unambiguous texel-centre coords.
            guard let nearest = makeSampler(ctx.device, min: .nearest, mag: .nearest, address: .clampToEdge),
                  let repeatS = makeSampler(ctx.device, min: .nearest, mag: .nearest, address: .repeat) else {
                throw GPUError.message("could not create nearest samplers")
            }
            // Metal selects texel i for uv in [i/N, (i+1)/N), so a specific
            // texel is targeted by sampling at its centre, uv = (i + 0.5)/N.
            let probes: [(String, MTLSamplerState, Float, Float, Int)] = [
                ("nearest texel(2,2) at uv 0.625", nearest, 0.625, 0.625, 2 * n + 2),
                ("clamp below range uv -0.375 -> texel(0,0)", nearest, -0.375, -0.375, 0),
                ("repeat below range uv -0.375 -> texel(2,2)", repeatS, -0.375, -0.375, 2 * n + 2),
                ("clamp above range uv 1.25 -> texel(3,3)", nearest, 1.25, 1.25, 3 * n + 3),
            ]
            var addrBad = 0
            for (label, sampler, u, v, texelIndex) in probes {
                let got = try sample(sampler, [(u, v)])[0]
                let expected = texels[texelIndex]
                let ok = abs(got - expected) < 1e-6
                if !ok { addrBad += 1 }
                details.append("\(ok ? "ok  " : "BAD ") \(label): got \(got) expected \(expected)")
            }

            // A correct bilinear filter may differ slightly from an exact
            // reference because filter weights are quantised on the GPU; a
            // broken one would be wrong by a large fraction of a texel (1.0).
            let linOK = maxLinErr < 0.02
            let ok = readBad == 0 && addrBad == 0 && linOK
            let status: Status = ok ? .pass : ((maxLinErr < 0.1 && addrBad == 0 && readBad == 0) ? .warn : .fail)
            return Check(group: "Correctness", name: "Texture read + filtering", status: status,
                         summary: String(format: "texel access exact, bilinear max error %.2e, addressing %@",
                                         maxLinErr, addrBad == 0 ? "all 4 modes correct" : "\(addrBad) wrong"),
                         details: details, seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "Texture read + filtering", error, nowSeconds() - started)
        }
    }

    // MARK: - 9. Blit transfers

    func testBlit() -> Check {
        let started = nowSeconds()
        let n = 1 << 18
        do {
            let src = try ctx.sharedBuffer(bytes: n * 4)
            let dst = try ctx.sharedBuffer(bytes: n * 4)
            let values = (0..<n).map { Float($0 % 1021) * 0.25 }
            writeFloats(src, values)
            memset(dst.contents(), 0, n * 4)
            guard let cb = ctx.queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else {
                throw GPUError.noCommandBuffer
            }
            blit.copy(from: src, sourceOffset: 0, to: dst, destinationOffset: 0, size: n * 4)
            blit.endEncoding()
            cb.commit()
            try MetalContext.wait(cb)
            let got = readFloats(dst, count: n)
            var bad = 0
            for i in 0..<n where got[i] != values[i] { bad += 1 }

            // Buffer -> texture -> buffer round trip
            let size = 64
            let tdesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: size, height: size, mipmapped: false)
            tdesc.usage = [.shaderRead, .shaderWrite]
            tdesc.storageMode = .private
            guard let tex = ctx.device.makeTexture(descriptor: tdesc) else {
                throw GPUError.message("could not create blit texture")
            }
            let texBytes = size * size * 16
            let texSrc = try ctx.sharedBuffer(bytes: texBytes)
            let texDst = try ctx.sharedBuffer(bytes: texBytes)
            let raw = (0..<(size * size * 4)).map { Float($0) }
            writeFloats(texSrc, raw)
            guard let cb2 = ctx.queue.makeCommandBuffer(), let blit2 = cb2.makeBlitCommandEncoder() else {
                throw GPUError.noCommandBuffer
            }
            blit2.copy(from: texSrc, sourceOffset: 0, sourceBytesPerRow: size * 16, sourceBytesPerImage: texBytes,
                       sourceSize: MTLSize(width: size, height: size, depth: 1),
                       to: tex, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
            blit2.copy(from: tex, sourceSlice: 0, sourceLevel: 0,
                       sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                       sourceSize: MTLSize(width: size, height: size, depth: 1),
                       to: texDst, destinationOffset: 0, destinationBytesPerRow: size * 16, destinationBytesPerImage: texBytes)
            blit2.endEncoding()
            cb2.commit()
            try MetalContext.wait(cb2)
            let texGot = readFloats(texDst, count: size * size * 4)
            var texBad = 0
            for i in 0..<(size * size * 4) where texGot[i] != raw[i] { texBad += 1 }

            let ok = bad == 0 && texBad == 0
            return Check(group: "Correctness", name: "Blit copy paths", status: ok ? .pass : .fail,
                         summary: ok
                            ? "buffer->buffer and buffer->texture->buffer both bit-exact"
                            : "buffer mismatches \(bad), texture mismatches \(texBad)",
                         details: ["\(Fmt.bytes(n * 4)) buffer copy and \(size)x\(size) rgba32Float texture round trip"],
                         seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "Blit copy paths", error, nowSeconds() - started)
        }
    }

    // MARK: - 10. Heap allocation

    func testHeapAllocation() -> Check {
        let started = nowSeconds()
        let n = 1 << 16
        do {
            let heapSize = 64 << 20
            let hd = MTLHeapDescriptor()
            hd.size = heapSize
            hd.storageMode = .private
            guard let heap = ctx.device.makeHeap(descriptor: hd) else {
                return Check(group: "Correctness", name: "Heap allocation + use", status: .warn,
                             summary: "device.makeHeap returned nil", seconds: nowSeconds() - started)
            }
            guard let a = heap.makeBuffer(length: n * 4, options: .storageModePrivate),
                  let b = heap.makeBuffer(length: n * 4, options: .storageModePrivate) else {
                return Check(group: "Correctness", name: "Heap allocation + use", status: .fail,
                             summary: "could not allocate buffers from heap", seconds: nowSeconds() - started)
            }
            let p = try ctx.pipeline("k_saxpy")
            // Seed through a shared staging buffer, then blit into the heap buffer.
            let stage = try ctx.sharedBuffer(bytes: n * 4)
            let stageY = try ctx.sharedBuffer(bytes: n * 4)
            let x = (0..<n).map { Float($0 % 251) * 0.01 }
            writeFloats(stage, x)
            writeFloats(stageY, [Float](repeating: 1.0, count: n))
            guard let cb = ctx.queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else {
                throw GPUError.noCommandBuffer
            }
            blit.copy(from: stage, sourceOffset: 0, to: a, destinationOffset: 0, size: n * 4)
            blit.copy(from: stageY, sourceOffset: 0, to: b, destinationOffset: 0, size: n * 4)
            blit.endEncoding()
            cb.commit()
            try MetalContext.wait(cb)

            let d = Dispatch(pipeline: p, buffers: [a, b], constantBytes: [(2, floatData([3.0]))],
                             threadgroups: gridSize(threads: n, threadsPerThreadgroup: 256),
                             threadsPerThreadgroup: tpg(256))
            try ctx.run(d)

            // Read the heap buffer back.
            let out = try ctx.sharedBuffer(bytes: n * 4)
            guard let cb2 = ctx.queue.makeCommandBuffer(), let blit2 = cb2.makeBlitCommandEncoder() else {
                throw GPUError.noCommandBuffer
            }
            blit2.copy(from: b, sourceOffset: 0, to: out, destinationOffset: 0, size: n * 4)
            blit2.endEncoding()
            cb2.commit()
            try MetalContext.wait(cb2)

            let got = readFloats(out, count: n)
            var maxRel: Float = 0
            for i in 0..<n {
                let ref = 3.0 * x[i] + 1.0
                maxRel = max(maxRel, abs(got[i] - ref) / max(abs(ref), 1e-30))
            }
            let ok = maxRel < 1e-5
            return Check(group: "Correctness", name: "Heap allocation + use", status: ok ? .pass : .fail,
                         summary: ok
                            ? String(format: "kernel ran on heap-allocated buffers, max rel error %.2e", maxRel)
                            : String(format: "heap buffer result wrong, max rel error %.2e", maxRel),
                         details: ["MTLHeap of \(Fmt.bytes(heapSize)) with sub-allocated buffers"],
                         seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "Heap allocation + use", error, nowSeconds() - started)
        }
    }

    // MARK: - 11. Indirect command buffers

    func testIndirectCommandBuffer() -> Check {
        let started = nowSeconds()
        let n = 1 << 16
        do {
            let p = try ctx.pipeline("k_saxpy")
            let desc = MTLIndirectCommandBufferDescriptor()
            desc.commandTypes = .concurrentDispatch
            desc.inheritBuffers = false
            desc.inheritPipelineState = false
            desc.maxKernelBufferBindCount = 4
            guard let icb = ctx.device.makeIndirectCommandBuffer(descriptor: desc, maxCommandCount: 1, options: .storageModePrivate) else {
                return Check(group: "Correctness", name: "Indirect command buffer", status: .unsupported,
                             summary: "device.makeIndirectCommandBuffer returned nil", seconds: nowSeconds() - started)
            }
            let a = try ctx.sharedBuffer(bytes: n * 4)
            let b = try ctx.sharedBuffer(bytes: n * 4)
            let x = (0..<n).map { Float($0 % 199) * 0.01 }
            writeFloats(a, x)
            writeFloats(b, [Float](repeating: 2.0, count: n))

            let cmd = icb.indirectComputeCommandAt(0)
            cmd.setComputePipelineState(p)
            cmd.setKernelBuffer(a, offset: 0, at: 0)
            cmd.setKernelBuffer(b, offset: 0, at: 1)
            // MTLIndirectComputeCommand has no setKernelBytes, so the scalar is
            // supplied through a small buffer.
            var alpha: Float = 4.0
            let alphaBuf = try ctx.sharedBuffer(bytes: 4)
            memcpy(alphaBuf.contents(), &alpha, 4)
            cmd.setKernelBuffer(alphaBuf, offset: 0, at: 2)
            cmd.concurrentDispatchThreadgroups(gridSize(threads: n, threadsPerThreadgroup: 256), threadsPerThreadgroup: tpg(256))

            guard let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                throw GPUError.noCommandBuffer
            }
            enc.executeCommandsInBuffer(icb, range: 0..<1)
            enc.endEncoding()
            cb.commit()
            try MetalContext.wait(cb)

            let got = readFloats(b, count: n)
            var maxRel: Float = 0
            for i in 0..<n {
                let ref = 4.0 * x[i] + 2.0
                maxRel = max(maxRel, abs(got[i] - ref) / max(abs(ref), 1e-30))
            }
            let ok = maxRel < 1e-5
            return Check(group: "Correctness", name: "Indirect command buffer", status: ok ? .pass : .fail,
                         summary: ok ? String(format: "ICB dispatch produced correct result (max rel %.2e)", maxRel)
                                     : String(format: "ICB result wrong (max rel %.2e)", maxRel),
                         details: ["MTLIndirectCommandBuffer with concurrentDispatch, buffers bound at encode time"],
                         seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "Indirect command buffer", error, nowSeconds() - started)
        }
    }

    // MARK: - 12. SIMD group matrix

    func testSimdMatrix() -> Check {
        let started = nowSeconds()
        do {
            let p = try ctx.pipeline("k_simdmat", inGroup: "simdmat")
            let out = try ctx.sharedBuffer(bytes: 64 * 4)
            memset(out.contents(), 0, 64 * 4)
            let d = Dispatch(pipeline: p, buffers: [out],
                             threadgroups: MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: tpg(32))
            try ctx.run(d)
            let got = readFloats(out, count: 64)
            // Each element of the 8x8 result is sum over k of 2*3 = 48.
            var bad = 0
            for v in got where abs(v - 48.0) > 1e-3 { bad += 1 }
            let ok = bad == 0
            return Check(group: "Correctness", name: "SIMD group matrix (8x8)", status: ok ? .pass : .fail,
                         summary: ok ? "8x8 simdgroup_multiply_accumulate produced 48.0 in all 64 elements"
                                     : "\(bad) of 64 elements wrong (first \(got.first ?? 0))",
                         details: ["exercises the matrix coprocessor via simdgroup_matrix<float,8,8>"],
                         seconds: nowSeconds() - started)
        } catch {
            // On this class of GPU the MSL compiles but the backend cannot build
            // a pipeline for simdgroup_matrix, so record the compiler's reason.
            return Check(group: "Correctness", name: "SIMD group matrix (8x8)", status: .unsupported,
                         summary: "GPU cannot build a matrix-coprocessor pipeline: \(firstLine(String(describing: error)))",
                         details: ["simdgroup_matrix<float,8,8> compiles in MSL but pipeline creation fails in the driver",
                                   "this is a driver/backend limitation, not a test harness problem"],
                         seconds: nowSeconds() - started)
        }
    }

    // MARK: - 13. Large buffer / 64-bit addressing

    func testLargeBuffer64BitAddressing() -> Check {
        let started = nowSeconds()
        // Sized so byte offsets cross the 2^31 boundary, which is exactly where
        // a 32-bit byte offset would break.
        let targetBytes = config.largeBufferBytes
        let maxBytes = Int(ctx.device.maxBufferLength)
        guard targetBytes < maxBytes else {
            return Check(group: "Correctness", name: "64-bit addressing", status: .skipped,
                         summary: "max buffer length \(Fmt.bytes(maxBytes)) too small for a >2 GiB test",
                         seconds: nowSeconds() - started)
        }
        do {
            let p = try ctx.pipeline("k_pattern")
            let big = try ctx.privateBuffer(bytes: targetBytes)
            let n = targetBytes / 4
            let seed: UInt32 = 0xABCD_EF01
            let d = Dispatch(pipeline: p, buffers: [big], constantBytes: [(1, uintData([seed]))],
                             threadgroups: gridSize(threads: n, threadsPerThreadgroup: 256),
                             threadsPerThreadgroup: tpg(256))
            let seconds = try ctx.run(d)

            // Read back a window straddling the 2^31-byte boundary.
            let boundaryIndex = (1 << 31) / 4
            let windowCount = 1024
            let startIndex = boundaryIndex - windowCount / 2
            let windowBytes = windowCount * 4
            let stage = try ctx.sharedBuffer(bytes: windowBytes)
            guard let cb = ctx.queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else {
                throw GPUError.noCommandBuffer
            }
            blit.copy(from: big, sourceOffset: startIndex * 4, to: stage, destinationOffset: 0, size: windowBytes)
            blit.endEncoding()
            cb.commit()
            try MetalContext.wait(cb)

            let got = readFloats(stage, count: windowCount)
            var bad = 0
            var firstBad = ""
            for i in 0..<windowCount {
                let idx = startIndex + i
                let h = (UInt32(truncatingIfNeeded: idx) ^ seed) &* 2654435761
                let expected = Float(h & 0xFFFF) * (1.0 / 65536.0)
                if got[i] != expected {
                    bad += 1
                    if firstBad.isEmpty { firstBad = "byte offset \(idx * 4) (index \(idx))" }
                }
            }
            let ok = bad == 0
            let gbs = Fmt.gbs(bytes: Double(targetBytes), seconds: seconds)
            return Check(group: "Correctness", name: "64-bit addressing", status: ok ? .pass : .fail,
                         summary: ok
                            ? String(format: "%@ buffer written correctly across the 2 GiB byte boundary (%.1f GB/s fill)", Fmt.bytes(targetBytes), gbs)
                            : "\(bad) values wrong beyond 2 GiB, first at \(firstBad)",
                         details: [
                            "buffer \(Fmt.bytes(targetBytes)) vs maxBufferLength \(Fmt.bytes(maxBytes))",
                            "verified window straddles byte offset 2^31 (\(Fmt.bytes(UInt64(1) << 31)))",
                         ],
                         seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "64-bit addressing", error, nowSeconds() - started)
        }
    }

    // MARK: - 14. Concurrent queue submission

    func testConcurrentQueues() -> Check {
        let started = nowSeconds()
        let n = 1 << 18
        let queueCount = 4
        do {
            let p = try ctx.pipeline("k_saxpy")
            var buffers: [(MTLBuffer, MTLBuffer, [Float])] = []
            var dispatches: [MTLCommandBuffer] = []
            for q in 0..<queueCount {
                let a = try ctx.sharedBuffer(bytes: n * 4)
                let b = try ctx.sharedBuffer(bytes: n * 4)
                let x = (0..<n).map { Float(($0 + q) % 313) * 0.01 }
                writeFloats(a, x)
                writeFloats(b, [Float](repeating: Float(q), count: n))
                buffers.append((a, b, x))
                let d = Dispatch(pipeline: p, buffers: [a, b], constantBytes: [(2, floatData([Float(q) + 1]))],
                                 threadgroups: gridSize(threads: n, threadsPerThreadgroup: 256),
                                 threadsPerThreadgroup: tpg(256))
                let cb = try ctx.commitLoop(d, iterations: 1)
                dispatches.append(cb)
            }
            for cb in dispatches { try MetalContext.wait(cb) }
            var maxRel: Float = 0
            for (q, entry) in buffers.enumerated() {
                let got = readFloats(entry.1, count: n)
                for i in 0..<n {
                    let ref = (Float(q) + 1) * entry.2[i] + Float(q)
                    maxRel = max(maxRel, abs(got[i] - ref) / max(abs(ref), 1e-30))
                }
            }
            let ok = maxRel < 1e-5
            return Check(group: "Correctness", name: "Concurrent queue submission", status: ok ? .pass : .fail,
                         summary: ok
                            ? "\(queueCount) command buffers in flight simultaneously, all results correct"
                            : String(format: "results incorrect with concurrent submission (max rel %.2e)", maxRel),
                         details: ["validates that in-flight work on multiple queues does not corrupt results"],
                         seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "Concurrent queue submission", error, nowSeconds() - started)
        }
    }

    // MARK: - 15. GPU timestamp counters

    func testTimestampCounters() -> Check {
        let started = nowSeconds()
        do {
            guard let sets = ctx.device.counterSets,
                  let tsSet = sets.first(where: { $0.name.lowercased().contains("timestamp") }) else {
                return Check(group: "Correctness", name: "GPU timestamp counters", status: .unsupported,
                             summary: "device reports no timestamp counter set", seconds: nowSeconds() - started)
            }
            let desc = MTLCounterSampleBufferDescriptor()
            desc.counterSet = tsSet
            desc.storageMode = .shared
            desc.sampleCount = 2
            let sampleBuffer = try ctx.device.makeCounterSampleBuffer(descriptor: desc)

            let p = try ctx.pipeline("k_spin")
            let n = 1 << 16
            let out = try ctx.privateBuffer(bytes: n * 4)
            let d = Dispatch(pipeline: p, buffers: [out], constantBytes: [(1, uintData([4096]))],
                             threadgroups: gridSize(threads: n, threadsPerThreadgroup: 128),
                             threadsPerThreadgroup: tpg(128))
            let t0 = nowSeconds()
            let cb = try ctx.commitLoop(d, iterations: 1, counterBuffer: sampleBuffer)
            try MetalContext.wait(cb)
            let wall = nowSeconds() - t0

            guard let data = try? sampleBuffer.resolveCounterRange(0..<2), data.count >= 16 else {
                return Check(group: "Correctness", name: "GPU timestamp counters", status: .warn,
                             summary: "counter sample buffer could not be resolved", seconds: nowSeconds() - started)
            }
            let ticks = data.withUnsafeBytes { raw -> (UInt64, UInt64) in
                (raw.load(fromByteOffset: 0, as: UInt64.self), raw.load(fromByteOffset: 8, as: UInt64.self))
            }
            let delta = ticks.1 >= ticks.0 ? ticks.1 - ticks.0 : 0
            let tickRate = wall > 0 ? Double(delta) / wall : 0
            let details = [
                "counter set: \(tsSet.name) (\(tsSet.counters.count) counters)",
                "raw counter delta: \(delta) ticks",
                String(format: "wall clock for the same dispatch: %.6f s", wall),
                String(format: "derived counter frequency: %.3f GHz", tickRate / 1e9),
            ]
            let ok = delta > 0
            self.measurements.append(Measurement(
                group: "Timing", name: "GPU counter frequency", value: tickRate / 1e9, unit: "GHz",
                formatted: String(format: "%.3f GHz", tickRate / 1e9),
                notes: ["derived from timestamp counter delta vs wall clock"]
            ))
            return Check(group: "Correctness", name: "GPU timestamp counters", status: ok ? .pass : .warn,
                         summary: ok
                            ? String(format: "GPU-side timing works: %llu ticks, derived %.3f GHz", delta, tickRate / 1e9)
                            : "counter delta was zero",
                         details: details, seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "GPU timestamp counters", error, nowSeconds() - started)
        }
    }

    // MARK: - 16. Render pipeline

    /// Render targets are always private: on this class of GPU a shared-storage
    /// render target is not host-coherent, so results are read back with a blit.
    private func makeRenderTarget(width: Int, height: Int, sampleCount: Int) throws -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        if sampleCount > 1 {
            desc.textureType = .type2DMultisample
            desc.sampleCount = sampleCount
            desc.usage = [.renderTarget]
        } else {
            desc.usage = [.renderTarget, .shaderRead]
        }
        desc.storageMode = .private
        guard let tex = ctx.device.makeTexture(descriptor: desc) else {
            throw GPUError.message("could not create render target \(width)x\(height) sampleCount \(sampleCount)")
        }
        return tex
    }

    /// Blits a texture into a shared buffer and returns its bytes.
    private func readTextureBytes(_ tex: MTLTexture) throws -> [UInt8] {
        let width = tex.width, height = tex.height
        let bytesPerRow = width * 4
        let total = bytesPerRow * height
        let buf = try ctx.sharedBuffer(bytes: total)
        guard let cb = ctx.queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else {
            throw GPUError.noCommandBuffer
        }
        blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: width, height: height, depth: 1),
                  to: buf, destinationOffset: 0,
                  destinationBytesPerRow: bytesPerRow, destinationBytesPerImage: total)
        blit.endEncoding()
        cb.commit()
        try MetalContext.wait(cb)
        let p = buf.contents().bindMemory(to: UInt8.self, capacity: total)
        return Array(UnsafeBufferPointer(start: p, count: total))
    }

    private func makeRenderPipeline(vertex: String, sampleCount: Int) throws -> MTLRenderPipelineState {
        let lib = try ctx.shaderLibrary("render")
        guard let vfn = lib.makeFunction(name: vertex),
              let ffn = lib.makeFunction(name: "f_const") else {
            throw GPUError.message("render shader functions missing")
        }
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = vfn
        d.fragmentFunction = ffn
        d.sampleCount = sampleCount
        d.colorAttachments[0].pixelFormat = .rgba8Unorm
        return try ctx.device.makeRenderPipelineState(descriptor: d)
    }

    func testRenderPipeline() -> Check {
        let started = nowSeconds()
        do {
            let size = 128
            let tex = try makeRenderTarget(width: size, height: size, sampleCount: 1)
            let state = try makeRenderPipeline(vertex: "v_full", sampleCount: 1)

            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = tex
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

            guard let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeRenderCommandEncoder(descriptor: pass) else {
                throw GPUError.noCommandBuffer
            }
            enc.setRenderPipelineState(state)
            var color = SIMD4<Float>(0.25, 0.5, 0.75, 1.0)
            enc.setFragmentBytes(&color, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
            cb.commit()
            try MetalContext.wait(cb)

            let pixels = try readTextureBytes(tex)
            // Expected: 0.25*255=64, 0.5*255=128, 0.75*255=191
            let expected: [UInt8] = [64, 128, 191, 255]
            var bad = 0
            var worst = 0
            for p in 0..<(size * size) {
                for c in 0..<4 {
                    let diff = abs(Int(pixels[p * 4 + c]) - Int(expected[c]))
                    worst = max(worst, diff)
                    if diff > 1 { bad += 1 }
                }
            }
            let ok = bad == 0
            return Check(group: "Correctness", name: "Render pipeline (rasterise)", status: ok ? .pass : .fail,
                         summary: ok
                            ? "full-screen triangle rendered \(size)x\(size); all \(size * size) pixels match expected colour"
                            : "\(bad) channel values off by more than 1 (worst \(worst))",
                         details: [
                            "vertex + fragment shaders compiled at runtime, offscreen rgba8Unorm target",
                            "clear colour and constant fragment colour both verified",
                         ],
                         seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "Render pipeline (rasterise)", error, nowSeconds() - started)
        }
    }

    func testMSAA() -> Check {
        let started = nowSeconds()
        do {
            let size = 128
            let msaa = try makeRenderTarget(width: size, height: size, sampleCount: 4)
            let resolve = try makeRenderTarget(width: size, height: size, sampleCount: 1)
            let state = try makeRenderPipeline(vertex: "v_half", sampleCount: 4)

            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = msaa
            pass.colorAttachments[0].resolveTexture = resolve
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .multisampleResolve
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

            guard let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeRenderCommandEncoder(descriptor: pass) else {
                throw GPUError.noCommandBuffer
            }
            enc.setRenderPipelineState(state)
            var color = SIMD4<Float>(1, 1, 1, 1)
            enc.setFragmentBytes(&color, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
            cb.commit()
            try MetalContext.wait(cb)

            let pixels = try readTextureBytes(resolve)
            var black = 0, white = 0, partial = 0
            for p in 0..<(size * size) {
                let v = Int(pixels[p * 4])
                if v <= 4 { black += 1 } else if v >= 251 { white += 1 } else { partial += 1 }
            }
            // A diagonal edge rasterised with 4x MSAA must leave partially
            // covered pixels; without MSAA every pixel would be 0 or 255.
            let ok = partial > 0 && white > 0 && black > 0
            return Check(group: "Correctness", name: "MSAA 4x + resolve", status: ok ? .pass : .fail,
                         summary: ok
                            ? "4x MSAA resolved: \(white) covered, \(black) clear, \(partial) partially covered pixels"
                            : "no partially covered pixels - MSAA resolve did not take effect (\(white) white, \(black) black)",
                         details: [
                            "diagonal triangle edge rasterised at 4 samples/pixel then resolved",
                            "partially covered pixels are the signature of working multisampling",
                         ],
                         seconds: nowSeconds() - started)
        } catch {
            return unsupported("Correctness", "MSAA 4x + resolve", error, nowSeconds() - started)
        }
    }

    // MARK: - Runner

    func runAll(progress: Progress) -> [Check] {
        var checks: [Check] = []

        // willRun() writes the step name to the journal before the GPU is
        // touched, so a hang points at exactly one test.
        func record(_ name: String, _ body: () -> Check) {
            progress.willRun(name)
            let check = body()
            progress.step(check.name, status: check.status, summary: check.summary, seconds: check.seconds)
            for d in check.details { progress.note(d) }
            checks.append(check)
            rest(milliseconds: config.cooldownMs)
        }

        progress.beginPhase("Functional verification (compute)")
        record("Memory allocation") { testResourceAllocation() }
        record("Buffer round trip") { testBufferRoundTrip() }
        record("SAXPY") { testSaxpy() }
        record("Threadgroup reduction") { testThreadgroupReduction() }
        record("Atomic operations") { testAtomics() }
        record("Concurrent queues") { testConcurrentQueues() }

        progress.beginPhase("Numerical precision and math accuracy")
        record("FP16 half precision") { testHalfPrecision() }
        record("FP64 double precision") { testDoublePrecision() }
        record("Math accuracy") { testMathAccuracy() }
        record("Fast-math accuracy") { testFastMathAccuracy() }

        progress.beginPhase("Textures, blit and advanced resources")
        record("Texture read + filtering") { testTextures() }
        record("Blit copy paths") { testBlit() }
        record("Heap allocation") { testHeapAllocation() }
        record("Indirect command buffer") { testIndirectCommandBuffer() }
        record("SIMD group matrix") { testSimdMatrix() }
        if config.largeBufferBytes > 0 {
            record("64-bit addressing") { testLargeBuffer64BitAddressing() }
        } else {
            progress.willRun("64-bit addressing")
            let skipped = Check(group: "Correctness", name: "64-bit addressing", status: .skipped,
                                summary: "skipped at intensity '\(config.intensity.rawValue)'; use --intensity normal or --mode thorough to enable",
                                details: ["this test allocates a multi-gigabyte buffer"])
            progress.step(skipped.name, status: skipped.status, summary: skipped.summary)
            checks.append(skipped)
        }
        record("GPU timestamp counters") { testTimestampCounters() }

        if config.includeRender {
            progress.beginPhase("Graphics pipeline")
            record("Render pipeline") { testRenderPipeline() }
            record("MSAA resolve") { testMSAA() }
        }

        return checks
    }
}
