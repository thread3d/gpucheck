// MetalContext.swift — device context, shader compilation, and dispatch harness.
import Foundation
import Metal

struct Dispatch {
    var pipeline: MTLComputePipelineState
    var buffers: [MTLBuffer?] = []
    var constantBytes: [(index: Int, data: Data)] = []
    var textures: [MTLTexture?] = []
    var samplers: [MTLSamplerState?] = []
    var threadgroups: MTLSize
    var threadsPerThreadgroup: MTLSize
}

struct CompileReport {
    var group: String
    var ok: Bool
    var error: String?
}

/// Wraps a shader-compilation failure message so it can be cached in a Result.
struct ShaderFailure: Error, CustomStringConvertible {
    let message: String
    var description: String { message }
}

final class MetalContext {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let name: String

    private var libraries: [String: Result<MTLLibrary, ShaderFailure>] = [:]
    private var pipelineCache: [String: Result<MTLComputePipelineState, ShaderFailure>] = [:]
    private let groupForFunction: [String: ShaderGroup]
    private let groupByKey: [String: ShaderGroup]

    /// SIMD width of the device, read from a real pipeline state (the device
    /// does not expose it directly).
    private(set) var threadExecutionWidth: Int = 32
    private(set) var maxTotalThreadsPerThreadgroup: Int = 512

    init(device: MTLDevice) throws {
        self.device = device
        self.name = device.name
        guard let q = device.makeCommandQueue() else { throw GPUError.noCommandQueue }
        self.queue = q
        var map: [String: ShaderGroup] = [:]
        var byKey: [String: ShaderGroup] = [:]
        for g in shaderGroups {
            byKey[g.key] = g
            // First group wins: k_math_accuracy exists in both "core" (precise)
            // and "core_fast", and the precise one must stay the default.
            for f in g.functions where map[f] == nil { map[f] = g }
        }
        self.groupForFunction = map
        self.groupByKey = byKey
    }

    func group(for function: String) -> ShaderGroup? { groupForFunction[function] }

    /// Compiles (once) and returns a whole shader group's library. Needed for
    /// render pipelines, which use vertex and fragment functions.
    func shaderLibrary(_ groupKey: String) throws -> MTLLibrary {
        guard let g = groupByKey[groupKey] else {
            throw GPUError.message("unknown shader group \(groupKey)")
        }
        switch library(for: g) {
        case .success(let l): return l
        case .failure(let f): throw GPUError.message("shader group '\(groupKey)' failed to compile: \(firstLine(f.message))")
        }
    }

    /// Compiles (once) the shader group that owns `function`.
    private func library(for group: ShaderGroup) -> Result<MTLLibrary, ShaderFailure> {
        if let cached = libraries[group.key] { return cached }
        let result: Result<MTLLibrary, ShaderFailure>
        do {
            let options = MTLCompileOptions()
            if group.fastMath { options.fastMathEnabled = true }
            let lib = try device.makeLibrary(source: group.source, options: options)
            result = .success(lib)
        } catch {
            result = .failure(ShaderFailure(message: String(describing: error)))
        }
        libraries[group.key] = result
        return result
    }

    /// Returns a compiled pipeline state, or throws with the compiler message so
    /// the caller can record the feature as unsupported.
    func pipeline(_ function: String, inGroup groupKey: String? = nil) throws -> MTLComputePipelineState {
        let cacheKey = groupKey.map { "\($0)/\(function)" } ?? function
        if let cached = pipelineCache[cacheKey] {
            switch cached {
            case .success(let p): return p
            case .failure(let f): throw GPUError.message(f.message)
            }
        }
        let result: Result<MTLComputePipelineState, ShaderFailure>
        do {
            let group: ShaderGroup
            if let groupKey {
                guard let g = groupByKey[groupKey] else { throw GPUError.message("unknown shader group \(groupKey)") }
                group = g
            } else if let g = groupForFunction[function] {
                group = g
            } else {
                throw GPUError.missingFunction(function)
            }
            switch library(for: group) {
            case .failure(let failure):
                result = .failure(ShaderFailure(message: "shader group '\(group.key)' failed to compile: \(firstLine(failure.message))"))
            case .success(let lib):
                guard let fn = lib.makeFunction(name: function) else {
                    result = .failure(ShaderFailure(message: "function \(function) not present in group \(group.key)"))
                    break
                }
                let p = try device.makeComputePipelineState(function: fn)
                result = .success(p)
            }
        } catch {
            result = .failure(ShaderFailure(message: firstLine(String(describing: error))))
        }
        pipelineCache[cacheKey] = result
        switch result {
        case .success(let p): return p
        case .failure(let f): throw GPUError.message(f.message)
        }
    }

    /// Probes SIMD width using a cheap known-good kernel.
    func calibrate() {
        if let p = try? pipeline("k_copy") {
            threadExecutionWidth = p.threadExecutionWidth
            maxTotalThreadsPerThreadgroup = p.maxTotalThreadsPerThreadgroup
        }
    }

    /// Reports which shader groups compiled on this GPU, for the feature map.
    func compileReports() -> [CompileReport] {
        shaderGroups.map { g in
            switch library(for: g) {
            case .success: return CompileReport(group: g.key, ok: true, error: nil)
            case .failure(let f): return CompileReport(group: g.key, ok: false, error: firstLine(f.message))
            }
        }
    }

    // MARK: - Buffers

    func privateBuffer(bytes: Int) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(bytes, 16), options: .storageModePrivate) else {
            throw GPUError.message("could not allocate \(Fmt.bytes(bytes)) private buffer")
        }
        return b
    }

    func sharedBuffer(bytes: Int) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
            throw GPUError.message("could not allocate \(Fmt.bytes(bytes)) shared buffer")
        }
        return b
    }

    // MARK: - Dispatch

    func encode(_ d: Dispatch, into cb: MTLCommandBuffer) throws {
        guard let enc = cb.makeComputeCommandEncoder() else { throw GPUError.noEncoder }
        enc.setComputePipelineState(d.pipeline)
        for (i, b) in d.buffers.enumerated() { if let b { enc.setBuffer(b, offset: 0, index: i) } }
        for (i, t) in d.textures.enumerated() { if let t { enc.setTexture(t, index: i) } }
        for (i, s) in d.samplers.enumerated() { if let s { enc.setSamplerState(s, index: i) } }
        for (index, data) in d.constantBytes {
            data.withUnsafeBytes { raw in
                if let base = raw.baseAddress, raw.count > 0 {
                    enc.setBytes(base, length: raw.count, index: index)
                }
            }
        }
        enc.dispatchThreadgroups(d.threadgroups, threadsPerThreadgroup: d.threadsPerThreadgroup)
        enc.endEncoding()
    }

    /// Encodes `iterations` dispatches into one command buffer. Amortising
    /// encode/commit overhead is what makes throughput numbers meaningful.
    func commitLoop(_ d: Dispatch, iterations: Int, counterBuffer: MTLCounterSampleBuffer? = nil) throws -> MTLCommandBuffer {
        guard let cb = queue.makeCommandBuffer() else { throw GPUError.noCommandBuffer }
        if let counterBuffer {
            guard let enc = cb.makeComputeCommandEncoder() else { throw GPUError.noEncoder }
            enc.setComputePipelineState(d.pipeline)
            for (i, b) in d.buffers.enumerated() { if let b { enc.setBuffer(b, offset: 0, index: i) } }
            for (i, t) in d.textures.enumerated() { if let t { enc.setTexture(t, index: i) } }
            for (i, s) in d.samplers.enumerated() { if let s { enc.setSamplerState(s, index: i) } }
            for (index, data) in d.constantBytes {
                data.withUnsafeBytes { raw in
                    if let base = raw.baseAddress, raw.count > 0 {
                        enc.setBytes(base, length: raw.count, index: index)
                    }
                }
            }
            enc.sampleCounters(sampleBuffer: counterBuffer, sampleIndex: 0, barrier: true)
            enc.dispatchThreadgroups(d.threadgroups, threadsPerThreadgroup: d.threadsPerThreadgroup)
            enc.sampleCounters(sampleBuffer: counterBuffer, sampleIndex: 1, barrier: true)
            enc.endEncoding()
        } else {
            for _ in 0..<max(1, iterations) { try encode(d, into: cb) }
        }
        cb.commit()
        return cb
    }

    static func wait(_ cb: MTLCommandBuffer) throws {
        cb.waitUntilCompleted()
        if let e = cb.error { throw GPUError.message("GPU command buffer error: \(e.localizedDescription)") }
    }

    /// Runs a dispatch and returns the wall-clock duration.
    @discardableResult
    func run(_ d: Dispatch) throws -> Double {
        let t0 = nowSeconds()
        let cb = try commitLoop(d, iterations: 1)
        try MetalContext.wait(cb)
        return nowSeconds() - t0
    }

    /// Best-of-N timing: returns the fastest total wall time for one iteration.
    func bestTime(_ d: Dispatch, attempts: Int, iterationsPerAttempt: Int = 1) throws -> Double {
        var best = Double.greatestFiniteMagnitude
        // Warm-up: first dispatch includes lazy driver work.
        _ = try run(d)
        for _ in 0..<attempts {
            let t0 = nowSeconds()
            let cb = try commitLoop(d, iterations: iterationsPerAttempt)
            try MetalContext.wait(cb)
            let elapsed = (nowSeconds() - t0) / Double(iterationsPerAttempt)
            best = min(best, elapsed)
        }
        return best
    }
}

func firstLine(_ s: String) -> String {
    let line = s.split(separator: "\n").first.map(String.init) ?? s
    return Fmt.truncate(line.trimmingCharacters(in: .whitespaces), 160)
}

// MARK: - Shared helpers

func gridSize(threads: Int, threadsPerThreadgroup: Int) -> MTLSize {
    let tpg = max(1, threadsPerThreadgroup)
    return MTLSize(width: (threads + tpg - 1) / tpg, height: 1, depth: 1)
}

func makeSampler(_ device: MTLDevice, min: MTLSamplerMinMagFilter, mag: MTLSamplerMinMagFilter, address: MTLSamplerAddressMode) -> MTLSamplerState? {
    let d = MTLSamplerDescriptor()
    d.minFilter = min
    d.magFilter = mag
    d.mipFilter = .notMipmapped
    d.sAddressMode = address
    d.tAddressMode = address
    d.normalizedCoordinates = true
    return device.makeSamplerState(descriptor: d)
}

func floatData(_ values: [Float]) -> Data {
    values.withUnsafeBufferPointer { Data(buffer: $0) }
}

func uintData(_ values: [UInt32]) -> Data {
    values.withUnsafeBufferPointer { Data(buffer: $0) }
}

/// IEEE-754 binary16 -> binary32. Implemented explicitly because `Float16` is
/// not usable on this x86_64 SDK.
func halfToFloat(_ h: UInt16) -> Float {
    let sign = UInt32(h & 0x8000) << 16
    let exp = UInt32(h & 0x7C00) >> 10
    let mant = UInt32(h & 0x03FF)
    var bits: UInt32
    if exp == 0 {
        if mant == 0 {
            bits = sign
        } else {
            // Subnormal half: normalise into a float significand.
            var m = mant
            var e: Int32 = -14
            while (m & 0x0400) == 0 {
                m <<= 1
                e -= 1
            }
            m &= 0x03FF
            bits = sign | (UInt32(e + 127) << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        bits = sign | 0x7F80_0000 | (mant << 13) // inf / NaN
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13)
    }
    return Float(bitPattern: bits)
}

/// IEEE-754 binary32 -> binary16 with round-to-nearest-even, matching
/// conventional GPU half-precision conversion.
func floatToHalf(_ f: Float) -> UInt16 {
    let bits = f.bitPattern
    let sign = UInt16((bits >> 16) & 0x8000)
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 { return sign | 0x7E00 } // NaN
    let unbiased = Int32((bits >> 23) & 0xFF) - 127
    let halfExp = unbiased + 15
    let mant = bits & 0x007F_FFFF
    if halfExp >= 0x1F { return sign | 0x7C00 }                    // overflow -> inf
    if halfExp <= 0 {
        if halfExp < -10 { return sign }                            // underflow -> zero
        let shift = UInt32(14 - halfExp)                            // == -unbiased - 1
        let sig = mant | 0x0080_0000
        var m = sig >> shift
        // round to nearest even
        let rem = sig & ((1 << shift) - 1)
        let halfway = UInt32(1) << (shift - 1)
        if rem > halfway || (rem == halfway && (m & 1) == 1) { m += 1 }
        return sign | UInt16(m)
    }
    var m = mant >> 13
    let rem = mant & 0x1FFF
    if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) { m += 1 }
    var e = halfExp
    if m == 0x400 { m = 0; e += 1; if e >= 0x1F { return sign | 0x7C00 } }
    return sign | UInt16(e << 10) | UInt16(m)
}

func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let p = buffer.contents().bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: p, count: count))
}

func readHalfAsFloat(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let p = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
    return (0..<count).map { halfToFloat(p[$0]) }
}

func writeFloats(_ buffer: MTLBuffer, _ values: [Float]) {
    values.withUnsafeBufferPointer { src in
        buffer.contents().copyMemory(from: src.baseAddress!, byteCount: values.count * 4)
    }
}

/// ULP distance between two finite floats, used for math accuracy reporting.
func ulpDistance(_ a: Float, _ b: Float) -> Int {
    if a == b { return 0 }
    if a.isNaN || b.isNaN { return Int.max }
    if a.isInfinite || b.isInfinite { return Int.max }
    let ia = Int32(bitPattern: a.bitPattern)
    let ib = Int32(bitPattern: b.bitPattern)
    let x = ia < 0 ? Int32.min &- ia : ia
    let y = ib < 0 ? Int32.min &- ib : ib
    return Int(abs(Int(x) - Int(y)))
}
