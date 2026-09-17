// OpenCLProbe.swift — maps the same GPUs through OpenCL, an independent driver
// stack. This cross-check exposes hardware facts Metal does not report (such as
// the compute-unit count and clock) and confirms the GPUs work through a second
// API. macOS ships OpenCL 1.2, so only 1.2-era queries are used.
import Foundation
import OpenCL

struct OpenCLDevice {
    var id: cl_device_id
    var platformName: String
    var name: String
    var vendor: String
    var deviceVersion: String
    var driverVersion: String
    var openCLCVersion: String
    var computeUnits: Int
    var maxClockMHz: Int
    var globalMemBytes: UInt64
    var maxAllocBytes: UInt64
    var localMemBytes: UInt64
    var constantMemBytes: UInt64
    var maxWorkGroupSize: Int
    var maxWorkItemSizes: [Int]
    var addressBits: Int
    var littleEndian: Bool
    var imageSupport: Bool
    var unifiedMemory: Bool
    var preferredFloatWidth: Int
    var preferredHalfWidth: Int
    var extensions: [String]

    /// OpenCL calls the same physical GPU "<name> Compute Engine"; strip that
    /// suffix so the two inventories can be matched up.
    var baseName: String {
        name.replacingOccurrences(of: " Compute Engine", with: "").trimmingCharacters(in: .whitespaces)
    }
}

// MARK: - Typed queries

private func devString(_ d: cl_device_id, _ p: Int32) -> String {
    var sz = 0
    guard clGetDeviceInfo(d, cl_device_info(p), 0, nil, &sz) == CL_SUCCESS, sz > 0 else { return "" }
    var buf = [CChar](repeating: 0, count: sz + 1)
    guard clGetDeviceInfo(d, cl_device_info(p), sz, &buf, nil) == CL_SUCCESS else { return "" }
    return String(cString: buf).trimmingCharacters(in: .whitespaces)
}

private func devNum<T>(_ d: cl_device_id, _ p: Int32, _ type: T.Type) -> T? {
    var v = UnsafeMutablePointer<T>.allocate(capacity: 1)
    defer { v.deallocate() }
    guard clGetDeviceInfo(d, cl_device_info(p), MemoryLayout<T>.size, v, nil) == CL_SUCCESS else { return nil }
    return v.pointee
}

private func platString(_ p: cl_platform_id, _ f: Int32) -> String {
    var sz = 0
    guard clGetPlatformInfo(p, cl_platform_info(f), 0, nil, &sz) == CL_SUCCESS, sz > 0 else { return "" }
    var buf = [CChar](repeating: 0, count: sz + 1)
    guard clGetPlatformInfo(p, cl_platform_info(f), sz, &buf, nil) == CL_SUCCESS else { return "" }
    return String(cString: buf).trimmingCharacters(in: .whitespaces)
}

// MARK: - Enumeration

func enumerateOpenCLDevices() -> [OpenCLDevice] {
    var platformCount: UInt32 = 0
    guard clGetPlatformIDs(0, nil, &platformCount) == CL_SUCCESS, platformCount > 0 else { return [] }
    var platforms = [cl_platform_id?](repeating: nil, count: Int(platformCount))
    guard clGetPlatformIDs(platformCount, &platforms, nil) == CL_SUCCESS else { return [] }

    var found: [OpenCLDevice] = []
    for maybePlatform in platforms {
        guard let platform = maybePlatform else { continue }
        let platformName = platString(platform, CL_PLATFORM_NAME)
        var deviceCount: UInt32 = 0
        guard clGetDeviceIDs(platform, cl_device_type(CL_DEVICE_TYPE_GPU), 0, nil, &deviceCount) == CL_SUCCESS,
              deviceCount > 0 else { continue }
        var devices = [cl_device_id?](repeating: nil, count: Int(deviceCount))
        guard clGetDeviceIDs(platform, cl_device_type(CL_DEVICE_TYPE_GPU), deviceCount, &devices, nil) == CL_SUCCESS else { continue }

        for maybeDevice in devices {
            guard let d = maybeDevice else { continue }
            let itemSizes = devNum(d, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, cl_uint.self).map { Int($0) } ?? 0
            var sizes: [Int] = []
            if itemSizes > 0 {
                var raw = [Int](repeating: 0, count: itemSizes)
                if clGetDeviceInfo(d, cl_device_info(CL_DEVICE_MAX_WORK_ITEM_SIZES),
                                   MemoryLayout<Int>.size * itemSizes, &raw, nil) == CL_SUCCESS {
                    sizes = raw
                }
            }
            found.append(OpenCLDevice(
                id: d,
                platformName: platformName,
                name: devString(d, CL_DEVICE_NAME),
                vendor: devString(d, CL_DEVICE_VENDOR),
                deviceVersion: devString(d, CL_DEVICE_VERSION),
                driverVersion: devString(d, CL_DRIVER_VERSION),
                openCLCVersion: devString(d, CL_DEVICE_OPENCL_C_VERSION),
                computeUnits: Int(devNum(d, CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint.self) ?? 0),
                maxClockMHz: Int(devNum(d, CL_DEVICE_MAX_CLOCK_FREQUENCY, cl_uint.self) ?? 0),
                globalMemBytes: devNum(d, CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong.self) ?? 0,
                maxAllocBytes: devNum(d, CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong.self) ?? 0,
                localMemBytes: devNum(d, CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong.self) ?? 0,
                constantMemBytes: devNum(d, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, cl_ulong.self) ?? 0,
                maxWorkGroupSize: Int(devNum(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, Int.self) ?? 0),
                maxWorkItemSizes: sizes,
                addressBits: Int(devNum(d, CL_DEVICE_ADDRESS_BITS, cl_uint.self) ?? 0),
                littleEndian: (devNum(d, CL_DEVICE_ENDIAN_LITTLE, cl_bool.self) ?? 0) != 0,
                imageSupport: (devNum(d, CL_DEVICE_IMAGE_SUPPORT, cl_bool.self) ?? 0) != 0,
                unifiedMemory: (devNum(d, CL_DEVICE_HOST_UNIFIED_MEMORY, cl_bool.self) ?? 0) != 0,
                preferredFloatWidth: Int(devNum(d, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, cl_uint.self) ?? 0),
                preferredHalfWidth: Int(devNum(d, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, cl_uint.self) ?? 0),
                extensions: devString(d, CL_DEVICE_EXTENSIONS).split(separator: " ").map(String.init).sorted()
            ))
        }
    }
    return found
}

func openCLSections(_ d: OpenCLDevice) -> [Section] {
    let fp64 = d.extensions.contains("cl_khr_fp64")
    let fp16 = d.extensions.contains("cl_khr_fp16")
    let atomics = d.extensions.contains { $0.hasPrefix("cl_khr_global_int32_base_atomics") }
    var items: [Section] = []
    items.append(Section(title: "OpenCL", items: [
        KV(key: "Platform", value: d.platformName),
        KV(key: "Device name", value: d.name),
        KV(key: "Vendor", value: d.vendor),
        KV(key: "Device version", value: d.deviceVersion),
        KV(key: "Driver version", value: d.driverVersion),
        KV(key: "OpenCL C version", value: d.openCLCVersion),
        KV(key: "Compute units", value: "\(d.computeUnits)", note: "hardware CUs, not exposed by Metal"),
        KV(key: "Max clock", value: "\(d.maxClockMHz) MHz"),
        KV(key: "Global memory", value: Fmt.bytes(d.globalMemBytes)),
        KV(key: "Max allocation", value: Fmt.bytes(d.maxAllocBytes)),
        KV(key: "Local memory", value: Fmt.bytes(d.localMemBytes)),
        KV(key: "Constant memory", value: Fmt.bytes(d.constantMemBytes)),
        KV(key: "Max work-group size", value: "\(d.maxWorkGroupSize)"),
        KV(key: "Max work-item sizes", value: d.maxWorkItemSizes.map(String.init).joined(separator: " x ")),
        KV(key: "Address bits", value: "\(d.addressBits)"),
        KV(key: "Little endian", value: d.littleEndian ? "yes" : "no"),
        KV(key: "Image support", value: d.imageSupport ? "yes" : "no"),
        KV(key: "Host unified memory", value: d.unifiedMemory ? "yes" : "no"),
        KV(key: "Preferred vector width", value: "float \(d.preferredFloatWidth), half \(d.preferredHalfWidth)"),
        KV(key: "FP64 (double)", value: fp64 ? "supported" : "not supported"),
        KV(key: "FP16 (half)", value: fp16 ? "supported" : "not supported"),
        KV(key: "Int32 atomics", value: atomics ? "supported" : "not supported"),
        KV(key: "Extensions (\(d.extensions.count))", value: d.extensions.joined(separator: ", ")),
    ]))
    return items
}

// MARK: - OpenCL functional + bandwidth test

private func clErrorName(_ code: cl_int) -> String {
    switch code {
    case CL_SUCCESS: return "CL_SUCCESS"
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND"
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE"
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE"
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES"
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY"
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL"
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT"
    default: return "CL error \(code)"
    }
}

/// Carries an OpenCL failure reason (String is not an Error, so it is wrapped).
struct OpenCLFailure: Error, CustomStringConvertible {
    let message: String
    var description: String { message }
}

struct OpenCLBandwidthResult {
    var deviceName: String
    var bytesPerSecond: Double
    var bytesMoved: UInt64
    var seconds: Double
    var iterations: Int
    var buildLog: String
}

let openCLCopySource = """
__kernel void gpucheck_copy(__global const float4 *in, __global float4 *out) {
    size_t i = get_global_id(0);
    out[i] = in[i];
}
"""

/// Compiles an OpenCL C kernel and measures device-local copy bandwidth.
/// Returns nil with a reason string when the OpenCL stack cannot run.
func openCLCopyBandwidth(_ d: OpenCLDevice, bufferBytes: Int, iterations: Int) -> Result<OpenCLBandwidthResult, OpenCLFailure> {
    var err: cl_int = 0
    var deviceID: cl_device_id? = d.id
    guard let context = clCreateContext(nil, 1, &deviceID, nil, nil, &err) else {
        return .failure(OpenCLFailure(message: "clCreateContext failed: \(clErrorName(err))"))
    }
    defer { clReleaseContext(context) }

    guard let queue = clCreateCommandQueue(context, d.id, 0, &err) else {
        return .failure(OpenCLFailure(message: "clCreateCommandQueue failed: \(clErrorName(err))"))
    }
    defer { clReleaseCommandQueue(queue) }

    var source = openCLCopySource
    var buildLog = ""
    let program: cl_program? = source.withCString { cstr -> cl_program? in
        var ptr: UnsafePointer<CChar>? = cstr
        return clCreateProgramWithSource(context, 1, &ptr, nil, &err)
    }
    guard let program else { return .failure(OpenCLFailure(message: "clCreateProgramWithSource failed: \(clErrorName(err))")) }
    defer { clReleaseProgram(program) }

    let buildStatus = clBuildProgram(program, 1, &deviceID, nil, nil, nil)
    // Retrieve the build log regardless of outcome; it is useful evidence.
    var logSize = 0
    if clGetProgramBuildInfo(program, d.id, cl_program_build_info(CL_PROGRAM_BUILD_LOG), 0, nil, &logSize) == CL_SUCCESS, logSize > 1 {
        var buf = [CChar](repeating: 0, count: logSize + 1)
        if clGetProgramBuildInfo(program, d.id, cl_program_build_info(CL_PROGRAM_BUILD_LOG), logSize, &buf, nil) == CL_SUCCESS {
            buildLog = String(cString: buf).trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }
    guard buildStatus == CL_SUCCESS else {
        return .failure(OpenCLFailure(message: "OpenCL kernel build failed: \(clErrorName(buildStatus))\(buildLog.isEmpty ? "" : " — \(buildLog)")"))
    }

    guard let kernel = clCreateKernel(program, "gpucheck_copy", &err) else {
        return .failure(OpenCLFailure(message: "clCreateKernel failed: \(clErrorName(err))"))
    }
    defer { clReleaseKernel(kernel) }

    guard let src = clCreateBuffer(context, cl_mem_flags(CL_MEM_READ_ONLY), bufferBytes, nil, &err),
          let dst = clCreateBuffer(context, cl_mem_flags(CL_MEM_WRITE_ONLY), bufferBytes, nil, &err) else {
        return .failure(OpenCLFailure(message: "clCreateBuffer failed: \(clErrorName(err))"))
    }
    defer { clReleaseMemObject(src); clReleaseMemObject(dst) }

    // Seed the source buffer once, outside the timed region.
    let seed = [Float](repeating: 0.5, count: min(bufferBytes, 1 << 20) / 4)
    guard clEnqueueWriteBuffer(queue, src, 1, 0, seed.count * 4, seed, 0, nil, nil) == CL_SUCCESS else {
        return .failure(OpenCLFailure(message: "clEnqueueWriteBuffer failed"))
    }

    var srcArg: cl_mem? = src
    var dstArg: cl_mem? = dst
    guard clSetKernelArg(kernel, 0, MemoryLayout<cl_mem?>.size, &srcArg) == CL_SUCCESS,
          clSetKernelArg(kernel, 1, MemoryLayout<cl_mem?>.size, &dstArg) == CL_SUCCESS else {
        return .failure(OpenCLFailure(message: "clSetKernelArg failed"))
    }

    let vectorCount = bufferBytes / 16
    var global = vectorCount
    var local = max(1, min(d.maxWorkGroupSize, 256))

    func enqueue() -> cl_int {
        clEnqueueNDRangeKernel(queue, kernel, 1, nil, &global, &local, 0, nil, nil)
    }

    // Warm-up
    for _ in 0..<2 {
        let e = enqueue()
        if e != CL_SUCCESS { return .failure(OpenCLFailure(message: "clEnqueueNDRangeKernel failed: \(clErrorName(e))")) }
    }
    guard clFinish(queue) == CL_SUCCESS else { return .failure(OpenCLFailure(message: "clFinish after warm-up failed")) }

    let start = nowSeconds()
    for _ in 0..<iterations {
        let e = enqueue()
        if e != CL_SUCCESS { return .failure(OpenCLFailure(message: "clEnqueueNDRangeKernel failed: \(clErrorName(e))")) }
    }
    guard clFinish(queue) == CL_SUCCESS else { return .failure(OpenCLFailure(message: "clFinish failed")) }
    let elapsed = nowSeconds() - start

    let moved = UInt64(bufferBytes) * 2 * UInt64(iterations)
    return .success(OpenCLBandwidthResult(
        deviceName: d.name,
        bytesPerSecond: Double(moved) / elapsed,
        bytesMoved: moved,
        seconds: elapsed,
        iterations: iterations,
        buildLog: buildLog
    ))
}

// MARK: - OpenCL compute cross-check

/// Same FMA workload shape as the Metal benchmark (8 independent float4 chains),
/// so the two APIs can be compared directly. If Metal reports much less than
/// OpenCL on the same GPU, the limit is in the Metal driver rather than the
/// silicon; if both agree, the device itself is delivering that throughput.
let openCLFMASource = """
__kernel void gpucheck_fma(__global float4 *out, const uint iters) {
    size_t gid = get_global_id(0);
    float4 a = (float4)(1.0000001f, 0.9999999f, 1.0000002f, 0.9999998f);
    float4 b = (float4)(1.0e-7f, 2.0e-7f, 3.0e-7f, 4.0e-7f);
    float base = (float)(gid & 1023u) * 1.0e-3f;
    float4 c0 = (float4)(base, base + 0.1f, base + 0.2f, base + 0.3f);
    float4 c1 = c0 + 0.5f, c2 = c0 + 0.7f, c3 = c0 + 0.9f;
    float4 c4 = c0 + 1.1f, c5 = c0 + 1.3f, c6 = c0 + 1.5f, c7 = c0 + 1.7f;
    for (uint i = 0; i < iters; ++i) {
        c0 = fma(a, c0, b); c1 = fma(a, c1, b);
        c2 = fma(a, c2, b); c3 = fma(a, c3, b);
        c4 = fma(a, c4, b); c5 = fma(a, c5, b);
        c6 = fma(a, c6, b); c7 = fma(a, c7, b);
    }
    out[gid] = (c0 + c1) + (c2 + c3) + (c4 + c5) + (c6 + c7);
}
"""

struct OpenCLComputeResult {
    var flopsPerSecond: Double
    var seconds: Double
    var iterations: Int
    var buildLog: String
}

func openCLFMAThroughput(_ d: OpenCLDevice, threads: Int, iters: UInt32, iterations: Int) -> Result<OpenCLComputeResult, OpenCLFailure> {
    var err: cl_int = 0
    var deviceID: cl_device_id? = d.id
    guard let context = clCreateContext(nil, 1, &deviceID, nil, nil, &err) else {
        return .failure(OpenCLFailure(message: "clCreateContext failed: \(clErrorName(err))"))
    }
    defer { clReleaseContext(context) }
    guard let queue = clCreateCommandQueue(context, d.id, 0, &err) else {
        return .failure(OpenCLFailure(message: "clCreateCommandQueue failed: \(clErrorName(err))"))
    }
    defer { clReleaseCommandQueue(queue) }

    var buildLog = ""
    let source = openCLFMASource
    let program: cl_program? = source.withCString { cstr -> cl_program? in
        var ptr: UnsafePointer<CChar>? = cstr
        return clCreateProgramWithSource(context, 1, &ptr, nil, &err)
    }
    guard let program else { return .failure(OpenCLFailure(message: "clCreateProgramWithSource failed: \(clErrorName(err))")) }
    defer { clReleaseProgram(program) }

    let buildStatus = clBuildProgram(program, 1, &deviceID, nil, nil, nil)
    var logSize = 0
    if clGetProgramBuildInfo(program, d.id, cl_program_build_info(CL_PROGRAM_BUILD_LOG), 0, nil, &logSize) == CL_SUCCESS, logSize > 1 {
        var buf = [CChar](repeating: 0, count: logSize + 1)
        if clGetProgramBuildInfo(program, d.id, cl_program_build_info(CL_PROGRAM_BUILD_LOG), logSize, &buf, nil) == CL_SUCCESS {
            buildLog = String(cString: buf).trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }
    guard buildStatus == CL_SUCCESS else {
        return .failure(OpenCLFailure(message: "OpenCL FMA kernel build failed: \(clErrorName(buildStatus))\(buildLog.isEmpty ? "" : " — \(buildLog)")"))
    }
    guard let kernel = clCreateKernel(program, "gpucheck_fma", &err) else {
        return .failure(OpenCLFailure(message: "clCreateKernel failed: \(clErrorName(err))"))
    }
    defer { clReleaseKernel(kernel) }

    let bufferBytes = threads * 16
    guard let out = clCreateBuffer(context, cl_mem_flags(CL_MEM_WRITE_ONLY), bufferBytes, nil, &err) else {
        return .failure(OpenCLFailure(message: "clCreateBuffer failed: \(clErrorName(err))"))
    }
    defer { clReleaseMemObject(out) }

    var outArg: cl_mem? = out
    var itersArg = iters
    guard clSetKernelArg(kernel, 0, MemoryLayout<cl_mem?>.size, &outArg) == CL_SUCCESS,
          clSetKernelArg(kernel, 1, MemoryLayout<cl_uint>.size, &itersArg) == CL_SUCCESS else {
        return .failure(OpenCLFailure(message: "clSetKernelArg failed"))
    }

    var global = threads
    // The global size must be a multiple of the local size.
    var local = max(1, min(d.maxWorkGroupSize, 256))
    while global % local != 0 && local > 1 { local /= 2 }

    func enqueue() -> cl_int {
        clEnqueueNDRangeKernel(queue, kernel, 1, nil, &global, &local, 0, nil, nil)
    }
    for _ in 0..<2 {
        let e = enqueue()
        if e != CL_SUCCESS { return .failure(OpenCLFailure(message: "clEnqueueNDRangeKernel failed: \(clErrorName(e))")) }
    }
    guard clFinish(queue) == CL_SUCCESS else { return .failure(OpenCLFailure(message: "clFinish after warm-up failed")) }

    let start = nowSeconds()
    for _ in 0..<iterations {
        let e = enqueue()
        if e != CL_SUCCESS { return .failure(OpenCLFailure(message: "clEnqueueNDRangeKernel failed: \(clErrorName(e))")) }
    }
    guard clFinish(queue) == CL_SUCCESS else { return .failure(OpenCLFailure(message: "clFinish failed")) }
    let elapsed = nowSeconds() - start

    let flops = Double(threads) * Double(iters) * 64.0 * Double(iterations)
    return .success(OpenCLComputeResult(flopsPerSecond: flops / elapsed, seconds: elapsed / Double(iterations),
                                       iterations: iterations, buildLog: buildLog))
}
