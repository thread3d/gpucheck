// Inventory.swift — maps the capabilities and features of each Metal GPU.
import Foundation
import Metal

/// Raw values verified against MTLDevice.h in the active SDK.
let gpuFamilyTable: [(name: String, raw: Int)] = [
    ("Apple1", 1001), ("Apple2", 1002), ("Apple3", 1003), ("Apple4", 1004),
    ("Apple5", 1005), ("Apple6", 1006), ("Apple7", 1007), ("Apple8", 1008),
    ("Apple9", 1009), ("Apple10", 1010),
    ("Mac1", 2001), ("Mac2", 2002),
    ("Common1", 3001), ("Common2", 3002), ("Common3", 3003),
    ("Metal3", 5001), ("Metal4", 5002),
]

func locationName(_ device: MTLDevice) -> String {
    switch device.location {
    case .builtIn: return "built-in"
    case .slot: return "PCIe slot \(device.locationNumber)"
    case .external: return "external (eGPU)"
    default: return "unspecified"
    }
}

func readWriteTierName(_ t: MTLReadWriteTextureTier) -> String {
    switch t {
    case .tierNone: return "none"
    case .tier1: return "tier 1"
    case .tier2: return "tier 2"
    default: return "unknown (\(t.rawValue))"
    }
}

func argumentBufferTierName(_ t: MTLArgumentBuffersTier) -> String {
    switch t {
    case .tier1: return "tier 1"
    case .tier2: return "tier 2"
    default: return "unknown (\(t.rawValue))"
    }
}

/// Pixel formats probed empirically by attempting to allocate textures with
/// each intended usage. This turns an abstract capability table into a
/// measured "works / does not work" map for this specific GPU.
/// `depthStencil` marks formats that must never be passed to
/// minimumLinearTextureAlignmentForPixelFormat, which asserts on them.
let pixelFormatProbeList: [(name: String, format: MTLPixelFormat, depthStencil: Bool)] = [
    ("rgba8Unorm", .rgba8Unorm, false),
    ("rgba8Unorm_srgb", .rgba8Unorm_srgb, false),
    ("bgra8Unorm", .bgra8Unorm, false),
    ("rgba16Float", .rgba16Float, false),
    ("rgba32Float", .rgba32Float, false),
    ("rgba16Unorm", .rgba16Unorm, false),
    ("r16Float", .r16Float, false),
    ("r32Float", .r32Float, false),
    ("rg16Float", .rg16Float, false),
    ("rg32Float", .rg32Float, false),
    ("rgb10a2Unorm", .rgb10a2Unorm, false),
    ("rg11b10Float", .rg11b10Float, false),
    ("depth32Float", .depth32Float, true),
    ("depth16Unorm", .depth16Unorm, true),
    ("stencil8", .stencil8, true),
    ("depth24Unorm_stencil8", .depth24Unorm_stencil8, true),
]

struct PixelFormatCapability {
    var name: String
    var shaderReadWrite: Bool
    var renderTarget: Bool
    var msaa4: Bool
    var minLinearAlignment: Int

    var compact: String {
        var flags: [String] = []
        if shaderReadWrite { flags.append("rw") }
        if renderTarget { flags.append("rt") }
        if msaa4 { flags.append("msaa4") }
        if flags.isEmpty { return "not usable" }
        return flags.joined(separator: "+")
    }
}

func probePixelFormats(_ device: MTLDevice) -> [PixelFormatCapability] {
    var results: [PixelFormatCapability] = []
    for (name, format, isDepthStencil) in pixelFormatProbeList {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: 64, height: 64, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        let rw = device.makeTexture(descriptor: d) != nil

        let r = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: 64, height: 64, mipmapped: false)
        r.usage = [.renderTarget]
        r.storageMode = .private
        let rt = device.makeTexture(descriptor: r) != nil

        let m = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: 64, height: 64, mipmapped: false)
        m.textureType = .type2DMultisample
        m.sampleCount = 4
        m.usage = [.renderTarget]
        m.storageMode = .private
        let msaa = device.makeTexture(descriptor: m) != nil

        results.append(PixelFormatCapability(
            name: name,
            shaderReadWrite: rw,
            renderTarget: rt,
            msaa4: msaa,
            // Depth/stencil formats have no linear layout, and querying
            // their alignment trips an internal assertion.
            minLinearAlignment: isDepthStencil ? 0 : device.minimumLinearTextureAlignment(for: format)
        ))
    }
    return results
}

/// Builds the per-device Metal capability sections for the report.
func metalInventory(_ device: MTLDevice, pixelFormats: [PixelFormatCapability]) -> [Section] {
    var sections: [Section] = []

    // Identity
    sections.append(Section(title: "Identity", items: [
        KV(key: "Name", value: device.name),
        KV(key: "Registry ID", value: "\(device.registryID)"),
        KV(key: "Location", value: locationName(device)),
        KV(key: "Headless", value: device.isHeadless ? "yes" : "no"),
        KV(key: "Low power", value: device.isLowPower ? "yes" : "no"),
        KV(key: "Removable", value: device.isRemovable ? "yes" : "no"),
        KV(key: "Unified memory", value: device.hasUnifiedMemory ? "yes" : "no (discrete)"),
    ]))

    // Memory
    let hb = device.heapBufferSizeAndAlign(length: 1 << 20, options: .storageModePrivate)
    let ht = device.heapTextureSizeAndAlign(descriptor: {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 256, height: 256, mipmapped: false)
        d.usage = [.shaderRead]
        d.storageMode = .private
        return d
    }())
    let workingSet = device.recommendedMaxWorkingSetSize
    let singleBuffer = device.maxBufferLength
    sections.append(Section(title: "Memory & resources", items: [
        KV(key: "Dedicated video memory", value: Fmt.bytes(workingSet), note: "recommendedMaxWorkingSetSize"),
        KV(key: "Max single buffer", value: Fmt.bytes(singleBuffer)),
        KV(key: "Max threadgroup memory", value: Fmt.bytes(device.maxThreadgroupMemoryLength)),
        KV(key: "Buffer alignment", value: "\(hb.align) B", note: "1 MB heap alloc"),
        KV(key: "Texture alignment", value: "\(ht.align) B", note: "256x256 heap texture"),
        KV(key: "Sparse tile size", value: device.sparseTileSizeInBytes > 0 ? Fmt.bytes(device.sparseTileSizeInBytes) : "not supported"),
        KV(key: "Current allocated", value: Fmt.bytes(device.currentAllocatedSize)),
    ]))

    // Compute capabilities
    let tpt = device.maxThreadsPerThreadgroup
    let sampleCounts = (1...8).filter { device.supportsTextureSampleCount($0) }
    sections.append(Section(title: "Compute capabilities", items: [
        KV(key: "Max threads / threadgroup", value: "\(tpt.width) x \(tpt.height) x \(tpt.depth)"),
        KV(key: "Max threads per group (1D)", value: "\(tpt.width)"),
        KV(key: "Argument buffer tier", value: argumentBufferTierName(device.argumentBuffersSupport)),
        KV(key: "Max arg-buffer samplers", value: "\(device.maxArgumentBufferSamplerCount)"),
        KV(key: "Read-write texture tier", value: readWriteTierName(device.readWriteTextureSupport)),
        KV(key: "Raster order groups", value: device.areRasterOrderGroupsSupported ? "yes" : "no"),
        KV(key: "Programmable sample positions", value: device.areProgrammableSamplePositionsSupported ? "yes" : "no"),
        KV(key: "Texture MSAA sample counts", value: sampleCounts.map(String.init).joined(separator: ", ")),
        KV(key: "BC texture compression", value: device.supportsBCTextureCompression ? "yes" : "no"),
        KV(key: "Depth24Stencil8 pixel format", value: device.isDepth24Stencil8PixelFormatSupported ? "yes" : "no"),
    ]))

    // Feature families / modern features
    let families = gpuFamilyTable
        .filter { MTLGPUFamily(rawValue: $0.raw).map { device.supportsFamily($0) } ?? false }
        .map { $0.name }
    sections.append(Section(title: "Feature families", items: [
        KV(key: "GPU families", value: families.isEmpty ? "none reported" : families.joined(separator: ", ")),
        KV(key: "Metal ray tracing", value: device.supportsRaytracing ? "yes" : "no"),
        KV(key: "Ray tracing from render", value: device.supportsRaytracingFromRender ? "yes" : "no"),
        KV(key: "Primitive motion blur", value: device.supportsPrimitiveMotionBlur ? "yes" : "no"),
        KV(key: "Function pointers", value: device.supportsFunctionPointers ? "yes" : "no"),
        KV(key: "Function pointers from render", value: device.supportsFunctionPointersFromRender ? "yes" : "no"),
        KV(key: "Dynamic libraries", value: device.supportsDynamicLibraries ? "yes" : "no"),
    ]))

    // Counters
    let counterItems: [KV]
    if let sets = device.counterSets, !sets.isEmpty {
        counterItems = sets.map { KV(key: $0.name, value: "\($0.counters.count) counters", note: nil) }
    } else {
        counterItems = [KV(key: "Counter sets", value: "none reported", note: nil)]
    }
    sections.append(Section(title: "Performance counters", items: counterItems))

    // Empirical pixel format map
    let usable = pixelFormats.filter { $0.shaderReadWrite || $0.renderTarget }
    let items = usable.map { pf -> KV in
        KV(key: pf.name, value: pf.compact + (pf.minLinearAlignment > 0 ? "  linear-align \(pf.minLinearAlignment) B" : ""))
    }
    sections.append(Section(title: "Pixel formats (empirically probed)", items: items))

    return sections
}
