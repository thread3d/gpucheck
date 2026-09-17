// main.swift — gpucheck: comprehensive GPU testing, verification and feature mapping.
import Foundation
import Metal
import OpenCL

// MARK: - Command line

struct Options {
    var mode = "standard"
    var outDir = "gpucheck-report"
    var deviceSelection: [Int] = []
    var color = true
    var includeOpenCL = true
    var includeRender = true
    var listOnly = false
    var sustainOverride: Double? = nil
    var bandwidthOverride: Int? = nil
}

let usage = """
gpucheck — comprehensive GPU test, verification and feature mapping

USAGE
  gpucheck [options]

OPTIONS
  --mode quick|standard|thorough   Test depth. standard is the default.
                                     quick    ~30 s, minimal sustained load
                                     standard benchmarks + 8 s sustained per GPU
                                     thorough more repetitions + 20 s sustained per GPU
  --device N[,N...]                Only test the given (1-based) GPU indices.
  --out DIR                        Directory for reports (default: gpucheck-report).
  --bandwidth MB                   Working-set size for memory benchmarks.
  --sustained SECONDS              Override the sustained-load duration per GPU.
  --no-opencl                      Skip the OpenCL cross-check.
  --no-render                      Skip graphics-pipeline tests.
  --no-color                       Disable ANSI colour.
  --list                           List detected GPUs and exit.
  -h, --help                       Show this help.

OUTPUT
  Console progress while running, then a summary plus JSON, Markdown and HTML
  reports written to the output directory.
"""

func parseOptions(_ args: [String]) -> Options? {
    var o = Options()
    var i = 0
    func next(_ name: String) -> String? {
        i += 1
        guard i < args.count else { print("error: \(name) requires a value"); return nil }
        return args[i]
    }
    while i < args.count {
        let a = args[i]
        switch a {
        case "--mode":
            guard let v = next(a) else { return nil }
            guard ["quick", "standard", "thorough"].contains(v) else {
                print("error: unknown mode '\(v)'"); return nil
            }
            o.mode = v
        case "--out":
            guard let v = next(a) else { return nil }
            o.outDir = v
        case "--device":
            guard let v = next(a) else { return nil }
            let parts = v.split(separator: ",").map(String.init)
            for p in parts {
                guard let n = Int(p), n > 0 else { print("error: bad device index '\(p)'"); return nil }
                o.deviceSelection.append(n - 1)
            }
        case "--bandwidth":
            guard let v = next(a), let n = Int(v), n > 0 else { print("error: bad --bandwidth value"); return nil }
            o.bandwidthOverride = n
        case "--sustained":
            guard let v = next(a), let d = Double(v), d >= 0 else { print("error: bad --sustained value"); return nil }
            o.sustainOverride = d
        case "--no-opencl": o.includeOpenCL = false
        case "--no-render": o.includeRender = false
        case "--no-color": o.color = false
        case "--list": o.listOnly = true
        case "-h", "--help": print(usage); exit(0)
        default:
            print("error: unknown option '\(a)'\n")
            print(usage)
            return nil
        }
        i += 1
    }
    return o
}

// MARK: - Entry point

let rawArgs = Array(CommandLine.arguments.dropFirst())
guard var options = parseOptions(rawArgs) else { exit(64) }

let console = Console(useColor: options.color && isatty(STDOUT_FILENO) == 1)
let progress = Progress(console: console)
let runStart = nowSeconds()

console.out("")
console.out(console.bold(console.cyan("gpucheck")) + console.grey("  v1.0  —  GPU test, verification and feature map"))
console.out(console.grey("  " + console.rule(60)))

let host = hostInfo()

// MARK: - Device discovery

let allMetalDevices = MTLCopyAllDevices()
guard !allMetalDevices.isEmpty else {
    console.out(console.red("No Metal devices found on this system."))
    exit(1)
}

let selectedMetal: [MTLDevice]
if options.deviceSelection.isEmpty {
    selectedMetal = allMetalDevices
} else {
    selectedMetal = options.deviceSelection.compactMap { idx in
        (idx >= 0 && idx < allMetalDevices.count) ? allMetalDevices[idx] : nil
    }
    if selectedMetal.isEmpty {
        console.out(console.red("No valid device indices; system has \(allMetalDevices.count) GPUs."))
        exit(1)
    }
}

// OpenCL cross-check enumeration (independent driver stack).
var openCLDevices: [OpenCLDevice] = []
if options.includeOpenCL {
    openCLDevices = enumerateOpenCLDevices()
}

func matchOpenCL(_ metalName: String) -> OpenCLDevice? {
    let target = metalName.lowercased().trimmingCharacters(in: .whitespaces)
    if let exact = openCLDevices.first(where: { $0.baseName.lowercased() == target }) { return exact }
    return openCLDevices.first { $0.baseName.lowercased().contains(target) || target.contains($0.baseName.lowercased()) }
}

// MARK: - List mode

if options.listOnly {
    console.out("")
    console.out(console.bold("Detected GPUs (\(allMetalDevices.count) Metal devices)"))
    for (i, d) in allMetalDevices.enumerated() {
        let ocl = matchOpenCL(d.name)
        let cu = ocl.map { "\($0.computeUnits) CUs" } ?? "CUs n/a"
        let mem = ocl.map { Fmt.bytes($0.globalMemBytes) } ?? Fmt.bytes(d.recommendedMaxWorkingSetSize)
        let slot = d.location == .slot ? "slot \(d.locationNumber)" : locationName(d)
        console.out("  [\(i + 1)] \(d.name)")
        console.out(console.grey("       \(slot) · \(mem) · \(cu) · Metal 3 \(d.supportsFamily(MTLGPUFamily(rawValue: 5001) ?? .common3) ? "yes" : "no")"))
    }
    if !openCLDevices.isEmpty {
        console.out("")
        console.out(console.bold("OpenCL devices (\(openCLDevices.count))"))
        for (i, d) in openCLDevices.enumerated() {
            console.out("  [\(i + 1)] \(d.name)  \(console.grey("CU \(d.computeUnits) @ \(d.maxClockMHz) MHz, \(Fmt.bytes(d.globalMemBytes))"))")
        }
    }
    exit(0)
}

// MARK: - Suite configuration

var config = SuiteConfig.make(mode: options.mode)
config.includeOpenCL = options.includeOpenCL
config.includeRender = options.includeRender
if let s = options.sustainOverride {
    config.sustainedSeconds = s
    config.includeSustained = s > 0
}
if let b = options.bandwidthOverride { config.bandwidthMB = b }

console.out("")
console.out("  \(console.grey("Host"))    \(host.model) · \(host.cpu)")
console.out("  \(console.grey("Memory"))  \(Fmt.bytes(host.memoryBytes)) · \(host.physicalCores) cores")
console.out("  \(console.grey("OS"))      macOS \(host.osVersion) (\(host.osBuild))")
console.out("  \(console.grey("GPUs"))    \(selectedMetal.count)" + (selectedMetal.count != allMetalDevices.count ? " of \(allMetalDevices.count) selected" : ""))
if openCLDevices.count != allMetalDevices.count && options.includeOpenCL {
    console.out(console.grey("  note: OpenCL reports \(openCLDevices.count) GPU device(s), Metal reports \(allMetalDevices.count)"))
}
console.out("  \(console.grey("Mode"))    \(config.mode)")

// MARK: - Per-device suite

var deviceReports: [DeviceReport] = []
var contexts: [MetalContext] = []
var reportWarnings: [String] = []
var displayAttached = false

for (index, device) in selectedMetal.enumerated() {
    let subtitle = "\(locationName(device)) · \(Fmt.bytes(device.recommendedMaxWorkingSetSize)) · Metal 3: \(device.supportsFamily(MTLGPUFamily(rawValue: 5001) ?? .common3) ? "yes" : "no")"
    progress.beginDevice(index, total: selectedMetal.count, name: device.name, subtitle: subtitle)

    let ctx: MetalContext
    do {
        ctx = try MetalContext(device: device)
    } catch {
        console.out(console.red("  could not create a Metal command queue: \(error)"))
        deviceReports.append(DeviceReport(index: index + 1, name: device.name, subtitle: subtitle,
                                          checks: [Check(group: "Setup", name: "Command queue", status: .fail,
                                                         summary: "\(error)")]))
        continue
    }
    contexts.append(ctx)

    // Feature mapping -------------------------------------------------------
    progress.beginPhase("Feature mapping")
    progress.note("probing pixel formats and compiling shader groups…")
    let pixelFormats = probePixelFormats(device)
    ctx.calibrate()
    var sections = metalInventory(device, pixelFormats: pixelFormats)

    let oclMatch = options.includeOpenCL ? matchOpenCL(device.name) : nil
    if let ocl = oclMatch {
        sections.append(contentsOf: openCLSections(ocl))
    } else if options.includeOpenCL {
        sections.append(Section(title: "OpenCL", items: [
            KV(key: "Status", value: "no matching OpenCL device for this GPU"),
        ]))
        reportWarnings.append("\(device.name): no matching OpenCL device found for cross-checking")
    }

    let compileReports = ctx.compileReports()
    sections.append(Section(title: "Shader compilation (runtime MSL)", items: compileReports.map { r in
        KV(key: r.group, value: r.ok ? "compiled" : "FAILED", note: r.error)
    }))

    let identity = Dictionary(uniqueKeysWithValues: sections.first { $0.title == "Identity" }?.items.map { ($0.key, $0.value) } ?? [])
    let memoryItems = Dictionary(uniqueKeysWithValues: sections.first { $0.title == "Memory & resources" }?.items.map { ($0.key, $0.value) } ?? [])
    let familyItems = Dictionary(uniqueKeysWithValues: sections.first { $0.title == "Feature families" }?.items.map { ($0.key, $0.value) } ?? [])
    let cuText = oclMatch.map { "\($0.computeUnits)" } ?? "unknown"
    let clkText = oclMatch.map { "\($0.maxClockMHz) MHz" } ?? "unknown"

    var mapChecks: [Check] = []
    mapChecks.append(Check(group: "Mapping", name: "GPU identity",
                           status: .info,
                           summary: "\(device.name) at \(identity["Location"] ?? "?") · registry \(device.registryID)",
                           details: ["video memory \(memoryItems["Dedicated video memory"] ?? "?")",
                                     "max single buffer \(memoryItems["Max single buffer"] ?? "?")",
                                     "unified memory: \(identity["Unified memory"] ?? "?")"]))
    mapChecks.append(Check(group: "Mapping", name: "Hardware mapping",
                           status: oclMatch != nil ? .info : .warn,
                           summary: oclMatch != nil
                              ? "\(cuText) compute units @ \(clkText) (via OpenCL cross-check)"
                              : "compute-unit count unavailable (OpenCL did not report this GPU)",
                           details: ["Metal exposes no CU or clock query; OpenCL is used as the second source",
                                     "GPU families: \(familyItems["GPU families"] ?? "-")",
                                     "SIMD width: \(ctx.threadExecutionWidth) lanes",
                                     "max threads/threadgroup: \(ctx.maxTotalThreadsPerThreadgroup)"]))
    let failedGroups = compileReports.filter { !$0.ok }
    let optionalKeys = Set(shaderGroups.filter { $0.optional }.map { $0.key })
    let failedRequired = failedGroups.filter { !optionalKeys.contains($0.group) }
    let failedOptional = failedGroups.filter { optionalKeys.contains($0.group) }
    var shaderStatus: Status = .pass
    var shaderSummary = "all \(compileReports.count) MSL shader groups compiled on this GPU"
    if !failedRequired.isEmpty {
        shaderStatus = .fail
        shaderSummary = "\(failedRequired.count) required shader group(s) failed: \(failedRequired.map { $0.group }.joined(separator: ", "))"
    } else if !failedOptional.isEmpty {
        shaderSummary = "core shaders compiled; optional group(s) unavailable: \(failedOptional.map { $0.group }.joined(separator: ", "))"
    }
    mapChecks.append(Check(group: "Mapping", name: "Shader groups compiled",
                           status: shaderStatus,
                           summary: shaderSummary,
                           details: compileReports.map { r in
                               let tag = optionalKeys.contains(r.group) ? " (optional)" : ""
                               return "\(r.group)\(tag): \(r.ok ? "ok" : (r.error ?? "failed"))"
                           }))

    for c in mapChecks {
        progress.step(c.name, status: c.status, summary: c.summary)
        for d in c.details { progress.note(d) }
    }

    // Functional tests ------------------------------------------------------
    let tests = FunctionalTests(ctx: ctx, config: config)
    let checks = tests.runAll(progress: progress)

    // Performance -----------------------------------------------------------
    progress.beginPhase("Performance")
    let bench = Benchmarks(ctx: ctx, config: config, openCLMatch: oclMatch)
    bench.runAll(progress: progress)
    for m in bench.measurements {
        progress.measure(m.name, value: m.formatted, detail: m.notes.first)
    }
    for c in bench.checks {
        progress.step(c.name, status: c.status, summary: c.summary, seconds: c.seconds)
    }

    let deviceReport = DeviceReport(index: index + 1, name: device.name, subtitle: subtitle,
                                    sections: sections,
                                    checks: mapChecks + checks + bench.checks,
                                    measurements: tests.measurements + bench.measurements)
    if device.isHeadless == false && device.location == .slot {
        // On a Mac Pro every MPX GPU can drive displays; note it for the report.
        displayAttached = true
    }
    deviceReports.append(deviceReport)
}

// MARK: - Multi-GPU concurrency

var crossMeasurements: [Measurement] = []
var crossChecks: [Check] = []

if contexts.count > 1 && config.includeConcurrency {
    progress.beginPhase("Multi-GPU concurrency")
    progress.note("measuring each GPU alone, then all \(contexts.count) GPUs simultaneously…")
    let outcome = MultiGPU.measure(contexts: contexts, config: config)

    let soloSumF = outcome.soloFLOPS.reduce(0, +)
    let soloList = zip(contexts, outcome.soloFLOPS).map { String(format: "%@: %.2f", $0.0.name, $0.1 / 1e12) }.joined(separator: ", ")
    let efficiencyPct = outcome.efficiency * 100

    crossMeasurements.append(Measurement(
        group: "Multi-GPU", name: "Sum of single-GPU rates", value: soloSumF / 1e12, unit: "TFLOP/s",
        formatted: String(format: "%.2f TFLOP/s", soloSumF / 1e12),
        notes: ["measured one GPU at a time", soloList]))
    crossMeasurements.append(Measurement(
        group: "Multi-GPU", name: "Concurrent aggregate", value: outcome.concurrentFLOPS / 1e12, unit: "TFLOP/s",
        formatted: String(format: "%.2f TFLOP/s", outcome.concurrentFLOPS / 1e12),
        notes: [String(format: "all GPUs busy at once for %.3f s", outcome.wallSeconds)]))
    crossMeasurements.append(Measurement(
        group: "Multi-GPU", name: "Parallel scaling efficiency", value: efficiencyPct, unit: "%",
        formatted: String(format: "%.1f%%", efficiencyPct),
        notes: ["concurrent aggregate divided by the sum of single-GPU rates",
                "100% means the GPUs run fully in parallel with no contention"]))
    if !outcome.soloCopyGBs.isEmpty {
        let copySoloSum = outcome.soloCopyGBs.reduce(0, +)
        crossMeasurements.append(Measurement(
            group: "Multi-GPU", name: "Sum of single-GPU copy", value: copySoloSum, unit: "GB/s",
            formatted: String(format: "%.1f GB/s", copySoloSum),
            notes: ["device-local copy measured one GPU at a time"]))
        crossMeasurements.append(Measurement(
            group: "Multi-GPU", name: "Concurrent copy aggregate", value: outcome.concurrentCopyGBs, unit: "GB/s",
            formatted: String(format: "%.1f GB/s", outcome.concurrentCopyGBs),
            notes: [String(format: "copy scaling efficiency %.1f%%", outcome.copyEfficiency * 100)]))
    }

    let scaleStatus: Status = efficiencyPct > 90 ? .pass : (efficiencyPct > 70 ? .warn : .fail)
    crossChecks.append(Check(
        group: "Multi-GPU", name: "Parallel execution across all GPUs", status: scaleStatus,
        summary: String(format: "%.2f TFLOP/s aggregate across %d GPUs = %.1f%% of the sum of single-GPU rates",
                        outcome.concurrentFLOPS / 1e12, contexts.count, efficiencyPct),
        details: [
            "single-GPU rates: \(soloList)",
            String(format: "concurrent wall time for the same work on all GPUs: %.4f s", outcome.wallSeconds),
            efficiencyPct > 90
                ? "the GPUs execute fully in parallel - this machine can use all of its GPU resources at once"
                : "scaling below 90% suggests shared power, thermal or host-bandwidth limits",
        ],
        seconds: outcome.wallSeconds))
    for c in crossChecks {
        progress.step(c.name, status: c.status, summary: c.summary, seconds: c.seconds)
        for d in c.details { progress.note(d) }
    }
    for m in crossMeasurements {
        progress.measure(m.name, value: m.formatted, detail: m.notes.first)
    }
} else if contexts.count > 1 {
    reportWarnings.append("multi-GPU concurrency test disabled by configuration")
}

// MARK: - Report assembly

let stamp = ISO8601DateFormatter().string(from: Date())
var report = RunReport(timestamp: stamp, mode: config.mode,
                       durationSeconds: nowSeconds() - runStart,
                       host: host,
                       devices: deviceReports,
                       crossDeviceMeasurements: crossMeasurements,
                       crossDeviceChecks: crossChecks,
                       warnings: reportWarnings,
                       notes: [
                        "Every functional check compares GPU output against an independent host-side reference computed on the CPU.",
                        "Throughput figures are best-of-N repetitions to reduce noise from background activity.",
                        "Host<->device transfer rates are estimates: measured host memcpy time is subtracted from the round trip.",
                        "The theoretical FP32 peak uses the compute-unit count and clock reported by OpenCL; Metal exposes neither.",
                        "Sustained-load tests compare the first and last quarter of a continuous run to expose thermal throttling.",
                        "Reports are written as JSON, Markdown and a self-contained HTML file.",
                       ])
if displayAttached {
    report.notes.append("At least one tested GPU also drives a display, so brief visual stutter is expected during the sustained-load phase.")
}

consoleReport(report, console: console)

// MARK: - Write outputs

let outDir = URL(fileURLWithPath: options.outDir)
do {
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    var written: [String] = []
    if let json = jsonReport(report) {
        let u = outDir.appendingPathComponent("gpucheck-report.json")
        try json.write(to: u)
        written.append(u.path)
    }
    let md = markdownReport(report)
    let mdURL = outDir.appendingPathComponent("gpucheck-report.md")
    try md.write(to: mdURL, atomically: true, encoding: .utf8)
    written.append(mdURL.path)

    let html = htmlReport(report)
    let htmlURL = outDir.appendingPathComponent("gpucheck-report.html")
    try html.write(to: htmlURL, atomically: true, encoding: .utf8)
    written.append(htmlURL.path)

    console.header("REPORTS WRITTEN")
    for w in written { console.out("  \(console.green("\u{2713}")) \(w)") }
    console.out("")
} catch {
    console.out(console.red("Could not write reports to \(options.outDir): \(error)"))
    exit(1)
}

let allChecks = report.devices.flatMap { $0.checks } + report.crossDeviceChecks
exit(allChecks.contains { $0.status == .fail } ? 2 : 0)
