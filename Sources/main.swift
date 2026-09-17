// main.swift — gpucheck: comprehensive GPU testing, verification and feature mapping.
import Foundation
import Metal
import OpenCL

// MARK: - Command line

struct Options {
    var mode = "standard"
    var intensity = Intensity.safe
    var allowHeavy = false
    var outDir = "gpucheck-report"
    var deviceSelection: [Int] = []
    var deviceIDSelection: [UInt64] = []
    var deviceNameSelection: [String] = []
    var skipDisplayGPUs = false
    var color = true
    var noOpenCL = false
    var noRender = false
    var listOnly = false
    var dryRun = false
    var sustainOverride: Double? = nil
    var bandwidthOverride: Int? = nil
    var journalPath: String? = nil
    var noJournal = false
}

let usage = """
gpucheck — comprehensive GPU test, verification and feature mapping

USAGE
  gpucheck [options]

SAFETY
  This tool drives the GPUs hard. On a machine whose GPUs also drive the
  displays, that can starve WindowServer and hang the whole system, and a GPU
  with a history of driver resets can be pushed into one. The defaults are
  therefore conservative: --intensity safe runs every correctness test and every
  benchmark with reduced workloads and a cooldown between them, and skips the
  two riskiest tests (sustained full-occupancy load and driving every GPU at
  once). Those two require --allow-heavy.

  Every step is written to a journal and fsync'd before the GPU is touched, and
  reports are rewritten after each GPU completes, so if the machine does hang the
  journal names the test that caused it and the partial report survives a reboot.

OPTIONS
  --mode quick|standard|thorough   How deep the run goes. standard is default.
  --intensity safe|normal|full     How hard any single workload may push the GPU.
                                     safe (default)  small workloads, 250 ms cooldown,
                                                     no sustained, no multi-GPU load,
                                                     no multi-gigabyte allocations
                                     normal          larger workloads, 120 ms cooldown
                                     full            largest workloads, 60 ms cooldown
  --allow-heavy                    Enable sustained-load and simultaneous all-GPU
                                     tests. These are the tests most likely to hang
                                     a marginal machine.
  --device N[,N...]                Only test the given (1-based) GPU indices.
                                     NOTE: Metal device order is not stable across
                                     reboots. Prefer --device-id for reproducible
                                     runs.
  --device-name TEXT[,TEXT...]     Only test GPUs whose name contains TEXT
                                     (case-insensitive). GPU names are the only
                                     identifier stable across reboots.
  --device-id ID[,ID...]           Only test GPUs with the given Metal registry IDs
                                     (see --list). Stable within a boot, but macOS
                                     may renumber them after a reboot.
  --skip-display-gpus              Skip GPUs that currently drive a display. This
                                     is the lowest-risk way to exercise the machine,
                                     because display GPUs are the ones whose load
                                     can starve WindowServer.
  --out DIR                        Report directory (default: gpucheck-report).
  --journal FILE                   Journal path (default: <out>/gpucheck-journal.log).
  --no-journal                     Do not write a journal.
  --bandwidth MB                   Override the memory benchmark working set.
  --sustained SECONDS              Override sustained duration (needs --allow-heavy).
  --no-opencl                      Skip the OpenCL cross-check.
  --no-render                      Skip graphics-pipeline tests.
  --no-color                       Disable ANSI colour.
  --list                           List detected GPUs and exit.
  --dry-run                        Show the selected GPUs and the workload sizes
                                     that would run, then exit without touching
                                     the GPU.
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
        case "--device-id":
            guard let v = next(a) else { return nil }
            for p in v.split(separator: ",").map(String.init) {
                guard let n = UInt64(p) else { print("error: bad registry id '\(p)'"); return nil }
                o.deviceIDSelection.append(n)
            }
        case "--device-name":
            guard let v = next(a) else { return nil }
            o.deviceNameSelection.append(contentsOf: v.split(separator: ",").map { String($0).lowercased() })
        case "--skip-display-gpus": o.skipDisplayGPUs = true
        case "--intensity":
            guard let v = next(a) else { return nil }
            guard let intensity = Intensity(rawValue: v) else {
                print("error: unknown intensity '\(v)' (expected safe, normal or full)"); return nil
            }
            o.intensity = intensity
        case "--allow-heavy": o.allowHeavy = true
        case "--out":
            guard let v = next(a) else { return nil }
            o.outDir = v
        case "--journal":
            guard let v = next(a) else { return nil }
            o.journalPath = v
        case "--no-journal": o.noJournal = true
        case "--device":
            guard let v = next(a) else { return nil }
            for p in v.split(separator: ",").map(String.init) {
                guard let n = Int(p), n > 0 else { print("error: bad device index '\(p)'"); return nil }
                o.deviceSelection.append(n - 1)
            }
        case "--bandwidth":
            guard let v = next(a), let n = Int(v), n > 0 else { print("error: bad --bandwidth value"); return nil }
            o.bandwidthOverride = n
        case "--sustained":
            guard let v = next(a), let d = Double(v), d >= 0 else { print("error: bad --sustained value"); return nil }
            o.sustainOverride = d
        case "--no-opencl": o.noOpenCL = true
        case "--no-render": o.noRender = true
        case "--no-color": o.color = false
        case "--list": o.listOnly = true
        case "--dry-run": o.dryRun = true
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
guard let options = parseOptions(rawArgs) else { exit(64) }

let console = Console(useColor: options.color && isatty(STDOUT_FILENO) == 1)
let progress = Progress(console: console)
let outDir = URL(fileURLWithPath: options.outDir)
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

// Neither the device order nor the registry IDs survive a reboot, so matching on
// the GPU name is the reproducible way to pick a device across sessions.
let displayModels = displayAttachedGPUModels()

var selectedMetal: [MTLDevice]
if !options.deviceNameSelection.isEmpty {
    selectedMetal = allMetalDevices.filter { d in
        let name = d.name.lowercased()
        return options.deviceNameSelection.contains { name.contains($0) }
    }
    if selectedMetal.isEmpty {
        console.out(console.red("No GPU name matches '\(options.deviceNameSelection.joined(separator: ", "))'."))
        exit(1)
    }
} else if !options.deviceIDSelection.isEmpty {
    selectedMetal = allMetalDevices.filter { options.deviceIDSelection.contains($0.registryID) }
    if selectedMetal.isEmpty {
        console.out(console.red("No GPU matches registry id(s) \(options.deviceIDSelection.map(String.init).joined(separator: ", "))."))
        exit(1)
    }
} else if !options.deviceSelection.isEmpty {
    selectedMetal = options.deviceSelection.compactMap { idx in
        (idx >= 0 && idx < allMetalDevices.count) ? allMetalDevices[idx] : nil
    }
    if selectedMetal.isEmpty {
        console.out(console.red("No valid device indices; system has \(allMetalDevices.count) GPUs."))
        exit(1)
    }
} else {
    selectedMetal = allMetalDevices
}

if options.skipDisplayGPUs {
    let kept = selectedMetal.filter { !displayModels.contains($0.name) }
    if kept.isEmpty {
        console.out(console.red("--skip-display-gpus would leave no GPUs to test (every detected GPU drives a display)."))
        exit(1)
    }
    selectedMetal = kept
}

// Test GPUs that drive no display first. A display GPU is the one whose load can
// starve WindowServer, so ordering it last means the journalled results and
// partial reports from the safer GPUs are already on disk if it hangs the system.
let requestedOrder = selectedMetal
selectedMetal = selectedMetal.enumerated().sorted { a, b in
    let aDisplay = displayModels.contains(a.element.name)
    let bDisplay = displayModels.contains(b.element.name)
    if aDisplay != bDisplay { return !aDisplay }
    return a.offset < b.offset
}.map { $0.element }
if selectedMetal.map({ $0.registryID }) != requestedOrder.map({ $0.registryID }) {
    console.out("")
    console.out(console.grey("  note: reordered so non-display GPUs are tested first (a display GPU is the one that can hang the system)"))
}

var openCLDevices: [OpenCLDevice] = []
if !options.noOpenCL {
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
    let displays = displayModels
    for (i, d) in allMetalDevices.enumerated() {
        let ocl = matchOpenCL(d.name)
        let cu = ocl.map { "\($0.computeUnits) CUs" } ?? "CUs n/a"
        let mem = ocl.map { Fmt.bytes($0.globalMemBytes) } ?? Fmt.bytes(d.recommendedMaxWorkingSetSize)
        let slot = d.location == .slot ? "slot \(d.locationNumber)" : locationName(d)
        console.out("  [\(i + 1)] \(d.name)")
        console.out(console.grey("       registry \(d.registryID) · \(slot) · \(mem) · \(cu) · Metal 3 \(d.supportsFamily(MTLGPUFamily(rawValue: 5001) ?? .common3) ? "yes" : "no")\(displays.contains(d.name) ? " · drives a display" : "")"))
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

// MARK: - Environment risk assessment

let risk = assessEnvironment()

// MARK: - Journal

var journal: Journal?
if !options.noJournal {
    let path = options.journalPath ?? outDir.appendingPathComponent("gpucheck-journal.log").path
    try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    journal = Journal(path: path)
}
progress.journal = journal
journal?.log("gpucheck run start — mode=\(options.mode) intensity=\(options.intensity.rawValue) allowHeavy=\(options.allowHeavy)")

// MARK: - Suite configuration

var config = SuiteConfig.make(mode: options.mode, intensity: options.intensity, allowHeavy: options.allowHeavy)
config.includeOpenCL = !options.noOpenCL
config.includeRender = !options.noRender
if let s = options.sustainOverride {
    config.sustainedSeconds = s
    config.includeSustained = s > 0 && options.allowHeavy
}
if let b = options.bandwidthOverride { config.bandwidthMB = b }

// MARK: - Configuration banner

console.out("")
console.out("  \(console.grey("Host"))        \(host.model) · \(host.cpu)")
console.out("  \(console.grey("Memory"))      \(Fmt.bytes(host.memoryBytes)) · \(host.physicalCores) cores")
console.out("  \(console.grey("OS"))          macOS \(host.osVersion) (\(host.osBuild))")
console.out("  \(console.grey("GPUs"))        \(selectedMetal.count)" + (selectedMetal.count != allMetalDevices.count ? " of \(allMetalDevices.count) selected" : ""))
console.out("  \(console.grey("Mode"))        \(config.mode)   \(console.grey("intensity \(config.intensity.rawValue)"))")
console.out("  \(console.grey("Workloads"))   \(Fmt.bytes(config.bandwidthMB << 20)) memory · \(Fmt.count(Double(config.computeThreads))) threads · \(config.cooldownMs) ms cooldown")
if let j = journal {
    console.out("  \(console.grey("Journal"))     \(j.path)")
}

if risk.hasHistory {
    console.out("")
    console.out("  \(console.yellow("! system stability history"))")
    if risk.gpuResetReportCount > 0 {
        let models = risk.resetsByModel.sorted { $0.value > $1.value }
            .map { "\($0.key) ×\($0.value)" }.joined(separator: ", ")
        console.out(console.grey("      GPU reset reports: \(risk.gpuResetReportCount) (\(risk.gpuResetsToday) today)\(models.isEmpty ? "" : " — \(models)")"))
        if let last = risk.mostRecentReset {
            console.out(console.grey("      most recent reset: \(last)"))
        }
    }
    if risk.windowServerHangCount > 0 {
        console.out(console.grey("      WindowServer hang reports: \(risk.windowServerHangCount)"))
    }
    if !risk.displayAttachedModels.isEmpty {
        console.out(console.grey("      display-driving GPUs: \(risk.displayAttachedModels.sorted().joined(separator: ", "))"))
    }
    console.out(console.yellow("      Heavy GPU load on this machine has previously reset the GPU and hung the"))
    console.out(console.yellow("      display. Staying at --intensity \(config.intensity.rawValue)\(config.allowHeavy ? "" : "; sustained and all-GPU tests are off")."))
}
if !config.allowHeavy {
    console.out("")
    console.out(console.grey("  Heavy tests (sustained load, simultaneous all-GPU load) are disabled. Add --allow-heavy to enable."))
}

// MARK: - Dry run

if options.dryRun {
    console.out("")
    console.out(console.bold("Selected GPUs (\(selectedMetal.count))"))
    for (i, d) in selectedMetal.enumerated() {
        let tag = displayModels.contains(d.name) ? console.yellow(" · drives a display") : ""
        console.out("  [\(i + 1)] \(d.name)  \(console.grey("registry \(d.registryID) · \(locationName(d))"))\(tag)")
    }
    console.out("")
    console.out(console.bold("Planned workload"))
    console.out("  memory working set      \(Fmt.bytes(config.bandwidthMB << 20))")
    console.out("  compute threads         \(Fmt.count(Double(config.computeThreads)))")
    console.out("  loop iterations         \(config.computeIters)")
    console.out("  benchmark repetitions   \(config.attempts)")
    console.out("  cooldown between tests  \(config.cooldownMs) ms")
    console.out("  large-buffer test       \(config.largeBufferBytes > 0 ? Fmt.bytes(config.largeBufferBytes) : "skipped")")
    console.out("  sustained load          \(config.includeSustained ? String(format: "%.0f s", config.sustainedSeconds) : "skipped")")
    console.out("  multi-GPU concurrency   \(config.includeConcurrency ? "enabled" : "skipped")")
    console.out("  OpenCL cross-check      \(config.includeOpenCL ? "enabled" : "disabled")")
    console.out("  graphics tests          \(config.includeRender ? "enabled" : "disabled")")
    if !config.allowHeavy {
        console.out("")
        console.out(console.grey("  Sustained and multi-GPU tests need --allow-heavy."))
    }
    console.out("")
    exit(0)
}

// MARK: - Report scaffolding

var deviceReports: [DeviceReport] = []
var contexts: [MetalContext] = []
var crossMeasurements: [Measurement] = []
var crossChecks: [Check] = []
var reportWarnings: [String] = []

if risk.gpuResetReportCount > 0 {
    reportWarnings.append("this system has \(risk.gpuResetReportCount) GPU reset report(s) on file (\(risk.gpuResetsToday) today); "
        + "GPU driver resets under load are a pre-existing condition on this machine and are not necessarily caused by this run")
}
if risk.windowServerHangCount > 0 {
    reportWarnings.append("\(risk.windowServerHangCount) WindowServer watchdog hang report(s) on file; heavy GPU load can starve the display compositor on this system")
}
if !risk.displayAttachedModels.isEmpty {
    reportWarnings.append("displays are driven by: \(risk.displayAttachedModels.sorted().joined(separator: ", ")); brief visual stutter is expected during GPU load")
}

var reportNotes: [String] = [
    "Every functional check compares GPU output against an independent host-side reference computed on the CPU.",
    "Throughput figures are best-of-N repetitions to reduce noise from background activity.",
    "Host<->device transfer figures time only the blit between system and GPU memory, so no host staging correction is needed.",
    "The theoretical FP32 peak uses the compute-unit count and clock reported by OpenCL; Metal exposes neither.",
    "Sustained-load tests compare the first and last quarter of a continuous run to expose thermal throttling.",
    "Reports are written as JSON, Markdown and a self-contained HTML file.",
    "Reports are rewritten after each GPU finishes, so a partial report survives a system hang.",
]

/// Builds the report as it currently stands, so it can be written mid-run.
func currentReport(partial: Bool) -> RunReport {
    var r = RunReport(timestamp: ISO8601DateFormatter().string(from: Date()),
                      mode: config.mode,
                      durationSeconds: nowSeconds() - runStart,
                      host: host,
                      devices: deviceReports,
                      crossDeviceMeasurements: crossMeasurements,
                      crossDeviceChecks: crossChecks,
                      warnings: reportWarnings,
                      notes: reportNotes)
    r.partial = partial
    return r
}

// MARK: - Per-device suite

for (index, device) in selectedMetal.enumerated() {
    let subtitle = "\(locationName(device)) · \(Fmt.bytes(device.recommendedMaxWorkingSetSize)) · Metal 3: \(device.supportsFamily(MTLGPUFamily(rawValue: 5001) ?? .common3) ? "yes" : "no")"
        + (risk.displayAttachedModels.contains(device.name) ? " · drives a display" : "")
    progress.beginDevice(index, total: selectedMetal.count, name: device.name, subtitle: subtitle)
    journal?.log("DEVICE \(index + 1)/\(selectedMetal.count) \(device.name)")

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
    progress.willRun("pixel format probe")
    progress.note("probing pixel formats and compiling shader groups…")
    let pixelFormats = probePixelFormats(device)
    ctx.calibrate()
    var sections = metalInventory(device, pixelFormats: pixelFormats)

    let oclMatch = options.noOpenCL ? nil : matchOpenCL(device.name)
    if let ocl = oclMatch {
        sections.append(contentsOf: openCLSections(ocl))
    } else if !options.noOpenCL {
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
    mapChecks.append(Check(group: "Mapping", name: "GPU identity", status: .info,
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
    mapChecks.append(Check(group: "Mapping", name: "Shader groups compiled", status: shaderStatus,
                           summary: shaderSummary,
                           details: compileReports.map { r in
                               let tag = optionalKeys.contains(r.group) ? " (optional)" : ""
                               return "\(r.group)\(tag): \(r.ok ? "ok" : (r.error ?? "failed"))"
                           }))

    for c in mapChecks {
        progress.step(c.name, status: c.status, summary: c.summary)
        for d in c.details { progress.note(d) }
    }
    rest(milliseconds: config.cooldownMs)

    // Functional tests ------------------------------------------------------
    let tests = FunctionalTests(ctx: ctx, config: config)
    let checks = tests.runAll(progress: progress)

    // Performance -----------------------------------------------------------
    let bench = Benchmarks(ctx: ctx, config: config, openCLMatch: oclMatch)
    bench.runAll(progress: progress)
    for m in bench.measurements {
        progress.measure(m.name, value: m.formatted, detail: m.notes.first)
    }
    for c in bench.checks {
        progress.step(c.name, status: c.status, summary: c.summary, seconds: c.seconds)
    }

    deviceReports.append(DeviceReport(index: index + 1, name: device.name, subtitle: subtitle,
                                      sections: sections,
                                      checks: mapChecks + checks + bench.checks,
                                      measurements: tests.measurements + bench.measurements))

    // Persist what we have before moving on to the next GPU.
    do {
        try writeReports(currentReport(partial: true), to: outDir)
        journal?.log("partial reports written after device \(index + 1)")
        console.out(console.grey("  partial reports written to \(outDir.path) (survives a hang)"))
    } catch {
        console.out(console.yellow("  could not write partial reports: \(error)"))
    }
    rest(milliseconds: config.cooldownMs * 2)
}

// MARK: - Multi-GPU concurrency

if contexts.count > 1 && config.includeConcurrency {
    progress.beginPhase("Multi-GPU concurrency")
    progress.willRun("multi-GPU concurrency")
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
    journal?.log("DONE   multi-GPU concurrency")
} else if contexts.count > 1 {
    let reason = config.allowHeavy
        ? "disabled by configuration"
        : "skipped: requires --allow-heavy (driving every GPU at once is the highest-risk test on this class of machine)"
    reportWarnings.append("multi-GPU concurrency test \(reason)")
    console.out("")
    console.out(console.grey("  multi-GPU concurrency \(reason)"))
    journal?.log("SKIP   multi-GPU concurrency — \(reason)")
}

// MARK: - Report assembly and output

let report = currentReport(partial: false)
consoleReport(report, console: console)
journal?.log("run complete — \(report.devices.count) device(s)")

do {
    let written = try writeReports(report, to: outDir)
    console.header("REPORTS WRITTEN")
    for w in written { console.out("  \(console.green("\u{2713}")) \(w)") }
    if let j = journal?.path {
        console.out("  \(console.green("\u{2713}")) \(j)")
    }
    console.out("")
} catch {
    console.out(console.red("Could not write reports to \(options.outDir): \(error)"))
    journal?.log("ERROR writing reports: \(error)")
    journal?.close()
    exit(1)
}
journal?.close()

let allChecks = report.devices.flatMap { $0.checks } + report.crossDeviceChecks
exit(allChecks.contains { $0.status == .fail } ? 2 : 0)
