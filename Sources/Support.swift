// Support.swift — shared data model, formatting helpers, and live progress reporting.
import Foundation

// MARK: - Result model

enum Status: String, Codable {
    case pass, fail, warn, info, unsupported, skipped

    var glyph: String {
        switch self {
        case .pass: return "PASS"
        case .fail: return "FAIL"
        case .warn: return "WARN"
        case .info: return "info"
        case .unsupported: return "n/a "
        case .skipped: return "skip"
        }
    }
}

struct KV: Codable {
    var key: String
    var value: String
    var note: String?
}

struct Section: Codable {
    var title: String
    var items: [KV]
}

struct Check: Codable {
    var group: String
    var name: String
    var status: Status
    var summary: String
    var details: [String] = []
    var seconds: Double? = nil
}

struct Measurement: Codable {
    var group: String
    var name: String
    var value: Double
    var unit: String
    var higherIsBetter: Bool = true
    var formatted: String
    var notes: [String] = []
}

struct DeviceReport: Codable {
    var index: Int
    var name: String
    var subtitle: String
    var sections: [Section] = []
    var checks: [Check] = []
    var measurements: [Measurement] = []
}

struct HostInfo: Codable {
    var model: String
    var cpu: String
    var physicalCores: Int
    var logicalCores: Int
    var memoryBytes: UInt64
    var osVersion: String
    var osBuild: String
    var architecture: String
}

struct RunReport: Codable {
    var tool: String = "gpucheck"
    var version: String = "1.0"
    var timestamp: String = ""
    var mode: String = ""
    var durationSeconds: Double = 0
    var host: HostInfo
    var devices: [DeviceReport] = []
    var crossDeviceMeasurements: [Measurement] = []
    var crossDeviceChecks: [Check] = []
    var warnings: [String] = []
    var notes: [String] = []
    /// True while reports are being written mid-run; if the machine hangs, the
    /// last partial report on disk still holds everything measured so far.
    var partial: Bool = false
}

// MARK: - Formatting

enum Fmt {
    static func bytes(_ v: Double) -> String {
        let units = ["B", "KB", "MB", "GB", "TB"]
        var x = v
        var i = 0
        while x >= 1024 && i < units.count - 1 { x /= 1024; i += 1 }
        return String(format: x >= 100 || i == 0 ? "%.0f %@" : "%.1f %@", x, units[i])
    }

    static func bytes(_ v: UInt64) -> String { bytes(Double(v)) }

    static func bytes(_ v: Int) -> String { bytes(Double(v)) }

    static func seconds(_ v: Double) -> String {
        if v < 1e-6 { return String(format: "%.0f ns", v * 1e9) }
        if v < 1e-3 { return String(format: "%.1f us", v * 1e6) }
        if v < 1 { return String(format: "%.2f ms", v * 1e3) }
        return String(format: "%.2f s", v)
    }

    /// Throughput in GB/s from a byte count and a duration.
    static func gbs(bytes: Double, seconds: Double) -> Double { bytes / seconds / 1e9 }

    static func rate(_ v: Double, unit: String) -> String {
        if v >= 1000 { return String(format: "%.2f T%@", v / 1000, unit) }
        if v >= 1 { return String(format: "%.2f G%@", v, unit) }
        return String(format: "%.1f M%@", v * 1000, unit)
    }

    static func count(_ v: Double) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.maximumFractionDigits = 0
        return f.string(from: NSNumber(value: v)) ?? String(format: "%.0f", v)
    }

    static func align(_ s: String, _ width: Int) -> String {
        if s.count >= width { return s }
        return s + String(repeating: " ", count: width - s.count)
    }

    static func alignLeft(_ s: String, _ width: Int) -> String {
        if s.count >= width { return s }
        return String(repeating: " ", count: width - s.count) + s
    }

    static func truncate(_ s: String, _ width: Int) -> String {
        if s.count <= width { return s }
        return String(s.prefix(max(0, width - 1))) + "\u{2026}"
    }
}

// MARK: - Crash-resistant journal

/// Appends a line to disk and fsyncs it before returning.
///
/// GPU work can hang a machine hard enough to require a power cycle. When that
/// happens nothing buffered in memory survives, so each step is recorded on disk
/// first: after a hang the journal shows exactly which test was running and
/// which one never completed.
final class Journal {
    private let handle: FileHandle?
    private let lock = NSLock()
    private let start = Date()
    let path: String

    init(path: String) {
        self.path = path
        if !FileManager.default.fileExists(atPath: path) {
            FileManager.default.createFile(atPath: path, contents: nil)
        }
        handle = FileHandle(forWritingAtPath: path)
        handle?.seekToEndOfFile()
    }

    func log(_ message: String) {
        lock.lock()
        defer { lock.unlock() }
        guard let handle else { return }
        let elapsed = Date().timeIntervalSince(start)
        let stamp = String(format: "+%8.3fs", elapsed)
        if let data = "\(stamp)  \(message)\n".data(using: .utf8) {
            handle.write(data)
            // Force the record to stable storage so it survives a hard reset.
            try? handle.synchronize()
        }
    }

    func close() {
        lock.lock()
        defer { lock.unlock() }
        try? handle?.close()
    }
}

// MARK: - Duty cycling

/// Pauses between GPU workloads.
///
/// On a machine whose GPUs also drive the displays, back-to-back full-load
/// dispatches can starve WindowServer of the GPU. A short gap between tests
/// keeps the compositor responsive at negligible cost to the measurements.
func rest(milliseconds: Int) {
    guard milliseconds > 0 else { return }
    usleep(useconds_t(milliseconds) * 1000)
}

// MARK: - Environment risk assessment

struct EnvironmentRisk {
    var gpuResetReportCount = 0
    var gpuResetsToday = 0
    var resetsByModel: [String: Int] = [:]
    var mostRecentReset: String?
    var windowServerHangCount = 0
    var displayAttachedModels: Set<String> = []

    var hasHistory: Bool { gpuResetReportCount > 0 || windowServerHangCount > 0 }
}

/// Which GPU models currently drive a display, parsed from system_profiler.
func displayAttachedGPUModels() -> Set<String> {
    guard let text = shell("/usr/sbin/system_profiler", ["SPDisplaysDataType"]) else { return [] }
    var attached = Set<String>()
    var current: String?
    var hasDisplay = false
    func flush() {
        if let model = current, hasDisplay { attached.insert(model) }
        current = nil
        hasDisplay = false
    }
    for raw in text.split(separator: "\n", omittingEmptySubsequences: false) {
        let line = String(raw).trimmingCharacters(in: .whitespaces)
        if line.hasPrefix("Chipset Model:") {
            flush()
            current = line.replacingOccurrences(of: "Chipset Model:", with: "").trimmingCharacters(in: .whitespaces)
        } else if line.hasPrefix("Displays:") || line.hasPrefix("Online: Yes") {
            hasDisplay = true
        }
    }
    flush()
    return attached
}

/// Reads the system's own GPU-reset and display-hang history.
///
/// A machine that has already been resetting its GPU under load is a machine
/// where an aggressive benchmark can hang the whole system, so this history is
/// surfaced before any heavy work is attempted.
func assessEnvironment() -> EnvironmentRisk {
    var risk = EnvironmentRisk()
    let dir = "/Library/Logs/DiagnosticReports"
    let df = DateFormatter()
    df.dateFormat = "yyyy-MM-dd"
    let todayPrefix = df.string(from: Date())

    guard let files = try? FileManager.default.contentsOfDirectory(atPath: dir) else {
        risk.displayAttachedModels = displayAttachedGPUModels()
        return risk
    }

    // The report stamps look like "Thu Sep 17 14:32:37 2026".
    let stampFormat = DateFormatter()
    stampFormat.locale = Locale(identifier: "en_US_POSIX")
    stampFormat.dateFormat = "EEE MMM d HH:mm:ss yyyy"
    var mostRecent: Date?

    for name in files {
        if name.contains("userspace_watchdog_timeout") && name.hasSuffix(".spin") {
            risk.windowServerHangCount += 1
            continue
        }
        guard name.hasSuffix(".gpuRestart") else { continue }
        risk.gpuResetReportCount += 1
        if name.contains(todayPrefix) { risk.gpuResetsToday += 1 }
        guard let text = try? String(contentsOfFile: dir + "/" + name, encoding: .utf8) else { continue }
        for raw in text.split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("Graphics Hardware:") {
                let model = line.replacingOccurrences(of: "Graphics Hardware:", with: "").trimmingCharacters(in: .whitespaces)
                if !model.isEmpty { risk.resetsByModel[model, default: 0] += 1 }
            } else if line.hasPrefix("Date/Time:") {
                let stamp = line.replacingOccurrences(of: "Date/Time:", with: "").trimmingCharacters(in: .whitespaces)
                // Keep the genuinely latest stamp, not the last file enumerated.
                if let date = stampFormat.date(from: stamp), mostRecent == nil || date > mostRecent! {
                    mostRecent = date
                    risk.mostRecentReset = stamp
                } else if mostRecent == nil, !stamp.isEmpty {
                    risk.mostRecentReset = stamp
                }
            }
        }
    }
    risk.displayAttachedModels = displayAttachedGPUModels()
    return risk
}

// MARK: - Console output

final class Console {
    let useColor: Bool
    private let lock = NSLock()

    init(useColor: Bool) { self.useColor = useColor }

    private func wrap(_ code: String, _ s: String) -> String {
        useColor ? "\u{1B}[\(code)m\(s)\u{1B}[0m" : s
    }

    func bold(_ s: String) -> String { wrap("1", s) }
    func dim(_ s: String) -> String { wrap("2", s) }
    func red(_ s: String) -> String { wrap("31", s) }
    func green(_ s: String) -> String { wrap("32", s) }
    func yellow(_ s: String) -> String { wrap("33", s) }
    func blue(_ s: String) -> String { wrap("34", s) }
    func magenta(_ s: String) -> String { wrap("35", s) }
    func cyan(_ s: String) -> String { wrap("36", s) }
    func grey(_ s: String) -> String { wrap("90", s) }

    func statusColor(_ s: Status) -> String {
        switch s {
        case .pass: return green(s.glyph)
        case .fail: return red(s.glyph)
        case .warn: return yellow(s.glyph)
        case .info: return blue(s.glyph)
        case .unsupported: return grey(s.glyph)
        case .skipped: return grey(s.glyph)
        }
    }

    func out(_ s: String) {
        lock.lock(); defer { lock.unlock() }
        print(s)
        fflush(stdout)
    }

    func rule(_ width: Int = 78, char: String = "\u{2500}") -> String {
        String(repeating: char, count: width)
    }

    func header(_ title: String) {
        out("")
        out(bold(cyan(title)))
        out(cyan(rule(title.count)))
    }
}

// MARK: - Live progress reporting

/// Prints one aligned line per completed step, flushed immediately so the
/// application provides continuous feedback while a long suite is running.
final class Progress {
    private let console: Console
    var journal: Journal?
    private(set) var stepCount = 0
    private(set) var currentPhase = ""
    private var phaseStart = Date()
    private var failed = 0
    private var warned = 0

    init(console: Console) { self.console = console }

    func beginPhase(_ title: String) {
        currentPhase = title
        phaseStart = Date()
        journal?.log("PHASE  \(title)")
        console.out("")
        console.out(console.bold(console.blue("\u{25B6} " + title)))
    }

    func beginDevice(_ index: Int, total: Int, name: String, subtitle: String) {
        console.out("")
        console.out("  \(console.bold("[\(index + 1)/\(total)] \(name)")) \(console.grey("(\(subtitle))"))")
    }

    /// Records that a named step is about to run. If the machine dies during
    /// this step, the journal's last line names it.
    func willRun(_ name: String) {
        journal?.log("START  \(name)")
    }

    /// Reports a finished check with its measured duration.
    func step(_ name: String, status: Status, summary: String, seconds: Double? = nil, indent: Int = 2) {
        stepCount += 1
        journal?.log("END    \(name)  [\(status.rawValue)] \(summary)")
        if status == .fail { failed += 1 }
        if status == .warn { warned += 1 }
        let pad = String(repeating: "  ", count: indent)
        let label = Fmt.align(Fmt.truncate(name, 30), 30)
        var line = "\(pad)\(console.grey("\u{251C}\u{2500}")) \(console.statusColor(status))  \(label) \(summary)"
        if let s = seconds, s > 0.0005 {
            line += console.grey("  [\(Fmt.seconds(s))]")
        }
        console.out(line)
    }

    func note(_ text: String, indent: Int = 3) {
        console.out("\(String(repeating: "  ", count: indent))\(console.grey("\u{2502}  " + text))")
    }

    /// Reports a measurement line (a benchmark number, not a pass/fail check).
    func measure(_ name: String, value: String, detail: String? = nil, indent: Int = 2) {
        stepCount += 1
        journal?.log("MEASURE \(name) = \(value)")
        let pad = String(repeating: "  ", count: indent)
        let label = Fmt.align(Fmt.truncate(name, 30), 30)
        var line = "\(pad)\(console.grey("\u{251C}\u{2500}")) \(console.statusColor(.info))  \(label) \(console.bold(value))"
        if let d = detail, !d.isEmpty { line += console.grey("  " + d) }
        console.out(line)
    }

    var summaryLine: String {
        let failPart = failed > 0 ? console.red("\(failed) failed") : console.green("0 failed")
        let warnPart = warned > 0 ? console.yellow("\(warned) warned") : "0 warned"
        return "\(stepCount) checks: \(failPart), \(warnPart)"
    }

    var failureCount: Int { failed }
    var warningCount: Int { warned }
}

// MARK: - Utilities

func nowSeconds() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) / 1e9
}

func measureSeconds<T>(_ body: () throws -> T) rethrows -> (value: T, seconds: Double) {
    let t0 = nowSeconds()
    let v = try body()
    return (v, nowSeconds() - t0)
}

enum GPUError: Error, CustomStringConvertible {
    case message(String)
    case noDevice
    case noCommandQueue
    case noCommandBuffer
    case noEncoder
    case missingFunction(String)

    var description: String {
        switch self {
        case .message(let m): return m
        case .noDevice: return "no Metal device available"
        case .noCommandQueue: return "could not create command queue"
        case .noCommandBuffer: return "could not create command buffer"
        case .noEncoder: return "could not create compute command encoder"
        case .missingFunction(let f): return "missing pipeline function \(f)"
        }
    }
}

/// Deterministic pseudo-random generator so runs are reproducible.
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

func shell(_ launchPath: String, _ args: [String]) -> String? {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: launchPath)
    p.arguments = args
    let pipe = Pipe()
    p.standardOutput = pipe
    p.standardError = FileHandle.nullDevice
    do { try p.run() } catch { return nil }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
}

func sysctlString(_ name: String) -> String {
    shell("/usr/sbin/sysctl", ["-n", name]) ?? "unknown"
}

func hostInfo() -> HostInfo {
    var h = HostInfo(
        model: sysctlString("hw.model"),
        cpu: sysctlString("machdep.cpu.brand_string"),
        physicalCores: Int(sysctlString("hw.physicalcpu")) ?? 0,
        logicalCores: Int(sysctlString("hw.logicalcpu")) ?? 0,
        memoryBytes: UInt64(sysctlString("hw.memsize")) ?? 0,
        osVersion: sysctlString("kern.osproductversion"),
        osBuild: sysctlString("kern.osversion"),
        architecture: sysctlString("hw.machine")
    )
    if h.model == "unknown" || h.model.isEmpty { h.model = "Mac" }
    return h
}
