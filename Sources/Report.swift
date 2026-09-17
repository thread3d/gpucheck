// Report.swift — renders the collected results as console output, JSON,
// Markdown and a self-contained HTML report.
import Foundation

// MARK: - Console rendering

func renderTable(headers: [String], rows: [[String]], console: Console, indent: Int = 0) -> String {
    guard !rows.isEmpty else { return "" }
    var widths = headers.map { $0.count }
    for row in rows {
        for (i, cell) in row.enumerated() where i < widths.count {
            widths[i] = max(widths[i], cell.count)
        }
    }
    let pad = String(repeating: " ", count: indent)
    var out = ""
    var head = pad
    for (i, h) in headers.enumerated() {
        head += i == 0 ? Fmt.align(h, widths[i]) : "  " + Fmt.align(h, widths[i])
    }
    out += console.bold(head) + "\n"
    out += pad + console.grey(widths.map { String(repeating: "\u{2500}", count: $0) }.joined(separator: "  ")) + "\n"
    for row in rows {
        var line = pad
        for (i, cell) in row.enumerated() {
            let c = i < widths.count ? Fmt.align(cell, widths[i]) : cell
            line += i == 0 ? c : "  " + c
        }
        out += line.trimmingCharacters(in: .whitespaces) + "\n"
    }
    return out
}

/// The headline metrics compared across all GPUs.
let comparisonMetrics: [(name: String, short: String)] = [
    ("FP32 FMA", "FP32"),
    ("FP16 FMA", "FP16"),
    ("FP64 FMA", "FP64"),
    ("Copy bandwidth (r+w)", "copy GB/s"),
    ("Read bandwidth (read)", "read GB/s"),
    ("Fill bandwidth (write)", "fill GB/s"),
    ("Texture sampling", "tex Gs/s"),
    ("Dispatch round trip", "latency us"),
    ("Host to device", "H2D GB/s"),
    ("Sustained vs burst", "sustained %"),
]

func consoleReport(_ report: RunReport, console: Console) {
    console.header("GPU DIAGNOSTIC REPORT")
    let h = report.host
    console.out("  \(console.grey("Host"))      \(h.model)  \(h.cpu)")
    console.out("  \(console.grey("CPU"))       \(h.physicalCores) physical / \(h.logicalCores) logical cores")
    console.out("  \(console.grey("Memory"))    \(Fmt.bytes(h.memoryBytes))")
    console.out("  \(console.grey("OS"))        macOS \(h.osVersion) (\(h.osBuild))  \(h.architecture)")
    console.out("  \(console.grey("GPUs"))      \(report.devices.count) Metal devices")
    console.out("  \(console.grey("Mode"))      \(report.mode)   \(console.grey(String(format: "total runtime %.1f s", report.durationSeconds)))")

    // Per-device summary
    for device in report.devices {
        console.out("")
        console.out(console.bold(console.cyan("\u{2588} GPU \(device.index): \(device.name)")))
        console.out(console.grey("  \(device.subtitle)"))

        // Key identity/memory facts
        var facts: [String] = []
        for section in device.sections where ["Identity", "Memory & resources", "OpenCL"].contains(section.title) {
            for item in section.items where [
                "Dedicated video memory", "Max single buffer", "Unified memory", "Location",
                "Compute units", "Max clock", "GPU families", "Max threads / threadgroup",
            ].contains(item.key) {
                facts.append("\(console.grey(item.key))=\(item.value)")
            }
        }
        if !facts.isEmpty {
            console.out("  " + facts.joined(separator: "  "))
        }

        // Measurements
        let byGroup = Dictionary(grouping: device.measurements, by: { $0.group })
        for group in ["Compute", "Memory", "Transfer", "Latency", "Timing", "OpenCL", "Sustained"] {
            guard let items = byGroup[group], !items.isEmpty else { continue }
            let rows = items.map { m -> [String] in
                [m.name, m.formatted, m.notes.first ?? ""]
            }
            console.out("")
            console.out("  \(console.bold(group))")
            console.out(renderTable(headers: ["Metric", "Value", "Note"], rows: rows, console: console, indent: 4))
        }

        // Checks
        let problems = device.checks.filter { $0.status == .fail || $0.status == .warn || $0.status == .unsupported }
        let passed = device.checks.filter { $0.status == .pass }.count
        console.out("")
        console.out("  \(console.bold("Verification"))  \(console.green("\(passed) passed"))" +
                    (problems.isEmpty ? "" : ", \(problems.count) to review") +
                    "  \(console.grey("of \(device.checks.count) checks"))")
        for c in device.checks {
            let mark: String
            switch c.status {
            case .pass: mark = console.green("\u{2713}")
            case .fail: mark = console.red("\u{2717}")
            case .warn: mark = console.yellow("!")
            case .unsupported: mark = console.grey("-")
            case .skipped: mark = console.grey("-")
            case .info: mark = console.blue("i")
            }
            if c.status == .pass {
                console.out("    \(mark) \(Fmt.align(Fmt.truncate(c.name, 30), 30)) \(console.grey(c.summary))")
            } else {
                console.out("    \(mark) \(Fmt.align(Fmt.truncate(c.name, 30), 30)) \(c.summary)")
            }
        }
    }

    // Cross-device comparison
    console.header("COMPARISON ACROSS GPUs")
    var headers = ["Metric"]
    for d in report.devices { headers.append(Fmt.truncate(d.name.replacingOccurrences(of: "AMD Radeon ", with: ""), 20)) }
    var rows: [[String]] = []
    for (name, _) in comparisonMetrics {
        var row = [name]
        var any = false
        for d in report.devices {
            if let m = d.measurements.first(where: { $0.name == name }) {
                row.append(m.formatted)
                any = true
            } else {
                row.append("-")
            }
        }
        if any { rows.append(row) }
    }
    if report.devices.count > 1 {
        let sumRow = ["Sum (all GPUs)"] + report.devices.map { _ in "" }
        rows.append(sumRow)
    }
    console.out(renderTable(headers: headers, rows: rows, console: console, indent: 2))

    // Multi-GPU scaling
    if !report.crossDeviceMeasurements.isEmpty || !report.crossDeviceChecks.isEmpty {
        console.out("")
        console.out(console.bold(console.cyan("MULTI-GPU CONCURRENCY")))
        for c in report.crossDeviceChecks {
            let mark = c.status == .pass ? console.green("PASS") : (c.status == .warn ? console.yellow("WARN") : console.red("FAIL"))
            console.out("  \(mark)  \(c.name): \(c.summary)")
            for d in c.details { console.out(console.grey("        " + d)) }
        }
        let rows = report.crossDeviceMeasurements.map { [$0.name, $0.formatted, $0.notes.first ?? ""] }
        console.out(renderTable(headers: ["Metric", "Value", "Note"], rows: rows, console: console, indent: 2))
    }

    // Totals
    let allChecks = report.devices.flatMap { $0.checks } + report.crossDeviceChecks
    let fails = allChecks.filter { $0.status == .fail }
    let warns = allChecks.filter { $0.status == .warn }
    let unsupported = allChecks.filter { $0.status == .unsupported }
    let passed = allChecks.filter { $0.status == .pass }
    console.header("SUMMARY")
    console.out("  Checks run:      \(allChecks.count)")
    console.out("  Passed:          \(console.green("\(passed.count)"))")
    console.out("  Warnings:        \(warns.isEmpty ? "0" : console.yellow("\(warns.count)"))")
    console.out("  Failed:          \(fails.isEmpty ? console.green("0") : console.red("\(fails.count)"))")
    console.out("  Not supported:   \(unsupported.isEmpty ? "0" : console.grey("\(unsupported.count)"))")
    if !fails.isEmpty {
        console.out("")
        for f in fails { console.out("  \(console.red("FAIL")) \(f.name): \(f.summary)") }
    }
    if !warns.isEmpty {
        console.out("")
        for w in warns { console.out("  \(console.yellow("WARN")) \(w.name): \(w.summary)") }
    }
    console.out("")
    let verdict = fails.isEmpty
        ? console.green("All functional checks passed on \(report.devices.count) GPU(s).")
        : console.red("\(fails.count) functional check(s) failed.")
    console.out("  \(verdict)")
}

// MARK: - JSON

func jsonReport(_ report: RunReport) -> Data? {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    return try? encoder.encode(report)
}

// MARK: - Markdown

func markdownReport(_ report: RunReport) -> String {
    var out = "# GPU Diagnostic Report\n\n"
    let h = report.host
    out += "**Host:** \(h.model) — \(h.cpu)  \n"
    out += "**CPU:** \(h.physicalCores) physical / \(h.logicalCores) logical cores  \n"
    out += "**Memory:** \(Fmt.bytes(h.memoryBytes))  \n"
    out += "**OS:** macOS \(h.osVersion) (\(h.osBuild)), \(h.architecture)  \n"
    out += "**GPUs detected:** \(report.devices.count) Metal devices  \n"
    out += "**Mode:** \(report.mode) — total runtime \(String(format: "%.1f", report.durationSeconds)) s  \n"
    out += "**Generated:** \(report.timestamp)  \n\n"

    let allChecks = report.devices.flatMap { $0.checks } + report.crossDeviceChecks
    let fails = allChecks.filter { $0.status == .fail }.count
    let warns = allChecks.filter { $0.status == .warn }.count
    let passed = allChecks.filter { $0.status == .pass }.count
    out += "## Summary\n\n"
    out += "| Result | Count |\n|---|---|\n"
    out += "| Passed | \(passed) |\n| Warnings | \(warns) |\n| Failed | \(fails) |\n"
    out += "| Not supported | \(allChecks.filter { $0.status == .unsupported }.count) |\n\n"

    if report.devices.count > 1 {
        out += "## Cross-GPU comparison\n\n"
        var headers = ["Metric"]
        for d in report.devices { headers.append(d.name) }
        out += "| " + headers.joined(separator: " | ") + " |\n"
        out += "|" + headers.map { _ in "---" }.joined(separator: "|") + "|\n"
        for (name, _) in comparisonMetrics {
            var cells = [name]
            var any = false
            for d in report.devices {
                if let m = d.measurements.first(where: { $0.name == name }) {
                    cells.append(m.formatted)
                    any = true
                } else { cells.append("-") }
            }
            if any { out += "| " + cells.joined(separator: " | ") + " |\n" }
        }
        out += "\n"
    }

    if !report.crossDeviceChecks.isEmpty {
        out += "## Multi-GPU concurrency\n\n"
        for c in report.crossDeviceChecks {
            out += "- **\(c.name)** — \(c.status.rawValue.uppercased()): \(c.summary)\n"
            for d in c.details { out += "  - \(d)\n" }
        }
        out += "\n"
        if !report.crossDeviceMeasurements.isEmpty {
            out += "| Metric | Value | Note |\n|---|---|---|\n"
            for m in report.crossDeviceMeasurements {
                out += "| \(m.name) | \(m.formatted) | \(m.notes.first ?? "") |\n"
            }
            out += "\n"
        }
    }

    for device in report.devices {
        out += "## GPU \(device.index): \(device.name)\n\n"
        out += "*\(device.subtitle)*\n\n"

        out += "### Feature map\n\n"
        for section in device.sections {
            out += "#### \(section.title)\n\n"
            out += "| Property | Value |\n|---|---|\n"
            for item in section.items {
                let note = item.note.map { " _(\($0))_" } ?? ""
                out += "| \(item.key) | \(item.value)\(note) |\n"
            }
            out += "\n"
        }

        out += "### Functional verification\n\n"
        out += "| Check | Result | Detail | Time |\n|---|---|---|---|\n"
        for c in device.checks {
            let t = c.seconds.map { Fmt.seconds($0) } ?? ""
            let summary = c.summary.replacingOccurrences(of: "|", with: "\\|")
            out += "| \(c.name) | \(c.status.rawValue.uppercased()) | \(summary) | \(t) |\n"
        }
        out += "\n"

        let detailChecks = device.checks.filter { !$0.details.isEmpty }
        if !detailChecks.isEmpty {
            out += "<details><summary>Check details</summary>\n\n"
            for c in detailChecks {
                out += "**\(c.name)** (\(c.status.rawValue))\n\n"
                for d in c.details { out += "- \(d)\n" }
                out += "\n"
            }
            out += "</details>\n\n"
        }

        out += "### Measurements\n\n"
        let byGroup = Dictionary(grouping: device.measurements, by: { $0.group })
        for group in ["Compute", "Memory", "Transfer", "Latency", "Timing", "OpenCL", "Sustained"] {
            guard let items = byGroup[group], !items.isEmpty else { continue }
            out += "#### \(group)\n\n"
            out += "| Metric | Value | Notes |\n|---|---|---|\n"
            for m in items {
                out += "| \(m.name) | \(m.formatted) | \(m.notes.joined(separator: "; ")) |\n"
            }
            out += "\n"
        }
    }

    if !report.warnings.isEmpty {
        out += "## Warnings\n\n"
        for w in report.warnings { out += "- \(w)\n" }
        out += "\n"
    }
    if !report.notes.isEmpty {
        out += "## Methodology notes\n\n"
        for n in report.notes { out += "- \(n)\n" }
        out += "\n"
    }
    return out
}

// MARK: - HTML

private func htmlEscape(_ s: String) -> String {
    s.replacingOccurrences(of: "&", with: "&amp;")
        .replacingOccurrences(of: "<", with: "&lt;")
        .replacingOccurrences(of: ">", with: "&gt;")
        .replacingOccurrences(of: "\"", with: "&quot;")
}

func htmlReport(_ report: RunReport) -> String {
    let statusClass: (Status) -> String = { s in
        switch s {
        case .pass: return "pass"
        case .fail: return "fail"
        case .warn: return "warn"
        case .unsupported, .skipped: return "na"
        case .info: return "info"
        }
    }
    let allChecks = report.devices.flatMap { $0.checks } + report.crossDeviceChecks
    let fails = allChecks.filter { $0.status == .fail }.count
    let warns = allChecks.filter { $0.status == .warn }.count
    let passed = allChecks.filter { $0.status == .pass }.count

    var html = """
    <!DOCTYPE html>
    <html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>GPU Diagnostic Report</title>
    <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 0 0 4rem; background: #14161a; color: #e6e8ec;
           font: 14px/1.55 -apple-system, "SF Pro Text", Segoe UI, Roboto, sans-serif; }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 0 1.5rem; }
    header { padding: 2.5rem 0 1.5rem; border-bottom: 1px solid #2a2f38; margin-bottom: 2rem; }
    h1 { margin: 0 0 .4rem; font-size: 1.9rem; letter-spacing: -.02em; }
    h2 { margin: 2.8rem 0 .8rem; font-size: 1.25rem; color: #9fd0ff;
         border-bottom: 1px solid #2a2f38; padding-bottom: .45rem; }
    h3 { margin: 2rem 0 .7rem; font-size: 1.05rem; }
    h4 { margin: 1.4rem 0 .5rem; font-size: .82rem; text-transform: uppercase;
         letter-spacing: .07em; color: #8b93a3; }
    .meta { color: #8b93a3; }
    .meta b { color: #e6e8ec; font-weight: 600; }
    .cards { display: flex; flex-wrap: wrap; gap: .8rem; margin: 1.2rem 0; }
    .card { background: #1b1f26; border: 1px solid #2a2f38; border-radius: 10px;
            padding: .9rem 1.1rem; min-width: 150px; }
    .card .n { font-size: 1.7rem; font-weight: 650; }
    .card .l { font-size: .74rem; text-transform: uppercase; letter-spacing: .07em; color: #8b93a3; }
    .card.ok .n { color: #4ade80; } .card.bad .n { color: #f87171; } .card.warn .n { color: #fbbf24; }
    table { border-collapse: collapse; width: 100%; margin: .6rem 0 1.2rem; font-size: .86rem; }
    th, td { text-align: left; padding: .5rem .7rem; border-bottom: 1px solid #252a33; vertical-align: top; }
    th { color: #8b93a3; font-weight: 600; font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; }
    tr:hover td { background: #1a1e25; }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    .pill { display: inline-block; padding: .1rem .5rem; border-radius: 999px;
            font-size: .7rem; font-weight: 700; letter-spacing: .04em; }
    .pill.pass { background: #14301f; color: #4ade80; }
    .pill.fail { background: #3a1719; color: #f87171; }
    .pill.warn { background: #3a2e12; color: #fbbf24; }
    .pill.na { background: #24282f; color: #8b93a3; }
    .pill.info { background: #16283a; color: #7cc4ff; }
    .gpu { background: #1b1f26; border: 1px solid #2a2f38; border-radius: 12px;
           padding: 1.2rem 1.4rem; margin: 1.4rem 0; }
    .gpu h3 { margin-top: 0; }
    .bars { margin: .4rem 0 1rem; }
    .bar-row { display: grid; grid-template-columns: 190px 1fr 110px; gap: .7rem;
               align-items: center; margin: .3rem 0; }
    .bar-label { color: #b9c0cc; font-size: .82rem; }
    .bar-track { background: #232830; border-radius: 5px; height: 20px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 5px; background: linear-gradient(90deg,#2563eb,#60a5fa); }
    .bar-val { text-align: right; font-variant-numeric: tabular-nums; font-size: .82rem; color: #e6e8ec; }
    .bars h4 { margin-top: 1.2rem; }
    code { background: #232830; padding: .08rem .35rem; border-radius: 4px; font-size: .85em; }
    details { margin: .5rem 0 1rem; } summary { cursor: pointer; color: #9fd0ff; }
    ul { padding-left: 1.3rem; } li { margin: .15rem 0; }
    .hint { color: #8b93a3; font-size: .82rem; }
    </style></head><body><div class="wrap">
    <header>
      <h1>GPU Diagnostic Report</h1>
      <div class="meta">
        <b>\(htmlEscape(report.host.model))</b> — \(htmlEscape(report.host.cpu))<br>
        \(report.host.physicalCores) physical / \(report.host.logicalCores) logical cores ·
        \(Fmt.bytes(report.host.memoryBytes)) · macOS \(htmlEscape(report.host.osVersion)) (\(htmlEscape(report.host.osBuild))) · \(htmlEscape(report.host.architecture))<br>
        \(report.devices.count) Metal GPU device(s) · mode <code>\(htmlEscape(report.mode))</code> ·
        runtime \(String(format: "%.1f", report.durationSeconds)) s · generated \(htmlEscape(report.timestamp))
      </div>
    </header>
    <div class="cards">
      <div class="card ok"><div class="n">\(passed)</div><div class="l">Passed</div></div>
      <div class="card warn"><div class="n">\(warns)</div><div class="l">Warnings</div></div>
      <div class="card \(fails > 0 ? "bad" : "ok")"><div class="n">\(fails)</div><div class="l">Failed</div></div>
      <div class="card"><div class="n">\(report.devices.count)</div><div class="l">GPUs</div></div>
    </div>
    """

    // Cross-GPU bar charts
    if report.devices.count > 1 {
        html += "<h2>Cross-GPU comparison</h2>"
        html += "<p class=\"hint\">Each bar is scaled relative to the fastest GPU for that metric. Latency metrics are inverted so longer bars are always better.</p>"
        html += "<div class=\"bars\">"
        for (name, _) in comparisonMetrics {
            let values: [(String, Double, String)] = report.devices.compactMap { d in
                guard let m = d.measurements.first(where: { $0.name == name }) else { return nil }
                return (d.name.replacingOccurrences(of: "AMD Radeon ", with: ""), m.value, m.formatted)
            }
            guard !values.isEmpty else { continue }
            html += "<h4>\(htmlEscape(name))</h4>"
            let maxVal = values.map { $0.1 }.max() ?? 1
            for (label, value, formatted) in values {
                let pct = maxVal > 0 ? max(2.0, min(100.0, value / maxVal * 100)) : 0
                html += "<div class=\"bar-row\"><div class=\"bar-label\">\(htmlEscape(Fmt.truncate(label, 26)))</div>"
                html += "<div class=\"bar-track\"><div class=\"bar-fill\" style=\"width:\(String(format: "%.1f", pct))%\"></div></div>"
                html += "<div class=\"bar-val\">\(htmlEscape(formatted))</div></div>"
            }
        }
        html += "</div>"

        // Comparison table
        var headers = ["Metric"] + report.devices.map { $0.name }
        html += "<table><thead><tr>"
        for (i, hcell) in headers.enumerated() {
            html += "<th\(i > 0 ? " class=\"num\"" : "")>\(htmlEscape(hcell))</th>"
        }
        html += "</tr></thead><tbody>"
        for (name, _) in comparisonMetrics {
            var cells = [name]
            var any = false
            for d in report.devices {
                if let m = d.measurements.first(where: { $0.name == name }) {
                    cells.append(m.formatted); any = true
                } else { cells.append("—") }
            }
            guard any else { continue }
            html += "<tr>"
            for (i, c) in cells.enumerated() {
                html += "<td\(i > 0 ? " class=\"num\"" : "")>\(htmlEscape(c))</td>"
            }
            html += "</tr>"
        }
        html += "</tbody></table>"
    }

    // Multi-GPU concurrency
    if !report.crossDeviceChecks.isEmpty || !report.crossDeviceMeasurements.isEmpty {
        html += "<h2>Multi-GPU concurrency</h2>"
        for c in report.crossDeviceChecks {
            html += "<p><span class=\"pill \(statusClass(c.status))\">\(c.status.rawValue.uppercased())</span> <b>\(htmlEscape(c.name))</b> — \(htmlEscape(c.summary))</p>"
            if !c.details.isEmpty {
                html += "<ul>" + c.details.map { "<li>\(htmlEscape($0))</li>" }.joined() + "</ul>"
            }
        }
        if !report.crossDeviceMeasurements.isEmpty {
            html += "<table><thead><tr><th>Metric</th><th class=\"num\">Value</th><th>Note</th></tr></thead><tbody>"
            for m in report.crossDeviceMeasurements {
                html += "<tr><td>\(htmlEscape(m.name))</td><td class=\"num\">\(htmlEscape(m.formatted))</td><td>\(htmlEscape(m.notes.first ?? ""))</td></tr>"
            }
            html += "</tbody></table>"
        }
    }

    // Per device
    for device in report.devices {
        html += "<div class=\"gpu\">"
        html += "<h3>GPU \(device.index): \(htmlEscape(device.name))</h3>"
        html += "<p class=\"hint\">\(htmlEscape(device.subtitle))</p>"

        // Measurements first: the headline numbers.
        let byGroup = Dictionary(grouping: device.measurements, by: { $0.group })
        let presentGroups = ["Compute", "Memory", "Transfer", "Latency", "Timing", "OpenCL", "Sustained"].filter {
            byGroup[$0]?.isEmpty == false
        }
        if !presentGroups.isEmpty {
            html += "<h4>Measured performance</h4>"
            html += "<table><thead><tr><th>Metric</th><th class=\"num\">Value</th><th>Notes</th></tr></thead><tbody>"
            for group in presentGroups {
                html += "<tr><td colspan=\"3\" style=\"color:#8b93a3;font-size:.74rem;text-transform:uppercase;letter-spacing:.06em\">\(htmlEscape(group))</td></tr>"
                for m in byGroup[group] ?? [] {
                    html += "<tr><td>\(htmlEscape(m.name))</td><td class=\"num\">\(htmlEscape(m.formatted))</td><td class=\"hint\">\(htmlEscape(m.notes.joined(separator: "; ")))</td></tr>"
                }
            }
            html += "</tbody></table>"
        }

        // Feature map
        html += "<h4>Feature map</h4>"
        for section in device.sections {
            html += "<details><summary>\(htmlEscape(section.title))</summary>"
            html += "<table><tbody>"
            for item in section.items {
                let note = item.note.map { " <span class=\"hint\">(\(htmlEscape($0)))</span>" } ?? ""
                html += "<tr><td style=\"width:34%\">\(htmlEscape(item.key))</td><td>\(htmlEscape(item.value))\(note)</td></tr>"
            }
            html += "</tbody></table></details>"
        }

        // Verification
        html += "<h4>Functional verification</h4>"
        html += "<table><thead><tr><th>Check</th><th>Result</th><th>Detail</th><th class=\"num\">Time</th></tr></thead><tbody>"
        for c in device.checks {
            let t = c.seconds.map { Fmt.seconds($0) } ?? ""
            html += "<tr><td>\(htmlEscape(c.name))</td><td><span class=\"pill \(statusClass(c.status))\">\(c.status.rawValue.uppercased())</span></td>"
            html += "<td>\(htmlEscape(c.summary))"
            if !c.details.isEmpty {
                html += "<details><summary>details</summary><ul>" + c.details.map { "<li>\(htmlEscape($0))</li>" }.joined() + "</ul></details>"
            }
            html += "</td><td class=\"num\">\(t)</td></tr>"
        }
        html += "</tbody></table></div>"
    }

    if !report.warnings.isEmpty || !report.notes.isEmpty {
        html += "<h2>Notes and caveats</h2><ul>"
        for w in report.warnings { html += "<li>\(htmlEscape(w))</li>" }
        for n in report.notes { html += "<li>\(htmlEscape(n))</li>" }
        html += "</ul>"
    }

    html += "</div></body></html>"
    return html
}

// MARK: - Writing report files

/// Writes JSON, Markdown and HTML reports to `outDir`, creating it if needed.
/// Called after every completed device so partial results survive a hang.
@discardableResult
func writeReports(_ report: RunReport, to outDir: URL) throws -> [String] {
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    var written: [String] = []
    if let json = jsonReport(report) {
        let u = outDir.appendingPathComponent("gpucheck-report.json")
        try json.write(to: u)
        written.append(u.path)
    }
    let mdURL = outDir.appendingPathComponent("gpucheck-report.md")
    try markdownReport(report).write(to: mdURL, atomically: true, encoding: .utf8)
    written.append(mdURL.path)

    let htmlURL = outDir.appendingPathComponent("gpucheck-report.html")
    try htmlReport(report).write(to: htmlURL, atomically: true, encoding: .utf8)
    written.append(htmlURL.path)
    return written
}
