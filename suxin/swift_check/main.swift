// Golden check for the app's SuxinText: every line of the golden files (made by `frontend.py golden`) must come out identical.
// usage: swift_check <pack folder> <golden.json> [<golden.json> ...]
import Foundation

struct Line: Decodable { let text: String; let ids: [Int32]; let chars: [Int] }

let args = CommandLine.arguments
guard args.count >= 3 else { print("usage: swift_check <pack folder> <golden.json>..."); exit(2) }

func now() -> Double { Date().timeIntervalSinceReferenceDate }
var t0 = now()
guard let frontend = SuxinText(folder: URL(fileURLWithPath: args[1])) else { print("could not load the pack"); exit(2) }
print(String(format: "load: %.1f ms", (now() - t0) * 1000))

var failed = false
for path in args[2...] {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)), let lines = try? JSONDecoder().decode([Line].self, from: data) else {
        print("cannot read \(path)"); exit(2)
    }
    var same = 0, encodeTime = 0.0, characters = 0
    for (n, line) in lines.enumerated() {
        let t = now()
        let (ids, utf16) = frontend.encode(line.text)
        encodeTime += now() - t
        characters += line.text.unicodeScalars.count
        // UTF-16 offsets back to code point indices, as the Python gives them
        var index: [Int: Int] = [:], offset = 0
        for (i, c) in line.text.unicodeScalars.enumerated() { index[offset] = i; offset += c.value > 0xFFFF ? 2 : 1 }
        let chars = utf16.map { $0 < 0 ? -1 : (index[$0] ?? -2) }
        if ids == line.ids && chars == line.chars { same += 1; continue }
        failed = true
        print("DIFF line \(n + 1): \(line.text)")
        if ids != line.ids {
            let k = (0..<min(ids.count, line.ids.count)).first { ids[$0] != line.ids[$0] } ?? min(ids.count, line.ids.count)
            func ctx(_ a: [Int32]) -> String { a[max(0, k - 6)..<min(a.count, k + 6)].map(String.init).joined(separator: " ") }
            print("  ids differ at \(k) (swift \(ids.count) ids, python \(line.ids.count))")
            print("  swift : \(ctx(ids))")
            print("  python: \(ctx(line.ids))")
        } else {
            let k = (0..<chars.count).first { chars[$0] != line.chars[$0] } ?? 0
            print("  chars differ at \(k): swift \(chars[max(0, k - 3)..<min(chars.count, k + 4)]) python \(line.chars[max(0, k - 3)..<min(line.chars.count, k + 4)])")
        }
    }
    print(String(format: "%@: %d of %d identical; %d encodes in %.1f ms (%.2f ms each, %.3f ms per character)", (path as NSString).lastPathComponent,
                 same, lines.count, lines.count, encodeTime * 1000, encodeTime * 1000 / Double(lines.count), encodeTime * 1000 / Double(max(characters, 1))))
}
exit(failed ? 1 : 0)
