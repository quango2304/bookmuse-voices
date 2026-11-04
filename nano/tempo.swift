// The app's AudioTempo.stretch (iOS PhononPace.swift) on the Mac, for A/B clips: the system's time-pitch unit, offline.
//
//     swiftc -O tempo.swift -o ../work/nano/tempo && ../work/nano/tempo <factor> <in.wav> <out.wav> [<in.wav> <out.wav> ...]
import AVFoundation

func stretch(_ samples: [Float], rate: Int, factor: Double) -> [Float] {
    guard !samples.isEmpty, abs(factor - 1) > 0.01 else { return samples }
    let sampleRate = Double(rate)
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!
    let engine = AVAudioEngine()
    let player = AVAudioPlayerNode()
    let pitch = AVAudioUnitTimePitch()
    pitch.rate = Float(factor)
    pitch.pitch = 0
    engine.attach(player)
    engine.attach(pitch)
    engine.connect(player, to: pitch, format: format)
    engine.connect(pitch, to: engine.mainMixerNode, format: format)
    let block: AVAudioFrameCount = 4096
    try! engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: block)
    try! engine.start()
    defer { engine.stop() }
    let padded = samples.count + Int(0.3 * sampleRate)
    let input = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(padded))!
    let data = input.floatChannelData![0]
    let chunk = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: block)!
    input.frameLength = AVAudioFrameCount(padded)
    samples.withUnsafeBufferPointer { data.update(from: $0.baseAddress!, count: samples.count) }
    (data + samples.count).update(repeating: 0, count: padded - samples.count)
    player.scheduleBuffer(input, completionHandler: nil)
    player.play()
    let skip = Int(pitch.latency * sampleRate)
    let wanted = Int(Double(samples.count) / factor)
    var out: [Float] = []
    while out.count < wanted + skip {
        guard let status = try? engine.renderOffline(block, to: chunk), status == .success || status == .insufficientDataFromInputNode,
              chunk.frameLength > 0, let channel = chunk.floatChannelData?[0]
        else { break }
        out.append(contentsOf: UnsafeBufferPointer(start: channel, count: Int(chunk.frameLength)))
    }
    guard out.count > skip else { return samples }
    return Array(out[skip ..< min(out.count, skip + wanted)])
}

let args = CommandLine.arguments
let factor = Double(args[1])!
for i in stride(from: 2, to: args.count - 1, by: 2) {
    let file = try! AVAudioFile(forReading: URL(fileURLWithPath: args[i]), commonFormat: .pcmFormatFloat32, interleaved: false)
    let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
    try! file.read(into: buffer)
    let samples = Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
    let out = stretch(samples, rate: Int(file.processingFormat.sampleRate), factor: factor)
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: file.processingFormat.sampleRate, channels: 1, interleaved: false)!
    let outBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(out.count))!
    outBuffer.frameLength = AVAudioFrameCount(out.count)
    out.withUnsafeBufferPointer { outBuffer.floatChannelData![0].update(from: $0.baseAddress!, count: out.count) }
    let writer = try! AVAudioFile(forWriting: URL(fileURLWithPath: args[i + 1]), settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
    try! writer.write(from: outBuffer)
}
