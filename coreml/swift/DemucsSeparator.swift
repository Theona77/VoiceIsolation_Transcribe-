// DemucsSeparator.swift
// Runs the Demucs Core ML model (from coreml/export_demucs.py) over a whole song.
//
// The model works on fixed 7.8 s chunks (343,980 samples at 44.1 kHz). This class
// cuts the song into overlapping chunks and cross-fades them back together, the
// same way `demucs.apply.apply_model` does (overlap 0.25, triangular weights).

import Accelerate
import AVFoundation
import CoreML

public final class DemucsSeparator {
    public let model: MLModel
    public let stems: [String]
    public let segment: Int
    public let sampleRate: Double
    public let overlap: Double

    /// - Parameter modelURL: a compiled `.mlmodelc` (what Xcode puts in the app bundle
    ///   when you add the `.mlpackage` to a target) or a raw `.mlpackage`/`.mlmodel`.
    public init(modelURL: URL, computeUnits: MLComputeUnits = .all, overlap: Double = 0.25) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        var url = modelURL
        if url.pathExtension != "mlmodelc" {
            url = try MLModel.compileModel(at: url)
        }
        model = try MLModel(contentsOf: url, configuration: config)

        let meta = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        stems = (meta["sources"] ?? "vocals").components(separatedBy: ",")
        segment = Int(meta["segment_samples"] ?? "") ?? 343_980
        sampleRate = Double(meta["samplerate"] ?? "") ?? 44_100
        self.overlap = overlap
    }

    /// Convenience initializer for a model added to the app target, e.g.
    /// `try DemucsSeparator(bundleResource: "Demucs_htdemucs_ft_vocals_6bit")`.
    public convenience init(bundleResource name: String, computeUnits: MLComputeUnits = .all) throws {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw NSError(domain: "DemucsSeparator", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "\(name).mlmodelc not found in bundle"])
        }
        try self.init(modelURL: url, computeUnits: computeUnits)
    }

    /// Separates a stereo song sampled at `sampleRate` (44.1 kHz).
    /// - Returns: stem name -> (left, right) samples, same length as the input.
    public func separate(
        left: [Float], right: [Float], progress: ((Double) -> Void)? = nil
    ) throws -> [String: (left: [Float], right: [Float])] {
        precondition(left.count == right.count, "Channels must have the same length")
        let length = left.count

        // Normalize like demucs.separate: zero mean / unit std on the mono mix.
        let mono = vDSP.multiply(0.5, vDSP.add(left, right))
        let mean = vDSP.mean(mono)
        let std = max(sqrt(vDSP.meanSquare(vDSP.add(-mean, mono))), 1e-8)
        let l = vDSP.multiply(1 / std, vDSP.add(-mean, left))
        let r = vDSP.multiply(1 / std, vDSP.add(-mean, right))

        // Triangular cross-fade weights over one segment.
        let half = segment / 2
        var weight = (0..<half).map { Float($0 + 1) } + (0..<(segment - half)).map { Float(segment - half - $0) }
        weight = vDSP.divide(weight, vDSP.maximum(weight))

        let nStems = stems.count
        var out = [[Float]](repeating: [Float](repeating: 0, count: length), count: nStems * 2)
        var sumWeight = [Float](repeating: 0, count: length)
        let stride = max(1, Int((1 - overlap) * Double(segment)))

        let input = try MLMultiArray(shape: [1, 2, NSNumber(value: segment)], dataType: .float32)
        let inPtr = input.dataPointer.bindMemory(to: Float.self, capacity: 2 * segment)
        let offsets = Array(Swift.stride(from: 0, to: length, by: stride))

        for (i, offset) in offsets.enumerated() {
            let n = min(segment, length - offset)
            // Like demucs TensorChunk.padded: a short final chunk is centred in the
            // segment window and surrounded by real neighbouring audio, not zeros.
            let lead = (segment - n) / 2
            let start = offset - lead
            let srcStart = max(0, start)
            let count = min(length, start + segment) - srcStart
            let dstOffset = srcStart - start
            inPtr.initialize(repeating: 0, count: 2 * segment)
            l.withUnsafeBufferPointer { (inPtr + dstOffset).update(from: $0.baseAddress! + srcStart, count: count) }
            r.withUnsafeBufferPointer { (inPtr + segment + dstOffset).update(from: $0.baseAddress! + srcStart, count: count) }

            let features = try MLDictionaryFeatureProvider(dictionary: ["mix": MLFeatureValue(multiArray: input)])
            let result = try model.prediction(from: features)
            // MLShapedArray handles any output strides/precision Core ML picks.
            let sources = MLShapedArray<Float>(result.featureValue(for: "sources")!.multiArrayValue!)
            let scalars = sources.scalars  // [1, stems, 2, segment]

            scalars.withUnsafeBufferPointer { src in
                weight.withUnsafeBufferPointer { w in
                    for s in 0..<(nStems * 2) {
                        out[s].withUnsafeMutableBufferPointer { dst in
                            let d = dst.baseAddress! + offset
                            // d += src * w
                            vDSP_vma(src.baseAddress! + s * segment + lead, 1, w.baseAddress!, 1, d, 1, d, 1, vDSP_Length(n))
                        }
                    }
                    sumWeight.withUnsafeMutableBufferPointer { dst in
                        let d = dst.baseAddress! + offset
                        vDSP_vadd(d, 1, w.baseAddress!, 1, d, 1, vDSP_Length(n))
                    }
                }
            }
            progress?(Double(i + 1) / Double(offsets.count))
        }

        var separated: [String: (left: [Float], right: [Float])] = [:]
        for (k, name) in stems.enumerated() {
            let outL = vDSP.add(mean, vDSP.multiply(std, vDSP.divide(out[2 * k], sumWeight)))
            let outR = vDSP.add(mean, vDSP.multiply(std, vDSP.divide(out[2 * k + 1], sumWeight)))
            separated[name] = (outL, outR)
        }
        return separated
    }
}

// MARK: - Audio helpers

public enum AudioIO {
    /// Loads any audio file as non-interleaved stereo Float32 at `sampleRate`.
    public static func loadStereo(url: URL, sampleRate: Double = 44_100) throws -> (left: [Float], right: [Float]) {
        let file = try AVAudioFile(forReading: url)
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
                                   channels: 2, interleaved: false)!
        let buffer = try convert(file: file, to: format)
        let n = Int(buffer.frameLength)
        let ch = buffer.floatChannelData!
        return (Array(UnsafeBufferPointer(start: ch[0], count: n)),
                Array(UnsafeBufferPointer(start: ch[1], count: n)))
    }

    /// Downmixes to mono and resamples to 16 kHz, the input WhisperKit expects.
    public static func whisperInput(left: [Float], right: [Float], sampleRate: Double = 44_100) throws -> [Float] {
        let mono = vDSP.multiply(0.5, vDSP.add(left, right))
        let src = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!
        let dst = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
        let inBuf = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: AVAudioFrameCount(mono.count))!
        inBuf.frameLength = AVAudioFrameCount(mono.count)
        mono.withUnsafeBufferPointer { inBuf.floatChannelData![0].update(from: $0.baseAddress!, count: mono.count) }

        let converter = AVAudioConverter(from: src, to: dst)!
        let capacity = AVAudioFrameCount(Double(mono.count) * 16_000 / sampleRate) + 1024
        let outBuf = AVAudioPCMBuffer(pcmFormat: dst, frameCapacity: capacity)!
        var fed = false
        var error: NSError?
        converter.convert(to: outBuf, error: &error) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true
            status.pointee = .haveData
            return inBuf
        }
        if let error { throw error }
        return Array(UnsafeBufferPointer(start: outBuf.floatChannelData![0], count: Int(outBuf.frameLength)))
    }

    private static func convert(file: AVAudioFile, to format: AVAudioFormat) throws -> AVAudioPCMBuffer {
        let inFormat = file.processingFormat
        let inBuf = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: inBuf)
        if inFormat == format { return inBuf }

        let converter = AVAudioConverter(from: inFormat, to: format)!
        if inFormat.channelCount == 1 {
            converter.channelMap = [0, 0]  // mono file -> same signal on both channels
        }
        let capacity = AVAudioFrameCount(Double(inBuf.frameLength) * format.sampleRate / inFormat.sampleRate) + 1024
        let outBuf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity)!
        var fed = false
        var error: NSError?
        converter.convert(to: outBuf, error: &error) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true
            status.pointee = .haveData
            return inBuf
        }
        if let error { throw error }
        return outBuf
    }
}
