// WorshipTranscriber.swift
// Example pipeline: song file -> Demucs vocals -> WhisperKit lyrics.
//
// Xcode setup (see coreml/README.md):
//   1. Add WhisperKit via Swift Package Manager: https://github.com/argmaxinc/WhisperKit
//   2. Drag `Demucs_htdemucs_ft_vocals_6bit.mlpackage` into the app target.
//   3. Add the WhisperKit model folder (with the compiled .mlmodelc files and
//      tokenizer.json) as a *folder reference* (blue folder) named `WhisperModel`.

import Foundation
import WhisperKit

public final class WorshipTranscriber {
    private let separator: DemucsSeparator
    private let whisper: WhisperKit

    public init(demucsModel: String = "Demucs_htdemucs_ft_vocals_6bit",
                whisperFolder: String = "WhisperModel") async throws {
        separator = try DemucsSeparator(bundleResource: demucsModel)
        guard let folder = Bundle.main.url(forResource: whisperFolder, withExtension: nil) else {
            throw NSError(domain: "WorshipTranscriber", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "\(whisperFolder) folder not in bundle"])
        }
        // download: false keeps everything offline; the tokenizer is read from the folder.
        whisper = try await WhisperKit(WhisperKitConfig(modelFolder: folder.path, download: false))
    }

    /// - Parameter vocabularyHint: optional words to bias decoding toward, e.g.
    ///   "Haleluya, Yesus, Tuhan, kudus, Roh Kudus, kemuliaan".
    public func transcribe(songURL: URL, language: String = "id",
                           vocabularyHint: String? = nil) async throws -> String {
        let song = try AudioIO.loadStereo(url: songURL, sampleRate: separator.sampleRate)
        let stems = try separator.separate(left: song.left, right: song.right)
        guard let vocals = stems["vocals"] else { throw NSError(domain: "WorshipTranscriber", code: 2) }
        let audio16k = try AudioIO.whisperInput(left: vocals.left, right: vocals.right,
                                                sampleRate: separator.sampleRate)

        var options = DecodingOptions(task: .transcribe, language: language,
                                      temperatureFallbackCount: 3, usePrefillPrompt: true,
                                      skipSpecialTokens: true, wordTimestamps: false,
                                      chunkingStrategy: .vad)
        if let hint = vocabularyHint, let tokenizer = whisper.tokenizer {
            options.promptTokens = tokenizer.encode(text: " " + hint)
                .filter { $0 < tokenizer.specialTokens.specialTokenBegin }
        }
        let results = try await whisper.transcribe(audioArray: audio16k, decodeOptions: options)
        return results.map(\.text).joined(separator: "\n")
    }
}
