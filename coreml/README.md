# Core ML export (Demucs + WhisperKit) for Xcode

Converts the two models in this project to Core ML `.mlpackage` files:

| Script | Produces | Used from Swift via |
|---|---|---|
| `export_demucs.py` | `Demucs_<model>_vocals_6bit.mlpackage` (vocal isolation) | `swift/DemucsSeparator.swift` |
| `export_whisperkit.py` | `MelSpectrogram` / `AudioEncoder` / `TextDecoder.mlpackage` + tokenizer | [WhisperKit](https://github.com/argmaxinc/WhisperKit) |

## About "5-bit"

Core ML weight palettization supports **1, 2, 3, 4, 6 and 8 bits**. There is no
5-bit mode, and a whisper.cpp `ggml-small-q5_0.bin` cannot be converted (Q5_0 is a
ggml-only format). Both scripts start from the original float weights and
quantize with Core ML instead:

| `--nbits` | Whisper small size | Notes |
|---|---|---|
| 6 (default) | ~176 MB | Same size as your ggml Q5_0 (~181 MB, Q5_0 is ~5.5 bits/weight). Closest match. |
| 4 + `--group-size 16` | ~120 MB | Smaller. Per-group palettization, needs iOS 18 / macOS 15. Check WER. |
| 8 | ~235 MB | Practically lossless. |
| 0 | ~480 MB | float16, no palettization. |

Demucs `htdemucs_ft` vocals at 6-bit is ~95 MB. About 64 MB of that is the exact
STFT/iSTFT basis, which is deliberately left unquantized.

## 1. Install (Python 3.11, macOS or Linux)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r coreml/requirements.txt
pip install -e . --no-deps          # this repo's demucs package
```

## 2. Export

```bash
# Vocal-only Demucs (the htdemucs_ft vocal specialist), 6-bit
python coreml/export_demucs.py --model htdemucs_ft --stem vocals --nbits 6

# Whisper small for WhisperKit, 6-bit
python coreml/export_whisperkit.py --model openai/whisper-small --nbits 6
```

Outputs go to `coreml/build/`. To export your fine-tuned models, see
[`docs/finetune_worship.md`](../docs/finetune_worship.md).

Before converting, `export_demucs.py` checks that its Core ML-friendly rewrite
gives the same output as the original PyTorch model. The rewrite does the
STFT/iSTFT with convolutions because Core ML has no complex numbers. If the
outputs differ, it stops (SNR must be over 60 dB; in testing it was about
125 dB). Use `--precision fp16` for speed or `fp32` for maximum accuracy. The
default, `mixed`, keeps the normalization reductions in float32.

## 3. Use in Xcode

1. **Demucs:** drag `Demucs_htdemucs_ft_vocals_6bit.mlpackage` into your app
   target. Xcode compiles it into the bundle as `.mlmodelc`. Add
   `swift/DemucsSeparator.swift` to the target.
2. **WhisperKit:** add the package `https://github.com/argmaxinc/WhisperKit`. Then
   compile the model folder on your Mac:
   ```bash
   sh coreml/compile_whisperkit_folder.sh coreml/build/openai_whisper-small_6bit
   ```
   Add that folder to the app as a **folder reference** (blue folder) and name
   it `WhisperModel`. It must contain the `.mlmodelc` files and `tokenizer.json`.
3. Add `swift/WorshipTranscriber.swift` and call it:

```swift
let transcriber = try await WorshipTranscriber()
let lyrics = try await transcriber.transcribe(
    songURL: url, language: "id",
    vocabularyHint: "Haleluya, Yesus, Tuhan, kudus, Roh Kudus, kemuliaan")
```

Demucs works on 7.8 s chunks (343,980 samples at 44.1 kHz). `DemucsSeparator`
overlaps and cross-fades them in the same way as `demucs.apply.apply_model`. A
Python port of this logic matched `apply_model` at about 126 dB SNR.

## Testing without the real weights

Both scripts accept `--random-init`, which converts the same architecture with
random weights. This lets you check the conversion pipeline offline. The output
is **not** a usable model.
