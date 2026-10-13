# Fine-tuning for church worship songs

There are two models you can adapt. Whisper is the one that matters most for
lyric accuracy, so start there.

| | What it fixes | Data you need | Effort |
|---|---|---|---|
| **0. Prompting (no training)** | Worship vocabulary such as *Haleluya, Yesus, kudus, Roh Kudus* | A list of words | Minutes |
| **1. Whisper fine-tune** | Mis-heard sung words, spelling, repeated lines, Indonesian worship phrasing | Songs + timed lyrics (`.lrc`) | Hours (1 GPU) |
| **2. Demucs fine-tune** | Choir/backing vocals missed, keys/pads bleeding into vocals | Multitrack stems (vocals + full mix) | Hours to days |

## 0. Try prompting first

WhisperKit's `promptTokens` biases the decoder toward the words you give it. This
costs nothing; `WorshipTranscriber.transcribe(vocabularyHint:)` already supports it.
Use it as your baseline: measure WER with `lyricCheck.py` before and after each step.

## 1. Fine-tune Whisper

### Data

- **Amount:** 30–60 songs (about 2–4 hours) gives clear gains. 10 songs is enough
  to try the pipeline.
- **Audio:** use the **Demucs vocal stem**, because that is what the app
  transcribes. Training on full mixes and testing on stems (or the reverse) loses
  part of the gain.
- **Lyrics:** `.lrc` files with line timestamps (`[01:23.45] text`). Sources:
  - LRC files from lyric sites or apps (check the timing by ear).
  - An LRC editor on the vocal stem, which takes about 5 minutes per song.
  - Stock Whisper with timestamps on the vocal stem, then correcting the text by
    hand. The timestamps are usually good enough.
- **Write lyrics the way they are sung.** If the chorus is sung 4 times, it must
  appear 4 times with its own timestamps (you can stack timestamps on one line:
  `[01:10.00][02:40.00] Haleluya`). Keep the spelling consistent, for example
  always *Haleluya* and never *Halleluya*.
- Put whole songs in either train or test, never both. The script splits by
  song for you.

```bash
pip install -r coreml/requirements.txt -r finetune/requirements.txt
pip install -e . --no-deps

# 1) Isolate vocals for the training songs
python -m demucs -n htdemucs_ft --two-stems vocals -o stems songs/*.mp3
#    copy stems/htdemucs_ft/<song>/vocals.wav -> songs_vocals/<song>.wav and put <song>.lrc beside it

# 2) Cut into <=30 s clips aligned with the lyrics
python finetune/prepare_whisper_dataset.py --songs songs_vocals --out data/worship_whisper

# 3) Train (CUDA GPU; a Colab T4 works for small)
python finetune/train_whisper.py --data data/worship_whisper --base openai/whisper-small \
    --language indonesian --out whisper-small-worship

# 4) Convert for Xcode
python coreml/export_whisperkit.py --model whisper-small-worship --nbits 6
```

`train_whisper.py` prints the test-set WER before and after training. It also
keeps the checkpoint with the best WER.

**Defaults and why:**

- `--freeze-encoder` is on, so only the decoder is trained. The decoder handles
  vocabulary, spelling and phrasing, which cause most lyric errors. With little
  data, training the encoder mostly leads to overfitting. With 5+ hours of
  audio, try `--no-freeze-encoder`.
- `lr 1e-5`, `2000 steps` with SpecAugment. If test WER goes up after improving
  at first, use fewer steps. If WER is still dropping at the end, use more steps
  or add data.
- Keep `--language indonesian` when training, and use `language: "id"` when
  transcribing in the app.

## 2. Fine-tune Demucs (optional)

Fine-tune Demucs only if you can hear that the vocal stem is the problem. Typical
signs are choir or backing vocals going missing, or sustained keys/pads leaking
into the vocals. Demucs needs **multitrack** data: a vocal stem and the full mix
for each song. Worship-team practice multitracks and your own church's recordings
both work well.

```
data/worship_stems/
  train/<song>/vocals.wav    # ALL singing: lead + backing + choir
  train/<song>/mixture.wav   # full song (or accompaniment.wav = everything but vocals)
  valid/<song>/...           # 3-5 songs held out
```

```bash
python finetune/finetune_demucs.py --data data/worship_stems --out release_models
python -m demucs -n worship_vocals --repo release_models song.mp3       # listen to it
python coreml/export_demucs.py --model worship_vocals --repo release_models --stem vocals --nbits 6
```

The script starts from the `htdemucs_ft` vocal specialist and trains only the
vocal output with L1 loss. It also remixes vocals with other songs'
accompaniment (plus gain, channel-swap and polarity augmentation) to make the
most of a small dataset. A checkpoint is saved only when the vocal SDR on the
validation songs improves. If it never improves, the pretrained model is
already better for your data.

## Measuring progress

Update the paths in `lyricCheck.py` and compare WER on songs that were **not**
used for training:

1. Stock Whisper small on the original mix.
2. Stock Whisper on the Demucs vocals.
3. The same, with the vocabulary prompt.
4. Fine-tuned Whisper on the vocals.
5. Fine-tuned Whisper converted to Core ML at 6 bits. The difference from 4
   should be small; if it is not, try `--nbits 8`.
