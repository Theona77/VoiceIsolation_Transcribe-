"""Build a Whisper fine-tuning dataset from worship songs and timed lyrics.

Input folder: one audio file + one `.lrc` lyric file per song, same base name:

    songs/
      KuNyanyiHaleluya.wav     (ideally the Demucs vocal stem, see below)
      KuNyanyiHaleluya.lrc     [00:12.30] Ku nyanyi haleluya ...

Output: a Hugging Face "audiofolder" dataset of <=30 s, 16 kHz mono clips:

    out/train/metadata.csv + clips    out/test/metadata.csv + clips

Train on the *same kind of audio you will transcribe in the app*. If the app runs
Demucs first, isolate the vocals for the training songs too:

    python -m demucs -n htdemucs_ft --two-stems vocals -o stems songs/*.mp3
    # then use stems/htdemucs_ft/<song>/vocals.wav, renamed to <song>.wav

Usage:
    python finetune/prepare_whisper_dataset.py --songs songs/ --out data/worship_whisper
"""

import argparse
import csv
import random
import re
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

SR = 16000
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
LRC_LINE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")


def parse_lrc(path):
    """Returns sorted [(start_seconds, text)], including empty lines (instrumental gaps)."""
    entries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stamps = LRC_LINE.findall(raw)
        if not stamps:
            continue  # metadata tags like [ar:...] or untimed lines
        text = LRC_LINE.sub("", raw).strip()
        for mm, ss in stamps:  # a line may carry several timestamps (repeated chorus)
            entries.append((int(mm) * 60 + float(ss), text))
    return sorted(entries)


def make_clips(entries, duration, max_len, last_line_len):
    """Groups consecutive lyric lines into clips no longer than `max_len` seconds."""
    lines = []
    for i, (start, text) in enumerate(entries):
        end = entries[i + 1][0] if i + 1 < len(entries) else min(duration, start + last_line_len)
        if text and end > start:
            lines.append((start, end, text))

    clips, cur = [], []
    for line in lines:
        # Start a new clip on overflow, or when there is a gap (instrumental) before this line.
        if cur and (line[1] - cur[0][0] > max_len or line[0] - cur[-1][1] > 0.5):
            clips.append(cur)
            cur = []
        if line[1] - line[0] > max_len:
            continue  # a single "line" longer than max_len is almost always a timing error
        cur.append(line)
    if cur:
        clips.append(cur)
    return [(c[0][0], c[-1][1], " ".join(t for _, _, t in c)) for c in clips]


def load_mono16k(path):
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--songs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--test-fraction", type=float, default=0.15,
                   help="Fraction of *songs* held out for evaluation (split by song, not clip).")
    p.add_argument("--max-len", type=float, default=28.0, help="Max clip length in seconds (Whisper: 30).")
    p.add_argument("--last-line-len", type=float, default=6.0, help="Assumed length of the final lyric line.")
    p.add_argument("--pad", type=float, default=0.2, help="Seconds of audio context added on each side.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    songs = sorted(f for f in args.songs.iterdir()
                   if f.suffix.lower() in AUDIO_EXT and f.with_suffix(".lrc").exists())
    if not songs:
        raise SystemExit(f"No <song>.<audio> + <song>.lrc pairs found in {args.songs}")
    random.Random(args.seed).shuffle(songs)
    n_test = max(1, round(len(songs) * args.test_fraction)) if len(songs) > 1 else 0
    splits = {"test": songs[:n_test], "train": songs[n_test:]}

    for split, files in splits.items():
        out = args.out / split
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for song in files:
            audio = load_mono16k(song)
            duration = len(audio) / SR
            clips = make_clips(parse_lrc(song.with_suffix(".lrc")), duration, args.max_len, args.last_line_len)
            for i, (start, end, text) in enumerate(clips):
                a = max(0, int((start - args.pad) * SR))
                b = min(len(audio), int((end + args.pad) * SR))
                name = f"{song.stem}_{i:03d}.wav"
                sf.write(out / name, audio[a:b].numpy(), SR)
                rows.append({"file_name": name, "text": text, "song": song.stem})
            print(f"[{split}] {song.name}: {len(clips)} clips")
        with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "text", "song"])
            writer.writeheader()
            writer.writerows(rows)
        total = sum(sf.info(out / r["file_name"]).duration for r in rows) / 60
        print(f"{split}: {len(files)} songs, {len(rows)} clips, {total:.1f} min")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
