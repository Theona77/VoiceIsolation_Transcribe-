"""Fine-tune the Demucs vocal model on worship-song multitracks.

Dataset layout (44.1 kHz, any length; one folder per song):

    data/worship_stems/
      train/<song>/vocals.wav  + mixture.wav   (or accompaniment.wav instead of mixture)
      valid/<song>/vocals.wav  + mixture.wav

`vocals.wav` should hold *all* sung parts (lead + backing vocals + choir) and
`mixture.wav` the full song. Multitracks from worship-team practice packs, your
own church recordings, or DAW sessions are ideal; 20-50 songs already help.

Training starts from the htdemucs_ft vocal specialist and only optimizes the
vocal output. Each training example also remixes vocals with another song's
accompaniment, which multiplies the effective data.

    python finetune/finetune_demucs.py --data data/worship_stems --out release_models
    python -m demucs -n worship_vocals --repo release_models song.mp3     # try it
    python coreml/export_demucs.py --model worship_vocals --repo release_models --stem vocals

Needs a CUDA GPU (or Apple Silicon `--device mps`, slower). ~10 GB of GPU memory
with --batch-size 2.
"""

import argparse
import random
from pathlib import Path

import torch
import torchaudio
from torch.nn import functional as F

from demucs.apply import BagOfModels
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model
from demucs.states import get_state


def load_stereo(path, sr):
    wav, file_sr = torchaudio.load(str(path))
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    if wav.shape[0] == 1:
        wav = wav.expand(2, -1)
    return wav[:2].contiguous()


def load_split(root, sr):
    songs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        vocals = load_stereo(d / "vocals.wav", sr)
        if (d / "accompaniment.wav").exists():
            acc = load_stereo(d / "accompaniment.wav", sr)
        else:
            acc = load_stereo(d / "mixture.wav", sr) - vocals
        n = min(vocals.shape[-1], acc.shape[-1])
        songs.append((d.name, vocals[:, :n], acc[:, :n]))
    if not songs:
        raise SystemExit(f"No song folders found in {root}")
    return songs


def random_crop(wav, length):
    if wav.shape[-1] <= length:
        return F.pad(wav, (0, length - wav.shape[-1]))
    start = random.randrange(wav.shape[-1] - length)
    return wav[:, start:start + length]


def make_batch(songs, batch_size, length, remix_prob):
    mixes, targets = [], []
    for _ in range(batch_size):
        _, vocals, acc = random.choice(songs)
        if random.random() < remix_prob:
            _, _, acc_other = random.choice(songs)
            v, a = random_crop(vocals, length), random_crop(acc_other, length)
        else:
            # Same offset for vocals and accompaniment so the original mix is kept.
            both = random_crop(torch.cat([vocals, acc]), length)
            v, a = both[:2], both[2:]
        v = v * random.uniform(0.5, 1.5)
        a = a * random.uniform(0.5, 1.5)
        if random.random() < 0.5:  # channel swap
            v, a = v.flip(0), a.flip(0)
        if random.random() < 0.5:  # polarity flip
            v, a = -v, -a
        mixes.append(v + a)
        targets.append(v)
    return torch.stack(mixes), torch.stack(targets)


@torch.no_grad()
def evaluate(model, songs, vocal_idx, length, device):
    """Mean SDR (dB) of the vocal estimate over consecutive chunks of each valid song."""
    model.eval()
    sdrs = []
    for _, vocals, acc in songs:
        for start in range(0, vocals.shape[-1] - length + 1, length):
            v = vocals[:, start:start + length].to(device)
            mix = v + acc[:, start:start + length].to(device)
            est = model(mix[None])[0, vocal_idx]
            num = v.pow(2).sum() + 1e-8
            den = (v - est).pow(2).sum() + 1e-8
            sdrs.append(10 * torch.log10(num / den).item())
    model.train()
    return sum(sdrs) / max(1, len(sdrs))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--base", default="htdemucs_ft", help="Pretrained bag or signature to start from.")
    p.add_argument("--out", default="release_models", type=Path)
    p.add_argument("--name", default="worship_vocals", help="Output model name (no '-').")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--remix-prob", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    assert "-" not in args.name, "Demucs uses '-' to separate signature and checksum."

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = get_model(args.base)
    if isinstance(model, BagOfModels):
        vi = model.sources.index("vocals")
        best = max(range(len(model.models)), key=lambda i: model.weights[i][vi])
        model = model.models[best]
    assert isinstance(model, HTDemucs), "This script expects a Hybrid Transformer Demucs model."
    model.to(args.device).train()
    vocal_idx = model.sources.index("vocals")
    sr = model.samplerate
    length = int(model.segment * sr)

    train = load_split(args.data / "train", sr)
    valid = load_split(args.data / "valid", sr)
    print(f"{len(train)} train / {len(valid)} valid songs, segment {length / sr:.1f}s")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    best_sdr = evaluate(model, valid, vocal_idx, length, args.device)
    print(f"Baseline valid vocal SDR: {best_sdr:.2f} dB")
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.name}.th"

    for epoch in range(args.epochs):
        total = 0.0
        for _ in range(args.steps_per_epoch):
            mix, target = make_batch(train, args.batch_size, length, args.remix_prob)
            mix, target = mix.to(args.device), target.to(args.device)
            est = model(mix)[:, vocal_idx]
            loss = F.l1_loss(est, target)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            total += loss.item()
        sdr = evaluate(model, valid, vocal_idx, length, args.device)
        print(f"epoch {epoch + 1}: train L1 {total / args.steps_per_epoch:.4f}, valid vocal SDR {sdr:.2f} dB")
        if sdr > best_sdr:
            best_sdr = sdr
            init_args, init_kwargs = model._init_args_kwargs
            torch.save({
                "klass": model.__class__,
                "args": init_args,
                "kwargs": init_kwargs,
                "state": get_state(model, None, half=False),
            }, out_path)
            print(f"  saved {out_path}")

    if out_path.exists():
        print(f"Best valid vocal SDR {best_sdr:.2f} dB -> {out_path}")
    else:
        print("Fine-tuning never beat the pretrained model; nothing saved. Try more data or a lower --lr.")


if __name__ == "__main__":
    main()
