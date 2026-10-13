"""Export a Hybrid Transformer Demucs model to a Core ML `.mlpackage`.

Core ML has no complex-number support, so `torch.stft` / `torch.istft` cannot be
converted. This script wraps HTDemucs so the STFT and iSTFT are computed with
fixed DFT-basis convolutions instead (numerically equivalent, checked against the
original model before conversion). The resulting package takes raw stereo audio
and returns separated stems, so the Swift side only has to chunk the song.

Examples (run on a Mac, or any machine that can reach dl.fbaipublicfiles.com):

    # Vocal-only model from the fine-tuned bag, 6-bit palettized (~Q5_0 size)
    python coreml/export_demucs.py --model htdemucs_ft --stem vocals --nbits 6

    # Your own fine-tuned checkpoint (see docs/finetune_worship.md)
    python coreml/export_demucs.py --model <sig> --repo ./release_models --stem vocals

Core ML palettization supports 1, 2, 3, 4, 6 and 8 bits. There is no 5-bit
mode; 6-bit is the same size as whisper.cpp/ggml Q5_0 (~5.5 bits/weight).
"""

import argparse
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from demucs.apply import BagOfModels
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model


def _hann(n):
    return torch.hann_window(n, periodic=True, dtype=torch.float64)


class ConvSTFT(nn.Module):
    """`torch.stft(center=True, normalized=True, pad_mode='reflect')` as a conv1d.

    Returns the first `nfft // 2` bins (HTDemucs drops the Nyquist bin) with real
    parts in the first half of the channels and imaginary parts in the second.
    """

    def __init__(self, nfft, hop):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        n = torch.arange(nfft, dtype=torch.float64)
        k = torch.arange(nfft // 2, dtype=torch.float64)
        ang = 2 * math.pi * k[:, None] * n[None, :] / nfft
        win = _hann(nfft)[None, :] / math.sqrt(nfft)
        basis = torch.cat([torch.cos(ang) * win, -torch.sin(ang) * win], 0)
        self.register_buffer("basis", basis[:, None, :].float(), persistent=False)

    def forward(self, x):
        # x: [N, L]
        x = F.pad(x[:, None], (self.nfft // 2, self.nfft // 2), mode="reflect")
        return F.conv1d(x, self.basis, stride=self.hop)  # [N, nfft, frames]


class ConvISTFT(nn.Module):
    """`torch.istft(center=True, normalized=True)` as a conv_transpose1d.

    Expects the same channel layout ConvSTFT produces (Nyquist bin assumed zero).
    `frames` and the output crop are fixed at construction because Core ML wants
    static shapes.
    """

    def __init__(self, nfft, hop, frames, start, length):
        super().__init__()
        self.hop = hop
        self.start = start
        self.length = length
        n = torch.arange(nfft, dtype=torch.float64)
        k = torch.arange(nfft // 2, dtype=torch.float64)
        ang = 2 * math.pi * k[:, None] * n[None, :] / nfft
        coef = torch.full((nfft // 2, 1), 2.0, dtype=torch.float64)
        coef[0] = 1.0
        win = _hann(nfft)[None, :]
        # irfft scale (1/N) times the inverse of the forward normalization (sqrt(N)).
        scale = coef / math.sqrt(nfft)
        basis = torch.cat([scale * torch.cos(ang) * win, -scale * torch.sin(ang) * win], 0)
        self.register_buffer("basis", basis[:, None, :].float(), persistent=False)

        # Overlap-added squared window, inverted once here instead of dividing at runtime.
        ones = torch.ones(1, 1, frames, dtype=torch.float64)
        wsq = (_hann(nfft) ** 2)[None, None, :]
        env = F.conv_transpose1d(ones.expand(1, 1, frames), wsq, stride=hop)[0, 0]
        env = env[start:start + length]
        assert env.min() > 1e-8
        self.register_buffer("inv_env", (1.0 / env).float(), persistent=False)

    def forward(self, z):
        # z: [N, nfft, frames]
        x = F.conv_transpose1d(z, self.basis, stride=self.hop)[:, 0]
        return x[:, self.start:self.start + self.length] * self.inv_env


class HTDemucsCoreML(nn.Module):
    """Static-shape, real-valued re-implementation of `HTDemucs.forward` (eval, cac=True).

    Input:  mix     [1, 2, segment_samples]
    Output: sources [1, n_stems, 2, segment_samples]
    """

    def __init__(self, model: HTDemucs, stems=None):
        super().__init__()
        assert isinstance(model, HTDemucs), "Only HTDemucs (v4) models are supported."
        assert model.cac, "Only complex-as-channels HTDemucs models are supported."
        assert model.hop_length == model.nfft // 4
        self.model = model.eval()
        self.length = int(model.segment * model.samplerate)
        self.stem_idx = [model.sources.index(s) for s in (stems or model.sources)]

        hl, nfft = model.hop_length, model.nfft
        self.le = int(math.ceil(self.length / hl))
        self.pad = hl // 2 * 3
        self.stft = ConvSTFT(nfft, hl)
        # _ispec pads 2 frames on each side, then istft trims nfft//2 (center) and
        # _ispec trims `pad`, so the waveform starts at nfft//2 + pad in the OLA output.
        self.istft = ConvISTFT(nfft, hl, self.le + 4, nfft // 2 + self.pad, self.length)

    def forward(self, mix):
        m = self.model
        B, C, L = mix.shape
        S = len(m.sources)
        nfft2 = m.nfft // 2

        # ---- HTDemucs._spec + _magnitude (cac) ----
        hl = m.hop_length
        x = F.pad(mix, (self.pad, self.pad + self.le * hl - L), mode="reflect")
        z = self.stft(x.reshape(B * C, -1))[..., 2:2 + self.le]  # [B*C, 2*F, T]
        Fq, T = nfft2, self.le
        x = z.reshape(B, C * 2, Fq, T)  # channels: c0re, c0im, c1re, c1im

        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # ---- encoder / transformer / decoder: same as HTDemucs.forward ----
        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)

        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = m.channel_upsampler(x.reshape(b, c, f * t)).reshape(b, -1, f, t)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = m.channel_downsampler(x.reshape(b, c, f * t)).reshape(b, -1, f, t)
                xt = m.channel_downsampler_t(xt)

        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    xt, _ = tdec(xt, saved_t.pop(-1), length_t)

        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        xt = xt.view(B, S, -1, L) * stdt[:, None] + meant[:, None]

        # Keep only requested stems before the (expensive) iSTFT.
        idx = self.stem_idx
        if idx != list(range(S)):
            x = torch.cat([x[:, i:i + 1] for i in idx], 1)
            xt = torch.cat([xt[:, i:i + 1] for i in idx], 1)
        Sk = len(idx)

        # ---- HTDemucs._mask (cac) + _ispec ----
        x = x.reshape(B * Sk * C, 2 * Fq, T)  # [re bins..., im bins...] per channel
        x = F.pad(x, (2, 2))
        wav = self.istft(x).reshape(B, Sk, C, L)
        return xt + wav


def load_htdemucs(name, repo, stem):
    model = get_model(name, repo=Path(repo) if repo else None)
    if isinstance(model, BagOfModels):
        if stem is None:
            raise SystemExit(
                f"'{name}' is a bag of {len(model.models)} models. Pass --stem to export "
                "the specialist for that stem (e.g. --stem vocals for htdemucs_ft)."
            )
        si = model.sources.index(stem)
        weights = [w[si] for w in model.weights]
        best = int(np.argmax(weights))
        print(f"Using sub-model {best} of the bag (weight {weights[best]}) for '{stem}'.")
        model = model.models[best]
    return model


def random_htdemucs():
    """Same architecture as the released htdemucs, with random weights (pipeline test only)."""
    torch.manual_seed(0)
    return HTDemucs(sources=["drums", "bass", "other", "vocals"], segment=Fraction(39, 5))


def palettize(mlmodel, nbits, group_size):
    from coremltools.optimize.coreml import (OpPalettizerConfig, OptimizationConfig,
                                             get_weights_metadata, palettize_weights)

    # Leave the fixed STFT/iSTFT bases untouched: they are exact math, not learned
    # weights, and quantizing them adds error to every frequency bin.
    skip = {
        name: None
        for name, meta in get_weights_metadata(mlmodel, weight_threshold=2048).items()
        if meta.val.ndim == 3 and meta.val.shape[1:] == (1, meta.val.shape[0])
    }
    kwargs = dict(mode="kmeans", nbits=nbits, weight_threshold=2048)
    if group_size:
        kwargs.update(granularity="per_grouped_channel", group_size=group_size)
    cfg = OptimizationConfig(global_config=OpPalettizerConfig(**kwargs), op_name_configs=skip)
    print(f"Palettizing to {nbits} bits (skipping {len(skip)} STFT basis tensors)...")
    return palettize_weights(mlmodel, cfg)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="htdemucs_ft", help="Pretrained name or signature.")
    p.add_argument("--repo", help="Local folder with checkpoints (e.g. release_models/).")
    p.add_argument("--stem", help="Export only this stem (e.g. vocals). Default: all stems.")
    p.add_argument("--nbits", type=int, default=6, choices=[0, 1, 2, 3, 4, 6, 8],
                   help="Palettization bits (0 = keep float16). Default 6 (~ggml Q5_0 size).")
    p.add_argument("--group-size", type=int, default=0,
                   help="Per-grouped-channel palettization group size (needs iOS 18). "
                        "Use with --nbits 4 for a smaller model, e.g. --group-size 16.")
    p.add_argument("--precision", default="mixed", choices=["mixed", "fp16", "fp32"],
                   help="mixed: float16 except normalization reductions (default, safest). "
                        "fp16: fastest, may lose accuracy on loud input. fp32: GPU/CPU only.")
    p.add_argument("--target", default="iOS17", help="Minimum deployment target, e.g. iOS17, iOS18.")
    p.add_argument("--out", default="coreml/build", help="Output folder.")
    p.add_argument("--random-init", action="store_true",
                   help="Use random weights (tests the conversion pipeline offline).")
    p.add_argument("--skip-convert", action="store_true",
                   help="Only run the PyTorch equivalence check.")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    model = random_htdemucs() if args.random_init else load_htdemucs(args.model, args.repo, args.stem)
    stems = [args.stem] if args.stem else None
    wrapper = HTDemucsCoreML(model, stems).eval()
    L = wrapper.length
    print(f"Sources: {model.sources}, segment: {float(model.segment)}s = {L} samples @ {model.samplerate} Hz")

    # Equivalence check against the original (complex STFT) implementation.
    mix = torch.randn(1, model.audio_channels, L) * 0.1
    ref = model(mix)[:, wrapper.stem_idx]
    out = wrapper(mix)
    err = (out - ref).abs().max().item()
    snr = 10 * torch.log10(ref.pow(2).mean() / (out - ref).pow(2).mean()).item()
    print(f"Wrapper vs original HTDemucs: max abs err {err:.2e}, SNR {snr:.1f} dB")
    assert snr > 60, "Re-implementation does not match the original model."
    if args.skip_convert:
        return

    import coremltools as ct

    # nn.MultiheadAttention's fused eval fast path has no Core ML equivalent.
    torch.backends.mha.set_fastpath_enabled(False)
    traced = torch.jit.trace(wrapper, mix)
    # Reductions over the whole song chunk (mean/std and group norms) can overflow
    # in float16, so "mixed" keeps those ops in float32; the rest runs in float16.
    fp32_ops = {"reduce_mean", "reduce_sum", "reduce_sum_square", "sqrt", "rsqrt", "pow", "real_div"}
    precision = {
        "mixed": ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in fp32_ops),
        "fp16": ct.precision.FLOAT16,
        "fp32": ct.precision.FLOAT32,
    }[args.precision]
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mix", shape=mix.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="sources", dtype=np.float32)],
        minimum_deployment_target=getattr(ct.target, args.target),
        compute_precision=precision,
        convert_to="mlprogram",
        skip_model_load=True,
    )
    if args.nbits:
        mlmodel = palettize(mlmodel, args.nbits, args.group_size)

    stem_names = [model.sources[i] for i in wrapper.stem_idx]
    mlmodel.short_description = f"Hybrid Transformer Demucs ({args.model}) source separation"
    mlmodel.input_description["mix"] = f"Stereo audio chunk, {model.samplerate} Hz, {L} samples, float in [-1, 1]"
    mlmodel.output_description["sources"] = f"Separated stems {stem_names} as [1, stems, 2, {L}]"
    mlmodel.user_defined_metadata.update({
        "sources": ",".join(stem_names),
        "samplerate": str(model.samplerate),
        "segment_samples": str(L),
    })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "random" if args.random_init else args.model
    stem_tag = f"_{args.stem}" if args.stem else ""
    bits_tag = f"_{args.nbits}bit" if args.nbits else "_fp16"
    path = out_dir / f"Demucs_{tag}{stem_tag}{bits_tag}.mlpackage"
    mlmodel.save(str(path))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
