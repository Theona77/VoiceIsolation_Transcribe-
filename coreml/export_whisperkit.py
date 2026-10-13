"""Export a (fine-tuned) Hugging Face Whisper model to WhisperKit `.mlpackage` files.

Produces a folder that WhisperKit can load directly with `modelFolder:`:

    <out>/
      MelSpectrogram.mlpackage
      AudioEncoder.mlpackage
      TextDecoder.mlpackage
      config.json, generation_config.json, tokenizer.json, ...   (offline tokenizer)

It uses Argmax's own model definitions from whisperkittools, so the I/O names and
shapes are exactly what the WhisperKit Swift package expects. Unlike
`whisperkit-generate-model`, it does not run the macOS-only Core ML prediction
tests, so it also works on Linux/Colab.

Examples:

    # Stock whisper-small, 6-bit palettized (same size as whisper.cpp Q5_0)
    python coreml/export_whisperkit.py --model openai/whisper-small --nbits 6

    # Your fine-tuned checkpoint (see docs/finetune_worship.md)
    python coreml/export_whisperkit.py --model ./whisper-small-worship --nbits 6

Note: a whisper.cpp `ggml-small-q5_0.bin` cannot be converted. Q5_0 is a ggml-only
format; start from the Hugging Face weights and quantize with Core ML instead.
Core ML palettization supports 1, 2, 3, 4, 6 and 8 bits (no 5-bit).
"""

import argparse
import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from argmaxtools import _sdpa
from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration
from whisperkit import audio_encoder, test_utils, text_decoder

N_SAMPLES = 480000  # 30 s at 16 kHz, WhisperKit's fixed window
HF_FILES = [
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "normalizer.json", "added_tokens.json",
    "special_tokens_map.json", "preprocessor_config.json",
]


def to_coreml_inputs(torch_inputs):
    # Same dtypes WhisperKit feeds: float16 for floats, int32 for ints.
    return [
        ct.TensorType(k, shape=v.shape, dtype=np.float16 if v.dtype.is_floating_point else np.int32)
        for k, v in torch_inputs.items()
    ]


def convert(module, torch_inputs, output_names, target):
    traced = torch.jit.trace(module.eval(), example_kwarg_inputs=torch_inputs)
    return ct.convert(
        traced,
        inputs=to_coreml_inputs(torch_inputs),
        outputs=[ct.TensorType(n, dtype=np.float16) for n in output_names],
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        convert_to="mlprogram",
        skip_model_load=True,
    )


def palettize(mlmodel, nbits, group_size):
    from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights

    kwargs = dict(mode="kmeans", nbits=nbits, weight_threshold=2048)
    if group_size:
        kwargs.update(granularity="per_grouped_channel", group_size=group_size)
    return palettize_weights(mlmodel, OptimizationConfig(global_config=OpPalettizerConfig(**kwargs)))


def load(args):
    if args.random_init:
        # whisper-small architecture with random weights (pipeline test only).
        torch.manual_seed(0)
        cfg = WhisperConfig(
            vocab_size=51865, num_mel_bins=80, d_model=768,
            encoder_layers=12, decoder_layers=12,
            encoder_attention_heads=12, decoder_attention_heads=12,
            encoder_ffn_dim=3072, decoder_ffn_dim=3072,
        )
        gen = GenerationConfig(alignment_heads=[[5, 3], [5, 9], [8, 0], [8, 4], [8, 7], [8, 8],
                                                [9, 0], [9, 7], [9, 9], [10, 5]])
        return WhisperForConditionalGeneration(cfg).eval(), gen
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32)
    return model.eval(), model.generation_config


def copy_hf_files(model_id, out_dir):
    src = Path(model_id)
    for name in HF_FILES:
        if src.is_dir():
            path = src / name
            if not path.exists():
                continue
        else:
            from huggingface_hub import hf_hub_download
            try:
                path = Path(hf_hub_download(model_id, name))
            except Exception:
                continue
        shutil.copy(path, out_dir / name)
    if not (out_dir / "tokenizer.json").exists():
        print("WARNING: no tokenizer.json copied; WhisperKit will try to download the tokenizer.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="openai/whisper-small", help="HF hub id or local fine-tuned folder.")
    p.add_argument("--nbits", type=int, default=6, choices=[0, 1, 2, 3, 4, 6, 8],
                   help="Palettization bits for encoder/decoder (0 = float16). Default 6 (~Q5_0 size).")
    p.add_argument("--group-size", type=int, default=0,
                   help="Per-grouped-channel palettization (iOS 18+), e.g. 16 with --nbits 4.")
    p.add_argument("--target", default=None,
                   help="Minimum deployment target. Default iOS16, or iOS18 when --group-size/3-bit is used.")
    p.add_argument("--out", default=None, help="Output folder (default coreml/build/<model>_<bits>).")
    p.add_argument("--no-token-timestamps", action="store_true",
                   help="Drop the alignment_heads_weights output (word timestamps).")
    p.add_argument("--random-init", action="store_true",
                   help="Use random whisper-small weights (tests the pipeline offline).")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    target_name = args.target or ("iOS18" if args.group_size or args.nbits == 3 else "iOS16")
    target = getattr(ct.target, target_name)

    name = "random-whisper-small" if args.random_init else Path(args.model.rstrip("/")).name
    bits = f"{args.nbits}bit" if args.nbits else "fp16"
    out_dir = Path(args.out or f"coreml/build/openai_whisper-{name.replace('whisper-', '')}_{bits}")
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_model, gen_cfg = load(args)
    cfg = hf_model.config

    # MelSpectrogram (not quantized: it only holds the mel filterbank and window)
    print("Converting MelSpectrogram...")
    mel = audio_encoder.WhisperMelSpectrogram(n_mels=cfg.num_mel_bins)
    mlmodel = convert(mel, {"audio": torch.randn(N_SAMPLES)}, ["melspectrogram_features"], target)
    mlmodel.save(str(out_dir / "MelSpectrogram.mlpackage"))

    # AudioEncoder (SDPA choices match whisperkit-generate-model defaults)
    print("Converting AudioEncoder...")
    audio_encoder.SDPA_IMPL = _sdpa.SplitHeadsQ
    enc = audio_encoder.WhisperAudioEncoder(cfg)
    enc.load_state_dict(hf_model.model.encoder.state_dict())
    enc_inputs, _ = test_utils._prepare_test_inputs_for_encoder(
        embed_dim=cfg.d_model, n_mels=cfg.num_mel_bins, batch_size=1,
        melspectrogram_seq_len=cfg.max_source_positions * 2,
    )
    mlmodel = convert(enc, enc_inputs, ["encoder_output_embeds"], target)
    if args.nbits:
        print(f"  palettizing to {args.nbits} bits...")
        mlmodel = palettize(mlmodel, args.nbits, args.group_size)
    mlmodel.save(str(out_dir / "AudioEncoder.mlpackage"))

    # TextDecoder
    print("Converting TextDecoder...")
    text_decoder.SDPA_IMPL = _sdpa.Cat
    dec = text_decoder.WhisperTextDecoder(cfg)
    dec.load_state_dict(hf_model.model.decoder.state_dict())
    outputs = ["logits", "key_cache_updates", "value_cache_updates"]
    if not args.no_token_timestamps and getattr(gen_cfg, "alignment_heads", None):
        dec.configure_for_token_timestamps(gen_cfg)
        outputs.append("alignment_heads_weights")
    dec_inputs, _ = test_utils._prepare_test_inputs_for_decoder(
        embed_dim=cfg.d_model, vocab_size=cfg.vocab_size,
        n_heads=cfg.decoder_attention_heads, n_layers=cfg.decoder_layers, batch_size=1,
        enc_seq_len=cfg.max_source_positions, dec_kv_seq_len=cfg.max_target_positions,
        active_dec_kv_seq_len=cfg.max_target_positions // 2,
    )
    mlmodel = convert(dec, dec_inputs, outputs, target)
    if args.nbits:
        print(f"  palettizing to {args.nbits} bits...")
        mlmodel = palettize(mlmodel, args.nbits, args.group_size)
    mlmodel.save(str(out_dir / "TextDecoder.mlpackage"))

    if args.random_init:
        (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    else:
        copy_hf_files(args.model, out_dir)
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
