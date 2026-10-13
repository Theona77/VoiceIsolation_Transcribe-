"""Fine-tune Whisper on worship-song clips made by prepare_whisper_dataset.py.

Run on a CUDA GPU (Colab T4 is enough for whisper-small with the defaults) or on
an Apple Silicon Mac (mps, slower). The result is a normal Hugging Face folder
that coreml/export_whisperkit.py converts to WhisperKit .mlpackage files.

    python finetune/train_whisper.py --data data/worship_whisper --out whisper-small-worship
    python coreml/export_whisperkit.py --model whisper-small-worship --nbits 6

With little data (< ~2 h of singing), keep --freeze-encoder on: it trains only the
decoder (vocabulary, spelling, line style), which is where most lyric errors are,
and it greatly reduces overfitting and catastrophic forgetting.
"""

import argparse
from dataclasses import dataclass

import evaluate
import torch
from datasets import Audio, load_dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                          WhisperProcessor)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


@dataclass
class Collator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features):
        inputs = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        labels = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        # The model prepends <|startoftranscript|> itself.
        if (ids[:, 0] == self.decoder_start_token_id).all():
            ids = ids[:, 1:]
        inputs["labels"] = ids
        return inputs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Folder with train/ and test/ (audiofolder format).")
    p.add_argument("--base", default="openai/whisper-small")
    p.add_argument("--language", default="indonesian")
    p.add_argument("--out", default="whisper-small-worship")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-every", type=int, default=200)
    args = p.parse_args()

    processor = WhisperProcessor.from_pretrained(args.base, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    # SpecAugment: cheap regularization that helps a lot on small datasets.
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05
    model.config.mask_feature_prob = 0.05
    if args.freeze_encoder:
        model.freeze_encoder()

    ds = load_dataset("audiofolder", data_dir=args.data)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    ds = ds.map(prepare, remove_columns=ds["train"].column_names)
    ds = ds.filter(lambda labels: len(labels) <= 448, input_columns=["labels"])

    wer_metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()

    def compute_metrics(pred):
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        preds = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        preds = [normalizer(x) for x in preds]
        refs = [normalizer(x) for x in refs]
        pairs = [(p_, r) for p_, r in zip(preds, refs) if r.strip()]
        return {"wer": 100 * wer_metric.compute(predictions=[a for a, _ in pairs],
                                                references=[b for _, b in pairs])}

    cuda = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=min(200, args.steps // 10),
        max_steps=args.steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=cuda,
        eval_strategy="steps",
        eval_steps=args.eval_every,
        save_steps=args.eval_every,
        save_total_limit=2,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
    print("Baseline:", trainer.evaluate())
    trainer.train()
    print("Fine-tuned:", trainer.evaluate())
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    print(f"Saved to {args.out}. Next: python coreml/export_whisperkit.py --model {args.out} --nbits 6")


if __name__ == "__main__":
    main()
