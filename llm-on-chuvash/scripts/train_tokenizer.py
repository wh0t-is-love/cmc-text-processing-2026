#!/usr/bin/env python3
"""Train a byte-level BPE tokenizer on the corpus (HF `tokenizers`, Rust backend).

Usage: python -I train_tokenizer.py <corpus.txt> <out_dir> [vocab_size]

Trains on a capped sample of the corpus (BPE needs far less data than an LM),
then reports compression over the *full* corpus so the numbers are honest.
"""
import json
import sys
import time

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

TRAIN_CAP_CHARS = 40_000_000  # достаточно для стабильного BPE-словаря
CHUNK = 1 << 22  # 4 MiB текста за проход


def main() -> None:
    corpus_path, out_dir = sys.argv[1], sys.argv[2]
    vocab_size = int(sys.argv[3]) if len(sys.argv) > 3 else 8192

    tok = Tokenizer(models.BPE())
    # NFC без lower-case: орфография чувашского важна (Ӑ/ӑ, Ҫ/ҫ различаются)
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>"],
        show_progress=False,
    )

    with open(corpus_path, encoding="utf-8") as f:
        sample = f.read(TRAIN_CAP_CHARS)
    sample_path = corpus_path + ".sample"
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample)

    t0 = time.time()
    tok.train([sample_path], trainer)
    train_s = time.time() - t0
    tok.save(f"{out_dir}/bpe-{vocab_size}.json")

    # Честная метрика компрессии — на всём корпусе, потоково
    n_tokens = n_chars = 0
    with open(corpus_path, encoding="utf-8") as f:
        for chunk in iter(lambda: f.read(CHUNK), ""):
            n_chars += len(chunk)
            pieces = [chunk[i:i + (1 << 16)] for i in range(0, len(chunk), 1 << 16)]
            n_tokens += sum(len(e.ids) for e in tok.encode_batch_fast(pieces))

    stats = {
        "vocab_size": vocab_size,
        "train_seconds": round(train_s, 1),
        "corpus_chars": n_chars,
        "corpus_tokens": n_tokens,
        "chars_per_token": round(n_chars / max(n_tokens, 1), 3),
    }
    with open(f"{out_dir}/bpe-{vocab_size}.stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
