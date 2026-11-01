#!/usr/bin/env python3
"""Encode the corpus to flat uint16 bins for fast mmap training.

Usage: python -I tokenize_data.py <corpus.txt> <tokenizer.json> <out_dir> [val_frac]

Documents are separated by an <|endoftext|> token; the val split is taken from
the tail of the corpus (it is ordered by dump order, which is close to random
for our purposes).
"""
import json
import os
import sys

import numpy as np
from tokenizers import Tokenizer

CHUNK = 1 << 22  # 4 MiB
EOT_ID = 0  # <|endoftext|> — первый special-токен в словаре


def encode_stream(tok: Tokenizer, path: str, prefix: str, out_dir: str):
    n = 0
    out = np.lib.format.open_memmap(
        f"{out_dir}/{prefix}.npy", mode="w+", dtype=np.uint16, shape=(1 << 30,)
    )
    with open(path, encoding="utf-8") as f:
        for chunk in iter(lambda: f.read(CHUNK), ""):
            # документы разделены \n\n — кодируем по абзацам, вставляя EOT
            paras = chunk.split("\n\n")
            texts = [p.replace("\n", " ") for p in paras]
            enc = tok.encode_batch_fast(texts, add_special_tokens=False)
            ids = []
            for e in enc[:-1]:
                ids.extend(e.ids)
                ids.append(EOT_ID)
            ids.extend(enc[-1].ids)  # хвост склеится со следующим куском
            arr = np.asarray(ids, dtype=np.uint16)
            out[n:n + len(arr)] = arr
            n += len(arr)
    del out  # flush
    return n


def main() -> None:
    corpus_path, tok_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    val_frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01

    os.makedirs(out_dir, exist_ok=True)
    tok = Tokenizer.from_file(tok_path)
    n = encode_stream(tok, corpus_path, "all", out_dir)

    all_arr = np.load(f"{out_dir}/all.npy", mmap_mode="r")
    n_val = int(n * val_frac)
    n_train = n - n_val
    train = np.lib.format.open_memmap(
        f"{out_dir}/train.npy", mode="w+", dtype=np.uint16, shape=(n_train,)
    )
    train[:] = all_arr[:n_train]
    val = np.lib.format.open_memmap(
        f"{out_dir}/val.npy", mode="w+", dtype=np.uint16, shape=(n_val,)
    )
    val[:] = all_arr[n_train:n]
    del train, val

    with open(f"{out_dir}/meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {"total_tokens": n, "train_tokens": n_train, "val_tokens": n_val,
             "tokenizer": tok_path.rsplit("/", 1)[-1], "eot_id": EOT_ID},
            f, ensure_ascii=False, indent=2,
        )
    print(f"total={n:,} train={n_train:,} val={n_val:,}")


if __name__ == "__main__":
    main()
