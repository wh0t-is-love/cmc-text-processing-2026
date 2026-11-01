#!/usr/bin/env python3
"""Инференс: генерация текста из чекпоинта.

Usage: python -I generate.py <ckpt.pt> <tokenizer.json> "промпт" [n_tokens] [temperature] [top_k]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # найти train.TinyGPT
from train import TinyGPT  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402


def load(ckpt_path: str, tokenizer_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"] if "args" in ck else ck["cfg"]
    vocab_size = Tokenizer.from_file(tokenizer_path).get_vocab_size()
    model = TinyGPT(vocab_size, a["d_model"], a["n_layer"], a["n_head"],
                    a["block_size"], a.get("swa_window", 0))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, Tokenizer.from_file(tokenizer_path)


@torch.no_grad()
def main() -> None:
    ckpt_path, tok_path, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
    n_new = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    temp = float(sys.argv[5]) if len(sys.argv) > 5 else 0.8
    top_k = int(sys.argv[6]) if len(sys.argv) > 6 else 50

    model, tok = load(ckpt_path, tok_path)
    ids = tok.encode(prompt).ids
    ctx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(ctx, n_new, temperature=temp, top_k=top_k)[0].tolist()
    text = tok.decode(out)
    print(text)


if __name__ == "__main__":
    main()
