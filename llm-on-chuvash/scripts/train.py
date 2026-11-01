#!/usr/bin/env python3
"""Обучение маленькой decoder-only LLM с нуля (pure PyTorch, CPU-friendly).

Архитектура: GPT-style трансформер — pre-LN, RoPE, GELU-MLP 4x, tied embeddings,
опционально sliding-window attention (SWA). Оптимизация: AdamW + cosine с warmup,
градиентный клип, bf16-autocast (mixed precision на CPU).

Usage:
  python -I train.py --data-dir data/tokenized --tokenizer tokenizer/bpe-8192.json \
      --out-dir checkpoints --exp-name base --epochs 5
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

try:
    from tqdm import tqdm
except ImportError:  # прогресс-бар опционален, логгирование работает и без него
    tqdm = None

# ----------------------------- модель -----------------------------


class RoPE(nn.Module):
    """Rotary Positional Embeddings (Su et al., 2021) — бонус по ДЗ."""

    def __init__(self, dim: int, max_seq: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv_freq)  # (T, D/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def rotate(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q,k: (B, H, T, Dh); cos/sin: (T, Dh/2) -> разворачиваем на Dh
        # срез [:T] обязателен: при генерации T < max_seq, иначе broadcast
        # растягивает q до max_seq (shape error)
        T = q.shape[2]
        cos = torch.cat((self.cos, self.cos), dim=-1)[None, None, :T]
        sin = torch.cat((self.sin, self.sin), dim=-1)[None, None, :T]
        return q * cos + self.rotate_half(q) * sin, k * cos + self.rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, block_size: int, swa_window: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.d_head, block_size)
        self.swa_window = swa_window  # 0 = полное внимание
        if swa_window:
            mask = torch.ones(block_size, block_size, dtype=torch.bool).tril()
            band = torch.ones(block_size, block_size, dtype=torch.bool).triu(-swa_window)
            mask = mask & band  # каузальность + окно
            self.register_buffer("mask", mask.view(1, 1, block_size, block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        # rotate, а не apply: не затенять nn.Module.apply (используется self.apply(init))
        q, k = self.rope.rotate(q, k)
        if self.swa_window:
            att = F.scaled_dot_product_attention(
                q, k, v, attn_mask=self.mask[:, :, :T, :T]  # bool: True = участвует, float-каст ломает маскировку
            )
        else:
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(att.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, block_size: int, swa_window: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, swa_window)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, n_head: int,
                 block_size: int, swa_window: int = 0, tie_emb: bool = True):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, n_head, block_size, swa_window) for _ in range(n_layer)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_emb:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)
        # scaled residual init (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, temperature: float = 0.8,
                 top_k: int = 50) -> torch.Tensor:
        for _ in range(max_new):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ----------------------------- данные -----------------------------


class DataLoader:
    """Случайные окна по mmap-массиву токенов (uint16)."""

    def __init__(self, path: str, block_size: int, batch_size: int, seed: int):
        self.data = np.load(path, mmap_mode="r")
        self.n = len(self.data)
        self.T = block_size
        self.B = batch_size
        self.rng = np.random.default_rng(seed)

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = self.rng.integers(0, self.n - self.T - 1, size=self.B)
        x = torch.from_numpy(np.asarray([self.data[i:i + self.T] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.asarray([self.data[i + 1:i + 1 + self.T] for i in ix]).astype(np.int64))
        return x, y


def evaluate(model: TinyGPT, val: DataLoader, n_batches: int, device, autocast_dtype) -> float:
    model.eval()
    losses = []
    with torch.no_grad(), torch.autocast(device.type, dtype=autocast_dtype,
                                         enabled=autocast_dtype != torch.float32):
        for _ in range(n_batches):
            x, y = (t.to(device) for t in val.batch())
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


# ----------------------------- train -----------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--exp-name", default="run")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--swa-window", type=int, default=0, help="0 = full attention")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2.5e-3)
    p.add_argument("--min-lr", type=float, default=3e-4)
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--max-steps", type=int, default=0, help="0 = по epochs")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50,
                   help="шаг между статус-строками (nohup-логи без tty)")
    p.add_argument("--tensorboard", action="store_true",
                   help="писать скаляры/сэмплы в TensorBoard (нужен пакет tensorboard)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=0, help="0 = все ядра")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    threads = args.threads or os.cpu_count() or 1
    torch.set_num_threads(threads)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS недоступен (torch.backends.mps.is_available() == False)")
    # на MPS автокаст идёт в fp16 (bf16-автокаст MPS не поддерживает), на CPU — bf16
    if device.type == "mps":
        autocast_dtype = torch.float16 if args.dtype != "fp32" else torch.float32
    else:
        autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    with open(f"{args.data_dir}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    vocab_size = Tokenizer.from_file(args.tokenizer).get_vocab_size()

    run_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    model = TinyGPT(vocab_size, args.d_model, args.n_layer, args.n_head,
                    args.block_size, args.swa_window).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab={vocab_size} params={n_params:,} threads={threads} "
          f"device={device.type} dtype={args.dtype}")

    train = DataLoader(f"{args.data_dir}/train.npy", args.block_size, args.batch_size, args.seed)
    val = DataLoader(f"{args.data_dir}/val.npy", args.block_size, args.batch_size, args.seed + 1)
    steps_per_epoch = max(train.n // (args.batch_size * args.block_size), 1)
    max_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    warmup = int(max_steps * args.warmup_frac)

    print(f"[config] {json.dumps(vars(args), ensure_ascii=False)}", flush=True)
    print(f"[model] TinyGPT d={args.d_model} L={args.n_layer} H={args.n_head} "
          f"ctx={args.block_size} swa={args.swa_window or 'off'} vocab={vocab_size} "
          f"params={n_params:,}", flush=True)
    print(f"[data] train={train.n:,} val={val.n:,} токенов | "
          f"{steps_per_epoch} шаг/эпоху x {args.epochs} эп. = {max_steps} шагов "
          f"(warmup {warmup}) | threads={threads} dtype={args.dtype}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    start_step = 0
    best_val = float("inf")
    best_step = 0

    ckpt_path = os.path.join(run_dir, "last.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        best_val = ck.get("best_val", float("inf"))
        print(f"[resume] step={start_step} best_val={best_val:.4f} из {ckpt_path}", flush=True)
        best_step = ck.get("best_step", start_step)

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / max(warmup, 1)
        t = (step - warmup) / max(max_steps - warmup, 1)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))

    tok = Tokenizer.from_file(args.tokenizer)
    prompts = ["Чăваш", "Халăх", "Раççей", "Чăваш Ен"]
    log_path = os.path.join(run_dir, "log.jsonl")
    samples_path = os.path.join(run_dir, "samples.log")

    tb = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
            print(f"[tb] TensorBoard: {run_dir}/tb (tensorboard --logdir {args.out_dir})", flush=True)
        except ImportError:
            print("[tb] пакет tensorboard не установлен — скаляры только в log.jsonl "
                  "(pip install tensorboard)", flush=True)

    use_bar = tqdm is not None and sys.stderr.isatty()
    bar = (tqdm(total=max_steps, initial=start_step, dynamic_ncols=True, unit="step")
           if use_bar else None)

    def log_status(step: int, done: int, ema: float, lr: float, tok_s: int, eta_s: float,
                   val_loss: float | None = None) -> None:
        """Статус: в tty — постфикс прогресс-бара, в nohup-логе — строка раз в log-every."""
        pct = 100.0 * (step + 1) / max_steps
        msg = (f"loss {ema:.3f} | lr {lr:.2e} | {tok_s:,} ток/с | ETA {eta_s/60:.0f} мин"
               + (f" | val {val_loss:.3f}" if val_loss is not None else ""))
        if bar is not None:
            bar.n = step + 1
            bar.set_postfix_str(msg)
            bar.refresh()
        elif (step + 1) % args.log_every == 0 or step == max_steps - 1:
            print(f"[{pct:5.1f}%] step {step+1}/{max_steps} | {msg}", flush=True)

    t_start = time.time()
    t0 = t_start
    tokens_seen = 0
    tokens_total = 0
    ema_loss = None
    grad_norm = float("nan")
    model.train()
    for step in range(start_step, max_steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = (t.to(device) for t in train.batch())
        with torch.autocast(device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype != torch.float32):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        tokens_seen += x.numel()
        tokens_total += x.numel()
        loss_val = loss.item()
        ema_loss = loss_val if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_val

        if (step + 1) % args.log_every == 0 or step == max_steps - 1:
            done = step + 1 - start_step
            eta_s = (time.time() - t_start) / max(done, 1) * (max_steps - step - 1)
            log_status(step, done, ema_loss, lr, int(tokens_seen / (time.time() - t0)),
                       eta_s)
        if tb is not None and (step + 1) % args.log_every == 0:
            tb.add_scalar("train/loss_ema", ema_loss, step + 1)
            tb.add_scalar("train/grad_norm", grad_norm, step + 1)
            tb.add_scalar("train/lr", lr, step + 1)

        if (step + 1) % args.eval_every == 0 or step == max_steps - 1:
            val_loss = evaluate(model, val, 16, device, autocast_dtype)
            dt = time.time() - t0
            done = step + 1 - start_step
            eta_s = (time.time() - t_start) / max(done, 1) * (max_steps - step - 1)
            rec = {
                "step": step + 1, "train_loss": round(loss_val, 4),
                "train_loss_ema": round(ema_loss, 4),
                "val_loss": round(val_loss, 4), "ppl": round(math.exp(min(val_loss, 20)), 1),
                "lr": round(lr, 6), "grad_norm": round(grad_norm, 3),
                "tok_s": int(tokens_seen / dt), "tokens_total": tokens_total,
                "elapsed_s": round(time.time() - t_start, 1), "eta_s": round(eta_s),
            }
            print(f"[eval] {json.dumps(rec, ensure_ascii=False)}", flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if tb is not None:
                tb.add_scalar("val/loss", val_loss, step + 1)
                tb.add_scalar("val/ppl", math.exp(min(val_loss, 20)), step + 1)
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "step": step + 1, "best_val": best_val, "best_step": best_step,
                        "args": vars(args)}, ckpt_path)
            if val_loss < best_val:
                best_val = val_loss
                best_step = step + 1
                torch.save({"model": model.state_dict(), "step": step + 1,
                            "val_loss": val_loss, "args": vars(args)},
                           os.path.join(run_dir, "best.pt"))
                print(f"[ckpt] new best: val_loss={val_loss:.4f} "
                      f"(ppl {math.exp(min(val_loss, 20)):.1f}) -> best.pt", flush=True)
            t0 = time.time()
            tokens_seen = 0

        if (step + 1) % args.sample_every == 0 or step == max_steps - 1:
            model.eval()
            prompt = prompts[step % len(prompts)]
            ctx = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
            with torch.autocast(device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype != torch.float32):
                out = model.generate(ctx, 60, temperature=0.8, top_k=50)[0].tolist()
            text = tok.decode(out)
            print(f"[sample @{step+1}] {text!r}", flush=True)
            with open(samples_path, "a", encoding="utf-8") as f:
                f.write(f"step {step+1} | T=0.8 top-k=50 | {text}\n")
            if tb is not None:
                tb.add_text("samples", f"```\n{text}\n```", step + 1)
            model.train()

    if bar is not None:
        bar.close()
    if tb is not None:
        tb.close()
    wall = time.time() - t_start
    print(f"[done] steps {start_step} -> {max_steps} | wall {wall/60:.1f} мин | "
          f"токенов {tokens_total:,} | best val_loss {best_val:.4f} "
          f"(ppl {math.exp(min(best_val, 20)):.1f}) @ step {best_step}", flush=True)
    print(f"[artifacts] {run_dir}/{{best.pt,last.pt,log.jsonl,samples.log}}", flush=True)


if __name__ == "__main__":
    main()
