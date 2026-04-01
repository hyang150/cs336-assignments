#!/usr/bin/env python3
"""
CS336 Assignment 1 — Training script skeleton.
Trains a Transformer LM on tokenized data with memmap, checkpointing, and logging.
Usage:
  uv run python train.py --train_data data/train.npy --val_data data/val.npy --ckpt_dir runs/exp1
  uv run python train.py --resume runs/exp1/checkpoint_latest.pt ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Import from cs336_basics (your implementation)
# -----------------------------------------------------------------------------
from cs336_basics.data import get_batch
from cs336_basics.nn import (
    run_cross_entropy,
    run_gradient_clipping,
    run_rmsnorm,
    run_transformer_lm,
)
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.training import load_checkpoint, save_checkpoint


# -----------------------------------------------------------------------------
# Model: Transformer LM as nn.Module (wraps run_transformer_lm from nn.py)
# -----------------------------------------------------------------------------
class _TransformerBlock(torch.nn.Module):
    """Single block parameters only (no forward)."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.attn_q_proj = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.attn_k_proj = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.attn_v_proj = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.attn_output_proj = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.ln1 = torch.nn.Parameter(torch.ones(d_model))
        self.ffn_w1 = torch.nn.Parameter(torch.empty(d_ff, d_model))
        self.ffn_w2 = torch.nn.Parameter(torch.empty(d_model, d_ff))
        self.ffn_w3 = torch.nn.Parameter(torch.empty(d_ff, d_model))
        self.ln2 = torch.nn.Parameter(torch.ones(d_model))


class TransformerLM(torch.nn.Module):
    """Transformer language model; forward uses cs336_basics.nn.run_transformer_lm."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = torch.nn.Parameter(torch.empty(vocab_size, d_model))
        self.ln_final = torch.nn.Parameter(torch.ones(d_model))
        self.lm_head = torch.nn.Parameter(torch.empty(vocab_size, d_model))
        self.layers = torch.nn.ModuleList([
            _TransformerBlock(d_model, d_ff) for _ in range(num_layers)
        ])

        self._init_weights(init_scale)

    def _init_weights(self, scale: float) -> None:
        for p in self.parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, mean=0.0, std=scale)
            else:
                torch.nn.init.ones_(p)

    def _state_dict_for_forward(self) -> dict[str, torch.Tensor]:
        """Build the state dict expected by run_transformer_lm (key names)."""
        d = {
            "token_embeddings.weight": self.token_embeddings,
            "ln_final.weight": self.ln_final,
            "lm_head.weight": self.lm_head,
        }
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}."
            d[prefix + "attn.q_proj.weight"] = layer.attn_q_proj
            d[prefix + "attn.k_proj.weight"] = layer.attn_k_proj
            d[prefix + "attn.v_proj.weight"] = layer.attn_v_proj
            d[prefix + "attn.output_proj.weight"] = layer.attn_output_proj
            d[prefix + "ln1.weight"] = layer.ln1
            d[prefix + "ffn.w1.weight"] = layer.ffn_w1
            d[prefix + "ffn.w2.weight"] = layer.ffn_w2
            d[prefix + "ffn.w3.weight"] = layer.ffn_w3
            d[prefix + "ln2.weight"] = layer.ln2
        return d

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        weights = self._state_dict_for_forward()
        return run_transformer_lm(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            weights=weights,
            in_indices=in_indices,
        )


# -----------------------------------------------------------------------------
# Data loading with np.memmap
# -----------------------------------------------------------------------------
def load_dataset(path: str | Path, dtype: np.dtype = np.int64) -> np.ndarray:
    """Load tokenized dataset; use memmap for large files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    raise ValueError(f"Unsupported format: {path.suffix}. Use .npy (saved with np.save).")


# -----------------------------------------------------------------------------
# Validation loss
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    max_batches: int | None = 50,
) -> float:
    """Compute average cross-entropy over validation batches."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    from cs336_basics.nn import run_cross_entropy
    while True:
        if max_batches is not None and num_batches >= max_batches:
            break
        try:
            x, y = get_batch(dataset, batch_size, context_length, device)
        except Exception:
            break
        logits = model(x)
        # Flatten for CE: (B, T, V) -> (B*T, V), targets (B*T)
        B, T, V = logits.shape
        loss = run_cross_entropy(
            logits.reshape(B * T, V),
            y.reshape(B * T),
        )
        total_loss += loss.item()
        num_batches += 1
    model.train()
    if num_batches == 0:
        return float("nan")
    return total_loss / num_batches


# -----------------------------------------------------------------------------
# Logging (console + optional JSONL)
# -----------------------------------------------------------------------------
def log_step(
    step: int,
    wallclock: float,
    train_loss: float,
    val_loss: float | None,
    lr: float,
    log_file: Path | None,
) -> None:
    msg = f"step={step} wallclock={wallclock:.1f}s train_loss={train_loss:.4f}"
    if val_loss is not None:
        msg += f" val_loss={val_loss:.4f}"
    msg += f" lr={lr:.2e}"
    print(msg, flush=True)
    if log_file is not None:
        record = {
            "step": step,
            "wallclock": wallclock,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.", file=sys.stderr)
        device = "cpu"
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("MPS not available, using CPU.", file=sys.stderr)
        device = "cpu"

    # Data (memmap for large arrays)
    train_data = load_dataset(args.train_data)
    val_data = load_dataset(args.val_data) if args.val_data else None
    print(f"Train tokens: {len(train_data)}, val tokens: {len(val_data) if val_data is not None else 'N/A'}")

    # Model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        init_scale=args.init_scale,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from step {start_step}")

    # Logging
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = (ckpt_dir / "train_log.jsonl") if ckpt_dir else None

    total_steps = args.total_steps
    warmup_iters = args.warmup_iters
    cosine_cycle_iters = args.cosine_cycle_iters
    max_lr = args.lr
    min_lr = args.min_lr
    val_every = args.val_every
    ckpt_every = args.ckpt_every

    train_start = time.perf_counter()
    for step in range(start_step, total_steps):
        # Learning rate
        lr = get_lr_cosine_schedule(
            step,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            device,
        )
        optimizer.zero_grad()
        logits = model(x)
        B, T, V = logits.shape
        loss = run_cross_entropy(
            logits.reshape(B * T, V),
            y.reshape(B * T),
        )
        loss.backward()
        run_gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        wallclock = time.perf_counter() - train_start
        train_loss = loss.item()

        val_loss = None
        if val_data is not None and (step + 1) % val_every == 0:
            val_loss = evaluate(
                model,
                val_data,
                args.batch_size,
                args.context_length,
                device,
                max_batches=args.val_batches,
            )

        if (step + 1) % args.log_every == 0 or step == start_step:
            log_step(step + 1, wallclock, train_loss, val_loss, lr, log_file)

        if ckpt_dir is not None and (step + 1) % ckpt_every == 0:
            path = ckpt_dir / f"checkpoint_{step + 1}.pt"
            save_checkpoint(model, optimizer, step + 1, path)
        if ckpt_dir is not None and (step + 1) % ckpt_every == 0:
            latest = ckpt_dir / "checkpoint_latest.pt"
            save_checkpoint(model, optimizer, step + 1, latest)

    print("Training finished.", flush=True)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Transformer LM (CS336 Assignment 1)")
    # Data
    p.add_argument("--train_data", type=str, required=True, help="Path to train token ids (.npy)")
    p.add_argument("--val_data", type=str, default=None, help="Path to val token ids (.npy)")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    # Output
    p.add_argument("--ckpt_dir", type=str, default=None, help="Directory to save checkpoints and log")

    # Model
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344, help="Often 8/3 * d_model, multiple of 64")
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--init_scale", type=float, default=0.02)

    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1)

    # Schedule
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--cosine_cycle_iters", type=int, default=5000)

    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")

    # Logging / eval
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--val_every", type=int, default=100)
    p.add_argument("--val_batches", type=int, default=20)
    p.add_argument("--ckpt_every", type=int, default=500)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
