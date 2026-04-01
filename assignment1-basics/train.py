#!/usr/bin/env python3
"""
CS336 Assignment 1 — Training script (enhanced)
Features:
  - wandb 实时训练曲线
  - tqdm 进度条
  - Early stopping（验证 loss 不再下降时自动停止）
  - 训练结束后自动生成 loss 曲线图 (PNG)

Usage:
  # 基础用法
  python train.py --train_data data/train.npy --val_data data/val.npy --ckpt_dir runs/exp1

  # 开启 wandb
  python train.py --train_data data/train.npy --val_data data/val.npy --ckpt_dir runs/exp1 --wandb

  # 开启 early stopping（连续 10 次验证 loss 不下降就停止）
  python train.py ... --early_stopping --patience 10

  # 从 checkpoint 恢复
  python train.py --resume runs/exp1/checkpoint_latest.pt ...
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
# Model: Transformer LM as nn.Module
# -----------------------------------------------------------------------------
class _TransformerBlock(torch.nn.Module):
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
# Data loading
# -----------------------------------------------------------------------------
def load_dataset(path: str | Path, dtype: np.dtype = np.int64) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    raise ValueError(f"Unsupported format: {path.suffix}")


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
    model.eval()
    total_loss = 0.0
    num_batches = 0
    while True:
        if max_batches is not None and num_batches >= max_batches:
            break
        try:
            x, y = get_batch(dataset, batch_size, context_length, device)
        except Exception:
            break
        logits = model(x)
        B, T, V = logits.shape
        loss = run_cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
        total_loss += loss.item()
        num_batches += 1
    model.train()
    if num_batches == 0:
        return float("nan")
    return total_loss / num_batches


# -----------------------------------------------------------------------------
# Early Stopping
# -----------------------------------------------------------------------------
class EarlyStopping:
    """验证 loss 连续 patience 次不下降就停止训练。"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_step = 0

    def should_stop(self, val_loss: float, step: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_step = step
            return False
        self.counter += 1
        return self.counter >= self.patience

    def status(self) -> str:
        return f"best_val_loss={self.best_loss:.4f} @ step {self.best_step}, no_improve={self.counter}/{self.patience}"


# -----------------------------------------------------------------------------
# Plot training curves (offline, no wandb needed)
# -----------------------------------------------------------------------------
def plot_training_curves(log_file: Path, out_dir: Path) -> None:
    """从 train_log.jsonl 读取数据，生成 loss 曲线图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无 GUI 模式
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过画图。可用 pip install matplotlib 安装。")
        return

    steps, train_losses, wallclocks = [], [], []
    val_steps, val_losses, val_wallclocks = [], [], []

    with open(log_file, "r") as f:
        for line in f:
            record = json.loads(line)
            steps.append(record["step"])
            train_losses.append(record["train_loss"])
            wallclocks.append(record["wallclock"])
            if record.get("val_loss") is not None:
                val_steps.append(record["step"])
                val_losses.append(record["val_loss"])
                val_wallclocks.append(record["wallclock"])

    if not steps:
        print("日志为空，跳过画图。")
        return

    # ---- 图1: Loss vs Steps ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(steps, train_losses, label="train loss", alpha=0.7, linewidth=0.8)
    if val_steps:
        axes[0].plot(val_steps, val_losses, label="val loss", linewidth=2, color="red")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs Steps")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---- 图2: Loss vs Wallclock ----
    axes[1].plot(wallclocks, train_losses, label="train loss", alpha=0.7, linewidth=0.8)
    if val_wallclocks:
        axes[1].plot(val_wallclocks, val_losses, label="val loss", linewidth=2, color="red")
    axes[1].set_xlabel("Wallclock Time (s)")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss vs Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"训练曲线已保存 → {plot_path}")


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def log_step(
    step: int,
    wallclock: float,
    train_loss: float,
    val_loss: float | None,
    lr: float,
    log_file: Path | None,
    wandb_run=None,
) -> None:
    # Console (handled by tqdm, only print if val_loss)
    if val_loss is not None:
        print(f"\n  [val] step={step} val_loss={val_loss:.4f} train_loss={train_loss:.4f} lr={lr:.2e}", flush=True)

    # JSONL file
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

    # wandb
    if wandb_run is not None:
        log_dict = {
            "train/loss": train_loss,
            "train/lr": lr,
            "train/wallclock": wallclock,
            "train/perplexity": math.exp(min(train_loss, 20)),  # 防止 overflow
        }
        if val_loss is not None:
            log_dict["val/loss"] = val_loss
            log_dict["val/perplexity"] = math.exp(min(val_loss, 20))
        wandb_run.log(log_dict, step=step)


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

    # ---- wandb ----
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config=vars(args),
            )
            print(f"wandb 已启动: {wandb_run.url}")
        except ImportError:
            print("wandb 未安装，跳过。可用 pip install wandb 安装。", file=sys.stderr)
        except Exception as e:
            print(f"wandb 初始化失败: {e}", file=sys.stderr)

    # ---- Data ----
    train_data = load_dataset(args.train_data)
    val_data = load_dataset(args.val_data) if args.val_data else None
    print(f"Train tokens: {len(train_data):,}, val tokens: {len(val_data) if val_data is not None else 'N/A'}")

    # ---- Model ----
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

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # ---- Optimizer ----
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

    # ---- Logging setup ----
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = (ckpt_dir / "train_log.jsonl") if ckpt_dir else None

    # ---- Early stopping ----
    early_stopper = None
    if args.early_stopping and val_data is not None:
        early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        print(f"Early stopping 已开启: patience={args.patience}, min_delta={args.min_delta}")

    total_steps = args.total_steps
    warmup_iters = args.warmup_iters
    cosine_cycle_iters = args.cosine_cycle_iters
    max_lr = args.lr
    min_lr = args.min_lr
    val_every = args.val_every
    ckpt_every = args.ckpt_every

    # ---- tqdm 进度条 ----
    try:
        from tqdm import tqdm
        pbar = tqdm(
            range(start_step, total_steps),
            desc="Training",
            dynamic_ncols=True,
            initial=start_step,
            total=total_steps,
        )
    except ImportError:
        print("tqdm 未安装，使用普通循环。可用 pip install tqdm 安装。")
        pbar = range(start_step, total_steps)

    # ---- Training loop ----
    train_start = time.perf_counter()
    best_val_loss = float("inf")

    for step in pbar:
        # Learning rate schedule
        lr = get_lr_cosine_schedule(
            step,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Forward + backward
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        optimizer.zero_grad()
        logits = model(x)
        B, T, V = logits.shape
        loss = run_cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
        loss.backward()
        run_gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        wallclock = time.perf_counter() - train_start
        train_loss = loss.item()

        # ---- Validation ----
        val_loss = None
        if val_data is not None and (step + 1) % val_every == 0:
            val_loss = evaluate(
                model, val_data, args.batch_size,
                args.context_length, device, max_batches=args.val_batches,
            )
            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if ckpt_dir is not None:
                    save_checkpoint(model, optimizer, step + 1, ckpt_dir / "checkpoint_best.pt")

            # Early stopping 检查
            if early_stopper is not None:
                if early_stopper.should_stop(val_loss, step + 1):
                    print(f"\n⏹ Early stopping triggered! {early_stopper.status()}")
                    log_step(step + 1, wallclock, train_loss, val_loss, lr, log_file, wandb_run)
                    if ckpt_dir is not None:
                        save_checkpoint(model, optimizer, step + 1, ckpt_dir / "checkpoint_early_stop.pt")
                    break

        # ---- Logging ----
        if (step + 1) % args.log_every == 0 or step == start_step:
            log_step(step + 1, wallclock, train_loss, val_loss, lr, log_file, wandb_run)

        # ---- tqdm 进度条更新 ----
        if hasattr(pbar, "set_postfix"):
            postfix = {"loss": f"{train_loss:.4f}", "lr": f"{lr:.2e}"}
            if val_loss is not None:
                postfix["val"] = f"{val_loss:.4f}"
            if early_stopper is not None:
                postfix["es"] = f"{early_stopper.counter}/{early_stopper.patience}"
            pbar.set_postfix(postfix)

        # ---- Checkpointing ----
        if ckpt_dir is not None and (step + 1) % ckpt_every == 0:
            save_checkpoint(model, optimizer, step + 1, ckpt_dir / f"checkpoint_{step + 1}.pt")
            save_checkpoint(model, optimizer, step + 1, ckpt_dir / "checkpoint_latest.pt")

    # ---- 训练结束 ----
    total_time = time.perf_counter() - train_start
    print(f"\nTraining finished! Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # ---- 生成训练曲线图 ----
    if log_file is not None and log_file.exists():
        plot_training_curves(log_file, ckpt_dir)

    # ---- 关闭 wandb ----
    if wandb_run is not None:
        wandb_run.finish()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Transformer LM (CS336 Assignment 1)")

    # Data
    p.add_argument("--train_data", type=str, required=True)
    p.add_argument("--val_data", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)

    # Output
    p.add_argument("--ckpt_dir", type=str, default=None)

    # Model
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
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

    # wandb
    p.add_argument("--wandb", action="store_true", help="开启 wandb 日志")
    p.add_argument("--wandb_project", type=str, default="cs336-assignment1")
    p.add_argument("--wandb_name", type=str, default=None, help="wandb run 名称")

    # Early stopping
    p.add_argument("--early_stopping", action="store_true", help="开启 early stopping")
    p.add_argument("--patience", type=int, default=10,
                    help="连续多少次验证 loss 不下降后停止")
    p.add_argument("--min_delta", type=float, default=0.0,
                    help="loss 下降幅度小于此值不算改善")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())