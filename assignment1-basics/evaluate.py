#!/usr/bin/env python3
"""
CS336 Assignment 1 — 模型评估与文本生成
功能：
  1. 计算验证集 perplexity
  2. 生成文本（支持 temperature / top-p sampling）
  3. 统计模型参数量

Usage:
  # 生成文本
  python evaluate.py --ckpt runs/exp1/checkpoint_best.pt --mode generate \
      --prompt "Once upon a time" --max_tokens 256

  # 计算 perplexity
  python evaluate.py --ckpt runs/exp1/checkpoint_best.pt --mode perplexity \
      --val_data data/tinystories/val.npy

  # 两个都做
  python evaluate.py --ckpt runs/exp1/checkpoint_best.pt --mode all \
      --val_data data/tinystories/val.npy \
      --tokenizer_dir data/tinystories \
      --prompt "Once upon a time"
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

from cs336_basics.data import get_batch
from cs336_basics.nn import run_cross_entropy, run_softmax

# Reuse model definition from train.py
from train import TransformerLM, load_dataset


# -----------------------------------------------------------------------------
# Load tokenizer from saved files
# -----------------------------------------------------------------------------
def load_tokenizer(tokenizer_dir: str, special_tokens=None):
    """从 tokenize_data.py 保存的文件加载 tokenizer。"""
    from cs336_basics.tokenizer import get_tokenizer

    special_tokens = special_tokens or ["<|endoftext|>"]
    vocab_path = os.path.join(tokenizer_dir, "tokenizer.vocab")
    merges_path = os.path.join(tokenizer_dir, "tokenizer.merges")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)
    vocab = {int(k): v.encode("latin-1") for k, v in vocab_raw.items()}

    with open(merges_path, "r", encoding="utf-8") as f:
        merges_raw = json.load(f)
    merges = [(p1.encode("latin-1"), p2.encode("latin-1")) for p1, p2 in merges_raw]

    return get_tokenizer(vocab, merges, special_tokens=special_tokens)


# -----------------------------------------------------------------------------
# Load model from checkpoint
# -----------------------------------------------------------------------------
def load_model(ckpt_path: str, device: str, **model_kwargs) -> tuple:
    """加载 checkpoint，返回 (model, step)。"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = TransformerLM(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    step = ckpt.get("iteration", -1)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型已加载: step={step}, params={num_params:,} ({num_params/1e6:.1f}M)")
    return model, step


# -----------------------------------------------------------------------------
# Text Generation (Decoding)
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: str = "cuda",
) -> str:
    """
    自回归文本生成，支持 temperature scaling 和 top-p (nucleus) sampling。

    Args:
        model: 训练好的 TransformerLM
        tokenizer: BPEInferenceTokenizer
        prompt: 输入提示文本
        max_tokens: 最多生成多少个 token
        temperature: 温度参数 (0 → greedy, 1 → 原始分布, >1 → 更随机)
        top_p: nucleus sampling 阈值 (1.0 = 关闭)
        device: 设备
    """
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    generated_ids = list(input_ids)

    # Find <|endoftext|> token id
    eos_token = "<|endoftext|>"
    eos_id = None
    if hasattr(tokenizer, 'special_token_to_id') and eos_token in tokenizer.special_token_to_id:
        eos_id = tokenizer.special_token_to_id[eos_token]

    context_length = model.context_length

    for _ in range(max_tokens):
        # Truncate to context_length
        if len(generated_ids) > context_length:
            context_ids = generated_ids[-context_length:]
        else:
            context_ids = generated_ids

        x = torch.tensor([context_ids], dtype=torch.long, device=device)
        logits = model(x)  # (1, seq_len, vocab_size)

        # Take logits at last position
        next_logits = logits[0, -1, :]  # (vocab_size,)

        # Temperature scaling
        if temperature > 0:
            next_logits = next_logits / temperature
        else:
            # Greedy: just take argmax
            next_id = next_logits.argmax().item()
            generated_ids.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
            continue

        # Softmax to get probabilities
        probs = run_softmax(next_logits, dim=-1)

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Find cutoff: smallest set where cumulative prob >= top_p
            cutoff_mask = cumulative_probs - sorted_probs >= top_p
            sorted_probs[cutoff_mask] = 0.0

            # Renormalize
            sorted_probs = sorted_probs / sorted_probs.sum()

            # Sample from filtered distribution
            next_idx = torch.multinomial(sorted_probs, num_samples=1).item()
            next_id = sorted_indices[next_idx].item()
        else:
            # Sample from full distribution
            next_id = torch.multinomial(probs, num_samples=1).item()

        generated_ids.append(next_id)

        if eos_id is not None and next_id == eos_id:
            break

    return tokenizer.decode(generated_ids)


# -----------------------------------------------------------------------------
# Perplexity evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_perplexity(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    max_batches: int = 100,
) -> dict:
    """计算验证集 perplexity。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for _ in range(max_batches):
        try:
            x, y = get_batch(dataset, batch_size, context_length, device)
        except Exception:
            break
        logits = model(x)
        B, T, V = logits.shape
        loss = run_cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
        total_loss += loss.item() * (B * T)
        total_tokens += B * T
        num_batches += 1

    if total_tokens == 0:
        return {"perplexity": float("nan"), "avg_loss": float("nan")}

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "num_batches": num_batches,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="CS336 模型评估与生成")

    # Required
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint 路径")
    p.add_argument("--mode", type=str, default="all",
                    choices=["generate", "perplexity", "all"])

    # Model config (需与训练时一致)
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--init_scale", type=float, default=0.02)

    # Data
    p.add_argument("--val_data", type=str, default=None)
    p.add_argument("--tokenizer_dir", type=str, default="data/tinystories",
                    help="tokenizer.vocab 和 tokenizer.merges 所在目录")

    # Generation
    p.add_argument("--prompt", type=str, default="Once upon a time",
                    help="生成文本的起始 prompt")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--num_samples", type=int, default=3,
                    help="生成几个不同的样本")

    # Perplexity
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_batches", type=int, default=100)

    # Device
    p.add_argument("--device", type=str, default="cuda")

    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ---- Load model ----
    model_kwargs = dict(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        init_scale=args.init_scale,
    )
    model, step = load_model(args.ckpt, device, **model_kwargs)

    # ---- Perplexity ----
    if args.mode in ("perplexity", "all"):
        if args.val_data is None:
            print("⚠️  需要 --val_data 来计算 perplexity")
        else:
            print("\n" + "=" * 60)
            print("Perplexity 评估")
            print("=" * 60)
            val_data = load_dataset(args.val_data)
            results = compute_perplexity(
                model, val_data, args.batch_size,
                args.context_length, device, args.max_batches,
            )
            print(f"  Avg Loss:    {results['avg_loss']:.4f}")
            print(f"  Perplexity:  {results['perplexity']:.2f}")
            print(f"  Tokens:      {results['total_tokens']:,}")
            print(f"  Batches:     {results['num_batches']}")

            # 课程目标参考
            print("\n  📊 课程目标参考:")
            print(f"     TinyStories val loss ≤ 1.45 → 你的: {results['avg_loss']:.4f}")
            if results['avg_loss'] <= 1.45:
                print("     ✅ 达标!")
            else:
                print(f"     ❌ 还差 {results['avg_loss'] - 1.45:.4f}")

    # ---- Text Generation ----
    if args.mode in ("generate", "all"):
        print("\n" + "=" * 60)
        print("文本生成")
        print("=" * 60)

        tokenizer = load_tokenizer(args.tokenizer_dir)

        prompts = [args.prompt]
        # 也可以加一些默认 prompt 来测试
        if args.prompt == "Once upon a time":
            prompts = [
                "Once upon a time",
                "The little girl",
                "One day, a boy named",
            ]

        for i in range(args.num_samples):
            for prompt in prompts:
                print(f"\n--- Sample {i+1} | prompt: \"{prompt}\" ---")
                print(f"    (temperature={args.temperature}, top_p={args.top_p})")
                text = generate(
                    model, tokenizer, prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                )
                print(text)
                print()

        # 不同 temperature 对比
        print("\n" + "=" * 60)
        print("Temperature 对比 (同一 prompt)")
        print("=" * 60)
        test_prompt = "Once upon a time"
        for temp in [0.3, 0.8, 1.2]:
            print(f"\n--- temperature={temp} ---")
            text = generate(
                model, tokenizer, test_prompt,
                max_tokens=128, temperature=temp,
                top_p=args.top_p, device=device,
            )
            print(text)


if __name__ == "__main__":
    main()