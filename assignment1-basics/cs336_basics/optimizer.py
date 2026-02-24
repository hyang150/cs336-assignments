from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m = state["m"]
                v = state["v"]

                # First and second moment updates.
                m.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                v.mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))

                # Bias correction as in Adam.
                alpha_t = lr * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)

                # Parameter update.
                p.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                # Decoupled weight decay (AdamW).
                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine schedule with linear warmup.

    Warmup:
        if it < T_w: lr = (it / T_w) * lr_max
    Cosine:
        if T_w <= it <= T_c:
            lr = lr_min + 0.5 * (1 + cos(pi * (it - T_w)/(T_c - T_w))) * (lr_max - lr_min)
    Floor:
        if it > T_c: lr = lr_min
    """
    if it < warmup_iters:
        if warmup_iters <= 0:
            return max_learning_rate
        return (it / warmup_iters) * max_learning_rate

    if it > cosine_cycle_iters:
        return min_learning_rate

    if cosine_cycle_iters == warmup_iters:
        return min_learning_rate

    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)
