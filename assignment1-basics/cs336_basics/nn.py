from __future__ import annotations

import torch
from torch import Tensor
from typing import Iterable, Any
from jaxtyping import Float, Int

# ==========================================
# 1. Linear (线性层) - 手写实现
# ==========================================
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    实现公式: Output = Input @ Weights^T
    不使用 nn.Linear，直接使用矩阵乘法。
    """
    # weights 的形状是 [d_out, d_in]，PyTorch 的 matmul 需要 [d_in, d_out]
    # 所以我们对 weights 进行转置 (.T)
    return in_features @ weights.T


# ==========================================
# 2. Softmax - 手写实现 (数值稳定版)
# ==========================================
def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    实现公式: exp(x_i) / sum(exp(x_j))
    为了数值稳定性，通常计算: exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    # 1. 找到最大值 (用于数值稳定性，防止 exp 溢出)
    # keepdim=True 保持维度以便广播相减
    x_max = in_features.max(dim=dim, keepdim=True).values
    
    # 2. 减去最大值后取指数
    x_exp = torch.exp(in_features - x_max)
    
    # 3. 计算分母的 sum
    x_sum = x_exp.sum(dim=dim, keepdim=True)
    
    # 4. 除法
    return x_exp / x_sum


# ==========================================
# 3. Cross Entropy - 手写实现
# ==========================================
def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], 
    targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """
    实现公式: Loss = - log(softmax(inputs)[target_index])
    等价于: - inputs[target_index] + log(sum(exp(inputs)))
    """
    batch_size = inputs.shape[0]
    
    # 1. 为了数值稳定性，先算 LogSumExp技巧
    # max_val: [batch_size, 1]
    max_val = inputs.max(dim=-1, keepdim=True).values
    # log_sum_exp = max + log(sum(exp(x - max)))
    log_sum_exp = max_val + torch.log(torch.exp(inputs - max_val).sum(dim=-1, keepdim=True))
    
    # 2. 获取正确类别的 logits
    # 我们需要从 inputs 中选出 targets 指定的那一列
    # inputs: [batch, vocab], targets: [batch]
    # 使用 gather 或者高级索引
    correct_logits = inputs[torch.arange(batch_size), targets]
    
    # 3. 计算每个样本的损失: log_sum_exp - correct_logit (因为公式是负对数似然)
    # log_softmax(x) = x - log_sum_exp
    # NLL = - log_softmax = log_sum_exp - x
    losses = log_sum_exp.squeeze() - correct_logits
    
    # 4. 返回平均损失
    return losses.mean()


# ==========================================
# 4. Gradient Clipping - 手写实现
# ==========================================
def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    实现梯度裁剪:
    1. 计算所有参数梯度的 L2 范数 total_norm
    2. 如果 total_norm > max_norm，则所有梯度乘以 (max_norm / total_norm)
    """
    # 过滤掉没有梯度的参数
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return

    # 1. 计算所有梯度的平方和 (L2 Norm 的平方)
    total_norm_sq = 0.0
    for p in params:
        total_norm_sq += p.grad.detach().data.norm(2).item() ** 2
    
    total_norm = total_norm_sq ** 0.5
    
    # 2. 计算缩放系数
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    
    # 3. 如果需要裁剪 (clip_coef < 1)，则修改梯度
    if clip_coef < 1:
        for p in params:
            p.grad.detach().data.mul_(clip_coef)

# ==========================================
# 5. 其他辅助函数
# ==========================================

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """
    SiLU (Swish) = x * sigmoid(x)
    """
    return in_features * torch.sigmoid(in_features)

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    RMSNorm = (x / RMS(x)) * weight
    RMS(x) = sqrt(mean(x^2) + eps)
    """
    # 计算均方根
    # mean over the last dimension
    mean_square = in_features.pow(2).mean(dim=-1, keepdim=True)
    rms = torch.rsqrt(mean_square + eps) # rsqrt 是 1/sqrt
    
    # 归一化并缩放
    return in_features * rms * weights