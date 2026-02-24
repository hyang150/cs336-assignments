from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random language-modeling batch from a 1D token dataset.

    Args:
        dataset: 1D numpy array (or memmap) of token ids.
        batch_size: Number of training examples to sample.
        context_length: Length of each input/target sequence.
        device: Target torch device string (e.g. "cpu", "cuda:0", "mps").

    Returns:
        x, y where:
            - x.shape == (batch_size, context_length)
            - y.shape == (batch_size, context_length)
            - y is x shifted by one token to the right in the source stream.
    """
    if dataset.ndim != 1:
        raise ValueError(f"Expected a 1D dataset, got shape {dataset.shape}.")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if len(dataset) <= context_length:
        raise ValueError(
            f"Dataset length ({len(dataset)}) must be greater than context_length ({context_length})."
        )

    # Valid starts satisfy: start + context_length < len(dataset)
    # so that we can read both x[start:start+context_length] and
    # y[start+1:start+1+context_length].
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=(batch_size,))

    x_np = np.stack([dataset[s : s + context_length] for s in starts], axis=0)
    y_np = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts], axis=0)

    # Token IDs should be integer tensors for embedding lookup / CE targets.
    x = torch.as_tensor(x_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    return x, y
