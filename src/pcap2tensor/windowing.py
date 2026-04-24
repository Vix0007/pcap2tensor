"""Sliding-window sequence construction."""

from __future__ import annotations

import torch


def sliding_window(
    tensor: torch.Tensor,
    window_size: int,
    stride: int,
) -> torch.Tensor:
    """Build sliding windows over a 2D tensor.

    Args:
        tensor: Shape ``(N, features)``.
        window_size: Packets per window. Must be >= 1.
        stride: Packets between consecutive window starts. Must be >= 1.

    Returns:
        Tensor of shape ``(num_windows, window_size, features)``. Empty
        (shape ``(0, window_size, features)``) if ``N < window_size``.

    Raises:
        ValueError: If ``tensor`` is not 2D or window/stride are invalid.
    """
    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D tensor of shape (N, features), got {tuple(tensor.shape)}")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    n, feats = tensor.shape
    if n < window_size:
        return torch.empty((0, window_size, feats), dtype=tensor.dtype)

    # unfold on dim 0 → shape (num_windows, features, window_size), then swap.
    windows = tensor.unfold(0, window_size, stride).transpose(1, 2).contiguous()
    return windows
