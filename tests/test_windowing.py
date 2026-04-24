"""Tests for sliding window construction."""

import pytest
import torch

from pcap2tensor import sliding_window


def test_basic_windowing():
    t = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    out = sliding_window(t, window_size=3, stride=1)
    assert out.shape == (8, 3, 2)


def test_stride_larger_than_one():
    t = torch.zeros(10, 4)
    out = sliding_window(t, window_size=4, stride=2)
    assert out.shape == (4, 4, 4)


def test_insufficient_packets_returns_empty():
    t = torch.zeros(5, 3)
    out = sliding_window(t, window_size=10, stride=5)
    assert out.shape == (0, 10, 3)


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        sliding_window(torch.zeros(3, 3, 3), window_size=2, stride=1)


def test_rejects_invalid_window_stride():
    t = torch.zeros(10, 2)
    with pytest.raises(ValueError):
        sliding_window(t, window_size=0, stride=1)
    with pytest.raises(ValueError):
        sliding_window(t, window_size=5, stride=0)
