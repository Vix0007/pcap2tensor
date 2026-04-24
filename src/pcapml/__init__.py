"""pcapml: PCAP → ML tensor extraction for network intrusion detection research.

Quick start:

    >>> from pcapml import extract
    >>> tensor = extract("capture.pcap", features="aegis-6d", window_size=1000, stride=500)
    >>> tensor.shape
    torch.Size([num_windows, 1000, 6])

See https://github.com/vicksonferrel/pcapml for full documentation.
"""
from __future__ import annotations

from pcapml.extractor import PCAPExtractor, batch_extract, extract
from pcapml.features import (
    Direction,
    Feature,
    IAT,
    PayloadRatio,
    PortCategory,
    ProtocolOneHot,
    Size,
    TCPFlags,
    TCPWindow,
)
from pcapml.presets import AEGIS_6D, BASIC_3D, EXTENDED_10D, FULL_13D, get_preset, list_presets
from pcapml.windowing import sliding_window

__version__ = "0.1.0"

__all__ = [
    # Core API
    "PCAPExtractor",
    "extract",
    "batch_extract",
    "sliding_window",
    # Feature classes
    "Feature",
    "Size",
    "IAT",
    "Direction",
    "TCPWindow",
    "TCPFlags",
    "PayloadRatio",
    "ProtocolOneHot",
    "PortCategory",
    # Presets
    "AEGIS_6D",
    "BASIC_3D",
    "EXTENDED_10D",
    "FULL_13D",
    "get_preset",
    "list_presets",
    # Meta
    "__version__",
]
