"""Named feature preset bundles.

Presets return fresh ``Feature`` instances with clean state. Resolve by name:

    >>> from pcap2tensor import get_preset
    >>> feats = get_preset("aegis-6d")

Or import the builder directly:

    >>> from pcap2tensor import AEGIS_6D
    >>> feats = AEGIS_6D()
"""

from __future__ import annotations

from pcap2tensor.features import (
    IAT,
    Direction,
    Feature,
    PayloadRatio,
    PortCategory,
    ProtocolOneHot,
    Size,
    TCPFlags,
    TCPWindow,
)


def AEGIS_6D() -> list[Feature]:
    """The 6-feature set used in the AEGIS paper (arXiv:2604.02149).

    Features: size, IAT, direction, TCP window, TCP flags, payload ratio.
    Output dimension: 6.
    """
    return [Size(), IAT(), Direction(), TCPWindow(), TCPFlags(), PayloadRatio()]


def BASIC_3D() -> list[Feature]:
    """Minimal 3-feature set for lightweight models.

    Features: size, IAT, direction.
    Output dimension: 3.
    """
    return [Size(), IAT(), Direction()]


def EXTENDED_10D() -> list[Feature]:
    """AEGIS_6D + one-hot transport protocol.

    Output dimension: 10.
    """
    return [
        Size(),
        IAT(),
        Direction(),
        TCPWindow(),
        TCPFlags(),
        PayloadRatio(),
        ProtocolOneHot(),
    ]


def FULL_13D() -> list[Feature]:
    """EXTENDED_10D + destination port category one-hot.

    Output dimension: 13.
    """
    return [
        Size(),
        IAT(),
        Direction(),
        TCPWindow(),
        TCPFlags(),
        PayloadRatio(),
        ProtocolOneHot(),
        PortCategory(),
    ]


_PRESETS = {
    "aegis-6d": AEGIS_6D,
    "basic-3d": BASIC_3D,
    "extended-10d": EXTENDED_10D,
    "full-13d": FULL_13D,
}


def get_preset(name: str) -> list[Feature]:
    """Resolve a preset name to a fresh list of ``Feature`` instances.

    Normalization: case-insensitive, underscores equivalent to hyphens.
    """
    key = name.lower().replace("_", "-")
    if key not in _PRESETS:
        raise ValueError(f"Unknown preset: {name!r}. Available: {sorted(_PRESETS.keys())}")
    return _PRESETS[key]()


def list_presets() -> list[str]:
    """Return all available preset names (sorted)."""
    return sorted(_PRESETS.keys())
