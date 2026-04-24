"""Per-packet feature extractors.

Each Feature is a stateful callable: ``Feature(pkt) -> float | list[float]``.
Features may maintain flow-level state (e.g. timing tables) that persists
across packets in the same PCAP. Call ``.reset()`` between PCAPs to clear it.

Built-in features:
    Size, IAT, Direction, TCPWindow, TCPFlags, PayloadRatio,
    ProtocolOneHot, PortCategory.

Custom features: subclass ``Feature``, implement ``__call__``, optionally
override ``name``, ``dim``, and ``reset``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6


class Feature(ABC):
    """Abstract base class for a packet-level feature extractor.

    Subclass and implement ``__call__``. Override ``dim`` if the feature
    returns a list of floats. Override ``reset`` if the feature holds state.
    """

    name: str = "feature"
    dim: int = 1

    @abstractmethod
    def __call__(self, pkt: Any) -> float | list[float]:
        """Extract the feature value(s) from a single Scapy packet."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any internal state. Called between PCAPs by the extractor."""
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, dim={self.dim})"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# All RFC1918 private prefixes for IPv4.
_PRIVATE_PREFIXES = (
    "10.",
    "192.168.",
    "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.",
    "172.28.", "172.29.", "172.30.", "172.31.",
)


def _ip_info(pkt: Any) -> tuple[str, str, int] | None:
    """Return (src, dst, protocol_number) or None if the packet has no IP layer."""
    if IP in pkt:
        ip = pkt[IP]
        return ip.src, ip.dst, int(ip.proto)
    if IPv6 in pkt:
        ip6 = pkt[IPv6]
        return ip6.src, ip6.dst, int(ip6.nh)
    return None


def _ports(pkt: Any) -> tuple[int, int]:
    """Return (sport, dport), or (0, 0) if no TCP/UDP layer."""
    if TCP in pkt:
        t = pkt[TCP]
        return int(t.sport), int(t.dport)
    if UDP in pkt:
        u = pkt[UDP]
        return int(u.sport), int(u.dport)
    return 0, 0


# --------------------------------------------------------------------------- #
# Built-in features
# --------------------------------------------------------------------------- #

class Size(Feature):
    """Normalized packet size, clipped to [``min_frac``, 1.0].

    Default: size / 1500 (Ethernet MTU), floor 0.026.
    """

    name = "size"
    dim = 1

    def __init__(self, mtu: float = 1500.0, min_frac: float = 0.026) -> None:
        self.mtu = float(mtu)
        self.min_frac = float(min_frac)

    def __call__(self, pkt: Any) -> float:
        return min(max(len(pkt) / self.mtu, self.min_frac), 1.0)


class IAT(Feature):
    """Per-flow inter-arrival time in seconds, clipped to ``max_iat``.

    Flow key is the 5-tuple (src, dst, sport, dport, proto). First packet
    of each flow returns 0.0.
    """

    name = "iat"
    dim = 1

    def __init__(self, max_iat: float = 10.0) -> None:
        self.max_iat = float(max_iat)
        self.flow_times: dict[tuple, float] = {}

    def __call__(self, pkt: Any) -> float:
        info = _ip_info(pkt)
        if info is None:
            return 0.0
        src, dst, proto = info
        sport, dport = _ports(pkt)
        key = (src, dst, sport, dport, proto)
        now = float(pkt.time)

        prev = self.flow_times.get(key)
        iat = 0.0 if prev is None else min(now - prev, self.max_iat)
        self.flow_times[key] = now
        return iat

    def reset(self) -> None:
        self.flow_times.clear()


class Direction(Feature):
    """Coarse flow direction sign: +1.0 if source is RFC1918 private, else -1.0.

    Returns 0.0 for packets with no IP layer.
    """

    name = "direction"
    dim = 1

    def __call__(self, pkt: Any) -> float:
        info = _ip_info(pkt)
        if info is None:
            return 0.0
        src = info[0]
        return 1.0 if src.startswith(_PRIVATE_PREFIXES) else -1.0


class TCPWindow(Feature):
    """Normalized TCP receive window (``window / 65535``). 0.0 for non-TCP."""

    name = "tcp_window"
    dim = 1

    def __call__(self, pkt: Any) -> float:
        if TCP in pkt:
            return float(pkt[TCP].window) / 65535.0
        return 0.0


class TCPFlags(Feature):
    """Normalized TCP flags byte (``flags / 255``). 0.0 for non-TCP."""

    name = "tcp_flags"
    dim = 1

    def __call__(self, pkt: Any) -> float:
        if TCP in pkt:
            return float(int(pkt[TCP].flags)) / 255.0
        return 0.0


class PayloadRatio(Feature):
    """Payload bytes / total packet bytes, for TCP or UDP. 0.0 otherwise."""

    name = "payload_ratio"
    dim = 1

    def __call__(self, pkt: Any) -> float:
        total = len(pkt)
        if total == 0:
            return 0.0
        if TCP in pkt:
            return len(pkt[TCP].payload) / total
        if UDP in pkt:
            return len(pkt[UDP].payload) / total
        return 0.0


class ProtocolOneHot(Feature):
    """One-hot of transport protocol: [TCP, UDP, ICMP, other]. Dim = 4."""

    name = "protocol_onehot"
    dim = 4

    def __call__(self, pkt: Any) -> list[float]:
        info = _ip_info(pkt)
        if info is None:
            return [0.0, 0.0, 0.0, 1.0]
        proto = info[2]
        if proto == 6:        # TCP
            return [1.0, 0.0, 0.0, 0.0]
        if proto == 17:       # UDP
            return [0.0, 1.0, 0.0, 0.0]
        if proto in (1, 58):  # ICMPv4 or ICMPv6
            return [0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]


class PortCategory(Feature):
    """One-hot of destination port range: [well-known, registered, dynamic]. Dim = 3.

    - Well-known: 1-1023
    - Registered: 1024-49151
    - Dynamic: 49152-65535

    All-zero if no transport layer.
    """

    name = "port_category"
    dim = 3

    def __call__(self, pkt: Any) -> list[float]:
        _, dport = _ports(pkt)
        if dport == 0:
            return [0.0, 0.0, 0.0]
        if dport <= 1023:
            return [1.0, 0.0, 0.0]
        if dport <= 49151:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]
