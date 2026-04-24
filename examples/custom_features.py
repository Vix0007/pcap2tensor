"""Custom feature example — payload byte entropy.

Shows two patterns:

1. Stateless feature: Shannon entropy of the payload.
2. Stateful feature: packets-per-second estimate maintained per flow.
"""
from __future__ import annotations

import math
from collections import Counter

from scapy.layers.inet import TCP, UDP

from pcap2tensor import Direction, Feature, IAT, PCAPExtractor, Size


# ---------------------------------------------------------------------------
# 1. Stateless custom feature
# ---------------------------------------------------------------------------
class PayloadEntropy(Feature):
    """Shannon entropy of the transport payload, normalized to [0, 1]."""

    name = "payload_entropy"
    dim = 1

    def __call__(self, pkt) -> float:
        if TCP in pkt:
            payload = bytes(pkt[TCP].payload)
        elif UDP in pkt:
            payload = bytes(pkt[UDP].payload)
        else:
            return 0.0

        if not payload:
            return 0.0

        total = len(payload)
        counts = Counter(payload)
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
        return entropy / 8.0  # 8 bits maximum → normalize to 0-1


# ---------------------------------------------------------------------------
# 2. Stateful custom feature
# ---------------------------------------------------------------------------
class FlowPacketRate(Feature):
    """Exponential moving average of packet rate (1/IAT) within each flow.

    Demonstrates a stateful feature — state is reset automatically between
    PCAPs by the extractor.
    """

    name = "flow_packet_rate"
    dim = 1

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = float(alpha)
        self._last_time: dict[tuple, float] = {}
        self._ema: dict[tuple, float] = {}

    def __call__(self, pkt) -> float:
        from pcap2tensor.features import _ip_info, _ports  # internal helpers
        info = _ip_info(pkt)
        if info is None:
            return 0.0
        src, dst, proto = info
        sport, dport = _ports(pkt)
        key = (src, dst, sport, dport, proto)
        now = float(pkt.time)

        prev = self._last_time.get(key)
        self._last_time[key] = now
        if prev is None or now <= prev:
            return 0.0

        inst_rate = 1.0 / (now - prev)
        ema_prev = self._ema.get(key, inst_rate)
        ema = self.alpha * inst_rate + (1 - self.alpha) * ema_prev
        self._ema[key] = ema
        # Cap at 1000 pps, normalize to 0-1
        return min(ema, 1000.0) / 1000.0

    def reset(self) -> None:
        self._last_time.clear()
        self._ema.clear()


# ---------------------------------------------------------------------------
# Compose with built-ins
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    feats = [
        Size(),
        IAT(),
        Direction(),
        PayloadEntropy(),
        FlowPacketRate(alpha=0.3),
    ]
    extractor = PCAPExtractor(features=feats, window_size=500, stride=250)
    tensor = extractor.extract("capture.pcap")
    print(f"custom feature tensor: {tuple(tensor.shape)}")
    #   → (num_windows, 500, 5)
