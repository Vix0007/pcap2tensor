"""Tests for built-in feature extractors using synthetic packets."""

import pytest
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether

from pcap2tensor import (
    IAT,
    Direction,
    PayloadRatio,
    PortCategory,
    ProtocolOneHot,
    Size,
    TCPFlags,
    TCPWindow,
)


def make_tcp_packet(
    src="192.168.1.10",
    dst="8.8.8.8",
    sport=12345,
    dport=443,
    flags="S",
    window=8192,
    payload=b"",
    timestamp=1.0,
):
    pkt = (
        Ether()
        / IP(src=src, dst=dst)
        / TCP(sport=sport, dport=dport, flags=flags, window=window)
        / payload
    )
    pkt.time = timestamp
    return pkt


def test_size_normalizes_and_clips():
    feat = Size()
    small = make_tcp_packet()
    result = feat(small)
    assert 0.0 <= result <= 1.0


def test_iat_first_packet_is_zero():
    feat = IAT()
    pkt = make_tcp_packet(timestamp=1.0)
    assert feat(pkt) == 0.0


def test_iat_computes_between_same_flow():
    feat = IAT()
    pkt1 = make_tcp_packet(timestamp=1.0)
    pkt2 = make_tcp_packet(timestamp=1.5)
    feat(pkt1)
    assert feat(pkt2) == pytest.approx(0.5)


def test_iat_reset_clears_state():
    feat = IAT()
    feat(make_tcp_packet(timestamp=1.0))
    feat.reset()
    assert feat.flow_times == {}


def test_direction_private_source():
    feat = Direction()
    assert feat(make_tcp_packet(src="192.168.1.1")) == 1.0
    assert feat(make_tcp_packet(src="10.0.0.5")) == 1.0
    assert feat(make_tcp_packet(src="172.20.1.1")) == 1.0


def test_direction_public_source():
    feat = Direction()
    assert feat(make_tcp_packet(src="8.8.8.8")) == -1.0


def test_tcp_window_normalized():
    feat = TCPWindow()
    pkt = make_tcp_packet(window=65535)
    assert feat(pkt) == pytest.approx(1.0)


def test_tcp_flags_normalized():
    feat = TCPFlags()
    pkt = make_tcp_packet(flags="S")
    assert 0.0 <= feat(pkt) <= 1.0


def test_payload_ratio_with_payload():
    feat = PayloadRatio()
    pkt = make_tcp_packet(payload=b"A" * 100)
    assert feat(pkt) > 0.0


def test_protocol_onehot_tcp():
    feat = ProtocolOneHot()
    assert feat(make_tcp_packet()) == [1.0, 0.0, 0.0, 0.0]


def test_protocol_onehot_udp():
    feat = ProtocolOneHot()
    pkt = Ether() / IP(src="192.168.1.1", dst="8.8.8.8") / UDP(sport=12345, dport=53)
    pkt.time = 1.0
    assert feat(pkt) == [0.0, 1.0, 0.0, 0.0]


def test_port_category_well_known():
    feat = PortCategory()
    pkt = make_tcp_packet(dport=443)
    assert feat(pkt) == [1.0, 0.0, 0.0]


def test_port_category_dynamic():
    feat = PortCategory()
    pkt = make_tcp_packet(dport=55000)
    assert feat(pkt) == [0.0, 0.0, 1.0]
