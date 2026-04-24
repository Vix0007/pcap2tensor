"""End-to-end pipeline tests using a temporary synthetic PCAP."""

import pytest
from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import Ether
from scapy.utils import wrpcap

from pcap2tensor import PCAPExtractor, extract


@pytest.fixture
def tiny_pcap(tmp_path):
    """Create a small synthetic PCAP file."""
    packets = []
    for i in range(50):
        pkt = (
            Ether()
            / IP(src="192.168.1.10", dst="8.8.8.8")
            / TCP(sport=12345, dport=443, flags="PA", window=8192)
            / (b"X" * 100)
        )
        pkt.time = 1.0 + i * 0.01
        packets.append(pkt)
    path = tmp_path / "tiny.pcap"
    wrpcap(str(path), packets)
    return str(path)


def test_extract_end_to_end(tiny_pcap):
    tensor = extract(tiny_pcap, window_size=10, stride=5, progress=False)
    assert tensor.dim() == 3
    assert tensor.shape[1] == 10
    assert tensor.shape[2] == 6  # aegis-6d


def test_extractor_produces_chunks(tiny_pcap):
    ext = PCAPExtractor(features="basic-3d", window_size=10, stride=5)
    chunks = list(ext.extract_chunks(tiny_pcap, progress=False))
    assert len(chunks) >= 1
    assert chunks[0].shape[2] == 3


def test_save_writes_files(tiny_pcap, tmp_path):
    ext = PCAPExtractor(window_size=10, stride=5)
    out_dir = tmp_path / "tensors"
    written = ext.save(tiny_pcap, out_dir, progress=False)
    assert len(written) >= 1
    for path in written:
        assert path.endswith(".pt")


def test_missing_pcap_raises():
    with pytest.raises(FileNotFoundError):
        extract("/does/not/exist.pcap", progress=False)


def test_feature_dim_matches_preset():
    ext = PCAPExtractor(features="extended-10d")
    assert ext.feature_dim == 10
    ext2 = PCAPExtractor(features="full-13d")
    assert ext2.feature_dim == 13
