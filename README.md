# pcapml

**PCAP → ML tensor extraction for network intrusion detection research.**

A fast, streaming, production-grade Python library for turning raw packet captures into training-ready tensors. Built for NIDS researchers tired of rolling their own extraction pipeline for every paper.

[![PyPI](https://img.shields.io/pypi/v/pcapml.svg)](https://pypi.org/project/pcapml)
[![Python](https://img.shields.io/pypi/pyversions/pcapml.svg)](https://pypi.org/project/pcapml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2604.02149-b31b1b.svg)](https://arxiv.org/abs/2604.02149)

---

## Why this exists

Every ML-based network intrusion detection paper reinvents the same pipeline:

1. Parse a PCAP
2. Extract per-packet features (size, inter-arrival time, direction, TCP flags, ...)
3. Slide a window across the sequence
4. Save as a tensor

Every implementation is a one-file script that doesn't handle PCAPs larger than RAM, doesn't expose clean extension points for custom features, and quietly crashes on the first malformed packet. `pcapml` is that pipeline, packaged properly — streaming, extensible, and published on PyPI.

## Install

```bash
pip install pcapml
```

Python ≥ 3.9. Depends on Scapy, PyTorch, NumPy, tqdm.

## Quickstart

```python
from pcapml import extract

tensor = extract("capture.pcap", features="aegis-6d", window_size=1000, stride=500)
print(tensor.shape)     # torch.Size([num_windows, 1000, 6])
```

Feed straight into any sequence model — Transformer, LSTM, SSM, CNN.

## Large PCAPs

Streaming chunked processing — never loads the full PCAP into memory:

```python
from pcapml import PCAPExtractor

extractor = PCAPExtractor(
    features="aegis-6d",
    window_size=1000,
    stride=500,
    chunk_size=2_000_000,   # flush every 2M packets
)

# Option A: save chunked .pt files
extractor.save("massive.pcap", output_dir="./tensors/")

# Option B: stream chunks into your training loop
for chunk in extractor.extract_chunks("massive.pcap"):
    train_step(chunk)
```

## Parallel batch

```python
from pcapml import batch_extract

batch_extract("./pcaps/", output_dir="./tensors/", features="aegis-6d", workers=8)
```

From the CLI:

```bash
pcapml batch ./pcaps/ -o ./tensors/ -n 8
```

## Feature presets

| Preset         | Dim | Features                                                                    |
| -------------- | --- | --------------------------------------------------------------------------- |
| `basic-3d`     | 3   | size, IAT, direction                                                        |
| `aegis-6d`     | 6   | size, IAT, direction, TCP window, TCP flags, payload ratio                  |
| `extended-10d` | 10  | `aegis-6d` + protocol one-hot (TCP/UDP/ICMP/other)                          |
| `full-13d`     | 13  | `extended-10d` + destination port category (well-known/registered/dynamic)  |

The `aegis-6d` preset matches the feature set in [AEGIS (Ferrel, 2026)](https://arxiv.org/abs/2604.02149) — a TVD-HL-SSM architecture achieving F1 0.9952 on encrypted traffic detection at 262 μs inference latency.

## Custom features

A `Feature` is any stateful callable returning a float or a flat list of floats. Subclass `Feature`, implement `__call__`, optionally override `reset` if you hold state:

```python
import math
from collections import Counter
from scapy.layers.inet import TCP
from pcapml import PCAPExtractor, Feature, Size, IAT, Direction


class PayloadEntropy(Feature):
    name = "payload_entropy"
    dim = 1

    def __call__(self, pkt):
        payload = bytes(pkt[TCP].payload) if TCP in pkt else b""
        if not payload:
            return 0.0
        counts = Counter(payload)
        n = len(payload)
        return -sum((c / n) * math.log2(c / n) for c in counts.values()) / 8.0


extractor = PCAPExtractor(
    features=[Size(), IAT(), Direction(), PayloadEntropy()],
)
tensor = extractor.extract("capture.pcap")
```

Return a `list[float]` and set `dim` accordingly for multi-valued features (e.g. one-hots).

## CLI

```bash
# Single PCAP
pcapml extract capture.pcap -o ./tensors/

# Parallel batch over a directory
pcapml batch ./pcaps/ -o ./tensors/ -n 8

# List presets
pcapml presets

# Override everything
pcapml extract capture.pcap -f extended-10d -w 2000 -s 1000 -c 5000000
```

## Design

| Concern              | How it's handled                                                            |
| -------------------- | --------------------------------------------------------------------------- |
| Memory               | Streaming `PcapReader`, chunked flush every `chunk_size` packets            |
| Malformed packets    | Caught per-packet, silently skipped — a 4-hour run doesn't die on one pkt  |
| Flow state           | Per-`Feature` instance, auto-reset between PCAPs                            |
| Parallelism          | `ProcessPoolExecutor` for batch mode                                        |
| IPv6                 | First-class (IPv6 src/dst, port extraction, protocol number)                |
| Reproducibility      | Same PCAP + same config = bit-identical tensor output                       |
| Output format        | PyTorch `.pt` on disk, `torch.Tensor` in memory                             |

## Performance

Rough single-core throughput with `aegis-6d` on a modern x86 machine:
roughly 50–120k packets/sec, TCP-heavy captures slower than UDP-heavy. With
8 workers in batch mode, processing 100 GB+ of PCAPs per hour is achievable.

Your bottleneck is Scapy parsing, not feature extraction.

## Output shape

Every extractor produces tensors of shape:

```
(num_windows, window_size, feature_dim)
```

where `feature_dim = sum(f.dim for f in features)`. For `aegis-6d`, that's 6.

## Citation

If you use this library in research, please cite the companion paper:

```bibtex
@article{ferrel2026aegis,
  title   = {AEGIS: Adversarial Entropy-Guided Immune System --
             Thermodynamic State Space Models for Zero-Day Network
             Evasion Detection},
  author  = {Ferrel, Vickson},
  journal = {arXiv preprint arXiv:2604.02149},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.02149}
}
```

## License

MIT © Vickson Ferrel — [Vixero Technology Enterprise](https://vixdev.cloud)

---

**Built in Sarawak. For network defenders everywhere.** 🛡️
