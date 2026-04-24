# Changelog

All notable changes to pcap2tensor are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-24

### Added
- Initial public release.
- Streaming `PCAPExtractor` with chunked output for PCAPs larger than RAM.
- Module-level helpers: `extract()`, `batch_extract()`.
- Four feature presets: `basic-3d`, `aegis-6d`, `extended-10d`, `full-13d`.
- Eight built-in `Feature` classes: `Size`, `IAT`, `Direction`, `TCPWindow`,
  `TCPFlags`, `PayloadRatio`, `ProtocolOneHot`, `PortCategory`.
- Parallel batch extraction via `ProcessPoolExecutor`.
- CLI: `pcap2tensor extract`, `pcap2tensor batch`, `pcap2tensor presets`.
- First-class IPv4 + IPv6 support.
- Malformed-packet tolerance — runs do not abort on a single bad packet.

[0.1.0]: https://github.com/Vix0007/pcap2tensor/releases/tag/v0.1.0