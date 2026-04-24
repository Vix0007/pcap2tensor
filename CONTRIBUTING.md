# Contributing to pcap2tensor

Thanks for your interest. Contributions are welcome — bug reports, features,
new `Feature` classes, documentation improvements.

## Development setup

```bash
git clone https://github.com/Vix0007/pcap2tensor.git
cd pcap2tensor
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
```

## Submitting changes

1. Open an issue first for non-trivial changes.
2. Fork, branch from `main`, make your changes.
3. Add tests for new behavior.
4. Ensure `ruff check` and `pytest` pass.
5. Open a PR with a clear description.

## Adding a new Feature

Subclass `Feature`, implement `__call__`, set `name` and `dim`, override
`reset` if stateful. See `examples/custom_features.py`.