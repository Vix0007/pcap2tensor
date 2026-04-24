"""Command-line interface: ``pcap2tensor extract``, ``pcap2tensor batch``, ``pcap2tensor presets``."""

from __future__ import annotations

import argparse
import sys

from pcap2tensor import __version__
from pcap2tensor.extractor import PCAPExtractor, batch_extract
from pcap2tensor.presets import list_presets


def _cmd_extract(args: argparse.Namespace) -> int:
    extractor = PCAPExtractor(
        features=args.features,
        window_size=args.window,
        stride=args.stride,
        chunk_size=args.chunk_size,
    )
    written = extractor.save(
        pcap_path=args.input,
        output_dir=args.output,
        prefix=args.prefix,
        progress=not args.quiet,
    )
    if not args.quiet:
        print(f"\nWrote {len(written)} chunk(s) to {args.output}")
        for p in written:
            print(f"  {p}")
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    results = batch_extract(
        input_spec=args.input,
        output_dir=args.output,
        features=args.features,
        window_size=args.window,
        stride=args.stride,
        chunk_size=args.chunk_size,
        prefix=args.prefix,
        workers=args.workers,
        recursive=not args.no_recursive,
    )
    total = sum(len(v) for v in results.values())
    if not args.quiet:
        print(f"\nProcessed {len(results)} PCAP(s) → {total} chunk(s) in {args.output}")
    return 0


def _cmd_presets(_: argparse.Namespace) -> int:
    for name in list_presets():
        print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pcap2tensor",
        description="PCAP → ML tensor extraction for network intrusion detection.",
    )
    p.add_argument("-V", "--version", action="version", version=f"pcap2tensor {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # extract
    ex = sub.add_parser("extract", help="Extract a single PCAP to tensor chunks.")
    ex.add_argument("input", help="Path to .pcap or .pcapng file.")
    ex.add_argument("-o", "--output", default="./tensors", help="Output directory.")
    ex.add_argument("-f", "--features", default="aegis-6d", help="Feature preset name.")
    ex.add_argument("-w", "--window", type=int, default=1000, help="Window size in packets.")
    ex.add_argument("-s", "--stride", type=int, default=500, help="Stride in packets.")
    ex.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="Packets buffered before flushing a chunk.",
    )
    ex.add_argument("-p", "--prefix", default="pcap2tensor", help="Output filename prefix.")
    ex.add_argument("-q", "--quiet", action="store_true", help="Suppress output.")
    ex.set_defaults(func=_cmd_extract)

    # batch
    ba = sub.add_parser("batch", help="Extract many PCAPs in parallel.")
    ba.add_argument("input", help="Directory or glob pattern for PCAPs.")
    ba.add_argument("-o", "--output", default="./tensors", help="Output directory.")
    ba.add_argument("-f", "--features", default="aegis-6d", help="Feature preset name.")
    ba.add_argument("-w", "--window", type=int, default=1000, help="Window size in packets.")
    ba.add_argument("-s", "--stride", type=int, default=500, help="Stride in packets.")
    ba.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="Packets buffered per chunk.",
    )
    ba.add_argument("-p", "--prefix", default="pcap2tensor", help="Output filename prefix.")
    ba.add_argument("-n", "--workers", type=int, default=4, help="Parallel worker processes.")
    ba.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse into subdirectories.",
    )
    ba.add_argument("-q", "--quiet", action="store_true", help="Suppress output.")
    ba.set_defaults(func=_cmd_batch)

    # presets
    pr = sub.add_parser("presets", help="List available feature presets.")
    pr.set_defaults(func=_cmd_presets)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
