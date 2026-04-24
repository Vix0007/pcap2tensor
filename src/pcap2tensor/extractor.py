"""PCAP → feature tensor extraction pipeline.

Core class: :class:`PCAPExtractor`. One-shot helpers: :func:`extract`,
:func:`batch_extract`.
"""
from __future__ import annotations

import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator, Union

import torch
from scapy.all import PcapReader
from scapy.layers.inet import IP
from scapy.layers.inet6 import IPv6
from tqdm import tqdm

from pcap2tensor.features import Feature
from pcap2tensor.presets import get_preset
from pcap2tensor.windowing import sliding_window

PathLike = Union[str, Path]


class PCAPExtractor:
    """PCAP → feature tensor pipeline with streaming and chunked output.

    Example:

        >>> from pcap2tensor import PCAPExtractor
        >>> extractor = PCAPExtractor(features="aegis-6d", window_size=1000, stride=500)
        >>> tensor = extractor.extract("capture.pcap")
        >>> tensor.shape
        torch.Size([num_windows, 1000, 6])

    For PCAPs too large to fit in memory, use :meth:`extract_chunks` or
    :meth:`save` which stream tensor chunks one at a time.
    """

    def __init__(
        self,
        features: Union[list[Feature], str] = "aegis-6d",
        window_size: int = 1000,
        stride: int = 500,
        chunk_size: int = 2_000_000,
        skip_non_ip: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Configure the extraction pipeline.

        Args:
            features: Either a preset name (e.g. ``"aegis-6d"``) or a list
                of :class:`Feature` instances. Default ``"aegis-6d"``.
            window_size: Packets per output window. Default 1000.
            stride: Packets between consecutive window starts. Default 500
                (50% overlap).
            chunk_size: Maximum packets buffered in RAM before a chunk is
                flushed. Tune down for memory-constrained systems. Default 2M.
            skip_non_ip: If True, non-IP packets (e.g. ARP) are silently
                skipped. Default True.
            dtype: Output tensor dtype. Default float32.
        """
        if isinstance(features, str):
            features = get_preset(features)
        if not features:
            raise ValueError("features must be a non-empty list.")

        self.features = list(features)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.chunk_size = int(chunk_size)
        self.skip_non_ip = bool(skip_non_ip)
        self.dtype = dtype

        self.feature_dim = sum(f.dim for f in self.features)
        self.feature_names = [f.name for f in self.features]

    # --------------------------------------------------------------------- #
    # Per-packet
    # --------------------------------------------------------------------- #

    def _extract_packet(self, pkt: Any) -> list[float]:
        """Run every configured feature on a single packet, flattening outputs."""
        out: list[float] = []
        for f in self.features:
            val = f(pkt)
            if isinstance(val, (list, tuple)):
                out.extend(float(v) for v in val)
            else:
                out.append(float(val))
        return out

    def _reset_features(self) -> None:
        """Reset state on all features. Called automatically between PCAPs."""
        for f in self.features:
            f.reset()

    # --------------------------------------------------------------------- #
    # Streaming core
    # --------------------------------------------------------------------- #

    def extract_chunks(
        self,
        pcap_path: PathLike,
        progress: bool = True,
    ) -> Iterator[torch.Tensor]:
        """Stream a PCAP as a sequence of windowed tensor chunks.

        Args:
            pcap_path: Path to a ``.pcap`` or ``.pcapng`` file.
            progress: Show a tqdm progress bar.

        Yields:
            Tensors of shape ``(num_windows, window_size, feature_dim)``.
            Only chunks with at least ``window_size`` packets are yielded.

        Raises:
            FileNotFoundError: If ``pcap_path`` does not exist.
        """
        path = str(pcap_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"PCAP not found: {path}")

        self._reset_features()
        buffer: list[list[float]] = []

        reader = PcapReader(path)
        pbar = None
        if progress:
            pbar = tqdm(desc=os.path.basename(path), unit="pkt", unit_scale=True)

        try:
            for pkt in reader:
                if pbar is not None:
                    pbar.update(1)

                if self.skip_non_ip and not (IP in pkt or IPv6 in pkt):
                    continue
                try:
                    buffer.append(self._extract_packet(pkt))
                except Exception:
                    # Malformed packets are skipped rather than aborting the run.
                    continue

                if len(buffer) >= self.chunk_size:
                    chunk = self._buffer_to_windows(buffer)
                    buffer.clear()
                    if chunk.size(0) > 0:
                        yield chunk
        finally:
            reader.close()
            if pbar is not None:
                pbar.close()

        if len(buffer) >= self.window_size:
            chunk = self._buffer_to_windows(buffer)
            if chunk.size(0) > 0:
                yield chunk

    def _buffer_to_windows(self, buffer: list[list[float]]) -> torch.Tensor:
        t = torch.tensor(buffer, dtype=self.dtype)
        return sliding_window(t, self.window_size, self.stride)

    # --------------------------------------------------------------------- #
    # High-level API
    # --------------------------------------------------------------------- #

    def extract(
        self,
        pcap_path: PathLike,
        progress: bool = True,
    ) -> torch.Tensor:
        """Extract a full PCAP into a single concatenated tensor.

        For very large PCAPs, prefer :meth:`extract_chunks` or :meth:`save` —
        this method holds every chunk in memory before concatenation.

        Returns:
            Tensor of shape ``(num_windows, window_size, feature_dim)``. Empty
            if the PCAP yields no valid windows.
        """
        chunks = list(self.extract_chunks(pcap_path, progress=progress))
        if not chunks:
            return torch.empty(
                (0, self.window_size, self.feature_dim), dtype=self.dtype
            )
        return torch.cat(chunks, dim=0)

    def save(
        self,
        pcap_path: PathLike,
        output_dir: PathLike,
        prefix: str = "pcap2tensor",
        progress: bool = True,
    ) -> list[str]:
        """Extract a PCAP and save each chunk as a separate ``.pt`` file.

        Args:
            pcap_path: Input PCAP path.
            output_dir: Directory to write chunks into (created if missing).
            prefix: Filename prefix for output chunks.
            progress: Show a tqdm progress bar.

        Returns:
            List of absolute paths of written ``.pt`` files.
        """
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.basename(str(pcap_path))
        for suffix in (".pcapng", ".pcap"):
            if base.lower().endswith(suffix):
                base = base[: -len(suffix)]
                break

        written: list[str] = []
        for idx, chunk in enumerate(
            self.extract_chunks(pcap_path, progress=progress), start=1
        ):
            fname = f"{prefix}_{base}_chunk{idx:03d}_w{self.window_size}.pt"
            out = os.path.abspath(os.path.join(str(output_dir), fname))
            torch.save(chunk, out)
            written.append(out)
        return written


# --------------------------------------------------------------------------- #
# Module-level convenience
# --------------------------------------------------------------------------- #

def extract(
    pcap_path: PathLike,
    features: Union[list[Feature], str] = "aegis-6d",
    window_size: int = 1000,
    stride: int = 500,
    progress: bool = True,
) -> torch.Tensor:
    """Extract a PCAP to a windowed tensor in one call.

    Args:
        pcap_path: Path to a ``.pcap`` or ``.pcapng`` file.
        features: Preset name or list of :class:`Feature` instances.
        window_size: Packets per window.
        stride: Packets between window starts.
        progress: Show a tqdm progress bar.

    Returns:
        Tensor of shape ``(num_windows, window_size, feature_dim)``.
    """
    extractor = PCAPExtractor(
        features=features, window_size=window_size, stride=stride
    )
    return extractor.extract(pcap_path, progress=progress)


# --------------------------------------------------------------------------- #
# Parallel batch processing
# --------------------------------------------------------------------------- #

def _worker(task: tuple) -> tuple[str, list[str], str | None]:
    """Process-pool worker. Returns (pcap_path, written_paths, error_or_None)."""
    pcap_path, out_dir, features_name, window, stride, chunk_size, prefix = task
    try:
        ext = PCAPExtractor(
            features=features_name,
            window_size=window,
            stride=stride,
            chunk_size=chunk_size,
        )
        written = ext.save(pcap_path, out_dir, prefix=prefix, progress=False)
        return (pcap_path, written, None)
    except Exception as e:  # noqa: BLE001
        return (pcap_path, [], f"{type(e).__name__}: {e}")


def batch_extract(
    input_spec: Union[str, Path, list[PathLike]],
    output_dir: PathLike,
    features: str = "aegis-6d",
    window_size: int = 1000,
    stride: int = 500,
    chunk_size: int = 2_000_000,
    prefix: str = "pcap2tensor",
    workers: int = 4,
    recursive: bool = True,
) -> dict[str, list[str]]:
    """Extract many PCAPs in parallel with a process pool.

    Args:
        input_spec: A directory (scanned for ``*.pcap`` / ``*.pcapng``),
            a glob pattern, or an explicit list of paths.
        output_dir: Directory to write tensor chunks into.
        features: Preset name, passed to each worker. Custom ``Feature``
            instances cannot cross process boundaries — use a preset here
            and monkey-patch features if you need custom extraction
            across a batch.
        window_size: Packets per window.
        stride: Packets between window starts.
        chunk_size: Packets per in-memory chunk before flushing.
        prefix: Output filename prefix.
        workers: Number of parallel worker processes.
        recursive: When ``input_spec`` is a directory, recurse into
            subdirectories.

    Returns:
        Dict mapping input PCAP path to list of written ``.pt`` paths.
        Inputs that failed produce an empty list and print a diagnostic.
    """
    os.makedirs(output_dir, exist_ok=True)
    pcaps = _resolve_inputs(input_spec, recursive=recursive)
    if not pcaps:
        raise ValueError(f"No PCAPs found for input: {input_spec!r}")

    tasks = [
        (p, str(output_dir), features, window_size, stride, chunk_size, prefix)
        for p in pcaps
    ]

    results: dict[str, list[str]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="PCAPs"):
            pcap_path, written, err = fut.result()
            if err:
                print(f"  [error] {os.path.basename(pcap_path)}: {err}")
            results[pcap_path] = written
    return results


def _resolve_inputs(
    spec: Union[str, Path, list[PathLike]],
    recursive: bool,
) -> list[str]:
    """Resolve a directory / glob / list into a concrete sorted list of paths."""
    if isinstance(spec, (list, tuple)):
        return sorted(str(p) for p in spec)

    spec_str = str(spec)
    if os.path.isdir(spec_str):
        if recursive:
            pats = [
                os.path.join(spec_str, "**", "*.pcap"),
                os.path.join(spec_str, "**", "*.pcapng"),
            ]
            found: list[str] = []
            for p in pats:
                found.extend(glob.glob(p, recursive=True))
            return sorted(found)
        pats = [
            os.path.join(spec_str, "*.pcap"),
            os.path.join(spec_str, "*.pcapng"),
        ]
        found = []
        for p in pats:
            found.extend(glob.glob(p))
        return sorted(found)

    # Treat as a glob pattern.
    return sorted(glob.glob(spec_str, recursive=recursive))
