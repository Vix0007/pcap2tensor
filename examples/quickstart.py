"""Quickstart: three ways to use pcap2tensor."""
from pathlib import Path

from pcap2tensor import AEGIS_6D, PCAPExtractor, batch_extract, extract

PCAP = "capture.pcap"
OUT = Path("./tensors")

# ---------------------------------------------------------------------------
# 1. One-shot API — everything fits in memory
# ---------------------------------------------------------------------------
tensor = extract(PCAP, features="aegis-6d", window_size=1000, stride=500)
print(f"[1] tensor shape: {tuple(tensor.shape)}")
#   → (num_windows, 1000, 6)

# ---------------------------------------------------------------------------
# 2. Streaming chunked — for PCAPs too large for RAM
# ---------------------------------------------------------------------------
extractor = PCAPExtractor(
    features=AEGIS_6D(),
    window_size=1000,
    stride=500,
    chunk_size=2_000_000,
)

for i, chunk in enumerate(extractor.extract_chunks(PCAP), start=1):
    print(f"[2] chunk {i}: {tuple(chunk.shape)}")
    # train_step(chunk) ...

# Or persist every chunk to disk:
paths = extractor.save(PCAP, output_dir=OUT, prefix="aegis")
print(f"[2] wrote {len(paths)} chunks to {OUT}")

# ---------------------------------------------------------------------------
# 3. Parallel batch — a whole directory of PCAPs
# ---------------------------------------------------------------------------
results = batch_extract(
    input_spec="./pcaps/",
    output_dir=OUT,
    features="aegis-6d",
    workers=4,
)
print(f"[3] processed {len(results)} PCAPs")
