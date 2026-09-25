"""
htv_io.py — reading .htv captures.

The only module that knows about the file format.  Everything downstream
works on plain numpy arrays.
"""

import struct
from dataclasses import dataclass

import numpy as np

HEADER_BYTES = 256


@dataclass(frozen=True)
class Capture:
    """Header metadata plus a memory-mapped view of the frame stack."""
    width:   int
    height:  int
    n_frames: int
    bpp:     int
    fps:     float
    frames:  np.memmap          # (n_frames, height, width)

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    @property
    def f_nyquist(self) -> float:
        return self.fps / 2.0


def load_header(path):
    """Parse the 256-byte header.  Returns (W, H, N, bpp, fps)."""
    with open(path, "rb") as fh:
        raw = fh.read(HEADER_BYTES)

    if raw[0:4] != b"HTV\x00":
        raise ValueError(f"bad magic {raw[0:4]!r} — not an .htv file")

    W   = struct.unpack_from("<I", raw,  8)[0]
    H   = struct.unpack_from("<I", raw, 12)[0]
    N   = struct.unpack_from("<I", raw, 16)[0]
    bpp = struct.unpack_from("<I", raw, 20)[0]

    fps_nom = struct.unpack_from("<d", raw, 32)[0]
    fps_ach = struct.unpack_from("<d", raw, 40)[0]
    fps = fps_ach if fps_nom * 0.5 < fps_ach < fps_nom * 1.5 else fps_nom

    return W, H, N, bpp, fps


def open_capture(path):
    """Parse the header and memory-map the frame stack."""
    W, H, N, bpp, fps = load_header(path)

    if bpp != 8:
        raise NotImplementedError(f"bpp={bpp} unsupported; frame reader assumes 8-bit")

    frames = np.memmap(path, dtype=np.uint8, mode="r",
                       offset=HEADER_BYTES, shape=(N, H, W))

    return Capture(width=W, height=H, n_frames=N, bpp=bpp, fps=fps, frames=frames)


def iter_chunks(capture, chunk):
    """Yield (start, end, block) with block as float32, block.shape = (n, H, W)."""
    N = capture.n_frames
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        yield start, end, capture.frames[start:end].astype(np.float32)
