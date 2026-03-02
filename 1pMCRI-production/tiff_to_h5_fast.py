#!/usr/bin/env python3
"""
Fast TIFF -> H5 converter using tifffile + h5py.

Supports:
  - Standard multipage TIFF
  - ImageJ stack TIFF

Usage:
  python tiff_to_h5_fast.py --input <input.tif> --output <output.h5> --dataset /mov --chunk-frames 1000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast TIFF to H5 converter.")
    p.add_argument("--input", required=True, help="Input TIFF path")
    p.add_argument("--output", required=True, help="Output H5 path")
    p.add_argument("--dataset", default="/mov", help="Dataset name in H5 (default: /mov)")
    p.add_argument("--chunk-frames", type=int, default=200, help="Target chunk frames")
    return p.parse_args()


def max_chunk_frames(height: int, width: int, dtype_bytes: int = 2) -> int:
    # HDF5 requires chunk byte-size < 4GB.
    bytes_per_frame = int(height) * int(width) * int(dtype_bytes)
    return max((4 * 1024**3 - 1) // bytes_per_frame, 1)


def normalize_dataset_name(name: str) -> str:
    return name if name.startswith("/") else "/" + name


def main() -> int:
    args = parse_args()
    in_path = args.input
    out_path = args.output
    dset_name = normalize_dataset_name(args.dataset)
    chunk_frames_req = int(args.chunk_frames)

    if chunk_frames_req < 1:
        raise ValueError("chunk-frames must be >= 1")
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input TIFF not found: {in_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.isfile(out_path):
        os.remove(out_path)

    with tifffile.TiffFile(in_path) as tif:
        series = tif.series[0]
        shape = series.shape
        dtype = np.dtype(series.dtype)

        # Expect (t, y, x) or (y, x) for single-frame.
        if len(shape) == 2:
            total_frames = 1
            height, width = shape
        elif len(shape) == 3:
            total_frames, height, width = shape
        else:
            raise ValueError(f"Unsupported TIFF shape: {shape}")

        if dtype != np.uint16:
            raise ValueError(f"Expected uint16 TIFF, got {dtype}")

        chunk_max = max_chunk_frames(height, width, 2)
        chunk_frames = min(chunk_frames_req, chunk_max, total_frames)

        print(f"Input TIFF: {in_path}")
        print(f"Shape (t,y,x): ({total_frames},{height},{width})")
        print(f"Output H5: {out_path}:{dset_name}")
        if chunk_frames < chunk_frames_req:
            print(f"chunk-frames={chunk_frames_req} exceeds HDF5 limit; using {chunk_frames}")

        with h5py.File(out_path, "w") as h5:
            ds = h5.create_dataset(
                dset_name,
                shape=(height, width, total_frames),
                dtype=np.uint16,
                chunks=(height, width, chunk_frames),
            )
            t0 = time.perf_counter()
            bytes_per_frame = height * width * 2

            if total_frames == 1:
                frame = series.asarray()
                ds[:, :, 0] = frame
                elapsed = max(time.perf_counter() - t0, 1e-9)
                mb_s = bytes_per_frame / elapsed / (1024**2)
                print(f"Chunk 1/1: frames 1-1 | 100.0% | {mb_s:.1f} MiB/s | ETA 0.0s", flush=True)
            else:
                num_chunks = (total_frames + chunk_frames - 1) // chunk_frames
                for i in range(num_chunks):
                    f_begin = i * chunk_frames
                    f_end = min((i + 1) * chunk_frames, total_frames)
                    # Read chunk as (n, y, x), then transpose to (y, x, n).
                    block = series.asarray(key=range(f_begin, f_end))
                    block = np.asarray(block, dtype=np.uint16).transpose(1, 2, 0)
                    ds[:, :, f_begin:f_end] = block
                    done_frames = f_end
                    progress = done_frames / total_frames * 100.0
                    elapsed = max(time.perf_counter() - t0, 1e-9)
                    done_bytes = done_frames * bytes_per_frame
                    rate_bps = done_bytes / elapsed
                    mb_s = rate_bps / (1024**2)
                    rem_frames = total_frames - done_frames
                    eta_s = (rem_frames * bytes_per_frame / rate_bps) if rate_bps > 0 else float("inf")
                    print(
                        f"Chunk {i+1}/{num_chunks}: frames {f_begin+1}-{f_end} | "
                        f"{progress:5.1f}% | {mb_s:7.1f} MiB/s | ETA {eta_s:7.1f}s",
                        flush=True,
                    )

    print(f"Done. Wrote H5: {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
