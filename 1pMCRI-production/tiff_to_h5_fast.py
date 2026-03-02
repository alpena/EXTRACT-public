#!/usr/bin/env python3
"""
Fast TIFF -> H5 converter using tifffile + h5py.

Supports:
  - Standard multipage TIFF
  - ImageJ stack TIFF

Usage:
  python tiff_to_h5_fast.py --input <input.tif> --output <output.h5> --dataset /mov --chunk-frames 1000
  python tiff_to_h5_fast.py --input <input.tif> --output <output.h5> --chunk-t 96 --chunk-x 256 --chunk-y 256
"""

from __future__ import annotations

import argparse
import itertools
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
    p.add_argument("--chunk-frames", type=int, default=200, help="Legacy time-chunk option (same as --chunk-t)")
    p.add_argument("--chunk-t", type=int, default=0, help="Chunk size along time axis")
    p.add_argument("--chunk-x", type=int, default=0, help="Chunk size along x axis (stored x)")
    p.add_argument("--chunk-y", type=int, default=0, help="Chunk size along y axis (stored y)")
    p.add_argument("--target-chunk-mb", type=float, default=16.0, help="Target chunk size in MiB for auto chunking")
    p.add_argument("--expected-height", type=int, default=0, help="Expected frame height from TIFF metadata")
    p.add_argument("--expected-width", type=int, default=0, help="Expected frame width from TIFF metadata")
    p.add_argument("--expected-frames", type=int, default=0, help="Expected total frames from TIFF metadata")
    return p.parse_args()


def max_chunk_frames(height: int, width: int, dtype_bytes: int = 2) -> int:
    # HDF5 requires chunk byte-size < 4GB.
    bytes_per_frame = int(height) * int(width) * int(dtype_bytes)
    return max((4 * 1024**3 - 1) // bytes_per_frame, 1)


def normalize_dataset_name(name: str) -> str:
    return name if name.startswith("/") else "/" + name


def normalize_block_to_nyx(block: np.ndarray, n_this: int, height: int, width: int) -> np.ndarray:
    arr = np.asarray(block, dtype=np.uint16)
    if arr.ndim == 2:
        if n_this != 1:
            raise ValueError(f"2D block cannot represent n_this={n_this}")
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D block, got shape {arr.shape}")

    target = (n_this, height, width)
    if tuple(arr.shape) == target:
        return arr

    for perm in itertools.permutations((0, 1, 2)):
        if tuple(arr.shape[idx] for idx in perm) == target:
            return np.transpose(arr, perm)

    raise ValueError(
        f"Cannot align block shape {arr.shape} to expected (n,y,x)={target}"
    )


def infer_tyx_from_series(shape: tuple[int, ...], axes: str | None) -> tuple[int, int, int]:
    if len(shape) == 2:
        h, w = shape
        return 1, h, w
    if len(shape) != 3:
        raise ValueError(f"Unsupported TIFF shape: {shape}")

    if axes:
        axes_u = axes.upper()
        if "Y" in axes_u and "X" in axes_u:
            idx_y = axes_u.index("Y")
            idx_x = axes_u.index("X")
            idx_t = None
            for key in ("T", "I", "Z"):
                if key in axes_u:
                    idx_t = axes_u.index(key)
                    break
            if idx_t is not None:
                return int(shape[idx_t]), int(shape[idx_y]), int(shape[idx_x])

    # Fallback for standard (t, y, x).
    t, h, w = shape
    return int(t), int(h), int(w)


def main() -> int:
    args = parse_args()
    in_path = args.input
    out_path = args.output
    dset_name = normalize_dataset_name(args.dataset)
    chunk_frames_req = int(args.chunk_frames)
    chunk_t_req = int(args.chunk_t)
    chunk_x_req = int(args.chunk_x)
    chunk_y_req = int(args.chunk_y)
    target_chunk_mb = float(args.target_chunk_mb)

    if chunk_frames_req < 1:
        raise ValueError("chunk-frames must be >= 1")
    if chunk_t_req < 0 or chunk_x_req < 0 or chunk_y_req < 0:
        raise ValueError("chunk-t/chunk-x/chunk-y must be >= 0")
    if target_chunk_mb <= 0:
        raise ValueError("target-chunk-mb must be > 0")
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input TIFF not found: {in_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.isfile(out_path):
        os.remove(out_path)

    with tifffile.TiffFile(in_path) as tif:
        series = tif.series[0]
        shape = series.shape
        axes = getattr(series, "axes", "")
        dtype = np.dtype(series.dtype)

        inferred_frames, inferred_height, inferred_width = infer_tyx_from_series(shape, axes)
        expected_height = int(args.expected_height)
        expected_width = int(args.expected_width)
        expected_frames = int(args.expected_frames)
        height = expected_height if expected_height > 0 else inferred_height
        width = expected_width if expected_width > 0 else inferred_width
        total_frames = expected_frames if expected_frames > 0 else inferred_frames

        if dtype != np.uint16:
            raise ValueError(f"Expected uint16 TIFF, got {dtype}")

        # Stored shape is (t, x, y), so width/height map to x/y chunking.
        chunk_t_base_req = chunk_t_req if chunk_t_req > 0 else chunk_frames_req
        chunk_t_max = max_chunk_frames(height, width, 2)
        chunk_t = min(chunk_t_base_req, chunk_t_max, total_frames)
        chunk_x = chunk_x_req if chunk_x_req > 0 else min(width, 256)
        chunk_y = chunk_y_req if chunk_y_req > 0 else min(height, 256)

        # Auto-adjust x/y when not explicitly specified to hit target chunk size.
        if chunk_x_req <= 0 or chunk_y_req <= 0:
            target_bytes = int(target_chunk_mb * 1024**2)
            bytes_per_t_plane = max(chunk_t * 2, 1)  # uint16
            target_xy_area = max(target_bytes // bytes_per_t_plane, 1)
            side = int(np.sqrt(target_xy_area))
            side = max(64, min(side, 512))
            if chunk_x_req <= 0:
                chunk_x = min(width, side)
            if chunk_y_req <= 0:
                chunk_y = min(height, side)

        # Hard clamp to valid ranges.
        chunk_x = max(1, min(chunk_x, width))
        chunk_y = max(1, min(chunk_y, height))

        # Ensure chunk byte-size stays within HDF5's <4GB chunk constraint.
        chunk_bytes = int(chunk_t) * int(chunk_x) * int(chunk_y) * 2
        max_chunk_bytes = 4 * 1024**3 - 1
        while chunk_bytes > max_chunk_bytes and chunk_t > 1:
            chunk_t = max(1, chunk_t // 2)
            chunk_bytes = int(chunk_t) * int(chunk_x) * int(chunk_y) * 2
        while chunk_bytes > max_chunk_bytes and (chunk_x > 1 or chunk_y > 1):
            if chunk_x >= chunk_y and chunk_x > 1:
                chunk_x = max(1, chunk_x // 2)
            elif chunk_y > 1:
                chunk_y = max(1, chunk_y // 2)
            chunk_bytes = int(chunk_t) * int(chunk_x) * int(chunk_y) * 2

        print(f"Input TIFF: {in_path}")
        print(f"Series shape/axes: {shape} / {axes}")
        print(f"Expected (t,y,x): ({total_frames},{height},{width})")
        print(f"Output H5: {out_path}:{dset_name}")
        print(
            f"Chunk (stored t,x,y): ({chunk_t},{chunk_x},{chunk_y}) "
            f"[{chunk_bytes / (1024**2):.1f} MiB]"
        )

        mm = tifffile.memmap(in_path, series=0)

        # MATLAB/HDF5 interoperability:
        # h5py shape order is interpreted differently by MATLAB h5info/h5read.
        # To make MATLAB see [height,width,frames], write as [frames,width,height].
        h5_shape = (total_frames, width, height)

        with h5py.File(out_path, "w") as h5:
            ds = h5.create_dataset(
                dset_name,
                shape=h5_shape,
                dtype=np.uint16,
                chunks=(chunk_t, chunk_x, chunk_y),
            )
            t0 = time.perf_counter()
            bytes_per_frame = height * width * 2

            if total_frames == 1:
                frame = mm[0, :, :] if mm.ndim == 3 else mm[:, :]
                block_nyx = normalize_block_to_nyx(frame, 1, height, width)
                ds[0, :, :] = block_nyx[0, :, :].T
                elapsed = max(time.perf_counter() - t0, 1e-9)
                mb_s = bytes_per_frame / elapsed / (1024**2)
                print(f"Chunk 1/1: frames 1-1 | 100.0% | {mb_s:.1f} MiB/s | ETA 0.0s", flush=True)
            else:
                num_chunks = (total_frames + chunk_t - 1) // chunk_t
                for i in range(num_chunks):
                    f_begin = i * chunk_t
                    f_end = min((i + 1) * chunk_t, total_frames)
                    n_this = f_end - f_begin
                    block = mm[f_begin:f_end, :, :]
                    block_nyx = normalize_block_to_nyx(block, n_this, height, width)
                    # (n,y,x) -> (n,x,y) for MATLAB-compatible storage
                    ds[f_begin:f_end, :, :] = block_nyx.transpose(0, 2, 1)
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

            # Enforce MATLAB-visible layout [height, width, frames] by writing
            # HDF5 as [frames, width, height] from Python.
            if tuple(ds.shape) != h5_shape:
                raise ValueError(
                    f"Invalid output shape {ds.shape}; expected stored (frames,width,height)="
                    f"({total_frames},{width},{height})"
                )
            print(f"H5 dataset stored shape (t,x,y): {ds.shape}", flush=True)
            print(
                f"MATLAB expected view (h,w,t): ({height}, {width}, {total_frames})",
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
