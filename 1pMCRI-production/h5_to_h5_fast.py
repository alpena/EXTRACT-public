#!/usr/bin/env python3
"""
Fast H5 -> H5 converter for masknmf output to EXTRACT-optimized layout.

Input dataset is expected to be /motion_corrected (fixed by MATLAB pipeline).
Output dataset is /mov with MATLAB-visible shape [height, width, frames].
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast H5 to H5 converter for EXTRACT.")
    p.add_argument("--input", required=True, help="Input H5 path")
    p.add_argument("--output", required=True, help="Output H5 path")
    p.add_argument("--input-dataset", default="/motion_corrected", help="Input dataset path")
    p.add_argument("--output-dataset", default="/mov", help="Output dataset path")
    p.add_argument("--expected-height", type=int, required=True, help="Expected movie height")
    p.add_argument("--expected-width", type=int, required=True, help="Expected movie width")
    p.add_argument("--expected-frames", type=int, required=True, help="Expected total frames")
    p.add_argument("--chunk-t", type=int, default=256, help="Output chunk t size")
    p.add_argument("--chunk-x", type=int, default=256, help="Output chunk x size")
    p.add_argument("--chunk-y", type=int, default=256, help="Output chunk y size")
    p.add_argument("--compression", type=int, default=0, help="Gzip compression level (0-9)")
    p.add_argument(
        "--orientation-fix",
        default="transpose_xy",
        choices=["none", "transpose_xy"],
        help="Apply orientation correction before writing output dataset.",
    )
    return p.parse_args()


def normalize_dataset_name(name: str) -> str:
    return name if name.startswith("/") else "/" + name


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def find_tyx_permutation(shape: tuple[int, ...], target_tyx: tuple[int, int, int]) -> tuple[int, int, int]:
    for perm in itertools.permutations((0, 1, 2)):
        if tuple(shape[idx] for idx in perm) == target_tyx:
            return perm
    raise ValueError(
        f"Cannot align input shape {shape} to expected (t,y,x)={target_tyx} via axis permutation."
    )


def cast_to_uint16(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint16:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
    if np.issubdtype(arr.dtype, np.floating):
        return np.clip(np.rint(arr), 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
    raise TypeError(f"Unsupported input dtype: {arr.dtype}")


def main() -> int:
    args = parse_args()
    in_path = args.input
    out_path = args.output
    in_dset = normalize_dataset_name(args.input_dataset)
    out_dset = normalize_dataset_name(args.output_dataset)
    h = int(args.expected_height)
    w = int(args.expected_width)
    t = int(args.expected_frames)
    orientation_fix = str(args.orientation_fix)
    chunk_t = max(1, int(args.chunk_t))
    chunk_x = max(1, int(args.chunk_x))
    chunk_y = max(1, int(args.chunk_y))
    compression = int(args.compression)

    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input H5 not found: {in_path}")
    if compression < 0 or compression > 9:
        raise ValueError("compression must be in [0, 9]")

    ensure_parent_dir(out_path)
    if os.path.isfile(out_path):
        os.remove(out_path)

    target_h = h
    target_w = w
    if orientation_fix == "transpose_xy":
        target_h, target_w = w, h

    expected_tyx = (t, target_h, target_w)
    out_shape_txy = (t, target_w, target_h)  # MATLAB-visible [h,w,t]
    bytes_per_voxel = 2
    max_chunk_bytes = 4 * 1024**3 - 1

    with h5py.File(in_path, "r") as h5_in:
        if in_dset not in h5_in:
            raise KeyError(f"Input dataset not found: {in_dset}")
        src = h5_in[in_dset]
        if src.ndim != 3:
            raise ValueError(f"Input dataset must be 3D, got shape {src.shape}")

        perm_tyx = find_tyx_permutation(tuple(src.shape), expected_tyx)
        src_dtype = src.dtype

        chunk_t = min(chunk_t, t)
        chunk_x = min(chunk_x, target_w)
        chunk_y = min(chunk_y, target_h)
        chunk_bytes = chunk_t * chunk_x * chunk_y * bytes_per_voxel
        while chunk_bytes > max_chunk_bytes and chunk_t > 1:
            chunk_t = max(1, chunk_t // 2)
            chunk_bytes = chunk_t * chunk_x * chunk_y * bytes_per_voxel
        while chunk_bytes > max_chunk_bytes and (chunk_x > 1 or chunk_y > 1):
            if chunk_x >= chunk_y and chunk_x > 1:
                chunk_x = max(1, chunk_x // 2)
            elif chunk_y > 1:
                chunk_y = max(1, chunk_y // 2)
            chunk_bytes = chunk_t * chunk_x * chunk_y * bytes_per_voxel

        print(f"Input H5: {in_path}:{in_dset}")
        print(f"Input dataset shape (h5py): {src.shape} dtype={src_dtype}")
        print(f"Orientation fix: {orientation_fix}")
        print(f"Expected (t,y,x): {expected_tyx}")
        print(f"Using axis permutation to (t,y,x): {perm_tyx}")
        print(f"Output H5: {out_path}:{out_dset}")
        print(
            f"Output chunk (stored t,x,y): ({chunk_t},{chunk_x},{chunk_y}) "
            f"[{chunk_bytes / (1024**2):.1f} MiB]"
        )

        with h5py.File(out_path, "w") as h5_out:
            create_kwargs: dict[str, object] = {
                "shape": out_shape_txy,
                "dtype": np.uint16,
                "chunks": (chunk_t, chunk_x, chunk_y),
            }
            if compression > 0:
                create_kwargs["compression"] = "gzip"
                create_kwargs["compression_opts"] = compression
            dst = h5_out.create_dataset(out_dset, **create_kwargs)

            num_chunks = (t + chunk_t - 1) // chunk_t
            t0 = time.perf_counter()
            bytes_per_frame = target_h * target_w * bytes_per_voxel
            src_t_axis = perm_tyx[0]

            for i in range(num_chunks):
                f_begin = i * chunk_t
                f_end = min((i + 1) * chunk_t, t)
                slicer = [slice(None), slice(None), slice(None)]
                slicer[src_t_axis] = slice(f_begin, f_end)
                block = src[tuple(slicer)]
                block_tyx = np.transpose(block, perm_tyx)  # (n,y,x)
                block_tyx = cast_to_uint16(block_tyx)
                dst[f_begin:f_end, :, :] = block_tyx.transpose(0, 2, 1)  # (n,x,y)

                done_frames = f_end
                progress = done_frames / t * 100.0
                elapsed = max(time.perf_counter() - t0, 1e-9)
                done_bytes = done_frames * bytes_per_frame
                rate_bps = done_bytes / elapsed
                mb_s = rate_bps / (1024**2)
                rem_frames = t - done_frames
                eta_s = (rem_frames * bytes_per_frame / rate_bps) if rate_bps > 0 else float("inf")
                print(
                    f"Chunk {i+1}/{num_chunks}: frames {f_begin+1}-{f_end} | "
                    f"{progress:5.1f}% | {mb_s:7.1f} MiB/s | ETA {eta_s:7.1f}s",
                    flush=True,
                )

            if tuple(dst.shape) != out_shape_txy:
                raise ValueError(
                    f"Invalid output shape {dst.shape}; expected (t,x,y)={out_shape_txy}"
                )

    print(f"Done. Wrote H5: {out_path}")
    print(f"MATLAB expected view (h,w,t): ({target_h}, {target_w}, {t})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
