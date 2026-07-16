"""
FIDVAC v2 - DCT compression path
==================================
Handles float32 elevation and uint8 accuracy-controlled (cm=3/cm=5/cm=6) data:
  - in-memory single-channel  : compress_channel_dct()          → cm=3 or cm=5
  - streaming single-channel  : compress_channel_dct_streaming() → cm=3 or cm=5
  - per-block RGB YCbCr       : compress_channel_rgb()           → cm=6
  - per-channel DCT loop with YCbCr decorrelation: compress_dct_path()

cm=3: float32/int16 elevation — DC Median3 + separate DC/AC streams, max error ≤ε
cm=5: uint8/int8 single-channel — binary search DCT, centering -128, max error ≤ε (grayscale)
cm=6: uint8 RGB accuracy — per-block YCbCr with cascading scaling factors, max error ≤ε (RGB)

All functions return the same 7-tuple:
    (lengths, coeffs_np, masks_data, max_error, validity, block_size, padded_shape)
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing
from typing import Optional, Tuple, List
from functools import partial
from copy import deepcopy

import numpy as np
from rasterio.windows import Window
from tqdm import tqdm

from engine.blocks import (
    process_block_batch,
    process_strip_batch,
    process_block_batch_rgb,
    rgb_to_ycbcr,
)
from config import Config
from .compression_utils import (
    pad_image,
    extract_blocks,
    auto_select_block_size,
    auto_select_uint8_parameters,
    serialize_compressed_blocks,
    serialize_compressed_blocks_rgb,
)

# Return type alias for clarity
_DctResult = Tuple[List, np.ndarray, List, float, bool, int, Tuple[int, int]]

_INTEGER_DTYPES = ("uint8", "int8", "uint16", "int16")


def compress_channel_dct(
    image: np.ndarray,
    block_size: int,
    nodata_value: Optional[float],
    config: Config,
    num_processes: int,
    src_dtype: Optional[str] = None,
) -> _DctResult:
    """Compress a single channel in-memory.

    For integer dtypes, forces binary search (accuracy mode).
    Retries with smaller block size if validity fails.
    """
    is_integer = src_dtype in _INTEGER_DTYPES
    if is_integer:
        config = deepcopy(config)
        config.model.minimize_backscan = 0
        config.model.backscan_break_after = 0

    if not config.output.quiet and is_integer:
        global_u8_accuracy = float(
            getattr(config.compression, "uint8_accuracy", config.compression.accuracy)
        )
        channel_accuracy = float(config.compression.accuracy)
        if abs(channel_accuracy - global_u8_accuracy) < 1e-9:
            print(f"  uint8 accuracy mode: binary search with accuracy_px={channel_accuracy:g}")
        else:
            print(
                "  uint8 accuracy mode: binary search with "
                f"uint8_accuracy_px={global_u8_accuracy:g}, channel_accuracy_px={channel_accuracy:g}"
            )

    if config.compression.auto_select_block_size:
        if is_integer:
            # For uint8/int8: auto-select BOTH scaling_factor and block_size
            sf, bs = auto_select_uint8_parameters(image, config, nodata_value)
            config = deepcopy(config)
            config.compression.block_size = bs
            config.compression.uint8_scaling_factor = [sf]
            block_size = bs
        else:
            block_size = auto_select_block_size(image, config, nodata_value)
            config = deepcopy(config)
            config.compression.block_size = block_size

    if is_integer:
        config.model.minimize_backscan = 0
        config.model.backscan_break_after = 0

    padded_image, padded_shape = pad_image(image, block_size)
    all_blocks = extract_blocks(padded_image, block_size, nodata_value)
    total_blocks = len(all_blocks)

    compressed_blocks, max_errors = _prefilter_trivial(all_blocks, total_blocks)
    blocks_to_process = _collect_nontrivial(all_blocks, compressed_blocks, is_integer, src_dtype)

    if not config.output.quiet:
        skipped = total_blocks - len(blocks_to_process)
        print(f"  Total blocks: {total_blocks} (skipped {skipped} trivial)")

    _run_block_batches(blocks_to_process, compressed_blocks, max_errors, num_processes, config)

    max_error_global = max(max_errors) if max_errors else 0.0
    # Use config.compression.accuracy (channel-specific for YCbCr, or global uint8_accuracy)
    validity = all(e <= config.compression.accuracy for e in max_errors)

    if not validity and config.compression.auto_select_block_size and block_size > 8:
        return _retry_with_smaller_n(
            image,
            block_size,
            nodata_value,
            config,
            num_processes,
            max_errors,
            src_dtype,
            in_memory=True,
        )

    lengths, coeffs_np, masks_data = serialize_compressed_blocks(compressed_blocks)
    return lengths, coeffs_np, masks_data, max_error_global, validity, block_size, padded_shape


def compress_channel_dct_streaming(
    src,
    band_idx: int,
    block_size: int,
    nodata_value: Optional[float],
    config: Config,
    num_processes: int,
    src_dtype: Optional[str] = None,
) -> _DctResult:
    """Compress a single channel via rasterio Window strips (low-memory mode).

    Sends whole strip numpy arrays to workers — eliminates per-block IPC overhead.
    Retries with smaller block size if validity fails.
    """
    H, W = src.height, src.width

    config, block_size = _streaming_detect_and_autoselect(
        src, band_idx, H, W, block_size, nodata_value, config, src_dtype
    )

    is_integer = src_dtype in _INTEGER_DTYPES
    if is_integer:
        config.model.minimize_backscan = 0
        config.model.backscan_break_after = 0

    n = block_size

    padded_H, padded_W, padded_shape = _calc_padded_dims(H, W, n)
    total_blocks = (padded_H // n) * (padded_W // n)

    compressed_blocks = [None] * total_blocks
    max_errors = [0.0] * total_blocks

    strip_rows = _calc_strip_rows(W, n)
    blocks_per_row = padded_W // n

    if not config.output.quiet:
        print(f"  Streaming mode: strip={strip_rows} rows, total blocks={total_blocks}")

    effective_processes = _calc_effective_processes(total_blocks, num_processes)

    if effective_processes > 1:
        _run_strips_multiprocess(
            src,
            band_idx,
            H,
            W,
            padded_H,
            padded_W,
            n,
            strip_rows,
            blocks_per_row,
            nodata_value,
            config,
            effective_processes,
            compressed_blocks,
            max_errors,
        )
    else:
        _run_strips_singleprocess(
            src,
            band_idx,
            H,
            W,
            padded_H,
            padded_W,
            n,
            strip_rows,
            blocks_per_row,
            nodata_value,
            config,
            compressed_blocks,
            max_errors,
        )

    trivial_count = sum(1 for b in compressed_blocks if not isinstance(b, np.ndarray))
    if not config.output.quiet:
        print(f"  Total blocks: {total_blocks} (skipped {trivial_count} trivial)")

    max_error_global = max(max_errors) if max_errors else 0.0
    # Use config.compression.accuracy (channel-specific for YCbCr, or global uint8_accuracy)
    validity = all(e <= config.compression.accuracy for e in max_errors)

    if not validity and config.compression.auto_select_block_size and block_size > 8:
        return _retry_with_smaller_n(
            src,
            block_size,
            nodata_value,
            config,
            num_processes,
            max_errors,
            src_dtype,
            in_memory=False,
            band_idx=band_idx,
        )

    lengths, coeffs_np, masks_data = serialize_compressed_blocks(compressed_blocks)
    return lengths, coeffs_np, masks_data, max_error_global, validity, block_size, padded_shape


def build_ycbcr_images(src, config, num_channels, all_uint8, _data_band_indices):
    """Pre-compute YCbCr images and per-channel accuracy budgets.

    Returns (ycbcr_images, ycbcr_acc, ycbcr_rgb_0).
    All dicts/lists are empty if conditions not met.
    """
    ycbcr_images = {}
    ycbcr_acc = {}
    ycbcr_rgb_0 = []

    rgb_idx_cfg = list(getattr(config.compression, "rgb_channel_indices", []))
    if not (
        len(rgb_idx_cfg) == 3
        and all_uint8
        and getattr(config.compression, "uint8_accuracy_mode", False)
    ):
        return ycbcr_images, ycbcr_acc, ycbcr_rgb_0

    r0, g0, b0 = [int(i) - 1 for i in rgb_idx_cfg]
    if not all(0 <= i < num_channels for i in [r0, g0, b0]):
        return ycbcr_images, ycbcr_acc, ycbcr_rgb_0

    r_arr = src.read(r0 + 1).astype(np.float32)
    g_arr = src.read(g0 + 1).astype(np.float32)
    b_arr = src.read(b0 + 1).astype(np.float32)
    u8a = float(getattr(config.compression, "uint8_accuracy", 1))

    ycbcr_images[r0], ycbcr_images[g0], ycbcr_images[b0] = rgb_to_ycbcr(r_arr, g_arr, b_arr)
    ycbcr_acc[r0] = u8a * config.compression.ycbcr_y_multiplier  # Y
    ycbcr_acc[g0] = u8a * config.compression.ycbcr_cb_multiplier  # Cb
    ycbcr_acc[b0] = u8a * config.compression.ycbcr_cr_multiplier  # Cr
    ycbcr_rgb_0 = [r0, g0, b0]

    del r_arr, g_arr, b_arr

    if not config.output.quiet:
        print(
            f"  YCbCr decorrelation: ch{r0+1}=Y(acc={ycbcr_acc[r0]:.2f}) "
            f"ch{g0+1}=Cb(acc={ycbcr_acc[g0]:.2f}) "
            f"ch{b0+1}=Cr(acc={ycbcr_acc[b0]:.2f}) "
            f"[guarantees ≤{u8a} in RGB]"
        )

    return ycbcr_images, ycbcr_acc, ycbcr_rgb_0


def compress_channel_rgb(
    src,
    r_idx: int,
    g_idx: int,
    b_idx: int,
    block_size: int,
    nodata_value: Optional[float],
    config: Config,
    num_processes: int,
) -> _DctResult:
    """Compress R, G, B channels via per-block YCbCr multiplier prediction (cm=6).

    For each block triplet (Y, Cb, Cr) we:
      1. Predict the starting multiplier index from chroma amplitude
         (skips multipliers that are analytically too large for complex blocks).
      2. Try the predicted multiplier; accept immediately if RGB error ≤ accuracy.
      3. Iterate down the fallback list only when needed (rare for good predictions).

    Returns the standard 7-tuple:
        (lengths, coeffs_dict, masks_data, max_error, validity, block_size, padded_shape)
    """
    # Read all three channels
    r = src.read(r_idx + 1).astype(np.float32)
    g = src.read(g_idx + 1).astype(np.float32)
    b = src.read(b_idx + 1).astype(np.float32)
    H, W = r.shape

    Y, Cb, Cr = rgb_to_ycbcr(r, g, b)

    # Pad to block_size multiples
    n = block_size
    pad_H = (n - H % n) % n
    pad_W = (n - W % n) % n
    padded_H, padded_W = H + pad_H, W + pad_W
    padded_shape = (padded_H, padded_W)

    def _pad(ch: np.ndarray) -> np.ndarray:
        return np.pad(ch, ((0, pad_H), (0, pad_W)), mode="edge")

    Y_p, Cb_p, Cr_p = _pad(Y), _pad(Cb), _pad(Cr)
    r_p, g_p, b_p = _pad(r), _pad(g), _pad(b)
    del Y, Cb, Cr, r, g, b  # free memory early

    # Extract all blocks (vectorised reshape)
    rows, cols = padded_H // n, padded_W // n
    total_blocks = rows * cols

    def _blocks(ch: np.ndarray) -> np.ndarray:
        return ch.reshape(rows, n, cols, n).transpose(0, 2, 1, 3).reshape(-1, n, n)

    # pylint: disable=invalid-name
    Y_blk = _blocks(Y_p)
    Cb_blk = _blocks(Cb_p)
    Cr_blk = _blocks(Cr_p)
    r_blk = _blocks(r_p)
    g_blk = _blocks(g_p)
    b_blk = _blocks(b_p)
    del Y_p, Cb_p, Cr_p, r_p, g_p, b_p

    # Build list of RGB block tuples: (idx, Y, Cb, Cr, r, g, b, nodata)
    all_blocks = [
        (i, Y_blk[i], Cb_blk[i], Cr_blk[i], r_blk[i], g_blk[i], b_blk[i], nodata_value)
        for i in range(total_blocks)
    ]
    del Y_blk, Cb_blk, Cr_blk, r_blk, g_blk, b_blk

    # Batch and process
    blocks_per_batch = max(256, total_blocks // max(1, num_processes * 4))
    batches = [
        (i // blocks_per_batch, all_blocks[i : i + blocks_per_batch])
        for i in range(0, len(all_blocks), blocks_per_batch)
    ]

    effective = min(num_processes, len(batches))
    if total_blocks < 20_000:
        effective = 1

    compressed_map: dict = {}
    max_errors_map: dict = {}

    if effective > 1:
        batch_func = partial(process_block_batch_rgb, config=config)
        with multiprocessing.Pool(processes=effective) as pool:
            iter_ = pool.imap(batch_func, batches)
            if not config.output.quiet:
                iter_ = tqdm(iter_, total=len(batches), desc="RGB blocks", ascii=False)
            for batch_results in iter_:
                for idx, metadata, err in batch_results:
                    compressed_map[idx] = metadata
                    max_errors_map[idx] = err
    else:
        for batch_data in tqdm(
            batches, desc="RGB blocks", disable=config.output.quiet, ascii=False
        ):
            for idx, metadata, err in process_block_batch_rgb(batch_data, config):
                compressed_map[idx] = metadata
                max_errors_map[idx] = err

    max_error = max(max_errors_map.values()) if max_errors_map else 0.0
    base_acc = float(getattr(config.compression, "uint8_accuracy", 5))
    validity = max_error <= base_acc

    if not config.output.quiet:
        fallback_n = sum(1 for m in compressed_map.values() if m.get("fallback", False))
        print(
            f"  cm=6 per-block YCbCr: max_err={max_error:.2f}, valid={validity},"
            f" fallback_blocks={fallback_n}/{total_blocks}"
            f" ({100*fallback_n/max(1,total_blocks):.1f}%)"
        )

    ordered = [(i, compressed_map[i]) for i in range(total_blocks)]
    lengths, coeffs_dict, masks_data = serialize_compressed_blocks_rgb(ordered)

    return lengths, coeffs_dict, masks_data, max_error, validity, block_size, padded_shape


def _prefilter_trivial(_all_blocks, total_blocks):
    compressed_blocks = [None] * total_blocks
    max_errors = [0.0] * total_blocks
    return compressed_blocks, max_errors


def _collect_nontrivial(all_blocks, compressed_blocks, is_integer, src_dtype):
    blocks_to_process = []
    for idx, block, nodata in all_blocks:
        if nodata is not None and np.all(block == nodata):
            compressed_blocks[idx] = 1  # ALL_NODATA
        elif np.all(block == 0):
            compressed_blocks[idx] = 0  # ALL_ZEROS
        else:
            blocks_to_process.append((idx, block, nodata))

    if is_integer and blocks_to_process:
        blocks_to_process = [(idx, blk, nd, src_dtype) for idx, blk, nd in blocks_to_process]

    return blocks_to_process


def _run_block_batches(blocks_to_process, compressed_blocks, max_errors, num_processes, config):
    if not blocks_to_process:
        return

    blocks_per_batch = max(2048, len(blocks_to_process) // num_processes)
    batches = [
        (i // blocks_per_batch, blocks_to_process[i : i + blocks_per_batch])
        for i in range(0, len(blocks_to_process), blocks_per_batch)
    ]

    effective = min(num_processes, len(batches))

    # For small files the Pool spawn + IPC pickle overhead exceeds compute gains.
    # Serial path (effective=1) is faster when total blocks fit in one or two batches.
    # Threshold ~20k covers typical uint8 800×800 (10k blocks) and small DSM tiles.
    _SERIAL_THRESHOLD = 20_000
    if len(blocks_to_process) < _SERIAL_THRESHOLD:
        effective = 1

    if effective > 1:
        batch_func = partial(process_block_batch, config=config)
        with multiprocessing.Pool(processes=effective) as pool:
            if not config.output.quiet:
                results = list(
                    tqdm(
                        pool.imap(batch_func, batches),
                        total=len(batches),
                        desc="Compression",
                        ascii=False,
                    )
                )
            else:
                results = pool.map(batch_func, batches)
        for batch_results in results:
            for idx, compressed, error in batch_results:
                compressed_blocks[idx] = compressed
                max_errors[idx] = error
    else:
        for batch_data in tqdm(
            batches, desc="Compression", disable=config.output.quiet, ascii=False
        ):
            for idx, compressed, error in process_block_batch(batch_data, config):
                compressed_blocks[idx] = compressed
                max_errors[idx] = error


def _retry_with_smaller_n(
    image_or_src,
    block_size,
    nodata_value,
    config,
    num_processes,
    max_errors,
    src_dtype,
    in_memory,
    band_idx=None,
):
    smaller_n = {32: 16, 16: 8}[block_size]
    if not config.output.quiet:
        failed = sum(1 for e in max_errors if e > config.compression.accuracy)
        print(f"  ✗ Validity FAILED ({failed} blocks) - retrying with N={smaller_n}")

    retry_cfg = deepcopy(config)
    retry_cfg.compression.block_size = smaller_n
    retry_cfg.compression.auto_select_block_size = False

    if in_memory:
        return compress_channel_dct(
            image_or_src, smaller_n, nodata_value, retry_cfg, num_processes, src_dtype=src_dtype
        )
    return compress_channel_dct_streaming(
        image_or_src,
        band_idx,
        smaller_n,
        nodata_value,
        retry_cfg,
        num_processes,
        src_dtype=src_dtype,
    )


def _streaming_detect_and_autoselect(
    src, band_idx, H, W, block_size, nodata_value, config, src_dtype
):
    sample_h = min(1024, H)
    sample_w = min(1024, W)
    row_off = max(0, (H - sample_h) // 2)
    col_off = max(0, (W - sample_w) // 2)
    sample = src.read(band_idx, window=Window(col_off, row_off, sample_w, sample_h)).astype(
        np.float32
    )

    is_integer = src_dtype in _INTEGER_DTYPES
    if is_integer:
        config = deepcopy(config)
        config.model.minimize_backscan = 0
        config.model.backscan_break_after = 0

    if config.compression.auto_select_block_size:
        block_size = auto_select_block_size(sample, config, nodata_value)
        config = deepcopy(config)
        config.compression.block_size = block_size

    return config, block_size


def _calc_padded_dims(H, W, n):
    padded_H = H + (n - H % n) % n
    padded_W = W + (n - W % n) % n
    return padded_H, padded_W, (padded_H, padded_W)


def _calc_strip_rows(W, n):
    strip_rows = max(n, (64 * 1024 * 1024) // (W * 4))
    return ((strip_rows + n - 1) // n) * n


def _calc_effective_processes(total_blocks, num_processes):
    min_per_worker = max(256, total_blocks // (num_processes * 4))
    return min(num_processes, max(1, total_blocks // min_per_worker))


def _read_and_pad_strip(src, band_idx, y_start, H, W, strip_rows, n, pad_W):
    y_end = min(y_start + strip_rows, H)
    read_h = y_end - y_start
    if read_h <= 0:
        return None, 0

    strip_data = src.read(band_idx, window=Window(0, y_start, W, read_h)).astype(np.float32)

    if pad_W > 0:
        strip_data = np.pad(strip_data, ((0, 0), (0, pad_W)), mode="edge")
    strip_pad_h = (n - read_h % n) % n if read_h % n != 0 else 0
    if strip_pad_h > 0:
        strip_data = np.pad(strip_data, ((0, strip_pad_h), (0, 0)), mode="edge")

    return strip_data, read_h


def _collect_strip_results(ar_or_results, compressed_blocks, max_errors):
    for idx, compressed, error in ar_or_results:
        compressed_blocks[idx] = compressed
        max_errors[idx] = error


def _run_strips_multiprocess(
    src,
    band_idx,
    H,
    W,
    padded_H,
    padded_W,
    n,
    strip_rows,
    blocks_per_row,
    nodata_value,
    config,
    effective_processes,
    compressed_blocks,
    max_errors,
):
    pad_W = padded_W - W
    total_blocks = len(compressed_blocks)
    strip_func = partial(process_strip_batch, config=config)

    pbar = (
        None
        if config.output.quiet
        else tqdm(total=total_blocks, desc="Compression", unit="blk", ascii=False)
    )
    pending = []

    with multiprocessing.Pool(processes=effective_processes) as pool:
        for y_start in range(0, padded_H, strip_rows):
            strip_data, _ = _read_and_pad_strip(src, band_idx, y_start, H, W, strip_rows, n, pad_W)
            if strip_data is None:
                break

            ar = pool.apply_async(
                strip_func, ((strip_data, y_start, blocks_per_row, nodata_value),)
            )
            pending.append(ar)

            if pbar:
                sH, sW = strip_data.shape
                pbar.update((sH // n) * (sW // n))

            del strip_data

            if len(pending) > effective_processes * 2:
                _flush_pending(pending, compressed_blocks, max_errors, limit=effective_processes)

        for ar in pending:
            _collect_strip_results(ar.get(), compressed_blocks, max_errors)

    if pbar:
        pbar.close()


def _run_strips_singleprocess(
    src,
    band_idx,
    H,
    W,
    padded_H,
    padded_W,
    n,
    strip_rows,
    blocks_per_row,
    nodata_value,
    config,
    compressed_blocks,
    max_errors,
):
    pad_W = padded_W - W
    total_blocks = len(compressed_blocks)
    pbar = (
        None
        if config.output.quiet
        else tqdm(total=total_blocks, desc="Compression", unit="blk", ascii=False)
    )

    for y_start in range(0, padded_H, strip_rows):
        strip_data, _ = _read_and_pad_strip(src, band_idx, y_start, H, W, strip_rows, n, pad_W)
        if strip_data is None:
            break

        results = process_strip_batch((strip_data, y_start, blocks_per_row, nodata_value), config)
        _collect_strip_results(results, compressed_blocks, max_errors)

        if pbar:
            sH, sW = strip_data.shape
            pbar.update((sH // n) * (sW // n))

        del strip_data

    if pbar:
        pbar.close()


def _flush_pending(pending, compressed_blocks, max_errors, limit):
    still_pending = []
    for ar in pending:
        if ar.ready():
            _collect_strip_results(ar.get(), compressed_blocks, max_errors)
        else:
            still_pending.append(ar)
            if len(still_pending) >= limit:
                _collect_strip_results(still_pending.pop(0).get(), compressed_blocks, max_errors)
    pending[:] = still_pending


__all__ = [
    "compress_channel_dct",
    "compress_channel_dct_streaming",
    "build_ycbcr_images",
    "compress_channel_rgb",
]
