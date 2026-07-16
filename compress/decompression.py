"""
FIDVAC v2 - Decompression logic (uniform grid, adaptive blocks)
"""

import os
import shutil
from typing import Optional, Dict, Any

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
import msgpack
import py7zr
from tqdm import tqdm

from engine.blocks import reconstruct_block, ycbcr_to_rgb
from config import Config, load_config
from core.codec import decode_vlq, dc_median_ac_decode


def load_compressed_data(file_path: str) -> Dict[str, Any]:
    """Extract and load compressed data from a .7z archive."""
    import tempfile
    import subprocess

    # Create temp directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Use system 7zz/7zip/7z or fallback to py7zr
        if shutil.which("7zz"):
            cmd = "7zz"
        elif shutil.which("7zip"):
            cmd = "7zip"
        elif shutil.which("7z"):
            cmd = "7z"
        else:
            cmd = None

        if cmd:
            subprocess.run(
                [cmd, "x", "-y", f"-o{temp_dir}", file_path],
                capture_output=True,
                check=True,
            )
            names = os.listdir(temp_dir)
        else:
            with py7zr.SevenZipFile(file_path, "r") as archive:
                names = archive.getnames()
                if not names:
                    raise ValueError(f"Archive {file_path} is empty")
                archive.extractall(path=temp_dir)

        if not names:
            raise ValueError(f"Archive {file_path} is empty")

        # Detect format:
        # A) jpeg_strips: rgb.bin + meta.msgpack
        # B) single GeoTIFF (legacy JPEG GeoTIFF path)
        # C) single msgpack (DCT path)
        has_rgb_bin = any(n.endswith(".bin") and "rgb" in n.lower() for n in names)
        has_meta_msgpack = any(n.endswith(".msgpack") and "meta" in n.lower() for n in names)
        tif_files = [n for n in names if n.lower().endswith((".tif", ".tiff"))]
        msgpack_files = [n for n in names if n.lower().endswith(".msgpack")]

        if has_rgb_bin and has_meta_msgpack:
            # Format A: jpeg_strips
            rgb_bin_data = None
            meta_data = None
            for name in names:
                p = os.path.join(temp_dir, name)
                if not os.path.exists(p):
                    continue
                if name.endswith(".bin") and "rgb" in name.lower():
                    with open(p, "rb") as f:
                        rgb_bin_data = f.read()
                elif name.endswith(".msgpack") and "meta" in name.lower():
                    with open(p, "rb") as f:
                        meta_data = f.read()
            if rgb_bin_data is None or meta_data is None:
                raise ValueError(
                    f"JPEG strips format: missing rgb.bin or meta.msgpack in {file_path}"
                )
            result = msgpack.unpackb(meta_data, raw=False, strict_map_key=False)
            result["_rgb_bin"] = rgb_bin_data
            return result

        if tif_files and not msgpack_files:
            # Format B: single GeoTIFF
            return os.path.join(temp_dir, tif_files[0])

        # Format C: single msgpack (DCT)
        msgpack_data = None
        for name in names:
            p = os.path.join(temp_dir, name)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    msgpack_data = f.read()
                break
        if msgpack_data is None:
            raise ValueError(f"msgpack data not found in {file_path}")
        return msgpack.unpackb(msgpack_data, raw=False, strict_map_key=False)

    finally:
        # Remove temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def _decode_dct_channel_rgb(
    ch_data: Dict[str, Any], block_size: int, padded_shape: tuple
) -> Dict[str, np.ndarray]:
    """Decode DCT coefficients for per-block RGB mode (cm=6).

    Returns dict with reconstructed R, G, B channels.
    """
    raw_l = ch_data.get("l", [])
    raw_c = ch_data.get("c", [])
    scaling_factor = ch_data.get("sf", 1)

    padded_H, padded_W = padded_shape
    blocks_per_row = padded_W // block_size

    # Flat format: l = [Y_lengths, Cb_lengths, Cr_lengths]
    # Flat format: c = [Y_dc, Y_ac, Cb_dc, Cb_ac, Cr_dc, Cr_ac, mult_y, mult_cb, mult_cr, fallback]
    if isinstance(raw_l, list) and len(raw_l) == 3 and not isinstance(raw_l[0], dict):
        lengths_map = {"Y": raw_l[0], "Cb": raw_l[1], "Cr": raw_l[2]}
    else:
        # Old dict format
        lengths_map = raw_l

    if isinstance(raw_c, list) and len(raw_c) == 10:
        encoded_map = {
            "Y": [raw_c[0], raw_c[1]],
            "Cb": [raw_c[2], raw_c[3]],
            "Cr": [raw_c[4], raw_c[5]],
        }
    elif isinstance(raw_c, list) and len(raw_c) == 6:
        encoded_map = {
            "Y": [raw_c[0], raw_c[1]],
            "Cb": [raw_c[2], raw_c[3]],
            "Cr": [raw_c[4], raw_c[5]],
        }
    else:
        # Old dict format
        encoded_map = {ch: raw_c.get(ch, b"") for ch in ["Y", "Cb", "Cr"]}

    # Decode each channel
    channels = {}
    for ch in ["Y", "Cb", "Cr"]:
        lengths = lengths_map[ch]
        encoded = encoded_map[ch]
        if isinstance(encoded, list) and len(encoded) == 2:
            dc_bytes, ac_bytes = encoded
            coeffs = dc_median_ac_decode(dc_bytes, ac_bytes, lengths, blocks_per_row)
        else:
            coeffs = np.array([], dtype=np.int64)

        # Reconstruct channel
        result = np.zeros((padded_H, padded_W), dtype=np.float32)
        coeff_pos = 0

        for idx, length in enumerate(lengths):
            row = idx // blocks_per_row
            col = idx % blocks_per_row
            y_start = row * block_size
            x_start = col * block_size

            if length == 0:
                block = np.zeros((block_size, block_size), dtype=np.float32)
            elif length == -1:
                block = np.full(
                    (block_size, block_size), -9999, dtype=np.float32
                )  # nodata placeholder
            else:
                block_coeffs = coeffs[coeff_pos : coeff_pos + length]
                coeff_pos += length

                # Reconstruct YCbCr without uint8_mode (coefficients already compressed with appropriate accuracy)
                block = reconstruct_block(
                    block_coeffs.tolist(),
                    block_size,
                    scaling_factor,
                    -9999,
                    uint8_mode=False,
                )
                # Clip to [0,255]: IDCT can overshoot; matches compression error check
                block = np.clip(block + 128.0, 0.0, 255.0)

            result[y_start : y_start + block_size, x_start : x_start + block_size] = block

        channels[ch] = result

    # Apply inverse YCbCr transform
    Y = channels["Y"]
    Cb = channels["Cb"]
    Cr = channels["Cr"]

    R, G, B = ycbcr_to_rgb(Y, Cb, Cr)
    R = R.astype(np.float32)
    G = G.astype(np.float32)
    B = B.astype(np.float32)

    return {"R": R, "G": G, "B": B}


def _decompress_single_channel(
    ch_data: Dict[str, Any],
    block_size: int,
    original_shape: tuple,
    nodata_value: float,
    scaling_factor: int,
    version: int,
) -> np.ndarray:
    """Decompress a single channel from flat format data. Returns 2D array."""
    lengths = ch_data.get("l", [])
    cm = ch_data.get("cm", 0)
    raw_c = ch_data.get("c", [])
    # Use per-channel scaling_factor if stored, otherwise fall back to global
    scaling_factor = ch_data.get("sf", scaling_factor)
    # Use per-channel block_size if available, otherwise fall back to global
    block_size = ch_data.get("bs", block_size)

    H, W = original_shape
    # Compute padded_shape from per-channel block_size (header value may differ)
    padded_H = ((H + block_size - 1) // block_size) * block_size
    padded_W = ((W + block_size - 1) // block_size) * block_size
    blocks_per_row = padded_W // block_size

    if cm == 4 and isinstance(raw_c, (bytes, bytearray)):
        # Lossless PNG mode: zlib decompress + inverse filter (if used)
        import zlib

        filter_used = ch_data.get("f", False)
        decompressed = zlib.decompress(raw_c)
        filtered = np.frombuffer(decompressed, dtype=np.uint8).reshape(H, W)
        if filter_used:
            # Inverse Sub filter: original[i] = filtered[i] + original[i-1]
            result = filtered.copy()
            result[:, 1:] = filtered[:, 1:] + result[:, :-1]
            result[:, 0] = filtered[:, 0]
        else:
            # No filter applied
            result = filtered
        return result.astype(np.float32)
    if cm == 5:
        # uint8 accuracy mode: DC-AC format, coefficients in centered space
        if isinstance(raw_c, list) and len(raw_c) == 2:
            dc_data, ac_data = raw_c
            coeffs_arr = dc_median_ac_decode(dc_data, ac_data, lengths, blocks_per_row)
            coeffs = coeffs_arr.tolist()
        else:
            coeffs = []
    elif cm == 3 and isinstance(raw_c, list) and len(raw_c) == 2:
        # New format: separate DC/AC streams with DC median prediction
        dc_data, ac_data = raw_c
        coeffs_arr = dc_median_ac_decode(dc_data, ac_data, lengths, blocks_per_row)
        coeffs = coeffs_arr.tolist()
    elif version >= 2 and isinstance(raw_c, (bytes, bytearray)):
        coeffs = decode_vlq(raw_c)
    else:
        coeffs = list(raw_c) if raw_c else []

    masks_data = ch_data.get("m", [])
    masks_dict = {item[0]: item[1] for item in masks_data}

    # Set result dtype based on compression mode
    result_dtype = np.uint8 if (cm == 5 or cm == 4) else np.float32
    result = np.zeros((padded_H, padded_W), dtype=result_dtype)

    coeff_pos = 0
    for idx, length in enumerate(lengths):

        row = idx // blocks_per_row
        col = idx % blocks_per_row
        y = row * block_size
        x = col * block_size

        if length == 0:
            result[y : y + block_size, x : x + block_size] = 0
        elif length == -1:
            result[y : y + block_size, x : x + block_size] = (
                nodata_value if nodata_value is not None else 0
            )
        elif length == -2:
            if idx in masks_dict:
                block = reconstruct_block(
                    masks_dict[idx],
                    block_size,
                    scaling_factor,
                    nodata_value,
                    uint8_mode=(cm == 5),
                )
                result[y : y + block_size, x : x + block_size] = block
        else:
            block_coeffs = coeffs[coeff_pos : coeff_pos + length]
            coeff_pos += length
            block = reconstruct_block(
                block_coeffs,
                block_size,
                scaling_factor,
                nodata_value,
                uint8_mode=(cm == 5),
            )
            result[y : y + block_size, x : x + block_size] = block

    return result[:H, :W]


def _decompress_single_channel_streaming(
    ch_data: Dict[str, Any],
    block_size: int,
    original_shape: tuple,
    nodata_value: float,
    scaling_factor: int,
    version: int,
    config: Config,
    dst,
    band_idx: int,
) -> None:
    """Decompress a single channel and write blocks directly to GeoTIFF via window.

    Avoids allocating the full padded image in RAM. Writes each reconstructed
    block directly to the output raster using rasterio's write(window=...).
    """
    from rasterio.windows import Window

    lengths = ch_data.get("l", [])
    cm = ch_data.get("cm", 0)
    raw_c = ch_data.get("c", [])
    # Use per-channel scaling_factor if stored, otherwise fall back to global
    scaling_factor = ch_data.get("sf", scaling_factor)
    # Use per-channel block_size if available, otherwise fall back to global
    block_size = ch_data.get("bs", block_size)

    H, W = original_shape
    # Compute padded_shape from per-channel block_size (header value may differ)
    _ = ((H + block_size - 1) // block_size) * block_size
    padded_W = ((W + block_size - 1) // block_size) * block_size
    blocks_per_row = padded_W // block_size

    if cm == 5:
        # uint8 accuracy mode: DC-AC format, coefficients in centered space
        if isinstance(raw_c, list) and len(raw_c) == 2:
            dc_data, ac_data = raw_c
            coeffs_arr = dc_median_ac_decode(dc_data, ac_data, lengths, blocks_per_row)
            coeffs = coeffs_arr.tolist()
        else:
            coeffs = []
    elif cm == 3 and isinstance(raw_c, list) and len(raw_c) == 2:
        # New format: separate DC/AC streams with DC median prediction
        dc_data, ac_data = raw_c
        coeffs_arr = dc_median_ac_decode(dc_data, ac_data, lengths, blocks_per_row)
        coeffs = coeffs_arr.tolist()
    elif version >= 2 and isinstance(raw_c, (bytes, bytearray)):
        coeffs = decode_vlq(raw_c)
    else:
        coeffs = list(raw_c) if raw_c else []

    masks_data = ch_data.get("m", [])
    masks_dict = {item[0]: item[1] for item in masks_data}

    # Set block dtype based on compression mode
    block_dtype = np.uint8 if (cm == 5 or cm == 4) else np.float32

    coeff_pos = 0
    for idx in tqdm(range(len(lengths)), desc=f"Ch{band_idx}", disable=config.output.quiet):
        length = lengths[idx]

        brow = idx // blocks_per_row
        bcol = idx % blocks_per_row
        y = brow * block_size
        x = bcol * block_size

        # Skip blocks outside original image bounds (padding-only)
        if y >= H or x >= W:
            coeff_pos += max(0, length)
            continue

        if length == 0:
            block = np.zeros((block_size, block_size), dtype=block_dtype)
        elif length == -1:
            block = np.full(
                (block_size, block_size),
                nodata_value if nodata_value is not None else 0,
                dtype=block_dtype,
            )
        elif length == -2:
            if idx in masks_dict:
                block = reconstruct_block(
                    masks_dict[idx],
                    block_size,
                    scaling_factor,
                    nodata_value,
                    uint8_mode=(cm == 5),
                )
            else:
                block = np.zeros((block_size, block_size), dtype=block_dtype)
        else:
            block_coeffs = coeffs[coeff_pos : coeff_pos + length]
            coeff_pos += length
            block = reconstruct_block(
                block_coeffs,
                block_size,
                scaling_factor,
                nodata_value,
                uint8_mode=(cm == 5),
            )

        # Clip to original image bounds
        write_h = min(block_size, H - y)
        write_w = min(block_size, W - x)
        if write_h <= 0 or write_w <= 0:
            continue

        window = Window(x, y, write_w, write_h)
        dst.write(block[:write_h, :write_w].astype(np.float32), band_idx, window=window)


def decompress_image(
    dcv_compress: Dict[str, Any],
    config: Optional[Config] = None,
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> str:
    """Decompress image from in-memory compression data dict. Returns output path."""
    if config is None:
        config = load_config()

    if output_dir is None:
        output_dir = config.results_dir

    os.makedirs(output_dir, exist_ok=True)

    # Get metadata
    if "n" in dcv_compress:
        # New flat format (v2)
        block_size = dcv_compress.get("n", config.compression.block_size)
        version = dcv_compress.get("v", 1)
        num_channels = dcv_compress.get("ch", 1)
        original_shape = dcv_compress.get("s", [0, 0])
        padded_shape = dcv_compress.get("p", original_shape)
        transform_list = dcv_compress.get("t", [1, 0, 0, 0, 1, 0])
        crs_str = dcv_compress.get("r", config.compression.crs)
        nodata_value = dcv_compress.get("d", -9999)
        # Separate: internal nodata for block reconstruction vs output nodata
        # If source had no nodata (d=None, dn=False), use -9999 internally but write None to GeoTIFF
        internal_nodata = nodata_value if nodata_value is not None else -9999.0
        output_nodata = nodata_value  # None if source had no nodata
        scaling_factor = dcv_compress.get("f", config.scaling_factor)
        is_flat_format = True
    else:
        # Old format
        block_size = dcv_compress.get("block_size", config.compression.block_size)
        version = 1
        original_shape = dcv_compress.get("original_shape", [0, 0])
        padded_shape = dcv_compress.get("padded_shape", original_shape)
        transform_list = dcv_compress.get("transform", [1, 0, 0, 0, 1, 0])
        crs_str = dcv_compress.get("crs", config.compression.crs)
        nodata_value = dcv_compress.get("nodata", -9999)
        scaling_factor = dcv_compress.get("scaling_factor", config.scaling_factor)
        num_channels = 1
        is_flat_format = False
        internal_nodata = nodata_value
        output_nodata = nodata_value

    H, W = original_shape
    padded_H, padded_W = padded_shape

    # Decide: streaming vs in-memory (>256 MB per channel → streaming)
    channel_bytes = H * W * 4
    use_streaming = channel_bytes > 256 * 1024 * 1024

    if not config.output.quiet:
        mode_str = "streaming" if use_streaming else "in-memory"
        print(
            f"  Decompressing: {W}x{H}, channels={num_channels}, block_size={block_size}, Mode: {mode_str}"
        )

    # Przygotuj transform
    if len(transform_list) >= 6:
        transform = Affine(
            transform_list[0],
            transform_list[1],
            transform_list[2],
            transform_list[3],
            transform_list[4],
            transform_list[5],
        )
    else:
        transform = Affine.identity()

    # Przygotuj CRS — None for non-georeferenced rasters (PNG, JPG, etc.)
    crs = None
    if crs_str and crs_str != "None":
        try:
            crs = CRS.from_string(crs_str)
        except Exception:
            try:
                crs = CRS.from_epsg(2180)
            except Exception:
                crs = None

    # Zapisz jako GeoTIFF
    if output_filename is None:
        output_filename = "decompressed.tif"

    output_path = os.path.join(output_dir, output_filename)

    tiff_compress = config.output.tiff_compression.upper()
    if tiff_compress not in ("DEFLATE", "LZW", "LZMA", "ZSTD", "NONE"):
        tiff_compress = "DEFLATE"

    # Detect uint8 accuracy mode (cm=5, cm=4, or cm=6) to set correct output dtype
    is_uint8_mode = False
    if is_flat_format:
        if num_channels == 1:
            cm = dcv_compress.get("cm", 0)
            is_uint8_mode = cm == 5 or cm == 4
        else:
            channel_list = dcv_compress.get("channels", [])
            if channel_list:
                # Check if all channels have cm=5, cm=4, or cm=6 (per-block RGB)
                is_uint8_mode = all(ch.get("cm", 0) in (5, 4, 6) for ch in channel_list)

    out_dtype = np.uint8 if is_uint8_mode else np.float32
    out_nodata = output_nodata

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=num_channels,
        dtype=out_dtype,
        crs=crs,
        transform=transform,
        nodata=out_nodata,
        compress=tiff_compress if tiff_compress != "NONE" else None,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        if use_streaming and is_flat_format:
            # Streaming mode: write blocks directly to GeoTIFF
            if num_channels == 1:
                ch_data = {
                    "l": dcv_compress.get("l", []),
                    "c": dcv_compress.get("c", []),
                    "m": dcv_compress.get("m", []),
                    "cm": dcv_compress.get("cm", 0),
                }
                _decompress_single_channel_streaming(
                    ch_data,
                    block_size,
                    (H, W),
                    internal_nodata,
                    scaling_factor,
                    version,
                    config,
                    dst,
                    1,
                )
            else:
                channel_list = dcv_compress.get("channels", [])
                for ch_idx, ch_data in enumerate(channel_list):
                    _decompress_single_channel_streaming(
                        ch_data,
                        block_size,
                        (H, W),
                        internal_nodata,
                        scaling_factor,
                        version,
                        config,
                        dst,
                        ch_idx + 1,
                    )
        else:
            # In-memory mode: decompress to arrays, then write
            channels = []

            # Check for per-block RGB mode (cm=6)
            if is_flat_format and num_channels == 3:
                channel_list = dcv_compress.get("channels", [])
                if channel_list and channel_list[0].get("cm") == 6:
                    # Per-block RGB mode
                    ch_data = channel_list[0]
                    rgb_channels = _decode_dct_channel_rgb(
                        ch_data, block_size, (padded_H, padded_W)
                    )
                    channels.append(rgb_channels["R"][:H, :W])
                    channels.append(rgb_channels["G"][:H, :W])
                    channels.append(rgb_channels["B"][:H, :W])
                else:
                    # Standard multi-channel
                    for ch_idx, ch_data in enumerate(channel_list):
                        ch_result = _decompress_single_channel(
                            ch_data,
                            block_size,
                            (H, W),
                            internal_nodata,
                            scaling_factor,
                            version,
                        )
                        channels.append(ch_result)
            elif is_flat_format and num_channels == 1:
                ch_data = {
                    "l": dcv_compress.get("l", []),
                    "c": dcv_compress.get("c", []),
                    "m": dcv_compress.get("m", []),
                    "cm": dcv_compress.get("cm", 0),
                }
                ch_result = _decompress_single_channel(
                    ch_data,
                    block_size,
                    (H, W),
                    internal_nodata,
                    scaling_factor,
                    version,
                )
                channels.append(ch_result)
            elif is_flat_format and num_channels > 1:
                channel_list = dcv_compress.get("channels", [])
                rgb_decoded = False
                for ch_idx, ch_data in enumerate(channel_list):
                    if ch_data.get("cm") == 6:
                        if not rgb_decoded and "c" in ch_data:
                            # First cm=6 entry with data: decode all RGB channels
                            if not config.output.quiet:
                                print(f"  Channel {ch_idx + 1}/{num_channels} (RGB per-block):")
                            rgb_channels = _decode_dct_channel_rgb(
                                ch_data, block_size, (padded_H, padded_W)
                            )
                            channels.append(rgb_channels["R"][:H, :W])
                            channels.append(rgb_channels["G"][:H, :W])
                            channels.append(rgb_channels["B"][:H, :W])
                            rgb_decoded = True
                        # Skip marker cm=6 entries (they have no "c" key)
                        continue
                    if not config.output.quiet:
                        print(f"  Channel {ch_idx + 1}/{num_channels}:")
                    ch_result = _decompress_single_channel(
                        ch_data,
                        block_size,
                        (H, W),
                        internal_nodata,
                        scaling_factor,
                        version,
                    )
                    channels.append(ch_result)
            else:
                # Old format (single channel only)
                compressed_blocks = dcv_compress.get("compressed_blocks", [])
                result = np.zeros((padded_H, padded_W), dtype=np.float32)
                blocks_per_row = padded_W // block_size

                for idx, compressed_data in enumerate(
                    tqdm(
                        compressed_blocks,
                        desc="Dekompresja",
                        disable=config.output.quiet,
                    )
                ):
                    if compressed_data is None:
                        continue

                    row = idx // blocks_per_row
                    col = idx % blocks_per_row
                    y = row * block_size
                    x = col * block_size

                    try:
                        block = reconstruct_block(
                            compressed_data, block_size, scaling_factor, internal_nodata
                        )
                        result[y : y + block_size, x : x + block_size] = block
                    except Exception as e:
                        if not config.output.quiet:
                            print(f"  Warning: Block {idx} reconstruction failed: {e}")

                channels.append(result[:H, :W])

            # Inverse YCbCr if channels were decorrelated during compression
            # Skip for cm=6 (per-block RGB) - channels are already R,G,B
            ycbcr_rgb = dcv_compress.get("ycbcr_rgb", [])
            has_cm6 = any(ch.get("cm") == 6 for ch in dcv_compress.get("channels", []))
            if len(ycbcr_rgb) == 3 and not has_cm6:
                r0, g0, b0 = ycbcr_rgb
                Y = channels[r0].astype(np.float32)
                Cb = channels[g0].astype(np.float32)
                Cr = channels[b0].astype(np.float32)
                R, G, B = ycbcr_to_rgb(Y, Cb, Cr)
                channels[r0] = R.astype(np.float32)
                channels[g0] = G.astype(np.float32)
                channels[b0] = B.astype(np.float32)

            for ch_idx, ch_result in enumerate(channels):
                if np.issubdtype(out_dtype, np.integer):
                    write_data = np.round(ch_result).astype(out_dtype)
                else:
                    write_data = ch_result.astype(out_dtype)
                dst.write(write_data, ch_idx + 1)

    if not config.output.quiet:
        print(f"  Saved: {output_path}")

    return output_path


def decompress_file(
    file_path: str, config: Optional[Config] = None, output_dir: Optional[str] = None
) -> str:
    """Decompress .7z archive to GeoTIFF. Returns output path."""
    if config is None:
        config = load_config()

    if output_dir is None:
        output_dir = config.results_dir

    if not config.output.quiet:
        print(f"Loading: {file_path}")

    dcv_compress = load_compressed_data(file_path)

    # Format B: single GeoTIFF (legacy JPEG GeoTIFF)
    if isinstance(dcv_compress, str) and os.path.exists(dcv_compress):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.tif")
        shutil.move(dcv_compress, output_path)
        if not config.output.quiet:
            print(f"  Saved: {output_path}")
        return output_path

    # Format A: jpeg_strips (rgb.bin + meta.msgpack)
    if isinstance(dcv_compress, dict) and dcv_compress.get("mode") == "jpeg_strips":
        from turbojpeg import TurboJPEG, TJPF_RGB
        from engine.utils import decode_mask_rle

        meta = dcv_compress
        rgb_bin = meta["_rgb_bin"]
        strip_sizes = meta["strip_sizes"]
        strip_offsets = meta["strip_offsets"]
        H, W = meta["original_shape"]
        num_bands = meta["num_bands"]
        nodata_rle = meta.get("nodata_rle")
        src_nodata = meta.get("nodata")
        crs_str = meta.get("crs")
        t = meta["transform"]
        crs = CRS.from_string(crs_str) if crs_str else None

        # Reconstruct nodata mask
        nodata_mask = None
        if nodata_rle:
            nodata_mask = decode_mask_rle(nodata_rle, (H, W))

        # Split rgb_bin into strips
        strips = []
        pos = 0
        for sz in strip_sizes:
            strips.append(rgb_bin[pos : pos + sz])
            pos += sz
        del rgb_bin

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.tif")

        tiff_compress = config.output.tiff_compression.upper()
        if tiff_compress not in ("DEFLATE", "LZW", "LZMA", "ZSTD", "NONE"):
            tiff_compress = "DEFLATE"

        tj = TurboJPEG()

        # Decode all strips into full RAM array, then write GeoTIFF in one shot
        if not config.output.quiet:
            print(f"  Allocating {H*W*num_bands/1024/1024:.0f} MB RAM...")
        full_img = np.empty((H, W, num_bands), dtype=np.uint8)

        pbar = (
            None
            if config.output.quiet
            else tqdm(total=len(strips), desc="Decompress", unit="strip", ascii=False)
        )

        for i, jpg_bytes in enumerate(strips):
            y_start = strip_offsets[i]
            rgb = tj.decode(jpg_bytes, pixel_format=TJPF_RGB)  # (strip_h, padded_W, 3) RGB
            strip_h = min(rgb.shape[0], H - y_start)
            if strip_h <= 0:
                continue
            full_img[y_start : y_start + strip_h, :, :] = rgb[:strip_h, :W, :]
            del rgb
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()
        del strips

        # Apply nodata mask to full image
        if nodata_mask is not None:
            fill = src_nodata if src_nodata is not None else 0
            full_img[nodata_mask] = fill

        if not config.output.quiet:
            print("  Encoding JPEG...")

        jpeg_quality = meta.get("quality", 85)
        from turbojpeg import TJSAMP_420

        TJFLAG_OPTIMIZE = 16
        jpg_data = tj.encode(
            full_img,
            quality=jpeg_quality,
            pixel_format=TJPF_RGB,
            jpeg_subsample=TJSAMP_420,
            flags=TJFLAG_OPTIMIZE,
        )
        del full_img

        # Write .jpg
        base_name = os.path.splitext(output_path)[0]
        jpg_path = base_name + ".jpg"
        with open(jpg_path, "wb") as f:
            f.write(jpg_data)
        del jpg_data

        # Write .jgw world file (georeferencing for JPEG)
        # Affine(t[0],t[1],t[2],t[3],t[4],t[5]) → world file order:
        # pixel_width, rotation_y, rotation_x, pixel_height, x_center_UL, y_center_UL
        with open(base_name + ".jgw", "w", encoding="utf-8") as f:
            f.write(f"{t[0]}\n{t[3]}\n{t[1]}\n{t[4]}\n{t[2]}\n{t[5]}\n")

        # Write .prj (CRS as WKT)
        if crs:
            with open(base_name + ".prj", "w", encoding="utf-8") as f:
                f.write(crs.to_wkt())

        if not config.output.quiet:
            print(f"  Saved: {jpg_path}")
        return jpg_path

    # Format C: DCT msgpack path
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{base_name}.tif"
    return decompress_image(dcv_compress, config, output_dir, output_filename)


# =============================================================================
# EKSPORT
# =============================================================================

__all__ = [
    "load_compressed_data",
    "decompress_image",
    "decompress_file",
]
