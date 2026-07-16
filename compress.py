#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIDVAC v2 - CLI
================
"""

import os
import re
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, Config
from compress.compression import compress_image
from compress.decompression import decompress_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FIDVAC v2 - DCT compression for geospatial data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compress.py -i input.tif -o output.7z               # Compression
  python compress.py -i input.png -o output.7z               # PNG + world file
  python compress.py -i input.tif -o output.7z --auto        # Auto-select block size
  python compress.py -i input.tif -o output.7z -a 0.01       # High accuracy
  python compress.py -i compressed.7z -o output.tif          # Decompression
  python compress.py -i input.tif -o output.7z --backend heuristic  # Use heuristic prediction

Supported input formats: GeoTIFF, PNG+PNGW, JPEG+JGW, BMP+BPW, ASC, Erdas Imagine (.img)
        """,
    )

    parser.add_argument("-i", "--input", required=True, help="Input file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file (optional for decompression: auto-derived from input)",
    )
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--accuracy", "-a", type=float, help="Compression accuracy (e.g., 0.05)")
    parser.add_argument("--block-size", "-n", type=int, choices=[8, 16, 32], help="Block size")
    parser.add_argument("--auto", action="store_true", help="Auto-select block size")
    parser.add_argument("--backend", choices=["binary", "heuristic"], help="Prediction backend")
    parser.add_argument(
        "--advanced-heuristic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable advanced heuristic (lookup table, 16 features)",
    )
    parser.add_argument(
        "--minimize-backscan", type=int, default=None, help="Minimize backscan threshold"
    )
    parser.add_argument(
        "--backscan-break-after",
        type=int,
        default=None,
        help="Break backscan after N consecutive failures (0 = never)",
    )
    parser.add_argument(
        "--accept-prediction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Accept prediction when within accuracy tolerance "
        "(skip binary search for float/uint8)",
    )
    parser.add_argument(
        "--decimal-places",
        type=int,
        default=None,
        help="Number of decimal places for scaling (default: 2)",
    )
    parser.add_argument("--processes", "-p", type=int, help="Number of processes")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    parser.add_argument(
        "--uint8-accuracy-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="uint8 accuracy mode (binary search)",
    )
    parser.add_argument(
        "--uint8-accuracy",
        type=int,
        default=None,
        help="Max color value error for uint8 accuracy (min=2, due to YCbCr rounding)",
    )
    parser.add_argument(
        "--rgb-quality", type=int, default=None, help="JPEG quality for RGB (1-100)"
    )
    parser.add_argument(
        "--lossless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Lossless PNG mode for uint8",
    )
    parser.add_argument(
        "--rgb-channel-indices",
        type=str,
        default=None,
        help="RGB channel indices R,G,B (1-based, e.g. 1,2,3)",
    )
    parser.add_argument(
        "--scaling-factor",
        type=str,
        default=None,
        help="Scaling factor or list of factors (comma-separated, e.g. 1,10)",
    )
    parser.add_argument(
        "--ycbcr-y-multiplier",
        type=float,
        default=None,
        help="YCbCr Y channel accuracy multiplier (default 0.9)",
    )
    parser.add_argument(
        "--ycbcr-cb-multiplier",
        type=float,
        default=None,
        help="YCbCr Cb channel accuracy multiplier (default 0.9)",
    )
    parser.add_argument(
        "--ycbcr-cr-multiplier",
        type=float,
        default=None,
        help="YCbCr Cr channel accuracy multiplier (default 0.9)",
    )
    parser.add_argument(
        "--ycbcr-per-block",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable per-block YCbCr multiplier selection " "(validity checked in RGB space)",
    )
    parser.add_argument(
        "--allowed-block-sizes",
        type=str,
        default=None,
        help="Allowed block sizes e.g. 8,16,32",
    )
    parser.add_argument(
        "--crs",
        type=str,
        default=None,
        help="CRS string (empty = no CRS override)",
    )
    parser.add_argument(
        "--compression-method",
        type=str,
        default=None,
        choices=["LZMA2", "PPMD", "BZIP2", "DEFLATE"],
        help="7z compression method",
    )
    parser.add_argument(
        "--verify-with-full-idct",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Validate final block error using full IDCT reconstruction",
    )
    parser.add_argument(
        "--l2-precheck",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable L2 precheck for fast rejection",
    )
    parser.add_argument(
        "--incremental-backscan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable incremental backscan",
    )
    parser.add_argument(
        "--uint8-L-prediction-scale",
        type=float,
        default=None,
        help="Single scale factor for uint8 L prediction (legacy shortcut)",
    )
    parser.add_argument(
        "--uint8-L-prediction-scales",
        type=str,
        default=None,
        help="Comma-separated uint8 L prediction cascade, " "e.g. 1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0",
    )

    return parser


def _apply_args_to_config(args: argparse.Namespace, config: Config) -> None:
    """Apply CLI argument overrides to the config object."""
    c, m, p, o = config.compression, config.model, config.performance, config.output

    # Compression settings
    if args.accuracy is not None:
        c.accuracy = args.accuracy
    if args.block_size is not None:
        c.block_size = args.block_size
        c.auto_select_block_size = False
    if args.auto:
        c.auto_select_block_size = True
    if args.decimal_places is not None:
        c.decimal_places = args.decimal_places
    if args.uint8_accuracy_mode is not None:
        c.uint8_accuracy_mode = args.uint8_accuracy_mode
    if args.uint8_accuracy is not None:
        c.uint8_accuracy = args.uint8_accuracy
    if args.rgb_quality is not None:
        c.rgb_quality = args.rgb_quality
    if args.lossless is not None:
        c.lossless = args.lossless
    if args.rgb_channel_indices is not None:
        c.rgb_channel_indices = [int(x) for x in args.rgb_channel_indices.split(",")]
    if args.scaling_factor is not None:
        sf_str = args.scaling_factor.strip("[]").strip()
        if "," in sf_str:
            c.uint8_scaling_factor = [int(x.strip()) for x in sf_str.split(",") if x.strip()]
        else:
            c.uint8_scaling_factor = int(sf_str)
    if args.ycbcr_y_multiplier is not None:
        c.ycbcr_y_multiplier = args.ycbcr_y_multiplier
    if args.ycbcr_cb_multiplier is not None:
        c.ycbcr_cb_multiplier = args.ycbcr_cb_multiplier
    if args.ycbcr_cr_multiplier is not None:
        c.ycbcr_cr_multiplier = args.ycbcr_cr_multiplier
    if args.ycbcr_per_block is not None:
        c.ycbcr_per_block_mode = args.ycbcr_per_block
    if args.allowed_block_sizes is not None:
        c.allowed_block_sizes = [int(x) for x in args.allowed_block_sizes.split(",")]
    if args.crs is not None:
        c.crs = args.crs

    # Model / prediction settings
    if args.backend is not None:
        m.backend = args.backend
    if args.advanced_heuristic is not None:
        m.advanced_heuristic = args.advanced_heuristic
    if args.minimize_backscan is not None:
        m.minimize_backscan = args.minimize_backscan
    if args.backscan_break_after is not None:
        m.backscan_break_after = args.backscan_break_after
    if args.accept_prediction is not None:
        m.accept_prediction_if_within_accuracy = args.accept_prediction
    if args.uint8_L_prediction_scales is not None:
        m.uint8_L_prediction_scales = [
            float(x.strip()) for x in args.uint8_L_prediction_scales.split(",") if x.strip()
        ]
    if args.uint8_L_prediction_scale is not None:
        m.uint8_L_prediction_scales = [args.uint8_L_prediction_scale]

    # Output settings
    if args.compression_method is not None:
        o.compression_method = args.compression_method
    if args.quiet:
        o.quiet = True

    # Performance settings
    if args.verify_with_full_idct is not None:
        p.verify_with_full_idct = args.verify_with_full_idct
    if args.l2_precheck is not None:
        p.l2_precheck_enabled = args.l2_precheck
    if args.incremental_backscan is not None:
        p.incremental_backscan = args.incremental_backscan


def _resolve_num_processes(args: argparse.Namespace) -> int:
    max_cpus = os.cpu_count() or 1
    if args.processes is not None:
        if args.processes < 1 or args.processes > max_cpus:
            print(f"Error: --processes must be between 1 and {max_cpus}")
            sys.exit(1)
        return args.processes
    return max_cpus


def _extract_validity_tag(result_path: str) -> str:
    bn = os.path.basename(result_path)
    if "_VT." in bn:
        return "VT"
    if "_VF." in bn:
        return "VF"
    m = re.search(r"_Q(\d+)[_.]", bn)
    return f"Q{m.group(1)}" if m else "OK"


def _print_banner(operation: str, input_path: str, output_path: str, config: Config) -> None:
    print("=" * 60)
    print(f"FIDVAC v2 - {operation}")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    if operation == "Compression":
        print(f"Auto-select: {config.compression.auto_select_block_size}")
        print(f"Backend: {config.model.backend}")
        print(f"Accuracy: {config.compression.accuracy}")
    print("=" * 60)


def main():
    """Main entry point for FIDVAC v2 CLI."""
    args = _build_parser().parse_args()

    # Resolve output path: derive output dir when -o is a dir or absent.
    # output_dir_only = True  → compress_image/decompress_file decides the filename
    #                            (keeps full descriptive name with params)
    # output_dir_only = False → user gave an explicit filename → rename to it
    input_is_archive = args.input.lower().endswith(".7z")
    output_dir_only = False

    if args.output is None:
        # no -o: use directory of input file, let compress/decompress name it
        args.output = os.path.dirname(os.path.abspath(args.input)) or "."
        output_dir_only = True
    elif os.path.isdir(args.output) or args.output.endswith(("/", os.sep)):
        # -o is an existing directory or trailing-slash path
        os.makedirs(args.output, exist_ok=True)
        output_dir_only = True
    # else: user gave explicit filename → output_dir_only stays False

    config = load_config(args.config)
    _apply_args_to_config(args, config)
    num_processes = _resolve_num_processes(args)

    # Check input file
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    output_path = args.output
    output_dir = output_path if output_dir_only else (os.path.dirname(output_path) or ".")
    if not os.path.exists(output_dir):
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    start_time = time.time()

    # Detect operation: compression vs decompression
    # When output_dir_only, infer from input extension.
    if output_dir_only:
        output_is_archive = not input_is_archive  # input .tif → compress; input .7z → decompress
    else:
        output_is_archive = output_path.lower().endswith(".7z")

    if not output_dir_only:
        if input_is_archive and output_is_archive:
            print("Error: Both input and output are archives - cannot determine operation")
            sys.exit(1)
        if not input_is_archive and not output_is_archive:
            print(
                "Error: Neither input nor output is an archive (.7z) - "
                "cannot determine operation"
            )
            sys.exit(1)

    operation = "Compression" if output_is_archive else "Decompression"
    out_dir_for_func = output_path if output_dir_only else (os.path.dirname(output_path) or ".")

    if not config.output.quiet:
        _print_banner(operation, input_path, output_path, config)

    try:
        if output_is_archive:
            result_path = compress_image(input_path, config, num_processes, out_dir_for_func)
            print(f"VALIDITY:{_extract_validity_tag(result_path)}", flush=True)
        else:
            result_path = decompress_file(input_path, config, out_dir_for_func)

        # Rename only when user gave an explicit output filename
        if not output_dir_only and result_path != output_path:
            os.replace(result_path, output_path)
        else:
            output_path = result_path
    except (IOError, OSError, RuntimeError) as e:
        print(f"Error during {operation.lower()}: {e}")
        sys.exit(1)

    if not config.output.quiet:
        elapsed = time.time() - start_time
        print(f"\n{operation} completed in {elapsed:.2f}s")
        print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
