#!/usr/bin/env python3
"""Environment and smoke-test verifier for FIDWAC v2."""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_MODULES = ["numpy", "scipy", "rasterio", "msgpack", "py7zr", "tqdm", "numba"]
OPTIONAL_MODULES = {
    "tkinter": "required for the desktop GUI",
    "turbojpeg": "required for JPEG quality mode (cm=1)",
}


def _print_result(status: str, label: str, detail: str = "") -> None:
    line = f"[{status}] {label}"
    if detail:
        line = f"{line}: {detail}"
    print(line)


def _import_module(name: str):
    try:
        module = importlib.import_module(name)
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
        return None, str(exc)
    version = getattr(module, "__version__", None)
    return module, version or "imported"


def check_imports(require_gui: bool, require_turbojpeg: bool) -> bool:
    ok = True

    print("== Python package imports ==")
    for name in REQUIRED_MODULES:
        module, detail = _import_module(name)
        if module is None:
            _print_result("FAIL", name, detail)
            ok = False
        else:
            _print_result("OK", name, str(detail))

    for name, purpose in OPTIONAL_MODULES.items():
        module, detail = _import_module(name)
        if module is None:
            is_required = (name == "tkinter" and require_gui) or (
                name == "turbojpeg" and require_turbojpeg
            )
            _print_result("FAIL" if is_required else "WARN", name, f"{detail} ({purpose})")
            ok = ok and not is_required
        else:
            _print_result("OK", name, str(detail))

    return ok


def check_archive_tools() -> None:
    print("\n== Archive backend ==")
    tool = next((candidate for candidate in ("7zz", "7zip", "7z") if shutil.which(candidate)), None)
    if tool:
        _print_result("OK", "7z backend", tool)
    else:
        _print_result("WARN", "7z backend", "system 7z not found; py7zr fallback will be used")


def run_lookup_smoke() -> bool:
    print("\n== Lookup-grid smoke test ==")
    grid_path = PROJECT_ROOT / "models" / "lookup_uint8_grid.npz"
    if not grid_path.exists():
        _print_result("WARN", "lookup_uint8_grid.npz", "file not found; skipping lookup test")
        return True

    try:
        import numpy as np
        predictor_uint8 = importlib.import_module("predictor.predictor_uint8")
        load_uint8_lookup_grid = predictor_uint8.load_uint8_lookup_grid
        predict_uint8_parameters = predictor_uint8.predict_uint8_parameters
        predict_uint8_parameters_batch = predictor_uint8.predict_uint8_parameters_batch
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
        _print_result("FAIL", "lookup imports", str(exc))
        return False

    grid = load_uint8_lookup_grid(str(grid_path))
    if grid is None:
        _print_result("FAIL", "lookup grid", "failed to load grid data")
        return False

    single_sf, single_bs = predict_uint8_parameters(10.0, 0.5, grid)
    batch_sf, batch_bs = predict_uint8_parameters_batch(
        np.array([5.0, 10.0, 20.0], dtype=np.float32),
        np.array([0.1, 0.5, 0.8], dtype=np.float32),
        grid,
    )
    if single_sf <= 0 or single_bs <= 0 or batch_sf.size != 3 or batch_bs.size != 3:
        _print_result("FAIL", "lookup predictions", "invalid prediction outputs")
        return False

    _print_result(
        "OK",
        "lookup predictions",
        f"single=(sf={single_sf}, bs={single_bs}), batch_bs={sorted(set(batch_bs.tolist()))}",
    )
    return True


def run_roundtrip_smoke(accuracy: float, keep_temp: bool) -> bool:
    print("\n== Compression round-trip smoke test ==")

    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        config_module = importlib.import_module("config")
        compression_module = importlib.import_module("compress.compression")
        decompression_module = importlib.import_module("compress.decompression")
        Config = config_module.Config
        compress_image = compression_module.compress_image
        decompress_file = decompression_module.decompress_file
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
        _print_result("FAIL", "smoke-test imports", str(exc))
        return False

    temp_dir = Path(tempfile.mkdtemp(prefix="fidwac-smoke-"))
    try:
        input_path = temp_dir / "synthetic_float32.tif"
        output_dir = temp_dir / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

        rows, cols = 32, 32
        yy, xx = np.mgrid[0:rows, 0:cols]
        yy = yy.astype(np.float32)
        xx = xx.astype(np.float32)
        data = ((xx * 0.2) + (yy * 0.1) + np.sin(xx / 4.0) + np.cos(yy / 5.0)).astype(np.float32)

        with rasterio.open(
            input_path,
            "w",
            driver="GTiff",
            height=rows,
            width=cols,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(10.0, 50.0, 1.0, 1.0),
        ) as dst:
            dst.write(data, 1)

        config = Config()
        config.compression.accuracy = accuracy
        config.compression.block_size = 8
        config.compression.auto_select_block_size = False
        config.compression.decimal_places = 2
        config.compression.crs = ""
        config.compression.uint8_accuracy_mode = False
        config.compression.lossless = False
        config.model.backend = "binary"
        config.model.advanced_heuristic = False
        config.model.accept_prediction_if_within_accuracy = False
        config.output.quiet = True

        archive_path = Path(
            compress_image(str(input_path), config=config, num_processes=1, output_dir=str(output_dir))
        )
        restored_path = Path(decompress_file(str(archive_path), config=config, output_dir=str(output_dir)))

        with rasterio.open(input_path) as src:
            original = src.read(1)
            original_transform = src.transform
            original_crs = src.crs

        with rasterio.open(restored_path) as src:
            restored = src.read(1)
            restored_transform = src.transform
            restored_crs = src.crs

        max_error = float(np.max(np.abs(restored.astype(np.float64) - original.astype(np.float64))))
        mean_error = float(np.mean(np.abs(restored.astype(np.float64) - original.astype(np.float64))))
        same_shape = restored.shape == original.shape
        same_transform = restored_transform == original_transform
        same_crs = restored_crs == original_crs

        if not same_shape or not same_transform or not same_crs:
            _print_result(
                "FAIL",
                "round-trip metadata",
                f"shape={same_shape}, transform={same_transform}, crs={same_crs}",
            )
            return False

        if max_error > accuracy + 1e-6:
            _print_result(
                "FAIL",
                "round-trip accuracy",
                f"max_error={max_error:.6f} exceeds requested accuracy={accuracy}",
            )
            return False

        _print_result(
            "OK",
            "round-trip accuracy",
            f"max_error={max_error:.6f}, mean_error={mean_error:.6f}, archive={archive_path.name}",
        )
        if keep_temp:
            _print_result("OK", "temporary files kept", str(temp_dir))
        return True
    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify FIDWAC v2 environment and run smoke tests.")
    parser.add_argument("--accuracy", type=float, default=0.05, help="Accuracy used in the round-trip smoke test.")
    parser.add_argument("--skip-roundtrip", action="store_true", help="Skip the compression/decompression smoke test.")
    parser.add_argument("--skip-lookup", action="store_true", help="Skip the uint8 lookup-grid smoke test.")
    parser.add_argument("--require-gui", action="store_true", help="Treat missing tkinter support as an error.")
    parser.add_argument(
        "--require-turbojpeg",
        action="store_true",
        help="Treat missing TurboJPEG support as an error.",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary smoke-test files for inspection.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ok = check_imports(args.require_gui, args.require_turbojpeg)
    check_archive_tools()

    if not args.skip_lookup:
        ok = run_lookup_smoke() and ok
    if not args.skip_roundtrip:
        ok = run_roundtrip_smoke(args.accuracy, args.keep_temp) and ok

    print("\n== Summary ==")
    if ok:
        _print_result("OK", "environment", "FIDWAC v2 verification passed")
        return 0

    _print_result("FAIL", "environment", "FIDWAC v2 verification failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
