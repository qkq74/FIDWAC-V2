"""
FIDVAC v2 - Configuration management
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class DirectoriesConfig:
    """Directory paths."""

    results: str = "./results/"
    source: str = "./source/"
    models: str = "./models/"
    cache: str = "../cache_train_v2/"


@dataclass
class CompressionConfig:
    """Compression parameters."""

    accuracy: float = 0.05
    block_size: int = 16
    decimal_places: int = 2
    dct_type: int = 2
    crs: str = "epsg:2180"
    auto_select_block_size: bool = False
    auto_select_sample_size: int = 1000  # config.json default
    auto_select_std_threshold_high: float = 10.0
    auto_select_std_threshold_medium: float = 5.0
    rgb_quality: int = 85  # JPEG-like quality 1-100 for 8-bit/RGB data
    lossless: bool = False  # Use lossless PNG compression for uint8 data
    uint8_accuracy_mode: bool = False  # Use binary search for uint8 (accuracy-controlled, cm=5)
    uint8_accuracy: int = (
        2  # Max pixel error for uint8 accuracy mode (integer pixels, min=2 due to YCbCr rounding)
    )
    uint8_scaling_factor: Any = (
        1  # Scaling factor or list of scaling factors for uint8 DCT coefficients (lower = smaller file, 1 = optimal for pixel accuracy)
    )
    ycbcr_y_multiplier: float = 0.9  # YCbCr Y channel accuracy multiplier (1.0 = use full accuracy)
    ycbcr_cb_multiplier: float = (
        0.9  # YCbCr Cb channel accuracy multiplier (1.0 = use full accuracy)
    )
    ycbcr_cr_multiplier: float = (
        0.9  # YCbCr Cr channel accuracy multiplier (1.0 = use full accuracy)
    )
    ycbcr_fallback_multipliers: list = field(
        default_factory=lambda: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    )
    ycbcr_per_block_mode: bool = True
    allowed_block_sizes: list = field(
        default_factory=lambda: [8, 16, 32]
    )  # Allowed N values for auto-select
    rgb_channel_indices: list = field(
        default_factory=list
    )  # [1,2,3] = R,G,B channels (1-based); empty = disabled
    rgb_strip_height_px: int = (
        2048  # Target RGB strip height in pixels (used as an upper bound, memory-safe capped)
    )


@dataclass
class PerformanceConfig:
    """Performance parameters."""

    num_processes: str = "auto"
    fast_eval_basis: bool = True
    verify_with_full_idct: bool = True
    l2_precheck_enabled: bool = True
    incremental_backscan: bool = True


@dataclass
class ModelConfig:
    """Model / search parameters."""

    backend: str = "heuristic"  # heuristic, binary
    advanced_heuristic: bool = False  # True = lookup table (16 features)
    minimize_backscan: int = 10
    backscan_break_after: int = 3
    accept_prediction_if_within_accuracy: bool = (
        False  # Accept trained L prediction for uint8 blocks without binary search
    )
    uint8_use_L_prediction: bool = (
        True  # Use per-block trained L lookup for uint8 (ac_abs_mean, zero_ratio) → start_L  # pylint: disable=invalid-name
    )
    uint8_L_prediction_scales: list = field(  # pylint: disable=invalid-name
        default_factory=lambda: [0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
    )


@dataclass
class OutputConfig:
    """Output parameters."""

    quiet: bool = False
    delete_temp_files: bool = True
    compression_method: str = "LZMA2"  # LZMA2, PPMD, BZIP2, DEFLATE
    tiff_compression: str = "DEFLATE"  # DEFLATE, LZW, LZMA, ZSTD, NONE


@dataclass
class Config:
    """Root configuration object."""

    directories: DirectoriesConfig = field(default_factory=DirectoriesConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    _scaling_factor_override: Optional[int] = None

    @property
    def results_dir(self) -> str:
        """Results directory path."""
        return self.directories.results

    @property
    def source_dir(self) -> str:
        """Source directory path."""
        return self.directories.source

    @property
    def models_dir(self) -> str:
        """Models directory path."""
        return self.directories.models

    @property
    def cache_dir(self) -> str:
        """Cache directory path."""
        return self.directories.cache

    @property
    def scaling_factor(self) -> int:
        """Scaling factor derived from decimal_places (10^n)."""
        if getattr(self, "_scaling_factor_override", None) is not None:
            return self._scaling_factor_override
        if self.compression.decimal_places == 0:
            return 1
        return 10**self.compression.decimal_places

    @property
    def num_processes_int(self) -> int:
        """Number of processes as int (resolves 'auto' to cpu_count)."""
        import multiprocessing  # pylint: disable=import-outside-toplevel

        if self.performance.num_processes == "auto":
            return multiprocessing.cpu_count()
        return int(self.performance.num_processes)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load config from JSON file; falls back to defaults if not found."""
    if config_path is None:
        possible_paths = [
            "./config.json",
            str(Path(__file__).parent / "config.json"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break

    if config_path is None or not os.path.exists(config_path):
        return Config()

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return _parse_config(data)


def _parse_config(data: Dict[str, Any]) -> Config:
    """Parse dict into Config object."""
    dirs = data.get("directories", {})
    directories = DirectoriesConfig(
        results=dirs.get("results", "./results/"),
        source=dirs.get("source", "./source/"),
        models=dirs.get("models", "./models/"),
        cache=dirs.get("cache", "../cache_train_v2/"),
    )

    comp = data.get("compression", {})
    compression = CompressionConfig(
        accuracy=comp.get("accuracy", 0.05),
        block_size=comp.get("block_size", 16),
        decimal_places=comp.get("decimal_places", 2),
        dct_type=comp.get("dct_type", 2),
        crs=comp.get("crs", "epsg:2180"),
        auto_select_block_size=comp.get("auto_select_block_size", False),
        auto_select_sample_size=comp.get("auto_select_sample_size", 1000),
        auto_select_std_threshold_high=comp.get("auto_select_std_threshold_high", 10.0),
        auto_select_std_threshold_medium=comp.get("auto_select_std_threshold_medium", 5.0),
        rgb_quality=comp.get("rgb_quality", 85),
        lossless=comp.get("lossless", False),
        uint8_accuracy_mode=comp.get("uint8_accuracy_mode", False),
        uint8_accuracy=comp.get("uint8_accuracy", 2),
        uint8_scaling_factor=comp.get("uint8_scaling_factor", 1),
        ycbcr_fallback_multipliers=comp.get(
            "ycbcr_fallback_multipliers", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        ),
        ycbcr_per_block_mode=comp.get("ycbcr_per_block_mode", True),
    )
    fallback_mults = compression.ycbcr_fallback_multipliers
    default_initial = fallback_mults[0] if fallback_mults else 0.9

    compression.ycbcr_y_multiplier = comp.get("ycbcr_y_multiplier", default_initial)
    compression.ycbcr_cb_multiplier = comp.get("ycbcr_cb_multiplier", default_initial)
    compression.ycbcr_cr_multiplier = comp.get("ycbcr_cr_multiplier", default_initial)
    compression.allowed_block_sizes = comp.get("allowed_block_sizes", [8, 16, 32])
    compression.rgb_channel_indices = comp.get("rgb_channel_indices", [])
    rgb_strip_height_px = comp.get(
        "rgb_strip_height_px", comp.get("rgb_strip_memory_fraction", 2048)
    )
    try:
        compression.rgb_strip_height_px = int(rgb_strip_height_px)
    except (TypeError, ValueError):
        compression.rgb_strip_height_px = 2048
    if compression.rgb_strip_height_px < 8:
        compression.rgb_strip_height_px = 2048

    perf = data.get("performance", {})
    parallel = perf.get("parallel", {})
    optimizations = perf.get("optimizations", {})

    performance = PerformanceConfig(
        num_processes=str(parallel.get("num_processes", "auto")),
        fast_eval_basis=optimizations.get("fast_eval_basis", True),
        verify_with_full_idct=optimizations.get("verify_with_full_idct", True),
        l2_precheck_enabled=optimizations.get("l2_precheck_enabled", True),
        incremental_backscan=optimizations.get("incremental_backscan", True),
    )

    mod = data.get("model", {})
    l_scales = mod.get("uint8_L_prediction_scales")
    if l_scales is None:
        l_scales = [0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
    elif not isinstance(l_scales, list):
        l_scales = [float(l_scales)]

    model = ModelConfig(
        backend=mod.get("backend", "heuristic"),
        advanced_heuristic=mod.get("advanced_heuristic", False),
        minimize_backscan=mod.get("minimize_backscan", 10),
        backscan_break_after=mod.get("backscan_break_after", 3),
        accept_prediction_if_within_accuracy=mod.get(
            "accept_prediction_if_within_accuracy", mod.get("accept_ai_if_within_accuracy", False)
        ),
        uint8_use_L_prediction=mod.get("uint8_use_L_prediction", True),
        uint8_L_prediction_scales=[float(x) for x in l_scales],
    )

    out = data.get("output", {})
    output = OutputConfig(
        quiet=out.get("quiet", False),
        delete_temp_files=out.get("delete_temp_files", True),
        compression_method=out.get("compression_method", "LZMA2"),
        tiff_compression=out.get("tiff_compression", "DEFLATE"),
    )

    return Config(
        directories=directories,
        compression=compression,
        performance=performance,
        model=model,
        output=output,
    )


def save_config(config: Config, config_path: str) -> None:
    """Save config to JSON file."""
    data = {
        "directories": {
            "results": config.results_dir,
            "source": config.source_dir,
            "models": config.models_dir,
            "cache": config.cache_dir,
        },
        "compression": {
            "accuracy": config.compression.accuracy,
            "block_size": config.compression.block_size,
            "decimal_places": config.compression.decimal_places,
            "dct_type": config.compression.dct_type,
            "crs": config.compression.crs,
            "auto_select_block_size": config.compression.auto_select_block_size,
            "auto_select_sample_size": config.compression.auto_select_sample_size,
            "auto_select_std_threshold_high": config.compression.auto_select_std_threshold_high,
            "auto_select_std_threshold_medium": config.compression.auto_select_std_threshold_medium,
            "rgb_quality": config.compression.rgb_quality,
            "lossless": config.compression.lossless,
            "uint8_accuracy_mode": config.compression.uint8_accuracy_mode,
            "uint8_accuracy": config.compression.uint8_accuracy,
            "uint8_scaling_factor": config.compression.uint8_scaling_factor,
            "ycbcr_fallback_multipliers": (config.compression.ycbcr_fallback_multipliers),
            "ycbcr_per_block_mode": config.compression.ycbcr_per_block_mode,
            "allowed_block_sizes": config.compression.allowed_block_sizes,
            "rgb_strip_height_px": config.compression.rgb_strip_height_px,
        },
        "performance": {
            "parallel": {
                "num_processes": config.performance.num_processes,
            },
            "optimizations": {
                "fast_eval_basis": config.performance.fast_eval_basis,
                "verify_with_full_idct": config.performance.verify_with_full_idct,
                "l2_precheck_enabled": config.performance.l2_precheck_enabled,
                "incremental_backscan": config.performance.incremental_backscan,
            },
        },
        "model": {
            "backend": config.model.backend,
            "advanced_heuristic": config.model.advanced_heuristic,
            "minimize_backscan": config.model.minimize_backscan,
            "backscan_break_after": config.model.backscan_break_after,
            "accept_prediction_if_within_accuracy": (
                config.model.accept_prediction_if_within_accuracy
            ),
            "uint8_use_L_prediction": config.model.uint8_use_L_prediction,
            "uint8_L_prediction_scales": config.model.uint8_L_prediction_scales,
        },
        "output": {
            "quiet": config.output.quiet,
            "delete_temp_files": config.output.delete_temp_files,
            "compression_method": config.output.compression_method,
            "tiff_compression": config.output.tiff_compression,
        },
    }

    # Preserve _descriptions section if it exists in original file
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                original_data = json.load(f)
            if "_descriptions" in original_data:
                data["_descriptions"] = original_data["_descriptions"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
