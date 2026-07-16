"""
FIDVAC v2 - Block processing
"""

from typing import Tuple, List, Optional, Dict, Any, Union

import numpy as np

from core.dct import dct2, to_zigzag, from_zigzag, idct2, _get_zigzag_indices, _get_idct_basis
from engine.utils import encode_mask_rle, decode_mask_rle, interpolate_zeros
from engine.refine import refine_dct_array

_predictor_cache: Dict[tuple, Any] = {}  # per-process predictor cache
from config import Config, load_config
from predictor.predictor_uint8 import load_uint8_L_grid, predict_uint8_L_batch


def process_block(
    block_data: Tuple[int, np.ndarray, Optional[float]],
    config: Optional[Config] = None,
    predicted_L: Optional[int] = None,
    skip_queue: Optional[list] = None,
) -> Tuple[int, Union[List, np.ndarray], float]:
    """Process single image block and return (idx, compressed_data, max_error)."""
    if config is None:
        config = load_config()

    idx, original_matrix, nodata_value = block_data[:3]
    src_dtype = block_data[3] if len(block_data) > 3 else None
    is_uint8_acc = src_dtype in ("uint8", "int8", "uint16", "int16")
    n = original_matrix.shape[0]
    accuracy = config.compression.accuracy
    dct_type = config.compression.dct_type
    decimal = config.compression.decimal_places
    scaling_factor = config.scaling_factor

    if np.all(original_matrix == 0):
        return idx, 0, 0.0

    if nodata_value is not None and np.all(original_matrix == nodata_value):
        return idx, 1, 0.0

    matrix_for_compression = original_matrix.copy()
    has_special_values = False
    masks = []

    if nodata_value is not None:
        nodata_mask = original_matrix == nodata_value
        if np.any(nodata_mask):
            has_special_values = True
            nodata_rle = encode_mask_rle(nodata_mask)
            masks.append([1] + nodata_rle)  # 1 = typ NoData
            matrix_for_compression[nodata_mask] = 0

    zero_mask = original_matrix == 0
    if nodata_value is not None:
        zero_mask = zero_mask & ~(original_matrix == nodata_value)

    if np.any(zero_mask):
        has_special_values = True
        zero_rle = encode_mask_rle(zero_mask)
        masks.append([2] + zero_rle)  # 2 = typ zero

    if has_special_values:
        nonzero_mask = matrix_for_compression != 0
        if np.any(nonzero_mask):
            matrix_for_compression = interpolate_zeros(matrix_for_compression, nonzero_mask)

    # For uint8/uint16 accuracy mode: center data before DCT (values 0-255 -> -128..127, uint16 -> -32768..32767)
    if is_uint8_acc:
        bit_depth = 16 if src_dtype in ("uint16", "int16") else 8
        center_val = float(1 << (bit_depth - 1))
        matrix_for_compression = matrix_for_compression - center_val

    dct_matrix = dct2(matrix_for_compression, dct_type)
    dct_matrix = np.round(dct_matrix, decimal)
    org_dct_zigzag = to_zigzag(dct_matrix)

    compressed_array, max_error, _ = refine_dct_array(
        org_dct_zigzag,
        accuracy,
        matrix_for_compression,
        n,
        dct_type,
        scaling_factor,
        config,
        predicted_L,
        skip_queue=skip_queue,
        src_dtype=src_dtype if is_uint8_acc else None,
    )

    if has_special_values:
        result = masks + [compressed_array.tolist()]
    else:
        result = compressed_array.tolist()

    return idx, result, max_error


def process_block_batch(
    batch_data: Tuple[int, List], config: Optional[Config] = None
) -> List[Tuple[int, Any, float]]:
    """Process a batch of blocks with optional L prediction. Returns list of (idx, data, error)."""
    if config is None:
        config = load_config()

    _batch_idx, blocks = batch_data
    # Detect uint8 accuracy mode from 4th element of first block tuple
    _src_dtype = blocks[0][3] if blocks and len(blocks[0]) > 3 else None
    _is_uint8_acc = _src_dtype in ("uint8", "int8", "uint16", "int16")
    L_predictions = {}
    skip_queues = {}
    results = []

    decimal = config.compression.decimal_places

    dct_candidates = []  # clean blocks
    special_candidates = []  # blocks with nodata — need interpolation
    for i, block_tuple in enumerate(blocks):
        try:
            _idx, _mat, _nodata = block_tuple[:3]
        except (TypeError, ValueError):
            continue
        if np.all(_mat == 0):
            continue
        if _nodata is not None and np.all(_mat == _nodata):
            continue
        if _nodata is not None and np.any(_mat == _nodata):
            _mat_interp = _mat.copy()
            nodata_mask = _mat == _nodata
            zero_mask = (_mat == 0) & ~nodata_mask
            _mat_interp[nodata_mask] = 0
            nonzero = _mat_interp != 0
            if np.any(nonzero):
                _mat_interp = interpolate_zeros(_mat_interp, nonzero)
            special_candidates.append(
                (
                    i,
                    _idx,
                    _mat_interp - 128.0 if _is_uint8_acc else _mat_interp,
                    _mat,
                    _nodata,
                )
            )
            continue

        dct_candidates.append((i, _idx, _mat - 128.0 if _is_uint8_acc else _mat))

    precomp_map = {}
    special_map = {}  # _idx -> (original_mat, nodata_val)
    all_for_dct = list(dct_candidates)
    if special_candidates:
        for i, _idx, _mat_interp, _mat_orig, _nodata in special_candidates:
            all_for_dct.append((i, _idx, _mat_interp))
            special_map[_idx] = (_mat_orig, _nodata)

    precomp_vectors = None
    if all_for_dct:
        dct_batch = np.array([m for (_, _, m) in all_for_dct])
        dct_batch = dct2(dct_batch, config.compression.dct_type)
        dct_batch = np.round(dct_batch, decimal)

        B = dct_batch.shape[0]
        Nloc = dct_batch.shape[1]
        idx_z = _get_zigzag_indices(Nloc)
        all_vectors = dct_batch.reshape(B, -1)[:, idx_z]

        for k, (i, _idx, _m) in enumerate(all_for_dct):
            precomp_map[_idx] = all_vectors[k]

        precomp_vectors = all_vectors[: len(dct_candidates)]

    # DCT length prediction
    backend = config.model.backend.lower()
    use_predictor = backend in ("heuristic",) or config.model.advanced_heuristic

    if use_predictor:
        try:
            from predictor.predictor import get_predictor, AdvancedHeuristicPredictor
            from predictor.predictor_uint8 import Uint8Predictor

            _sf_override = getattr(config, "_scaling_factor_override", None)
            _sf_cfg = getattr(config.compression, "uint8_scaling_factor", 1)
            if _sf_override is not None:
                _effective_sf = int(_sf_override)
            elif isinstance(_sf_cfg, (list, tuple)):
                _effective_sf = int(_sf_cfg[0]) if _sf_cfg else 1
            else:
                _effective_sf = int(_sf_cfg)
            _uint8_l_scales = tuple(
                float(x) for x in getattr(config.model, "uint8_L_prediction_scales", [1.5])
            )

            _cache_key = (
                backend,
                config.model.advanced_heuristic,
                config.compression.accuracy,
                config.compression.block_size,
                _src_dtype,  # Include src_dtype for uint8 vs float distinction
                _effective_sf,
                _uint8_l_scales,
            )
            if _cache_key not in _predictor_cache:
                _predictor_cache[_cache_key] = get_predictor(
                    backend=backend,
                    advanced_heuristic=config.model.advanced_heuristic,
                    accuracy=config.compression.accuracy,
                    block_size=config.compression.block_size,
                    config=config,
                    src_dtype=_src_dtype,  # Pass src_dtype for uint8 predictor selection
                )
            predictor = _predictor_cache[_cache_key]

            # Use predict_from_dct_batch for AdvancedHeuristicPredictor and Uint8Predictor
            if (
                isinstance(predictor, (AdvancedHeuristicPredictor, Uint8Predictor))
                and dct_candidates
            ):
                indices_for_pred = [_idx for (_, _idx, _) in dct_candidates]
                preds, queues = predictor.predict_from_dct_batch(precomp_vectors)
                for k, _idx in enumerate(indices_for_pred):
                    if k < len(preds):
                        L_predictions[_idx] = int(preds[k])
                        if queues and k < len(queues) and queues[k]:
                            skip_queues[_idx] = queues[k]
            else:
                # Fallback to predict_dct_length_batch for HeuristicPredictor
                eligible_blocks = [_mat for (_, _, _mat) in dct_candidates]
                eligible_indices = [_idx for (_, _idx, _) in dct_candidates]

                if eligible_blocks:
                    res = predictor.predict_dct_length_batch(eligible_blocks)
                    if isinstance(res, tuple):
                        preds, queues = res
                    else:
                        preds, queues = res, [[]] * len(res)
                    for k, idx in enumerate(eligible_indices):
                        if k < len(preds):
                            L_predictions[idx] = int(preds[k])
                            if queues and k < len(queues) and queues[k]:
                                skip_queues[idx] = queues[k]
        except (ImportError, RuntimeError):
            pass

    # Per-block L prediction for uint8 accuracy mode (analogous to float32 L prediction).
    # Uses trained lookup grid (ac_abs_mean, zero_ratio) → median optimal L from feature data.
    # Disabled when config.model.uint8_use_L_prediction=False (for benchmarking / fallback).
    if (
        _is_uint8_acc
        and config.model.uint8_use_L_prediction
        and precomp_vectors is not None
        and len(dct_candidates) > 0
    ):
        sf_override = getattr(config, "_scaling_factor_override", None)
        sf_cfg = getattr(config.compression, "uint8_scaling_factor", 1)
        if sf_override is not None:
            sf = int(sf_override)
        elif isinstance(sf_cfg, (list, tuple)):
            sf = int(sf_cfg[0]) if sf_cfg else 1
        else:
            sf = int(sf_cfg)
        cache_key = ("uint8_L_grid", Nloc, sf)
        if cache_key not in _predictor_cache:
            models_dir = config.models_dir if hasattr(config, 'models_dir') else "models"
            _predictor_cache[cache_key] = load_uint8_L_grid(Nloc, sf, models_dir)
        l_grid = _predictor_cache[cache_key]

        u8_indices = [_idx for (_, _idx, _) in dct_candidates]
        n_u8 = min(len(u8_indices), len(precomp_vectors))

        if n_u8 > 0:
            ac_abs_means = np.empty(n_u8, dtype=np.float32)
            zero_ratios = np.empty(n_u8, dtype=np.float32)
            for k in range(n_u8):
                zigzag = precomp_vectors[k]
                ac = zigzag[1:]  # skip DC
                ac_abs_means[k] = float(np.mean(np.abs(ac)))
                zero_ratios[k] = float(np.sum(ac == 0)) / max(1, len(ac))

            default_L = max(1, (Nloc * Nloc) // 2)
            L_preds = predict_uint8_L_batch(ac_abs_means, zero_ratios, l_grid, default_L)

            for k, _idx in enumerate(u8_indices[:n_u8]):
                if _idx not in L_predictions:
                    L_predictions[_idx] = int(L_preds[k])

    for block_tuple in blocks:
        try:
            idx = block_tuple[0]
            L_pred = L_predictions.get(idx)

            # Use precomputed zigzag if available
            if idx in precomp_map:
                _idx, _mat, _nodata = block_tuple[:3]

                if np.all(_mat == 0):
                    results.append((_idx, 0, 0.0))
                    continue
                if _nodata is not None and np.all(_mat == _nodata):
                    results.append((_idx, 1, 0.0))
                    continue

                org_dct_zigzag = precomp_map[idx]

                if idx in special_map:
                    _mat_orig, _nodata_val = special_map[idx]
                    nodata_mask = _mat_orig == _nodata_val
                    zero_mask = (_mat_orig == 0) & ~nodata_mask
                    masks = []
                    if np.any(nodata_mask):
                        masks.append([1] + encode_mask_rle(nodata_mask))
                    if np.any(zero_mask):
                        masks.append([2] + encode_mask_rle(zero_mask))
                    _mat_for_refine = _mat_orig.copy()
                    _mat_for_refine[nodata_mask] = 0
                    nonzero = _mat_for_refine != 0
                    if np.any(nonzero):
                        _mat_for_refine = interpolate_zeros(_mat_for_refine, nonzero)
                    if _is_uint8_acc:
                        _mat_for_refine = _mat_for_refine - 128.0
                    compressed_array, max_error, _ = refine_dct_array(
                        org_dct_zigzag,
                        config.compression.accuracy,
                        _mat_for_refine,
                        _mat_for_refine.shape[0],
                        config.compression.dct_type,
                        config.scaling_factor,
                        config,
                        L_pred,
                        skip_queue=skip_queues.get(idx),
                        src_dtype=_src_dtype,
                    )
                    result = masks + [compressed_array.tolist()]
                    results.append((_idx, result, max_error))
                else:
                    _mat_r = _mat - 128.0 if _is_uint8_acc else _mat
                    compressed_array, max_error, _ = refine_dct_array(
                        org_dct_zigzag,
                        config.compression.accuracy,
                        _mat_r,
                        _mat_r.shape[0],
                        config.compression.dct_type,
                        config.scaling_factor,
                        config,
                        L_pred,
                        skip_queue=skip_queues.get(idx),
                        src_dtype=_src_dtype,
                    )
                    results.append((_idx, compressed_array, max_error))
            else:
                skip_q = skip_queues.get(idx)
                result = process_block(block_tuple, config, L_pred, skip_queue=skip_q)
                results.append(result)
        except (TypeError, ValueError) as e:
            idx = block_tuple[0] if isinstance(block_tuple, tuple) else 0
            results.append((idx, {"type": "error", "msg": str(e)}, 1.0))

    return results


def reconstruct_block(
    compressed_data: Union[List, np.ndarray],
    n: int,
    scaling_factor: int,
    nodata_value: float = -9999,
    uint8_mode: bool = False,
) -> np.ndarray:
    """Reconstruct block from compressed DCT data.

    Coefficients are original DCT * scaling_factor.
    uint8_mode: True for cm=5 (uint8 accuracy path) - decenter +128, clip, round.
    """
    if compressed_data == 0:
        return np.zeros((n, n), dtype=np.float32)
    if compressed_data == 1:
        return np.full((n, n), nodata_value, dtype=np.float32)

    has_masks = False
    masks = []
    dct_coeffs = compressed_data

    if isinstance(compressed_data, list) and len(compressed_data) > 1:
        if isinstance(compressed_data[0], list):
            has_masks = True
            dct_coeffs = compressed_data[-1]
            masks = compressed_data[:-1]

    total_coeffs = n * n
    dct_zigzag = np.zeros(total_coeffs, dtype=np.float64)

    if isinstance(dct_coeffs, (list, np.ndarray)):
        L = min(len(dct_coeffs), total_coeffs)
        dct_zigzag[:L] = np.asarray(dct_coeffs[:L], dtype=np.float64) / scaling_factor

    dct_matrix = from_zigzag(dct_zigzag, n)
    reconstructed = idct2(dct_matrix).astype(np.float64)

    # For uint8 accuracy mode: decenter (+128), clip to [0,255], round to integer
    if uint8_mode:
        reconstructed = np.clip(np.round(reconstructed + 128.0), 0, 255).astype(np.float32)

    if has_masks:
        for mask_data in masks:
            if isinstance(mask_data, list) and len(mask_data) > 1:
                mask_type = mask_data[0]
                mask_rle = mask_data[1:]

                if mask_type == 1:  # NoData
                    nodata_mask = decode_mask_rle(mask_rle, (n, n))
                    reconstructed[nodata_mask] = nodata_value
                elif mask_type == 2:  # Zero
                    zero_mask = decode_mask_rle(mask_rle, (n, n))
                    reconstructed[zero_mask] = 0

    return reconstructed


__all__ = [
    "process_block",
    "process_block_batch",
    "process_block_batch_rgb",
    "process_strip_batch",
    "process_strip_batch_rgb",
    "reconstruct_block",
    "rgb_to_ycbcr",
    "ycbcr_to_rgb",
    "process_block_rgb",
]


# ---------------------------------------------------------------------------
# RGB <-> YCbCr conversion helpers (per-block)
# ---------------------------------------------------------------------------


def rgb_to_ycbcr(
    R: np.ndarray, G: np.ndarray, B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RGB block to YCbCr.

    Standard JPEG conversion:
    Y  = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.169*R - 0.331*G + 0.500*B + 128
    Cr = 0.500*R - 0.419*G - 0.081*B + 128
    """
    Y = np.clip(0.299 * R + 0.587 * G + 0.114 * B, 0, 255)
    Cb = np.clip(-0.169 * R - 0.331 * G + 0.500 * B + 128, 0, 255)
    Cr = np.clip(0.500 * R - 0.419 * G - 0.081 * B + 128, 0, 255)
    return Y, Cb, Cr


def ycbcr_to_rgb(
    Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert YCbCr block back to RGB.

    Inverse JPEG conversion:
    R = Y + 1.403*(Cr - 128)
    G = Y - 0.344*(Cb - 128) - 0.714*(Cr - 128)
    B = Y + 1.773*(Cb - 128)
    """
    R = np.clip(np.round(Y + 1.403 * (Cr - 128.0)), 0, 255)
    G = np.clip(np.round(Y - 0.344 * (Cb - 128.0) - 0.714 * (Cr - 128.0)), 0, 255)
    B = np.clip(np.round(Y + 1.773 * (Cb - 128.0)), 0, 255)
    return R, G, B


# ---------------------------------------------------------------------------
# Per-block RGB processing with YCbCr validity check
# ---------------------------------------------------------------------------


def process_block_rgb(
    block_data: Tuple[int, np.ndarray, np.ndarray, np.ndarray, Optional[float]],
    config: Config,
    multiplier: float,
    rgb_accuracy: float,
) -> Tuple[int, Dict[str, Any], float, bool]:
    """Process RGB block with single YCbCr multiplier.

    Args:
        block_data: (idx, R, G, B, nodata_value)
        config: Configuration object
        multiplier: Single multiplier to use (e.g., 0.9)
        rgb_accuracy: Target max error in RGB space (e.g., 5)

    Returns:
        (idx, block_metadata, max_error_rgb, success)

    block_metadata:
        {
            'Y': {'L': L_y, 'coeffs': coeffs_y, 'multiplier': mult_y},
            'Cb': {'L': L_cb, 'coeffs': coeffs_cb, 'multiplier': mult_cb},
            'Cr': {'L': L_cr, 'coeffs': coeffs_cr, 'multiplier': mult_cr},
            'sf': global_sf,
            'masks': masks (if any)
        }
    """
    idx, R, G, B, nodata_value = block_data
    n = R.shape[0]
    dct_type = config.compression.dct_type
    decimal = config.compression.decimal_places
    scaling_factor = config.scaling_factor

    # Check for trivial blocks
    if np.all(R == 0) and np.all(G == 0) and np.all(B == 0):
        return (
            idx,
            {
                "Y": {"L": 0, "coeffs": [], "multiplier": 1.0},
                "Cb": {"L": 0, "coeffs": [], "multiplier": 1.0},
                "Cr": {"L": 0, "coeffs": [], "multiplier": 1.0},
                "sf": scaling_factor,
                "masks": [],
            },
            0.0,
        )

    if (
        nodata_value is not None
        and np.all(R == nodata_value)
        and np.all(G == nodata_value)
        and np.all(B == nodata_value)
    ):
        return (
            idx,
            {
                "Y": {"L": -1, "coeffs": [], "multiplier": 1.0},
                "Cb": {"L": -1, "coeffs": [], "multiplier": 1.0},
                "Cr": {"L": -1, "coeffs": [], "multiplier": 1.0},
                "sf": scaling_factor,
                "masks": [],
            },
            0.0,
        )

    # Handle special values (nodata, zeros) - same logic as process_block
    masks = []

    for ch_name, ch_data in [("R", R), ("G", G), ("B", B)]:
        if nodata_value is not None:
            nodata_mask = ch_data == nodata_value
            if np.any(nodata_mask):
                nodata_rle = encode_mask_rle(nodata_mask)
                masks.append([1, ch_name] + nodata_rle)  # 1 = NoData, channel name

        zero_mask = ch_data == 0
        if nodata_value is not None:
            zero_mask = zero_mask & ~(ch_data == nodata_value)

        if np.any(zero_mask):
            zero_rle = encode_mask_rle(zero_mask)
            masks.append([2, ch_name] + zero_rle)  # 2 = Zero, channel name

    # Try single multiplier
    # Budget accuracy for YCbCr channels using individual multipliers from config
    acc_y = rgb_accuracy * multiplier * getattr(config.compression, "ycbcr_y_multiplier", 0.5)
    acc_cb = rgb_accuracy * multiplier * getattr(config.compression, "ycbcr_cb_multiplier", 2.0)
    acc_cr = rgb_accuracy * multiplier * getattr(config.compression, "ycbcr_cr_multiplier", 1.5)

    # RGB -> YCbCr
    Y, Cb, Cr = rgb_to_ycbcr(R, G, B)

    # Center data for DCT (0-255 -> -128..127)
    Y_centered = Y - 128.0
    Cb_centered = Cb - 128.0
    Cr_centered = Cr - 128.0

    # DCT on each channel
    dct_y = dct2(Y_centered, dct_type)
    dct_cb = dct2(Cb_centered, dct_type)
    dct_cr = dct2(Cr_centered, dct_type)

    dct_y = np.round(dct_y, decimal)
    dct_cb = np.round(dct_cb, decimal)
    dct_cr = np.round(dct_cr, decimal)

    # Zigzag
    zigzag_y = to_zigzag(dct_y)
    zigzag_cb = to_zigzag(dct_cb)
    zigzag_cr = to_zigzag(dct_cr)

    # Binary search for each channel with respective accuracy
    # Note: refine_dct_array expects original_matrix (centered)
    L_y, coeffs_y, _ = _refine_single_channel(
        zigzag_y, Y_centered, acc_y, n, dct_type, scaling_factor, config
    )
    L_cb, coeffs_cb, _ = _refine_single_channel(
        zigzag_cb, Cb_centered, acc_cb, n, dct_type, scaling_factor, config
    )
    L_cr, coeffs_cr, _ = _refine_single_channel(
        zigzag_cr, Cr_centered, acc_cr, n, dct_type, scaling_factor, config
    )

    # IDCT reconstruction
    # Coeffs are scaled by scaling_factor, need to unscale before IDCT
    coeffs_y_unscaled = (
        np.asarray(coeffs_y, dtype=np.float64) / scaling_factor if L_y > 0 else np.array([])
    )
    coeffs_cb_unscaled = (
        np.asarray(coeffs_cb, dtype=np.float64) / scaling_factor if L_cb > 0 else np.array([])
    )
    coeffs_cr_unscaled = (
        np.asarray(coeffs_cr, dtype=np.float64) / scaling_factor if L_cr > 0 else np.array([])
    )

    Y_rec = _idct_reconstruct(coeffs_y_unscaled, L_y, n) + 128.0
    Cb_rec = _idct_reconstruct(coeffs_cb_unscaled, L_cb, n) + 128.0
    Cr_rec = _idct_reconstruct(coeffs_cr_unscaled, L_cr, n) + 128.0

    # Inverse YCbCr -> RGB
    R_rec, G_rec, B_rec = ycbcr_to_rgb(Y_rec, Cb_rec, Cr_rec)

    # Round reconstructed RGB to match original uint8 format
    R_rec = np.round(R_rec)
    G_rec = np.round(G_rec)
    B_rec = np.round(B_rec)

    # Check RGB validity
    max_err_rgb = max(np.abs(R - R_rec).max(), np.abs(G - G_rec).max(), np.abs(B - B_rec).max())

    if max_err_rgb <= rgb_accuracy:
        # Success - return with this multiplier
        return (
            idx,
            {
                "Y": {
                    "L": L_y,
                    "coeffs": (coeffs_y.tolist() if isinstance(coeffs_y, np.ndarray) else coeffs_y),
                    "multiplier": multiplier,
                },
                "Cb": {
                    "L": L_cb,
                    "coeffs": (
                        coeffs_cb.tolist() if isinstance(coeffs_cb, np.ndarray) else coeffs_cb
                    ),
                    "multiplier": multiplier,
                },
                "Cr": {
                    "L": L_cr,
                    "coeffs": (
                        coeffs_cr.tolist() if isinstance(coeffs_cr, np.ndarray) else coeffs_cr
                    ),
                    "multiplier": multiplier,
                },
                "sf": scaling_factor,
                "masks": masks,
            },
            max_err_rgb,
            True,
        )

    # Failed - return None metadata with failure flag
    return idx, None, max_err_rgb, False


def _refine_single_channel(
    zigzag_coeffs: np.ndarray,
    original_matrix: np.ndarray,
    accuracy: float,
    n: int,
    dct_type: int,
    scaling_factor: int,
    config: Config,
) -> Tuple[int, np.ndarray, float]:
    """Helper: run refine_dct_array for a single channel."""
    compressed_array, max_error, _ = refine_dct_array(
        zigzag_coeffs,
        accuracy,
        original_matrix,
        n,
        dct_type,
        scaling_factor,
        config,
        None,  # predicted_L
        None,  # skip_queue
        src_dtype="uint8",  # Force uint8 mode for centering
    )
    L = len(compressed_array) if isinstance(compressed_array, (list, np.ndarray)) else 0
    return L, compressed_array, max_error


def _idct_reconstruct(coeffs: np.ndarray, L: int, n: int) -> np.ndarray:
    """Helper: reconstruct from DCT coefficients."""
    total_coeffs = n * n
    dct_zigzag = np.zeros(total_coeffs, dtype=np.float64)
    if L > 0:
        dct_zigzag[:L] = np.asarray(coeffs[:L], dtype=np.float64)
    dct_matrix = from_zigzag(dct_zigzag, n)
    return idct2(dct_matrix).astype(np.float64)


def process_strip_batch(
    strip_data: Tuple[np.ndarray, int, int, Optional[float]],
    config: Config,
) -> List[Tuple[int, Any, float]]:
    """Compress an entire strip using binary search path (for elevation data).

    Same interface as process_strip_quantize but uses the standard refine path.
    """
    strip_array, y_start, blocks_per_row, nodata_value = strip_data
    n = config.compression.block_size

    sH, sW = strip_array.shape
    rows_of_blocks = sH // n
    cols_of_blocks = sW // n

    blocks_3d = (
        strip_array.reshape(rows_of_blocks, n, cols_of_blocks, n)
        .transpose(0, 2, 1, 3)
        .reshape(-1, n, n)
    )

    global_row_offset = y_start // n

    # Build block tuples for process_block_batch
    blocks = []
    for local_idx in range(blocks_3d.shape[0]):
        local_row = local_idx // cols_of_blocks
        local_col = local_idx % cols_of_blocks
        global_idx = (global_row_offset + local_row) * blocks_per_row + local_col
        blocks.append((global_idx, blocks_3d[local_idx], nodata_value))

    # Delegate to existing batch function
    return process_block_batch((0, blocks), config)


def process_strip_batch_rgb(
    strip_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, Optional[float]],
    config: Config,
) -> List[Tuple[int, Any, float]]:
    """Compress a strip of RGB blocks using per-block YCbCr (cm=6).

    strip_data: (r_strip, g_strip, b_strip, Y_strip, Cb_strip, Cr_strip,
                 y_start, blocks_per_row, nodata_value)

    Extracts n×n blocks from each strip, builds RGB block tuples,
    and delegates to process_block_batch_rgb.
    """
    r_strip, g_strip, b_strip, Y_strip, Cb_strip, Cr_strip, y_start, blocks_per_row, nodata_value = strip_data
    n = config.compression.block_size

    sH, sW = r_strip.shape
    rows_of_blocks = sH // n
    cols_of_blocks = sW // n

    def _blocks(ch: np.ndarray) -> np.ndarray:
        return ch.reshape(rows_of_blocks, n, cols_of_blocks, n).transpose(0, 2, 1, 3).reshape(-1, n, n)

    r_blk = _blocks(r_strip)
    g_blk = _blocks(g_strip)
    b_blk = _blocks(b_strip)
    Y_blk = _blocks(Y_strip)
    Cb_blk = _blocks(Cb_strip)
    Cr_blk = _blocks(Cr_strip)

    global_row_offset = y_start // n

    blocks = []
    for local_idx in range(r_blk.shape[0]):
        local_row = local_idx // cols_of_blocks
        local_col = local_idx % cols_of_blocks
        global_idx = (global_row_offset + local_row) * blocks_per_row + local_col
        blocks.append((
            global_idx,
            Y_blk[local_idx], Cb_blk[local_idx], Cr_blk[local_idx],
            r_blk[local_idx], g_blk[local_idx], b_blk[local_idx],
            nodata_value,
        ))

    return process_block_batch_rgb((0, blocks), config)


# ---------------------------------------------------------------------------
# Per-block RGB / YCbCr compression (cm=6)
# ---------------------------------------------------------------------------


def process_block_batch_rgb(
    batch_data: Tuple[int, List],
    config: Optional[Config] = None,
) -> List[Tuple[int, Any, float]]:
    """Process RGB block triplets with per-block YCbCr multiplier selection (cm=6).

    Each input block: (idx, Y, Cb, Cr, r, g, b, nodata)
    Returns: [(idx, metadata, max_rgb_error), ...]

    For each block, we predict the starting multiplier from chroma amplitude and
    iterate DOWN only when needed.  Typical case: 1 attempt per block.

    metadata format (compatible with serialize_compressed_blocks_rgb):
        {
          'Y':  {'L': int, 'coeffs': np.ndarray, 'multiplier': float},
          'Cb': {'L': int, 'coeffs': np.ndarray, 'multiplier': float},
          'Cr': {'L': int, 'coeffs': np.ndarray, 'multiplier': float},
          'sf': int,
          'fallback': bool,   # True if start_idx prediction was not sufficient
          'masks': None,
        }
    """
    if config is None:
        config = load_config()

    _batch_idx, blocks = batch_data

    fallback_multipliers: list = list(
        getattr(
            config.compression,
            "ycbcr_fallback_multipliers",
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2],
        )
    )
    base_accuracy = float(getattr(config.compression, "uint8_accuracy", 5))
    sf = config.scaling_factor
    dct_tp = config.compression.dct_type
    dec = config.compression.decimal_places
    n_mult = len(fallback_multipliers)

    # OPTIMIZATION: Skip batch DCT for prediction - compute L prediction per-block instead
    # This avoids redundant DCT computation since DCT is computed per-block anyway
    use_predictor = (
        config.model.backend.lower() in ("heuristic",)
        or config.model.advanced_heuristic
        or config.model.uint8_use_L_prediction
    )

    L_predictions = {}  # idx -> {'Y': L_y, 'Cb': L_cb, 'Cr': L_cr}
    predictor = None

    if use_predictor:
        try:
            # Load predictor once for all blocks (handles all block sizes)
            n = blocks[0][1].shape[1] if blocks else 16

            from predictor.predictor import get_predictor

            cache_key = (
                config.model.backend.lower(),
                config.model.advanced_heuristic,
                base_accuracy,
                n,
                "uint8",  # cm=6 is always uint8
                True,  # is_ycbcr=True
            )
            if cache_key not in _predictor_cache:
                _predictor_cache[cache_key] = get_predictor(
                    backend=config.model.backend.lower(),
                    advanced_heuristic=config.model.advanced_heuristic,
                    accuracy=base_accuracy,
                    block_size=n,
                    config=config,
                    src_dtype="uint8",
                    is_ycbcr=True,
                )
            predictor = _predictor_cache[cache_key]

        except (ImportError, AttributeError, RuntimeError) as e:
            if not config.output.quiet:
                print(f"  Warning: Predictor initialization failed for cm=6: {e}")

    results: List[Tuple[int, Any, float]] = []

    # =========================================================================
    # VECTORIZED PRE-COMPUTATION: batch DCT, zigzag, L prediction, start_idx
    # for ALL blocks before the per-block refinement loop.
    # Blocks may have different sizes (auto_select_block_size) → group by n.
    # =========================================================================
    n_blocks = len(blocks)
    if n_blocks == 0:
        return results

    # Group blocks by block size (auto_select_block_size may mix 8/16/32)
    from collections import defaultdict

    size_groups = defaultdict(list)  # n -> [(k, block_data), ...]
    for k, bd in enumerate(blocks):
        bn = bd[1].shape[0]
        size_groups[bn].append((k, bd))

    from scipy.fftpack import dct as scipy_dct

    # Pre-allocate results arrays (indexed by original position k)
    all_zigzag_Y = [None] * n_blocks
    all_zigzag_Cb = [None] * n_blocks
    all_zigzag_Cr = [None] * n_blocks
    all_centered_Y = [None] * n_blocks
    all_centered_Cb = [None] * n_blocks
    all_centered_Cr = [None] * n_blocks
    all_start_idx = [0] * n_blocks
    L_predictions = {}

    # Batch DCT + zigzag helper function (defined outside loop to avoid cell variable issue)
    def _batch_dct_zigzag(blocks_arr, zigzag_idx, gn):
        dct_batch = scipy_dct(scipy_dct(blocks_arr, axis=1, norm="ortho"), axis=2, norm="ortho")
        dct_batch = np.round(dct_batch, dec)
        return dct_batch.reshape(gn, -1)[:, zigzag_idx]

    for bn, group in size_groups.items():
        gn = len(group)
        zigzag_idx = _get_zigzag_indices(bn)

        # Stack channel blocks for this size group
        Y_all = np.stack([bd[1] for _, bd in group]).astype(np.float64) - 128.0
        Cb_all = np.stack([bd[2] for _, bd in group]).astype(np.float64) - 128.0
        Cr_all = np.stack([bd[3] for _, bd in group]).astype(np.float64) - 128.0

        # Batch DCT + zigzag
        zz_Y = _batch_dct_zigzag(Y_all, zigzag_idx, gn)
        zz_Cb = _batch_dct_zigzag(Cb_all, zigzag_idx, gn)
        zz_Cr = _batch_dct_zigzag(Cr_all, zigzag_idx, gn)

        # Batch L prediction
        if use_predictor and predictor is not None:
            chroma_L_ratio = None
            try:
                # Get predictor for this block size
                cache_key_bn = (
                    config.model.backend.lower(),
                    config.model.advanced_heuristic,
                    base_accuracy,
                    bn,
                    "uint8",
                )
                if cache_key_bn not in _predictor_cache:
                    from predictor.predictor import get_predictor

                    _predictor_cache[cache_key_bn] = get_predictor(
                        backend=config.model.backend.lower(),
                        advanced_heuristic=config.model.advanced_heuristic,
                        accuracy=base_accuracy,
                        block_size=bn,
                        config=config,
                        src_dtype="uint8",
                        is_ycbcr=True,
                    )
                predictor_bn = _predictor_cache[cache_key_bn]

                all_zz = np.concatenate([zz_Y, zz_Cb, zz_Cr], axis=0)
                all_preds, _ = predictor_bn.predict_from_dct_batch(all_zz)

                # Compute chroma difficulty ratio from predicted L
                pred_Cb_L = all_preds[gn : 2 * gn]
                pred_Cr_L = all_preds[2 * gn : 3 * gn]
                chroma_L_max = np.maximum(pred_Cb_L, pred_Cr_L)
                chroma_L_ratio = chroma_L_max / (bn * bn)

                for gi, (k, _) in enumerate(group):
                    L_predictions[blocks[k][0]] = {
                        "Y": int(all_preds[gi]),
                        "Cb": int(all_preds[gn + gi]),
                        "Cr": int(all_preds[2 * gn + gi]),
                    }
            except (ValueError, IndexError, RuntimeError):
                pass

        # Batch start_idx based on chroma amplitude
        Cb_ac = zz_Cb[:, 1:]
        Cr_ac = zz_Cr[:, 1:]
        chroma_amp = np.maximum(np.mean(np.abs(Cb_ac), axis=1), np.mean(np.abs(Cr_ac), axis=1))
        start_idx_bn = np.zeros(gn, dtype=np.int32)
        start_idx_bn[chroma_amp >= 3.0] = 1
        start_idx_bn[chroma_amp >= 8.0] = 2
        start_idx_bn[chroma_amp >= 20.0] = 3

        # Upgrade start_idx using Advanced Heuristic's predicted L (chroma difficulty ratio)
        if use_predictor and predictor is not None and chroma_L_ratio is not None:
            pred_start_idx = np.zeros(gn, dtype=np.int32)
            pred_start_idx[chroma_L_ratio >= 0.15] = 1
            pred_start_idx[chroma_L_ratio >= 0.30] = 2
            pred_start_idx[chroma_L_ratio >= 0.45] = 3
            pred_start_idx[chroma_L_ratio >= 0.60] = 4
            pred_start_idx[chroma_L_ratio >= 0.75] = 5
            start_idx_bn = np.maximum(start_idx_bn, pred_start_idx)

        start_idx_bn = np.minimum(start_idx_bn, n_mult - 1)

        # Store into per-block arrays
        for gi, (k, _) in enumerate(group):
            all_zigzag_Y[k] = zz_Y[gi]
            all_zigzag_Cb[k] = zz_Cb[gi]
            all_zigzag_Cr[k] = zz_Cr[gi]
            all_centered_Y[k] = Y_all[gi]
            all_centered_Cb[k] = Cb_all[gi]
            all_centered_Cr[k] = Cr_all[gi]
            all_start_idx[k] = int(start_idx_bn[gi])

    # --- Per-block refinement with analytical bound optimization ---
    # For each block, try multipliers from start_idx. When per-channel errors
    # satisfy the analytical bound (bound ≤ accuracy), skip the expensive
    # IDCT + YCbCr→RGB reconstruction and accept immediately.

    for k, block_data in enumerate(blocks):
        idx, _Y_b, _Cb_b, _Cr_b, r_b, g_b, b_b, _nodata = block_data
        n = _Y_b.shape[0]

        zigzag_blocks = {
            "Y": all_zigzag_Y[k],
            "Cb": all_zigzag_Cb[k],
            "Cr": all_zigzag_Cr[k],
        }
        centered_blocks = {
            "Y": all_centered_Y[k],
            "Cb": all_centered_Cb[k],
            "Cr": all_centered_Cr[k],
        }

        start_idx = all_start_idx[k]

        # Pre-convert original RGB to float32 once
        r_f = r_b.astype(np.float32)
        g_f = g_b.astype(np.float32)
        b_f = b_b.astype(np.float32)

        best_metadata: Optional[Dict] = None
        best_err = float("inf")
        best_mult_idx = n_mult - 1

        for mult_idx in range(start_idx, n_mult):
            mult = fallback_multipliers[mult_idx]
            ch_acc = mult * base_accuracy

            try:
                ch_data: Dict[str, Dict] = {}
                ch_errs: Dict[str, float] = {}

                for ch_name in ("Y", "Cb", "Cr"):
                    zigzag = zigzag_blocks[ch_name]
                    centered = centered_blocks[ch_name]
                    L_pred = (
                        L_predictions.get(idx, {}).get(ch_name) if idx in L_predictions else None
                    )

                    compressed_arr, ch_err, _ = refine_dct_array(
                        zigzag,
                        ch_acc,
                        centered,
                        n,
                        dct_tp,
                        sf,
                        config,
                        predicted_L=L_pred,
                        src_dtype="uint8",
                    )

                    L = len(compressed_arr)
                    ch_data[ch_name] = {
                        "L": L,
                        "coeffs": compressed_arr,
                        "multiplier": mult,
                    }
                    ch_errs[ch_name] = ch_err

                # Analytical bound on max RGB error from per-channel errors (for debugging only)
                eY = ch_errs["Y"]
                eCb = ch_errs["Cb"]
                eCr = ch_errs["Cr"]
                _bound = max(
                    eY + 1.402 * eCr,
                    eY + 0.344136 * eCb + 0.714136 * eCr,
                    eY + 1.772 * eCb,
                )

                # Always perform full reconstruction - bound is not accurate enough
                # Need full reconstruction to check actual error.
                # Use fast tensordot basis (same math as reconstruct_block,
                # but ~10x faster — avoids .tolist() + scipy.idct overhead).
                basis = _get_idct_basis(n, dct_tp)
                ch_recons: Dict[str, np.ndarray] = {}
                for ch_name in ("Y", "Cb", "Cr"):
                    L = ch_data[ch_name]["L"]
                    compressed_arr = ch_data[ch_name]["coeffs"]
                    if L > 0:
                        coeffs_dq = compressed_arr.astype(np.float64) / sf
                        recon = np.tensordot(coeffs_dq, basis[:L], axes=(0, 0))
                    else:
                        recon = np.zeros((n, n), dtype=np.float64)
                    ch_recons[ch_name] = np.clip(recon + 128.0, 0.0, 255.0)

                Y_r = ch_recons["Y"]
                Cb = ch_recons["Cb"]
                Cr = ch_recons["Cr"]

                R_rec, G_rec, B_rec = ycbcr_to_rgb(Y_r, Cb, Cr)

                max_rgb_err = float(
                    max(
                        np.max(np.abs(r_f - R_rec)),
                        np.max(np.abs(g_f - G_rec)),
                        np.max(np.abs(b_f - B_rec)),
                    )
                )

                if max_rgb_err < best_err:
                    best_err = max_rgb_err
                    best_metadata = ch_data
                    best_mult_idx = mult_idx

                if max_rgb_err <= base_accuracy:
                    break  # accepted — stop iterating

            except (ValueError, IndexError, RuntimeError):
                continue

        if best_metadata is None:
            mult = fallback_multipliers[-1]
            empty = np.array([], dtype=np.int64)
            best_metadata = {
                ch: {"L": 0, "coeffs": empty, "multiplier": mult} for ch in ("Y", "Cb", "Cr")
            }
            best_err = 0.0
            best_mult_idx = n_mult - 1

        metadata = {
            "Y": best_metadata["Y"],
            "Cb": best_metadata["Cb"],
            "Cr": best_metadata["Cr"],
            "sf": sf,
            "fallback": best_mult_idx > start_idx,
            "masks": None,
        }
        results.append((idx, metadata, best_err))

    return results
