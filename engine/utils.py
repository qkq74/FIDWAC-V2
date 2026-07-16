"""
FIDVAC v2 - Utility functions: RLE masks, zero interpolation, numpy conversion
"""

from typing import List, Tuple, Any

import numpy as np
from scipy.interpolate import griddata

# RLE ENCODING / DECODING FOR MASKS


def encode_mask_rle(mask: np.ndarray) -> List[int]:
    """Encode binary mask as RLE: [start1, length1, start2, length2, ...]."""
    flat = mask.flatten().astype(np.int8)
    # Detect transitions: +1 = start of True run, -1 = end of True run
    changes = np.diff(flat, prepend=np.int8(0), append=np.int8(0))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    lengths = ends - starts
    rle = np.empty(len(starts) * 2, dtype=np.int64)
    rle[0::2] = starts
    rle[1::2] = lengths
    return rle.tolist()


def decode_mask_rle(rle: List[int], shape: Tuple[int, int]) -> np.ndarray:
    """Decode RLE [start1, length1, ...] into a boolean mask of given shape."""
    flat_mask = np.zeros(shape[0] * shape[1], dtype=bool)

    if len(rle) >= 2:
        rle_arr = np.asarray(rle, dtype=np.int64)
        starts = rle_arr[0::2]
        lengths = rle_arr[1::2]
        indices = np.concatenate(
            [np.arange(s, s + l, dtype=np.int64) for s, l in zip(starts.tolist(), lengths.tolist())]
        )
        if len(indices) > 0:
            flat_mask[indices] = True

    return flat_mask.reshape(shape)


def interpolate_zeros(matrix: np.ndarray, nonzero_mask: np.ndarray) -> np.ndarray:
    """Fill zero pixels via nearest-neighbour interpolation (smooths DCT input)."""
    compression_copy = matrix.copy()

    if not np.any(nonzero_mask):
        return compression_copy

    y_indices, x_indices = np.indices(matrix.shape)
    points = np.column_stack((y_indices[nonzero_mask], x_indices[nonzero_mask]))
    values = matrix[nonzero_mask]

    zero_mask = ~nonzero_mask
    if not np.any(zero_mask) or len(points) == 0:
        return compression_copy

    grid_y = y_indices[zero_mask]
    grid_x = x_indices[zero_mask]
    grid_points = np.column_stack((grid_y, grid_x))

    try:
        interpolated = griddata(points, values, grid_points, method="nearest")
        compression_copy[zero_mask] = interpolated
    except Exception:
        mean_value = np.mean(values) if len(values) > 0 else 0
        compression_copy[zero_mask] = mean_value

    return compression_copy


def numpy_to_python(obj: Any) -> Any:
    """Recursively convert numpy types to plain Python (for JSON/msgpack)."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    return obj


__all__ = [
    "encode_mask_rle",
    "decode_mask_rle",
    "interpolate_zeros",
    "numpy_to_python",
]
