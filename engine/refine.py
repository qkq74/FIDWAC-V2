"""
FIDVAC v2 - DCT Refinement (Binary Search + L Prediction)
===========================================================

Optimized implementation with:
- Incremental binary search (tensordot instead of full IDCT)
- L2 precheck for fast rejection
- Backscan for coefficient minimization
"""

import os
import math
from typing import Tuple, Optional

import numpy as np
import numba as nb

from core.dct import from_zigzag, idct2, _get_idct_basis
from config import Config, load_config

# ---------------------------------------------------------------------------
# Numba JIT helpers for hot-path binary search (8×8 blocks)
# Eliminates numpy dispatch overhead (~2.7 µs/step → ~0.4 µs/step, 7x faster)
#
# NOTE: numba cache must be on a Linux filesystem (not Windows /mnt/g/).
# NUMBA_CACHE_DIR is set to /tmp/fidwac_numba_cache to avoid ~500ms
# delay when reading __pycache__ through WSL from Windows NTFS.
# ---------------------------------------------------------------------------
# NOTE: numba cache must be on a Linux filesystem (not Windows /mnt/g/).
# /dev/shm is tmpfs - pure RAM, fastest possible read.
# Fallback to /tmp if /dev/shm is unavailable.
_NB_CACHE_DIR = (
    "/dev/shm/fidwac_numba_cache" if os.path.isdir("/dev/shm") else "/tmp/fidwac_numba_cache"
)
os.makedirs(_NB_CACHE_DIR, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", _NB_CACHE_DIR)

# Debug counters for skip rate measurement
_skip_count = 0
_total_count = 0


def reset_skip_counters():
    """Reset debug counters for skip rate measurement."""
    global _skip_count, _total_count
    _skip_count = 0
    _total_count = 0


def get_skip_stats():
    """Return skip counters for debug statistics."""
    return _skip_count, _total_count


@nb.njit(cache=True, fastmath=True)
def _nb_step_add(diff: np.ndarray, basis_k: np.ndarray, ck: float) -> float:
    """diff += basis_k * ck  (removing coefficient from recon → diff grows)"""
    n = diff.shape[0]
    mx, mn = diff[0, 0], diff[0, 0]
    for i in range(n):
        for j in range(n):
            v = diff[i, j] + basis_k[i, j] * ck
            diff[i, j] = v
            if v > mx:
                mx = v
            if v < mn:
                mn = v
    return mx if mx > -mn else -mn


@nb.njit(cache=True, fastmath=True)
def _nb_step_sub(diff: np.ndarray, basis_k: np.ndarray, ck: float) -> float:
    """diff -= basis_k * ck  (adding coefficient to recon → diff shrinks)"""
    n = diff.shape[0]
    mx, mn = diff[0, 0], diff[0, 0]
    for i in range(n):
        for j in range(n):
            v = diff[i, j] - basis_k[i, j] * ck
            diff[i, j] = v
            if v > mx:
                mx = v
            if v < mn:
                mn = v
    return mx if mx > -mn else -mn


@nb.njit(cache=True, fastmath=True)
def _nb_jump_diff(
    diff: np.ndarray, basis: np.ndarray, coeffs: np.ndarray, from_L: int, to_L: int
) -> float:
    """Incrementally shift diff from position from_L to to_L (arbitrary jump)."""
    n = diff.shape[0]
    if from_L < to_L:
        for k in range(from_L, to_L):
            ck = coeffs[k]
            for i in range(n):
                for j in range(n):
                    diff[i, j] -= basis[k, i, j] * ck
    elif from_L > to_L:
        for k in range(to_L, from_L):
            ck = coeffs[k]
            for i in range(n):
                for j in range(n):
                    diff[i, j] += basis[k, i, j] * ck
    mx, mn = diff[0, 0], diff[0, 0]
    for i in range(n):
        for j in range(n):
            v = diff[i, j]
            if v > mx:
                mx = v
            if v < mn:
                mn = v
    return mx if mx > -mn else -mn


@nb.njit(cache=True, fastmath=True)
def _nb_max_err(diff: np.ndarray) -> float:
    n = diff.shape[0]
    mx, mn = diff[0, 0], diff[0, 0]
    for i in range(n):
        for j in range(n):
            v = diff[i, j]
            if v > mx:
                mx = v
            if v < mn:
                mn = v
    return mx if mx > -mn else -mn


# Trigger JIT compilation on import (8×8 dummy block) so the first real block doesn't wait
_d = np.zeros((8, 8), dtype=np.float64)
_b = np.zeros((64, 8, 8), dtype=np.float64)
_c = np.zeros(64, dtype=np.float64)
_nb_max_err(_d)
_nb_step_add(_d, _b[0], 0.0)
_nb_step_sub(_d, _b[0], 0.0)
_nb_jump_diff(_d, _b, _c, 0, 1)
del _d, _b, _c


def _eval_error_for_len(
    org_dct_zigzag: np.ndarray,
    original_matrix: np.ndarray,
    L: int,
    n: int,
    dct_type: int,
    accuracy: float,
    use_fast_basis: bool = True,
    use_l2_precheck: bool = True,
    src_dtype: Optional[str] = None,
) -> float:
    """
    Compute max reconstruction error for L coefficients.

    OPTIMIZATIONS:
    - L2 precheck: fast rejection without reconstruction
    - Fast basis: tensordot instead of full IDCT
    """
    L = int(max(1, min(L, len(org_dct_zigzag))))

    # L2 precheck - only for fast rejection (error definitely too large)
    if use_l2_precheck:
        tail = org_dct_zigzag[L:].astype(np.float32, copy=False)
        tail_l2 = float(np.sqrt(np.sum(tail * tail)))

        # Only reject when error is definitely too large
        if tail_l2 > accuracy * np.sqrt(n * n):
            return accuracy + 1.0  # Error definitely too large
        # Do NOT return accuracy when tail_l2 <= accuracy - compute actual error

    _is_uint8 = src_dtype in ("uint8", "int8", "uint16", "int16")

    # Fast path with precomputed basis
    if use_fast_basis:
        coeffs = org_dct_zigzag[:L].astype(np.float64, copy=False)
        basis = _get_idct_basis(n, dct_type)
        recon = np.tensordot(coeffs, basis[:L], axes=(0, 0))
        if _is_uint8:
            # original_matrix is centered [-128..127]; decenter, clip, round, measure integer error
            orig_u8 = original_matrix + 128.0
            recon_u8 = np.clip(np.round(recon + 128.0), 0, 255)
            diff = orig_u8 - recon_u8
        else:
            diff = original_matrix.astype(np.float64, copy=False) - recon
        return max(float(diff.max()), float(-diff.min()))

    # Classic path
    array = np.zeros(len(org_dct_zigzag), dtype=np.float64)
    array[:L] = org_dct_zigzag[:L]
    reconstructed = from_zigzag(array, n)
    idct_reconstructed = idct2(reconstructed, dct_type)
    if _is_uint8:
        orig_u8 = original_matrix + 128.0
        recon_u8 = np.clip(np.round(idct_reconstructed + 128.0), 0, 255)
        diff = orig_u8 - recon_u8
    else:
        diff = original_matrix - idct_reconstructed
    return max(float(diff.max()), float(-diff.min()))


def _binary_search_min_len_from_start(
    org_dct_zigzag: np.ndarray,
    original_matrix: np.ndarray,
    accuracy: float,
    start_L: int,
    lo: int,
    hi: int,
    n: int,
    dct_type: int,
    skip_queue: Optional[list] = None,
    src_dtype: Optional[str] = None,
) -> Tuple[int, int]:
    lo = max(1, int(lo))
    hi = min(len(org_dct_zigzag), int(hi))
    if lo > hi:
        return hi, 0
    steps, ans = 0, hi
    if skip_queue is not None and len(skip_queue) > 0:
        for L_test in skip_queue:
            if L_test < lo or L_test > hi or L_test == 0:
                continue
            steps += 1
            err = _eval_error_for_len(
                org_dct_zigzag,
                original_matrix,
                L_test,
                n,
                dct_type,
                accuracy,
                use_fast_basis=True,
                src_dtype=src_dtype,
            )
            if err <= accuracy:
                ans = L_test
                hi = L_test - 1
            else:
                lo = L_test + 1
            if lo > hi:
                return ans, steps
    elif lo <= start_L <= hi:
        steps += 1
        err = _eval_error_for_len(
            org_dct_zigzag,
            original_matrix,
            start_L,
            n,
            dct_type,
            accuracy,
            use_fast_basis=True,
            src_dtype=src_dtype,
        )
        if err <= accuracy:
            ans = start_L
            hi = start_L - 1
        else:
            lo = start_L + 1
    if lo > hi:
        return ans, steps
    l, r = lo, hi
    while l <= r:
        mid = (l + r) // 2
        steps += 1
        err = _eval_error_for_len(
            org_dct_zigzag,
            original_matrix,
            mid,
            n,
            dct_type,
            accuracy,
            use_fast_basis=True,
            src_dtype=src_dtype,
        )
        if err <= accuracy:
            ans = mid
            r = mid - 1
        else:
            l = mid + 1
    return l, steps


def _binary_search_min_len_incremental(
    org_dct_zigzag: np.ndarray,
    original_matrix: np.ndarray,
    accuracy: float,
    start_L: int,
    lo: int,
    hi: int,
    n: int,
    dct_type: int,
) -> Tuple[int, int]:
    """
    Binary search with incremental reconstruction, starting from prediction.

    Same strategy as float path (_binary_search_min_len_from_start):
    1. Check start_L → narrow [lo, hi] range
    2. Standard binary search on narrowed range using _nb_jump_diff

    Returns
    -------
    Tuple[int, int]
        (L_found, steps)
    """
    lo = max(1, int(lo))
    hi = min(len(org_dct_zigzag), int(hi))

    if lo > hi:
        return hi, 0

    basis = _get_idct_basis(n, dct_type)
    coeffs = org_dct_zigzag.astype(np.float64, copy=False)

    # Initialize at start_L
    cur_L = max(lo, min(start_L, hi))
    if cur_L > 0:
        recon = np.tensordot(coeffs[:cur_L], basis[:cur_L], axes=(0, 0))
    else:
        recon = np.zeros((n, n), dtype=np.float64)

    diff = (original_matrix.astype(np.float64, copy=False) - recon).copy()
    steps = 1

    err_start = float(_nb_max_err(diff))

    # Step 1: Check start_L and narrow range
    if err_start <= accuracy:
        ans = cur_L
        hi = cur_L - 1
    else:
        ans = hi
        lo = cur_L + 1

    if lo > hi:
        return ans, steps

    # Step 2: Standard binary search on [lo, hi]
    l, r = lo, hi
    while l <= r:
        mid = (l + r) // 2
        steps += 1
        err = float(_nb_jump_diff(diff, basis, coeffs, cur_L, mid))
        cur_L = mid
        if err <= accuracy:
            ans = mid
            r = mid - 1
        else:
            l = mid + 1

    return ans, steps


def _binary_search_min_len(
    org_dct_zigzag: np.ndarray,
    original_matrix: np.ndarray,
    accuracy: float,
    n: int,
    dct_type: int,
    lo: Optional[int] = None,
    hi: Optional[int] = None,
    use_fast_basis: bool = True,
) -> Tuple[int, int]:
    """
    Standard binary search (non-incremental).
    """
    if lo is None:
        lo = 1
    if hi is None:
        hi = len(org_dct_zigzag)

    lo = max(1, int(lo))
    hi = min(len(org_dct_zigzag), int(hi))

    if lo > hi:
        return hi, 0

    steps = 0
    ans = hi

    while lo <= hi:
        mid = (lo + hi) // 2
        steps += 1

        try:
            err = _eval_error_for_len(
                org_dct_zigzag,
                original_matrix,
                mid,
                n,
                dct_type,
                accuracy,
                use_fast_basis=use_fast_basis,
            )
        except (ValueError, np.linalg.LinAlgError):
            err = float("inf")

        if err <= accuracy:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1

    return ans, steps


def refine_dct_array(
    org_dct_zigzag: np.ndarray,
    accuracy: float,
    original_matrix: np.ndarray,
    n: int,
    dct_type: int = 2,
    scaling_factor: int = 100,
    config: Optional[Config] = None,
    predicted_L: Optional[int] = None,
    skip_queue: Optional[list] = None,
    src_dtype: Optional[str] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Optimize the number of DCT coefficients for required accuracy.

    OPTIMIZED VERSION with:
    - Incremental binary search
    - L prediction as starting point
    - Backscan for minimization
    - Verification with full IDCT

    Parameters
    ----------
    org_dct_zigzag : np.ndarray
        DCT coefficients in zigzag order
    accuracy : float
        Required accuracy (max error)
    original_matrix : np.ndarray
        Original matrix before DCT
    n : int
        Block size
    dct_type : int
        DCT type
    scaling_factor : int
        Scaling factor for integer conversion
    config : Config, optional
        Configuration
    predicted_L : int, optional
        Predicted L value as starting point for binary search

    Returns
    -------
    Tuple[np.ndarray, float, np.ndarray]
        (compressed_array, max_error, reconstructed_matrix)
    """
    if config is None:
        config = load_config()

    total_len = len(org_dct_zigzag)

    # Options from config
    use_fast_basis = config.performance.fast_eval_basis
    use_incremental = config.performance.incremental_backscan
    verify_full = config.performance.verify_with_full_idct
    backscan_base = config.model.minimize_backscan
    backscan_break = config.model.backscan_break_after
    accept_if_within = config.model.accept_prediction_if_within_accuracy

    # Starting point
    if predicted_L is not None and predicted_L > 0:
        start_L = max(1, min(int(predicted_L), total_len))
    else:
        # Default starting point - middle of range
        start_L = total_len // 2

    _is_uint8 = src_dtype in ("uint8", "int8", "uint16", "int16")

    # Debug counters (module-level)
    global _skip_count, _total_count
    _total_count += 1

    # OPTIMIZATION: accept_prediction_if_within_accuracy (generic or uint8-specific)
    # If prediction meets accuracy → accept without binary search.
    # For uint8, try an increasing L-scale cascade before falling back to binary search.
    _check_accept = accept_if_within
    if _check_accept and predicted_L is not None and predicted_L > 0:
        try:
            if _is_uint8:
                raw_scales = getattr(config.model, "uint8_L_prediction_scales", [1.5])
                if not isinstance(raw_scales, (list, tuple)):
                    raw_scales = [raw_scales]
                candidate_lengths = []
                for scale in raw_scales:
                    try:
                        candidate_lengths.append(int(round(int(predicted_L) * float(scale))))
                    except (TypeError, ValueError):
                        continue
            else:
                candidate_lengths = [start_L]

            # Convert to a sorted list of unique candidate lengths
            candidates = sorted(
                list(set(max(1, min(int(l), total_len)) for l in candidate_lengths))
            )
            if _is_uint8:
                candidates = [l for l in candidates if l < total_len]

            if candidates:
                # 1. MATHEMATICAL PRE-CHECK:
                # Evaluate the LARGEST candidate first. If even the largest candidate L fails,
                # then absolutely NONE of the smaller candidates can meet the accuracy.
                # This drops failure checks from O(N) sequential IDCTs to exactly 1 check!
                max_cand = candidates[-1]
                if _is_uint8:
                    _c_dq = (
                        np.round(org_dct_zigzag[:max_cand] * scaling_factor).astype(np.float64)
                        / scaling_factor
                    )
                    _dq_z = np.zeros_like(org_dct_zigzag, dtype=np.float64)
                    _dq_z[:max_cand] = _c_dq
                    err_max = _eval_error_for_len(
                        _dq_z,
                        original_matrix,
                        max_cand,
                        n,
                        dct_type,
                        accuracy,
                        use_fast_basis=use_fast_basis,
                        use_l2_precheck=False,
                        src_dtype=src_dtype,
                    )
                else:
                    err_max = _eval_error_for_len(
                        org_dct_zigzag,
                        original_matrix,
                        max_cand,
                        n,
                        dct_type,
                        accuracy,
                        use_fast_basis=use_fast_basis,
                        src_dtype=src_dtype,
                    )

                if err_max <= accuracy:
                    # At least one candidate works.
                    if accept_if_within:
                        # accept_prediction_if_within_accuracy: iterate candidates in ascending order
                        # and accept first one that meets accuracy (no binary search)
                        for candidate_L in candidates:
                            if _is_uint8:
                                _c_dq = (
                                    np.round(org_dct_zigzag[:candidate_L] * scaling_factor).astype(
                                        np.float64
                                    )
                                    / scaling_factor
                                )
                                _dq_z = np.zeros_like(org_dct_zigzag, dtype=np.float64)
                                _dq_z[:candidate_L] = _c_dq
                                err_candidate = _eval_error_for_len(
                                    _dq_z,
                                    original_matrix,
                                    candidate_L,
                                    n,
                                    dct_type,
                                    accuracy,
                                    use_fast_basis=use_fast_basis,
                                    use_l2_precheck=False,
                                    src_dtype=src_dtype,
                                )
                            else:
                                err_candidate = _eval_error_for_len(
                                    org_dct_zigzag,
                                    original_matrix,
                                    candidate_L,
                                    n,
                                    dct_type,
                                    accuracy,
                                    use_fast_basis=use_fast_basis,
                                    src_dtype=src_dtype,
                                )
                            if err_candidate <= accuracy:
                                best_L = candidate_L
                                best_err = err_candidate
                                coeffs = org_dct_zigzag[:best_L]
                                result = np.round(coeffs * scaling_factor).astype(int)
                                _skip_count += 1
                                return result, best_err, None
                    else:
                        # Use binary search over candidate list to find the absolute smallest working candidate
                        lo_idx, hi_idx = 0, len(candidates) - 1
                        best_idx = hi_idx  # we know candidates[-1] works
                        best_err = err_max

                        while lo_idx <= hi_idx:
                            mid_idx = (lo_idx + hi_idx) // 2
                            candidate_L = candidates[mid_idx]

                            if mid_idx == len(candidates) - 1:
                                err_mid = err_max
                            else:
                                if _is_uint8:
                                    _c_dq = (
                                        np.round(org_dct_zigzag[:candidate_L] * scaling_factor).astype(
                                            np.float64
                                        )
                                        / scaling_factor
                                    )
                                    _dq_z = np.zeros_like(org_dct_zigzag, dtype=np.float64)
                                    _dq_z[:candidate_L] = _c_dq
                                    err_mid = _eval_error_for_len(
                                        _dq_z,
                                        original_matrix,
                                        candidate_L,
                                        n,
                                        dct_type,
                                        accuracy,
                                        use_fast_basis=use_fast_basis,
                                        use_l2_precheck=False,
                                        src_dtype=src_dtype,
                                    )
                                else:
                                    err_mid = _eval_error_for_len(
                                        org_dct_zigzag,
                                        original_matrix,
                                        candidate_L,
                                        n,
                                        dct_type,
                                        accuracy,
                                        use_fast_basis=use_fast_basis,
                                        src_dtype=src_dtype,
                                    )

                            if err_mid <= accuracy:
                                best_idx = mid_idx
                                best_err = err_mid
                                hi_idx = mid_idx - 1  # try to find a smaller working candidate
                            else:
                                lo_idx = mid_idx + 1  # try larger candidates

                        best_L = candidates[best_idx]

                    coeffs = org_dct_zigzag[:best_L]
                    result = np.round(coeffs * scaling_factor).astype(int)
                    _skip_count += 1
                    return result, best_err, None
        except (ValueError, IndexError):
            pass

    # Binary search
    if predicted_L is not None and predicted_L > 0 and use_fast_basis and use_incremental:
        # Heuristic + fast incremental path: use fast incremental search
        # with prediction as starting point (window allows linear check in the vicinity)
        L_found, _ = _binary_search_min_len_incremental(
            org_dct_zigzag,
            original_matrix,
            accuracy,
            start_L=int(predicted_L),
            lo=1,
            hi=total_len,
            n=n,
            dct_type=dct_type,
        )
    elif predicted_L is not None and predicted_L > 0:
        # Heuristic: binary search with prediction as starting point
        # (fallback when incremental unavailable)
        L_found, _ = _binary_search_min_len_from_start(
            org_dct_zigzag,
            original_matrix,
            accuracy,
            start_L=predicted_L,
            lo=1,
            hi=total_len,
            n=n,
            dct_type=dct_type,
            skip_queue=skip_queue,
            src_dtype=src_dtype,
        )
    else:
        # No prediction → pure binary search (always finds global minimum L)
        L_found, _ = _binary_search_min_len(
            org_dct_zigzag,
            original_matrix,
            accuracy,
            n,
            dct_type,
            lo=1,
            hi=total_len,
            use_fast_basis=use_fast_basis,
        )

    # Backscan for minimization
    if backscan_base > 0 and L_found > 1:
        # Scale backscan proportionally to block size
        scale_factor = (n / 8) * math.sqrt(n / 8)
        backscan = int(backscan_base * scale_factor)
        start = max(1, L_found - backscan)

        _is_uint8 = src_dtype in ("uint8", "int8", "uint16", "int16")

        if use_fast_basis and use_incremental:
            basis = _get_idct_basis(n, dct_type)
            coeffs = org_dct_zigzag.astype(np.float32, copy=False)
            recon = np.tensordot(coeffs[:L_found], basis[:L_found], axes=(0, 0))
            diff = original_matrix.astype(np.float32, copy=False) - recon
            _tmp = np.empty((n, n), dtype=np.float64)  # pre-alloc buffer

            new_L = L_found
            fail_streak = 0
            for k in range(L_found - 1, start - 1, -1):
                np.multiply(basis[k], coeffs[k], out=_tmp)
                np.add(diff, _tmp, out=diff)
                if _is_uint8:
                    # uint8: decenter, clip, round for error
                    # diff = original_matrix - new_recon  →  new_recon = original_matrix - diff
                    orig_u8 = original_matrix + 128.0
                    recon_u8 = np.clip(np.round(original_matrix - diff + 128.0), 0, 255)
                    err = max(
                        float((orig_u8 - recon_u8).max()),
                        float(-(orig_u8 - recon_u8).min()),
                    )
                else:
                    err = max(float(diff.max()), float(-diff.min()))
                if err <= accuracy:
                    new_L = k
                    fail_streak = 0
                else:
                    fail_streak += 1
                    # Break after N consecutive failures (0 = never break)
                    if backscan_break > 0 and fail_streak >= backscan_break:
                        break
            L_found = new_L
        else:
            for test_L in range(start, L_found):
                try:
                    test_error = _eval_error_for_len(
                        org_dct_zigzag,
                        original_matrix,
                        test_L,
                        n,
                        dct_type,
                        accuracy,
                        use_fast_basis=use_fast_basis,
                        src_dtype=src_dtype,
                    )
                    if test_error <= accuracy:
                        L_found = test_L
                        break
                except (ValueError, IndexError):
                    continue

    # Verification with full IDCT
    if verify_full:
        err_full = _eval_error_for_len(
            org_dct_zigzag,
            original_matrix,
            L_found,
            n,
            dct_type,
            accuracy,
            use_fast_basis=False,  # Full IDCT
            src_dtype=src_dtype,
        )

        # If error too large, increase L (max 5 iterations)
        refine_count = 0
        while err_full > accuracy and L_found < total_len and refine_count < 5:
            L_found += 1
            refine_count += 1
            err_full = _eval_error_for_len(
                org_dct_zigzag,
                original_matrix,
                L_found,
                n,
                dct_type,
                accuracy,
                use_fast_basis=False,
                src_dtype=src_dtype,
            )

    # Build result
    coeffs = org_dct_zigzag[:L_found]
    result = np.round(coeffs * scaling_factor).astype(int)

    # Verify error AFTER quantization (rounding error was not accounted for in binary search)
    # Dequantization: coeff_stored/sf ≠ org_dct_zigzag[:L] → actual error may exceed accuracy
    def _eval_quantized_error(L):
        """Compute reconstruction error from dequantized coefficients."""
        c = org_dct_zigzag[:L]
        dq = np.round(c * scaling_factor).astype(np.float64) / scaling_factor
        dq_zigzag = np.zeros_like(org_dct_zigzag, dtype=np.float64)
        dq_zigzag[:L] = dq
        return _eval_error_for_len(
            dq_zigzag,
            original_matrix,
            L,
            n,
            dct_type,
            accuracy,
            use_fast_basis=use_fast_basis,
            use_l2_precheck=False,
            src_dtype=src_dtype,
        )

    max_error = _eval_quantized_error(L_found)

    # If rounding introduced too large error — increase L until accuracy is met
    if max_error > accuracy:
        quant_refine = 0
        while max_error > accuracy and L_found < total_len and quant_refine < 16:
            L_found += 1
            quant_refine += 1
            result = np.round(org_dct_zigzag[:L_found] * scaling_factor).astype(int)
            max_error = _eval_quantized_error(L_found)

    return result, max_error, None  # Don't return reconstruction - not needed


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    "refine_dct_array",
]
