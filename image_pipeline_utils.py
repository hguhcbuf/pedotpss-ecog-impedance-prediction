"""
Utility functions for image-based feature extraction of printed PEDOT:PSS electrode patterns.

This module contains helper functions used by `pipeline.py`. The functions are
kept separate from the main pipeline so that the execution script remains clean
and easy to read.
"""

from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore


# =============================================================================
# Peak filtering and boundary refinement
# =============================================================================

def refine_peaks(
    peak_lists,           # start_peak_list or end_peak_list
    space_sign,           # start = -1, end = +1
    extra_space,          # pixel offset added to the refined boundary
    col, row,             # image size: number of columns and rows
    how_many_for_avg,     # number of neighboring columns used for averaging
    threshold,            # z-score threshold for outlier rejection
    img,                  # image copy used for optional point visualization
    dot_color=(0, 0, 255) # visualization color in BGR format
):
    """
    Refine column-wise peak positions by using neighboring valid peaks.

    Parameters
    ----------
    peak_lists : list of array-like
        Detected peak positions for each image column.
    space_sign : int
        Direction of the boundary offset. Use -1 for start boundaries and +1
        for end boundaries.
    extra_space : int
        Pixel offset applied after averaging neighboring peak locations.
    col, row : int
        Image width and height.
    how_many_for_avg : int
        Number of neighboring valid columns used for local averaging.
    threshold : float
        z-score threshold used to reject outlier peak positions.
    img : np.ndarray
        Image copy used to draw the refined points.
    dot_color : tuple
        BGR color used for point visualization.

    Returns
    -------
    final_mat : np.ndarray
        A 2-D array with shape (col, max_peaks). Missing values are marked as -1.
    """
    max_peaks = max(len(p) for p in peak_lists)
    peak_mat  = np.full((col, max_peaks), np.nan)

    # Convert column-wise peak lists into a padded 2-D matrix.
    for c, pks in enumerate(peak_lists):
        pks = np.sort(pks)
        peak_mat[c, :len(pks)] = pks

    final_mat = np.full_like(peak_mat, -1, dtype=int)

    for k in range(max_peaks):          # k = peak index / line index
        vec     = peak_mat[:, k]
        valid   = ~np.isnan(vec)
        zs      = np.full_like(vec, np.inf, dtype=float)
        zs[valid] = zscore(vec[valid])

        for idx in range(col):
            # Collect nearby valid peak positions while excluding z-score outliers.
            nbrs, d = [], 0
            while len(nbrs) < how_many_for_avg and d <= col:
                for off in (+d, -d):
                    j = idx + off
                    if 0 <= j < col and not np.isnan(vec[j]) \
                       and abs(zs[j]) < threshold:
                        nbrs.append(vec[j])
                d += 1

            # Calculate the final boundary coordinate.
            if nbrs:
                val = int(np.mean(nbrs)) + space_sign * extra_space
                val = max(0, min(row - 1, val))    # clamp to image boundary
                final_mat[idx, k] = val
                img[val, idx] = dot_color          # optional visualization point
            # Otherwise, keep -1 as a missing value.

    return final_mat


def filter_a_with_b(lista, listb):
    """
    Apply a sliding 1-D filter and return the absolute response.

    This function is mainly used to detect strong changes in a cumulative
    intensity profile. The absolute value is used when only the magnitude of the
    response is required.
    """
    filtered_list = []
    for i in range(len(lista)-len(listb)+1):
        zeros_list = [0]*i
        filtera = zeros_list + listb
        filtereda = [a*b for a,b in zip(lista, filtera)]
        filtersum = abs(sum(filtereda))
        filtered_list.append(filtersum)
    return(filtered_list)


def filter_a_with_b_no_abs(lista, listb):
    """
    Apply a sliding 1-D filter while preserving the sign of the response.

    The signed response is used to distinguish opposite edge directions, such as
    start and end boundaries.
    """
    filtered_list = []
    for i in range(len(lista)-len(listb)+1):
        zeros_list = [0]*i
        filtera = zeros_list + listb
        filtereda = [a*b for a,b in zip(lista, filtera)]
        filtersum = sum(filtereda)
        filtered_list.append(filtersum)
    return(filtered_list)


def smooth(y, window_size=10):
    """
    Smooth a 1-D signal using a moving-average window.
    """
    window = np.ones(window_size) / window_size
    return np.convolve(y, window, mode='same')


def trim_to_common_length(peak_lists):
    """
    Keep only columns with the most common number of detected peaks.

    The most frequently observed peak count is treated as the expected number of
    line boundaries. Columns with a different number of peaks are replaced with
    empty arrays and handled as missing values in the refinement step.

    Returns
    -------
    trimmed_lists : list[np.ndarray]
        Peak lists with only the most common peak count retained.
    expected_len : int
        Most common number of peaks among valid columns.
    """
    # Find the most frequent number of peaks.
    lengths = [len(p) for p in peak_lists if len(p) > 0]
    if not lengths:
        return peak_lists, 0

    expected_len = Counter(lengths).most_common(1)[0][0]

    # Replace columns with abnormal peak counts by empty arrays.
    trimmed = [
        np.sort(p) if len(p) == expected_len else np.array([])
        for p in peak_lists
    ]

    return trimmed, expected_len


# =============================================================================
# Visualization and boundary integrity checks
# =============================================================================

def show_image_with_red_line(image, n):
    """
    Display an image with a red vertical scanline at column n.

    This helper is intended for debugging column-wise scanline analysis.
    """
    image = image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    n = n
    image_rgb[:, n] = [255, 0, 0]  # red line in RGB

    # Optional thicker line:
    # image_rgb[:, n-1:n+2] = [255, 0, 0]

    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.title(f"Image with red vertical line at n={n}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def overlapping_columns(start_mat, end_mat, *, pitch_tol=0.4):
    """
    Identify columns where neighboring line regions overlap.

    For each column, this function constructs start-end intervals for all
    detected lines and checks whether adjacent intervals overlap. A very small
    gap relative to the estimated line pitch can also be flagged.
    """
    col, n_lines = start_mat.shape
    overlaps = []

    # Estimate the pitch from the median distance between line centers.
    dists = []
    for idx in range(col):
        centers = []
        for k in range(n_lines):
            s, e = start_mat[idx, k], end_mat[idx, k]
            if s >= 0 and e >= 0:
                centers.append(0.5*(s + e))
        centers = np.sort(centers)
        if len(centers) >= 2:
            dists += list(np.diff(centers))
    pitch = np.median(dists) if dists else 0

    for idx in range(col):
        intervals = []
        for k in range(n_lines):
            s, e = start_mat[idx, k], end_mat[idx, k]
            if s >= 0 and e >= 0:
                a, b = sorted((s, e))
                intervals.append((a, b))

        if len(intervals) < 2:
            continue

        intervals.sort(key=lambda t: t[0])

        # True overlap: the next interval starts before the previous interval ends.
        for i in range(len(intervals) - 1):
            a1, b1 = intervals[i]
            a2, b2 = intervals[i+1]
            gap = a2 - b1
            if gap < 0:
                overlaps.append(idx)
                break
            # Optional pitch-based tolerance for nearly overlapping intervals.
            elif pitch > 0 and gap < pitch * pitch_tol * 0.2:
                overlaps.append(idx)
                break
    return overlaps


# =============================================================================
# Missing-boundary correction
# =============================================================================

def _infer_missing_pairs(vec_s, vec_e, win=5):
    """
    Fill missing start/end pairs using nearby columns.

    Parameters
    ----------
    vec_s, vec_e : np.ndarray, shape=(W,)
        Start and end boundary vectors for one line across all columns.
    win : int
        Half-window size used to search neighboring columns.

    Notes
    -----
    If only one side of a boundary pair exists, the missing side is estimated
    from the median of nearby valid boundary positions. If not enough nearby
    values are available, the missing value remains -1.
    """
    W = len(vec_s)
    vec_s, vec_e = vec_s.copy(), vec_e.copy()

    for x in range(W):
        # Case 1: start exists, but end is missing.
        if vec_s[x] >= 0 and vec_e[x] < 0:
            neigh = [vec_e[j] for j in range(max(0, x-win), min(W, x+win+1))
                      if vec_e[j] >= 0]
            if len(neigh) >= win//2:
                vec_e[x] = int(np.median(neigh))

        # Case 2: end exists, but start is missing.
        elif vec_e[x] >= 0 and vec_s[x] < 0:
            neigh = [vec_s[j] for j in range(max(0, x-win), min(W, x+win+1))
                      if vec_s[j] >= 0]
            if len(neigh) >= win//2:
                vec_s[x] = int(np.median(neigh))

    return vec_s, vec_e


def _fill_orphans_with_thickness(s_vec, e_vec, thick_med, h):
    """
    Fill one-sided boundary detections using the global median thickness.

    If only the end boundary exists, the start boundary is estimated as
    end - median_thickness. If only the start boundary exists, the end boundary
    is estimated as start + median_thickness.
    """
    s_vec = s_vec.copy()
    e_vec = e_vec.copy()

    # End-only case: estimate start boundary.
    mask = (s_vec < 0) & (e_vec >= 0)
    s_vec[mask] = np.clip(e_vec[mask] - thick_med, 0, h - 1)

    # Start-only case: estimate end boundary.
    mask = (e_vec < 0) & (s_vec >= 0)
    e_vec[mask] = np.clip(s_vec[mask] + thick_med, 0, h - 1)

    return s_vec, e_vec


def fill_orphan_to_border(start_mat, end_mat, img_h):
    """
    Fill one-sided boundary pairs by extending them to the image border.

    End-only points are extended to the top border, and start-only points are
    extended to the bottom border.
    """
    s = start_mat.copy()
    e = end_mat.copy()
    m = (s < 0) & (e >= 0)   # end-only columns
    s[m] = 0
    m = (e < 0) & (s >= 0)   # start-only columns
    e[m] = img_h - 1
    return s, e


def fix_pairs_to_border(start_mat: np.ndarray,
                        end_mat:   np.ndarray,
                        img_h:     int):
    """
    Correct missing or reversed start/end pairs using image borders.

    For each column and line:
      - if start exists but end is missing, set end = img_h - 1
      - if end exists but start is missing, set start = 0
      - if both exist but end < start, swap the two values
    """
    s = start_mat.copy()
    e = end_mat.copy()

    m = (s >= 0) & (e < 0)
    e[m] = img_h - 1

    m = (e >= 0) & (s < 0)
    s[m] = 0

    m = (s >= 0) & (e >= 0) & (e < s)
    tmp = s[m].copy()
    s[m] = e[m]
    e[m] = tmp

    return s, e


def insert_edge_points(start_mat, end_mat, img_h, verbose=False):
    """
    Insert image-border points when a line is cut by the top or bottom boundary.

    If the first end boundary appears before the first start boundary, a top
    border start point is inserted. If the last start boundary appears after the
    last end boundary, a bottom border end point is appended.
    """
    # Convert NumPy arrays to list-of-lists if necessary.
    if isinstance(start_mat, np.ndarray):
        start_mat = [list(row[~np.isnan(row)]) for row in start_mat]
    if isinstance(end_mat, np.ndarray):
        end_mat = [list(row[~np.isnan(row)]) for row in end_mat]

    if verbose:
        print("Before insertion:")
        print("start_mat =", start_mat)
        print("end_mat   =", end_mat)
        print("-----")

    for i in range(len(start_mat)):
        start_row = start_mat[i]
        end_row = end_mat[i]

        if len(start_row) > 0 and len(end_row) > 0:
            if start_row[0] > end_row[0]:
                if verbose:
                    print(f"[{i}] insert 0 to start_row")
                start_row.insert(0, 0)

            if start_row[-1] > end_row[-1]:
                if verbose:
                    print(f"[{i}] append {img_h} to end_row")
                end_row.append(img_h)

    if verbose:
        print("After insertion:")
        print("start_mat =", start_mat)
        print("end_mat   =", end_mat)

    return start_mat, end_mat


def list_to_padded_array(mat_list, pad_val=np.nan, target_len=None):
    """
    Convert a list of variable-length rows into a padded 2-D NumPy array.

    Parameters
    ----------
    mat_list : list
        List of rows with potentially different lengths.
    pad_val : float
        Value used to fill missing entries.
    target_len : int or None
        If provided, all rows are padded to this length. Otherwise, the maximum
        row length in mat_list is used.
    """
    # Use the longest row length unless a target length is explicitly provided.
    if target_len is None:
        max_len = max(len(row) for row in mat_list)
    else:
        max_len = target_len

    padded = np.full((len(mat_list), max_len), pad_val)
    for i, row in enumerate(mat_list):
        padded[i, :len(row)] = row
    return padded


# =============================================================================
# Debug visualization
# =============================================================================

def draw_start_end(base_img, start_mat, end_mat,
                   start_color=(0,255,0), end_color=(0,0,255)):
    """
    Draw start and end boundary points on a copy of the input image.

    Start points are drawn in green by default, and end points are drawn in red.
    This function is intended for visual debugging of boundary detection.
    """
    out = base_img.copy()
    h, w, _ = out.shape
    W, K = start_mat.shape      # W == number of columns
    xs = np.arange(W)

    for k in range(K):
        s_col = start_mat[:, k]
        e_col = end_mat[:, k]

        cols = np.arange(W)

        m_start = ~np.isnan(s_col)
        m_end = ~np.isnan(e_col)

        # Convert to integer indices.
        s_col = s_col.astype(int)
        e_col = e_col.astype(int)

        # Draw detected boundary points.
        out[s_col[m_start], cols[m_start]] = start_color
        out[e_col[m_end]-1, cols[m_end]] = end_color

    return out


# =============================================================================
# Feature extraction and scoring
# =============================================================================

def calc_volume_multi(resized_image, start_mat, end_mat, *, alpha=0.20):
    """
    Calculate a volume-like image feature between start and end boundaries.

    For each detected line and column, the function integrates cumulative
    grayscale-intensity differences between the start and end boundaries. The
    inverse of the column-wise volume profile is summed to obtain a line-level
    score. The final output is the average score over valid lines.

    Returns
    -------
    avg_sigma : float
        Average volume-derived score over valid lines.
    sigmas : list[float]
        Line-wise volume-derived scores.
    out_bgr : np.ndarray
        Visualization image with calculated regions overlaid.
    """
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY).astype(np.int16)
    h, w  = gray.shape
    W, K  = start_mat.shape  # == (w, n_lines)
    overlay = np.zeros_like(resized_image, dtype=np.uint8)
    out_bgr = resized_image.copy()
    sigmas  = []

    use_the_line = True

    # Global median thickness calculated from valid start/end pairs.
    valid_pairs = (start_mat >= 0) & (end_mat >= 0)
    if np.any(valid_pairs):
        thick_med_global = np.median(np.abs(end_mat[valid_pairs] - start_mat[valid_pairs]))
    else:
        thick_med_global = 0

    for k in range(K):
        s_vec = start_mat[:, k]
        e_vec = end_mat[:, k]

        # Ignore lines that are likely cut by the image boundary.
        if np.mean(s_vec) == 0 or np.mean(e_vec) == 480:
            print("ignoring cutted lines")
            use_the_line = False


        # 1) Fill missing pairs using neighboring columns.
        s_vec, e_vec = _infer_missing_pairs(s_vec, e_vec, win=6)

        # 2) Fill remaining one-sided detections using median thickness.
        s_vec, e_vec = _fill_orphans_with_thickness(s_vec, e_vec,
                                                    thick_med_global, h)

        # 3) Skip columns with invalid or too-thin boundary pairs.
        bad_mask = (s_vec < 0) | (e_vec < 0) | (np.abs(e_vec - s_vec) < 2)

        volume_list = []
        for x in range(w):
            if bad_mask[x]:
                volume_list.append(0)
                continue

            y0, y1 = int(s_vec[x]), int(e_vec[x])
            if y0 > y1: y0, y1 = y1, y0

            # Cumulative grayscale-difference integration.
            cum = cum2 = 0
            prev = gray[y0, x]
            for r in range(y0 + 1, y1):
                cur  = int(gray[r, x])
                diff = prev - cur
                cum  += diff
                cum2 += cum
                prev = cur
            volume_list.append(cum2)

            # Draw calculated region and boundaries.
            overlay[y0:y1, x] = (0,0,255)
            out_bgr[y0, x]    = (0,0,255)
            out_bgr[y1-1, x]  = (0,0,255)

        vol_arr = np.asarray(volume_list, dtype=float)
        with np.errstate(divide="ignore"):
            inv = 1.0 / vol_arr
        inv[np.isinf(inv)] = 0.0
        sigma = float(abs(np.sum(inv)))
        if use_the_line:
            sigmas.append(round(sigma, 3))
        use_the_line = True

    if not sigmas:
        raise ValueError("No complete line was detected. Please check start/end pairs.")

    avg_sigma = float(np.mean(sigmas))
    out_bgr = cv2.addWeighted(out_bgr, 1.0, overlay, alpha, 0)

    return avg_sigma, sigmas, out_bgr


def interior_fill_ratio(start_mat: np.ndarray,
                        end_mat:   np.ndarray,
                        img_h: int,
                        img_w: int) -> float:
    """
    Calculate the percentage of image area covered by detected interior regions.

    The filled area is calculated from the vertical distance between start and
    end boundaries. If only one boundary exists, the area is extended to the
    corresponding image border.

    Returns
    -------
    float
        Interior fill ratio in percent, ranging from 0.0 to 100.0.
    """
    n_lines = start_mat.shape[1]
    filled_pixels = 0

    for x in range(img_w):                 # iterate over all columns
        for k in range(n_lines):           # iterate over detected lines
            s, e = int(start_mat[x, k]), int(end_mat[x, k])

            # Case 1: both start and end exist.
            if s >= 0 and e >= 0:
                y0, y1 = (s, e) if s < e else (e, s)

            # Case 2: only start exists; extend to the bottom border.
            elif s >= 0 and e < 0:
                y0, y1 = s, img_h

            # Case 3: only end exists; extend from the top border.
            elif s < 0 and e >= 0:
                y0, y1 = 0, e if e > 0 else 0

            # Case 4: both are missing.
            else:
                continue

            if y1 - y0 < 1:
                continue

            filled_pixels += (y1 - y0)     # boundaries are not double-counted

    total_pixels = img_h * img_w
    return filled_pixels / total_pixels * 100.0


def _post_filter_lines(start_mat, end_mat,
                       low_ratio=0.55, high_ratio=1.45,
                       center_tol=0.35, min_cov=0.70):
    """
    Remove abnormal boundary pairs using thickness and center-position criteria.

    This optional post-processing filter compares each boundary pair against the
    global median thickness and the line-wise center position. Points with
    abnormal thickness or center displacement are removed. Lines with insufficient
    valid-column coverage are also discarded.
    """
    W, K = start_mat.shape
    s = start_mat.astype(float).copy()
    e = end_mat.astype(float).copy()

    valid_mask = (s >= 0) & (e >= 0)
    width = np.where(valid_mask, np.abs(e - s), np.nan)
    center = np.where(valid_mask, 0.5 * (s + e), np.nan)

    # Global median thickness.
    thick_med = np.nanmedian(width)

    # Median center position for each line.
    ref_center = np.nanmedian(center, axis=0)

    # Estimate pitch from sorted line centers.
    sorted_centers = np.sort(ref_center[~np.isnan(ref_center)])
    pitch = np.median(np.diff(sorted_centers)) if len(sorted_centers) >= 2 else 0

    # Point-wise filtering.
    for k in range(K):
        for x in range(W):
            if not valid_mask[x, k]:
                continue
            w = width[x, k]
            c = center[x, k]

            bad_thick = (w < thick_med * low_ratio) or (w > thick_med * high_ratio)
            bad_center = (pitch > 0) and (abs(c - ref_center[k]) > pitch * center_tol)

            if bad_thick or bad_center:
                s[x, k] = -1
                e[x, k] = -1
                valid_mask[x, k] = False

    # Line-wise coverage filtering.
    keep_cols = []
    for k in range(K):
        cov = np.mean(valid_mask[:, k])
        if cov >= min_cov:
            keep_cols.append(k)

    if not keep_cols:
        # If all lines are removed, return the original input to avoid failure
        # in downstream steps.
        return start_mat, end_mat

    s = s[:, keep_cols].astype(int)
    e = e[:, keep_cols].astype(int)
    return s, e


def predict_impedance(volume_value):
    """
    Convert the image-derived volume feature to predicted impedance.

    The scaling factor is empirically determined from the calibration used in
    the study. The length correction converts the value to the target electrode
    length scale.
    """
    return 60620.155 * volume_value * 12.249 / 5


def calculate_imp_score(predicted_val, target):
    """
    Calculate the absolute error between predicted and target impedance.
    """
    return abs(target - predicted_val)


def min_max_normalize_imp(val):
    """
    Normalize impedance error to a score between 0 and 1.

    Smaller impedance error gives a higher score. Values larger than maxVal are
    clipped to maxVal.
    """
    maxVal = 100000
    if val > maxVal:
        val = maxVal
    nor = val/maxVal

    return nor * -1 +1


def min_max_unnormalize_imp(val):
    """
    Convert a normalized impedance score back to the impedance-error scale.
    """
    nor = (val - 1) * -1
    return nor * 100000


def min_max_normalize_area(val):
    """
    Normalize printed area percentage to a score between 0 and 1.
    """
    maxVal = 100
    if val > maxVal:
        val = maxVal
    nor = val/maxVal
    return nor


def min_max_unnormalize_area(val):
    """
    Convert a normalized area score back to percentage scale.
    """
    return val * 100


