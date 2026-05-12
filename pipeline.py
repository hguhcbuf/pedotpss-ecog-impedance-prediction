"""
Main execution pipeline for image-based feature extraction of printed PEDOT:PSS electrode patterns.

Only `run_full_pipeline` is kept in this file. Helper functions are imported
from `image_pipeline_utils.py`.
"""

import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from image_pipeline_utils import (
    calc_volume_multi,
    calculate_imp_score,
    draw_start_end,
    filter_a_with_b,
    filter_a_with_b_no_abs,
    insert_edge_points,
    interior_fill_ratio,
    list_to_padded_array,
    min_max_normalize_area,
    min_max_normalize_imp,
    overlapping_columns,
    predict_impedance,
    refine_peaks,
    smooth,
    trim_to_common_length,
)


# =============================================================================
# Main pipeline
# =============================================================================

def run_full_pipeline(file_path: str,
                      image_width: int = 640,
                      image_height: int = 480,
                      noise_filter_size: int = 5,
                      edge_filter_size: int = 2,
                      threshold: float = 1.2,
                      extra_space: int = 15,
                      how_many_for_avg: int = 10,
                      noise_prominence = (150, 1000),
                      gaussian_smooth = 40,
                      gaussian_sigma = 10,
                      prominence = (0.7, 20)):
    """
    Run the complete image-analysis pipeline.

    Steps
    -----
    1. Load and resize the image.
    2. Extract column-wise cumulative intensity-change profiles.
    3. Correct plateau-like noise regions.
    4. Detect start and end boundary peaks.
    5. Refine boundary positions using neighboring columns.
    6. Insert image-edge points when line regions are cut by image boundaries.
    7. Check overlapping columns.
    8. Calculate impedance-related and printed-area-related metrics.
    9. Visualize the original image and calculated regions.

    Returns
    -------
    dict
        Dictionary containing predicted impedance, printed area, and impedance
        error.
    """

    # -------------------------------------------------------------------------
    # 0. Load and resize image
    # -------------------------------------------------------------------------
    image = cv2.imread(file_path)
    resized_image = cv2.resize(image, (image_width, image_height),
                         interpolation=cv2.INTER_LANCZOS4)
    np_img  = np.asarray(resized_image)
    row, col, _ = np_img.shape

    # -------------------------------------------------------------------------
    # 1. Detect column-wise peaks
    # -------------------------------------------------------------------------
    edge_filter_noise = [-1]*noise_filter_size + [1]*noise_filter_size
    edge_filter       = [-1]*edge_filter_size   + [1]*edge_filter_size

    start_peak_list, end_peak_list = [], []
    for i in range(col):
        # 1-A. Calculate cumulative intensity differences along one column.
        accum, prev, first, change = 0, 0, True, []
        for pix in np_img[:, i, 0]:
            if first:
                prev, first = pix, False
            diff   = int(prev) - int(pix)
            accum += diff
            change.append(accum)
            prev   = pix

        # 1-B. Correct plateau-like noise regions.
        filt  = filter_a_with_b(change, edge_filter_noise)
        peaks = find_peaks(np.array(filt), prominence=noise_prominence, distance=1)[0]
        orig  = [p + noise_filter_size + (-2 if j % 2 == 0 else 2)
                 for j, p in enumerate(peaks)]
        if len(orig) % 2 == 0:
            for p0, p1 in zip(orig[::2], orig[1::2]):
                interp = np.linspace(change[p0], change[p1], p1-p0+1)
                change[p0:p1+1] = interp

        # 1-C. Detect start and end boundary peaks.
        sm  = gaussian_filter1d(smooth(change, gaussian_smooth), sigma=gaussian_sigma)
        flt = filter_a_with_b_no_abs(sm, edge_filter)
        start_peaks = find_peaks( np.array(flt),
                                  prominence=prominence, distance=6)[0]
        end_peaks   = find_peaks(-np.array(flt),
                                  prominence=prominence, distance=6)[0]

        if len(start_peaks)==len(end_peaks)==0:
            print(f"no peaks found at col {i}")

        start_peak_list.append(start_peaks)
        end_peak_list  .append(end_peaks)

    # -------------------------------------------------------------------------
    # 2. Match common peak counts and refine boundaries
    # -------------------------------------------------------------------------
    trim_start, expected_len = trim_to_common_length(start_peak_list)
    trim_end, _              = trim_to_common_length(end_peak_list)

    image_copy = resized_image.copy()
    final_start = refine_peaks(copy.deepcopy(trim_start), -1, extra_space,
                               col, row, how_many_for_avg, threshold,
                               image_copy)
    final_end   = refine_peaks(copy.deepcopy(trim_end), +1, extra_space,
                               col, row, how_many_for_avg, threshold,
                               image_copy)

    # -------------------------------------------------------------------------
    # 3. Insert top/bottom boundary points when the printed line is cut
    # -------------------------------------------------------------------------
    final_start, final_end = insert_edge_points(final_start, final_end, row)

    #print("number of start points : {}".format(len(final_start[0])))
    #print("number of end points : {}".format(len(final_end[0])))
    print("detected # of electrodes : {}".format(len(final_start[0])))

    # -------------------------------------------------------------------------
    # 4. Convert variable-length boundary lists into padded arrays
    # -------------------------------------------------------------------------
    max_len = max(
        max(len(row) for row in final_start),
        max(len(row) for row in final_end)
    )
    final_start = list_to_padded_array(final_start, target_len=max_len)
    final_end   = list_to_padded_array(final_end,   target_len=max_len)

    # Optional debugging plot:
    # debug_img = draw_start_end(resized_image, final_start, final_end)
    # plt.imshow(debug_img[..., ::-1])
    # plt.axis('off')
    # plt.title("Start=Green / End=Red")
    # plt.show()

    # -------------------------------------------------------------------------
    # 5. Check overlapping columns
    # -------------------------------------------------------------------------
    overlap_cols = overlapping_columns(final_start, final_end)
    if overlap_cols:
        print("overlapping columns:", overlap_cols)
    #     return {
    #             "predicted_impedance": 0,
    #             "printed_area": 0,
    #             }
    # else:
    #     print("continuous line, no overlaps.")
    #     print("-"*40)

    # -------------------------------------------------------------------------
    # 6. Calculate impedance and area metrics
    # -------------------------------------------------------------------------
    avg_v, per_line, colored = calc_volume_multi(
        resized_image, final_start, final_end, alpha = 0.1
    )
    predicted_impedance = predict_impedance(avg_v)

    # Set the target impedance value below.
    imp_score = calculate_imp_score(predicted_impedance, 100000)

    printed_area = interior_fill_ratio(start_mat= final_start, 
                                   end_mat= final_end,
                                   img_h= resized_image.shape[0],
                                   img_w= resized_image.shape[1])

    normalized_imp = min_max_normalize_imp(imp_score)
    normalized_area = min_max_normalize_area(printed_area)

    print("predicted_impedance  :", round(predicted_impedance, 3), "ohms")
    print("printed_area %  :", round(printed_area, 3), "%")
    #print("impedance error  :", round(imp_score, 3), "ohms")

    #print("impedance score normalized  :", round(normalized_imp, 3))
    #print("printed area score normalized  :", round(normalized_area, 3))

    print("-"*40)

    # -------------------------------------------------------------------------
    # 7. Visualize original image and calculated regions
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Original image: BGR to RGB conversion.
    ax[0].imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original image")
    ax[0].axis('off')

    # Calculated region image. 'colored' is in BGR format.
    ax[1].imshow(colored[..., ::-1])          # BGR to RGB
    ax[1].set_title("Calculated regions")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

    return {
        "predicted_impedance": round(predicted_impedance, 3),
        "printed_area": round(printed_area, 3),
        "impedance error": round(imp_score, 3),
    }
