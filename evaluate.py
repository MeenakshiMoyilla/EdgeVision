import os
import cv2
import numpy as np
import scipy.io as sio
import time
import csv

from edge_detection import (
    preprocess_image,
    sobel_edge_detection,
    prewitt_edge_detection,
    canny_edge_detection
)


# ==========================================================
# PATHS
# ==========================================================

IMAGE_FOLDER = os.path.join(
    "dataset",
    "BSDS500",
    "data",
    "images",
    "test"
)

GROUND_TRUTH_FOLDER = os.path.join(
    "dataset",
    "BSDS500",
    "data",
    "groundTruth",
    "test"
)


# ==========================================================
# SETTINGS
# ==========================================================

NUMBER_OF_IMAGES = 10

SOBEL_THRESHOLD = 100

PREWITT_THRESHOLD = 100

CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150

# A detected edge can be within this many pixels
# of the ground-truth edge and still be considered
# a match.
TOLERANCE = 1


# ==========================================================
# LOAD GROUND TRUTH
# ==========================================================

def load_ground_truth(mat_path):

    data = sio.loadmat(
        mat_path,
        squeeze_me=True,
        struct_as_record=False
    )

    ground_truth = data["groundTruth"]

    if not isinstance(ground_truth, np.ndarray):
        ground_truth = [ground_truth]

    boundary_maps = []

    for annotation in ground_truth:

        boundaries = annotation.Boundaries

        boundaries = np.asarray(
            boundaries,
            dtype=np.float32
        )

        boundary_maps.append(boundaries)

    combined = np.maximum.reduce(
        boundary_maps
    )

    combined = (
        combined > 0
    ).astype(np.uint8)

    return combined


# ==========================================================
# CREATE BINARY EDGE MAP
# ==========================================================

def create_binary_edge_map(edge_image, threshold):

    binary = (
        edge_image >= threshold
    ).astype(np.uint8)

    return binary


# ==========================================================
# DILATE EDGES FOR TOLERANCE
# ==========================================================

def dilate_edges(edge_map, tolerance):
    """
    Dilate an edge map to allow a small matching tolerance.
    """

    # OpenCV morphology requires uint8
    edge_map = (
        edge_map.astype(np.uint8)
        * 255
    )

    if tolerance == 0:
        return edge_map > 0

    kernel_size = (
        2 * tolerance + 1
    )

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8
    )

    dilated = cv2.dilate(
        edge_map,
        kernel,
        iterations=1
    )

    return dilated > 0


# ==========================================================
# CALCULATE METRICS
# ==========================================================

def calculate_metrics(
    predicted,
    ground_truth,
    tolerance=1
):

    predicted = predicted.astype(bool)

    ground_truth = ground_truth.astype(bool)


    # ------------------------------------------------------
    # Match predicted edges against ground truth
    # ------------------------------------------------------

    ground_truth_dilated = dilate_edges(
        ground_truth,
        tolerance
    )

    predicted_dilated = dilate_edges(
        predicted,
        tolerance
    )


    true_positive = np.logical_and(
        predicted,
        ground_truth_dilated
    ).sum()


    false_positive = (
        predicted.sum()
        - true_positive
    )


    true_positive_recall = np.logical_and(
        ground_truth,
        predicted_dilated
    ).sum()


    false_negative = (
        ground_truth.sum()
        - true_positive_recall
    )


    # ------------------------------------------------------
    # Precision
    # ------------------------------------------------------

    if true_positive + false_positive == 0:

        precision = 0.0

    else:

        precision = (
            true_positive
            /
            (true_positive + false_positive)
        )


    # ------------------------------------------------------
    # Recall
    # ------------------------------------------------------

    if true_positive_recall + false_negative == 0:

        recall = 0.0

    else:

        recall = (
            true_positive_recall
            /
            (true_positive_recall + false_negative)
        )


    # ------------------------------------------------------
    # F1 Score
    # ------------------------------------------------------

    if precision + recall == 0:

        f1_score = 0.0

    else:

        f1_score = (
            2
            * precision
            * recall
            /
            (precision + recall)
        )


    return precision, recall, f1_score


# ==========================================================
# GET TEST IMAGES
# ==========================================================

def get_test_images():

    files = []

    for filename in os.listdir(IMAGE_FOLDER):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            files.append(filename)

    files.sort()

    return files[:NUMBER_OF_IMAGES]


# ==========================================================
# EVALUATE ONE IMAGE
# ==========================================================

def evaluate_image(image_filename):

    image_name = os.path.splitext(
        image_filename
    )[0]


    image_path = os.path.join(
        IMAGE_FOLDER,
        image_filename
    )


    ground_truth_path = os.path.join(
        GROUND_TRUTH_FOLDER,
        image_name + ".mat"
    )


    if not os.path.exists(
        ground_truth_path
    ):

        print(
            f"Skipping {image_filename}: "
            f"ground truth not found."
        )

        return None


    # ------------------------------------------------------
    # Load image
    # ------------------------------------------------------

    image, gray, blurred = preprocess_image(
        image_path
    )


    # ------------------------------------------------------
    # Load ground truth
    # ------------------------------------------------------

    ground_truth = load_ground_truth(
        ground_truth_path
    )


    if gray.shape != ground_truth.shape:

        print(
            f"Skipping {image_filename}: "
            f"dimension mismatch."
        )

        return None


    # ======================================================
    # SOBEL
    # ======================================================

    start_time = time.perf_counter()

    sobel = sobel_edge_detection(
        blurred
    )

    sobel_time = (
        time.perf_counter()
        - start_time
    )


    sobel_binary = create_binary_edge_map(
        sobel,
        SOBEL_THRESHOLD
    )


    sobel_precision, sobel_recall, sobel_f1 = (
        calculate_metrics(
            sobel_binary,
            ground_truth,
            TOLERANCE
        )
    )


    # ======================================================
    # PREWITT
    # ======================================================

    start_time = time.perf_counter()

    prewitt = prewitt_edge_detection(
        blurred
    )

    prewitt_time = (
        time.perf_counter()
        - start_time
    )


    prewitt_binary = create_binary_edge_map(
        prewitt,
        PREWITT_THRESHOLD
    )


    prewitt_precision, prewitt_recall, prewitt_f1 = (
        calculate_metrics(
            prewitt_binary,
            ground_truth,
            TOLERANCE
        )
    )


    # ======================================================
    # CANNY
    # ======================================================

    start_time = time.perf_counter()

    canny = canny_edge_detection(
        blurred,
        CANNY_THRESHOLD1,
        CANNY_THRESHOLD2
    )

    canny_time = (
        time.perf_counter()
        - start_time
    )


    canny_binary = (
        canny > 0
    ).astype(np.uint8)


    canny_precision, canny_recall, canny_f1 = (
        calculate_metrics(
            canny_binary,
            ground_truth,
            TOLERANCE
        )
    )


    # ======================================================
    # RETURN RESULTS
    # ======================================================

    return {

        "image": image_name,

        "sobel_precision": sobel_precision,
        "sobel_recall": sobel_recall,
        "sobel_f1": sobel_f1,
        "sobel_time": sobel_time * 1000,

        "prewitt_precision": prewitt_precision,
        "prewitt_recall": prewitt_recall,
        "prewitt_f1": prewitt_f1,
        "prewitt_time": prewitt_time * 1000,

        "canny_precision": canny_precision,
        "canny_recall": canny_recall,
        "canny_f1": canny_f1,
        "canny_time": canny_time * 1000
    }


# ==========================================================
# SAVE CSV
# ==========================================================

def save_csv(results):

    filename = "evaluation_results.csv"

    with open(
        filename,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Image",

            "Sobel Precision",
            "Sobel Recall",
            "Sobel F1",
            "Sobel Time (ms)",

            "Prewitt Precision",
            "Prewitt Recall",
            "Prewitt F1",
            "Prewitt Time (ms)",

            "Canny Precision",
            "Canny Recall",
            "Canny F1",
            "Canny Time (ms)"
        ])


        for result in results:

            writer.writerow([

                result["image"],

                f"{result['sobel_precision']:.4f}",
                f"{result['sobel_recall']:.4f}",
                f"{result['sobel_f1']:.4f}",
                f"{result['sobel_time']:.3f}",

                f"{result['prewitt_precision']:.4f}",
                f"{result['prewitt_recall']:.4f}",
                f"{result['prewitt_f1']:.4f}",
                f"{result['prewitt_time']:.3f}",

                f"{result['canny_precision']:.4f}",
                f"{result['canny_recall']:.4f}",
                f"{result['canny_f1']:.4f}",
                f"{result['canny_time']:.3f}"
            ])


    print()
    print(
        f"Results saved to: {filename}"
    )


# ==========================================================
# PRINT AVERAGES
# ==========================================================

def print_averages(results):

    if len(results) == 0:

        print("No results available.")

        return


    # ------------------------------------------------------
    # Calculate averages
    # ------------------------------------------------------

    sobel_precision = np.mean([
        r["sobel_precision"]
        for r in results
    ])

    sobel_recall = np.mean([
        r["sobel_recall"]
        for r in results
    ])

    sobel_f1 = np.mean([
        r["sobel_f1"]
        for r in results
    ])

    sobel_time = np.mean([
        r["sobel_time"]
        for r in results
    ])


    prewitt_precision = np.mean([
        r["prewitt_precision"]
        for r in results
    ])

    prewitt_recall = np.mean([
        r["prewitt_recall"]
        for r in results
    ])

    prewitt_f1 = np.mean([
        r["prewitt_f1"]
        for r in results
    ])

    prewitt_time = np.mean([
        r["prewitt_time"]
        for r in results
    ])


    canny_precision = np.mean([
        r["canny_precision"]
        for r in results
    ])

    canny_recall = np.mean([
        r["canny_recall"]
        for r in results
    ])

    canny_f1 = np.mean([
        r["canny_f1"]
        for r in results
    ])

    canny_time = np.mean([
        r["canny_time"]
        for r in results
    ])


    # ------------------------------------------------------
    # Display
    # ------------------------------------------------------

    print()
    print("=" * 80)
    print("AVERAGE BSDS500 RESULTS")
    print("=" * 80)

    print()

    print(
        f"{'Method':<12}"
        f"{'Precision':<15}"
        f"{'Recall':<15}"
        f"{'F1-Score':<15}"
        f"{'Time (ms)':<15}"
    )

    print("-" * 80)


    print(
        f"{'Sobel':<12}"
        f"{sobel_precision:<15.4f}"
        f"{sobel_recall:<15.4f}"
        f"{sobel_f1:<15.4f}"
        f"{sobel_time:<15.3f}"
    )


    print(
        f"{'Prewitt':<12}"
        f"{prewitt_precision:<15.4f}"
        f"{prewitt_recall:<15.4f}"
        f"{prewitt_f1:<15.4f}"
        f"{prewitt_time:<15.3f}"
    )


    print(
        f"{'Canny':<12}"
        f"{canny_precision:<15.4f}"
        f"{canny_recall:<15.4f}"
        f"{canny_f1:<15.4f}"
        f"{canny_time:<15.3f}"
    )


    print("-" * 80)

    print()
    print(
        f"Images evaluated: {len(results)}"
    )


# ==========================================================
# MAIN
# ==========================================================

def evaluate():

    print()
    print("=" * 80)
    print("BSDS500 EDGE DETECTION EVALUATION")
    print("=" * 80)

    print()
    print(
        f"Images to evaluate: {NUMBER_OF_IMAGES}"
    )

    print(
        f"Sobel threshold: {SOBEL_THRESHOLD}"
    )

    print(
        f"Prewitt threshold: {PREWITT_THRESHOLD}"
    )

    print(
        f"Canny thresholds: "
        f"{CANNY_THRESHOLD1}, {CANNY_THRESHOLD2}"
    )

    print(
        f"Matching tolerance: {TOLERANCE} pixel(s)"
    )


    test_images = get_test_images()


    print()
    print(
        f"Found {len(test_images)} test images."
    )


    results = []


    for index, image_filename in enumerate(
        test_images,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(test_images)}] "
            f"Evaluating {image_filename}..."
        )


        result = evaluate_image(
            image_filename
        )


        if result is not None:

            results.append(result)

            print(
                f"    Sobel F1: "
                f"{result['sobel_f1']:.4f}"
            )

            print(
                f"    Prewitt F1: "
                f"{result['prewitt_f1']:.4f}"
            )

            print(
                f"    Canny F1: "
                f"{result['canny_f1']:.4f}"
            )


    # ------------------------------------------------------
    # Final Results
    # ------------------------------------------------------

    print_averages(
        results
    )


    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------

    if results:

        save_csv(
            results
        )


    print()
    print(
        "Evaluation completed successfully."
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    evaluate()