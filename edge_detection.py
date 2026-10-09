import cv2
import numpy as np


def preprocess_image(image_path):
    """
    Load an image, convert it to grayscale,
    and apply Gaussian blur.
    """

    # Load image
    image = cv2.imread(image_path)

    # Validate image
    if image is None:
        raise ValueError("Unable to load image.")

    # Convert BGR image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return image, gray, blurred


def sobel_edge_detection(gray):
    """
    Detect edges using the Sobel operator.
    """

    # Horizontal gradient
    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    # Vertical gradient
    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    # Calculate gradient magnitude
    magnitude = np.sqrt(
        sobel_x ** 2 +
        sobel_y ** 2
    )

    # Normalize to 0-255
    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Convert to 8-bit image
    magnitude = magnitude.astype(np.uint8)

    return magnitude


def prewitt_edge_detection(gray):
    """
    Detect edges using the Prewitt operator.
    """

    # Horizontal Prewitt kernel
    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    # Vertical Prewitt kernel
    kernel_y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ], dtype=np.float32)

    # Apply horizontal kernel
    prewitt_x = cv2.filter2D(
        gray,
        cv2.CV_32F,
        kernel_x
    )

    # Apply vertical kernel
    prewitt_y = cv2.filter2D(
        gray,
        cv2.CV_32F,
        kernel_y
    )

    # Calculate gradient magnitude
    magnitude = np.sqrt(
        prewitt_x ** 2 +
        prewitt_y ** 2
    )

    # Normalize to 0-255
    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Convert to 8-bit image
    magnitude = magnitude.astype(np.uint8)

    return magnitude


def canny_edge_detection(gray, threshold1=50, threshold2=150):
    """
    Detect edges using the Canny edge detector.
    """

    edges = cv2.Canny(
        gray,
        threshold1,
        threshold2
    )

    return edges


def process_image(image_path):
    """
    Process an image using Sobel, Prewitt, and Canny.

    Returns:
        image: Original image
        gray: Grayscale image
        sobel: Sobel edge image
        prewitt: Prewitt edge image
        canny: Canny edge image
    """

    # Preprocess image
    image, gray, blurred = preprocess_image(image_path)

    # Apply edge detection algorithms
    sobel = sobel_edge_detection(blurred)

    prewitt = prewitt_edge_detection(blurred)

    canny = canny_edge_detection(blurred)

    return image, gray, sobel, prewitt, canny