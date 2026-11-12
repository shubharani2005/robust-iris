"""Iris detection and segmentation using Hough circles and K-Means."""

import cv2
import numpy as np
from sklearn.cluster import KMeans

from config import HOUGH_MIN_DIST, HOUGH_PARAM1, HOUGH_PARAM2, KMEANS_CLUSTERS


def detect_iris_hough(gray_image: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
    """
    Detect iris region using Hough Circle Transform (OpenCV).
    Returns annotated image and (x, y, radius) if found.
    """
    blurred = cv2.medianBlur(gray_image, 5)
    h, w = blurred.shape[:2]
    min_radius = max(10, min(h, w) // 8)
    max_radius = min(h, w) // 2

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    output = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    if circles is None:
        return None, None

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]
    cv2.circle(output, (x, y), r, (0, 255, 0), 2)
    cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
    return output, (int(x), int(y), int(r))


def extract_iris_roi(
    gray_image: np.ndarray, circle: tuple[int, int, int]
) -> np.ndarray:
    """Mask and crop iris region from detected circle."""
    x, y, r = circle
    mask = np.zeros_like(gray_image)
    cv2.circle(mask, (x, y), r, 255, -1)
    masked = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    x1, y1 = max(x - r, 0), max(y - r, 0)
    x2, y2 = min(x + r, gray_image.shape[1]), min(y + r, gray_image.shape[0])
    return masked[y1:y2, x1:x2]


def segment_kmeans(gray_image: np.ndarray, n_clusters: int = KMEANS_CLUSTERS) -> np.ndarray:
    """
    Segment iris image using K-Means clustering.
    Returns a color-coded segmentation map.
    """
    if gray_image.size == 0:
        raise ValueError("Empty image for K-Means segmentation.")

    pixels = gray_image.reshape(-1, 1).astype(np.float32)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)

    centers = kmeans.cluster_centers_.flatten()
    order = np.argsort(centers)
    label_map = np.zeros_like(labels)
    for idx, label in enumerate(order):
        label_map[labels == label] = idx

    segmented = label_map.reshape(gray_image.shape)
    colors = np.array(
        [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 0, 255]],
        dtype=np.uint8,
    )
    colored = colors[segmented % len(colors)]
    return colored


def segmentation_pipeline(enhanced_gray: np.ndarray) -> dict[str, np.ndarray | tuple | None]:
    """Run iris detection and K-Means segmentation."""
    detected, circle = detect_iris_hough(enhanced_gray)
    result: dict[str, np.ndarray | tuple | None] = {
        "detected_iris": detected,
        "circle": circle,
        "iris_roi": None,
        "segmented": None,
    }

    if circle is not None:
        roi = extract_iris_roi(enhanced_gray, circle)
        if roi.size > 0:
            result["iris_roi"] = roi
            result["segmented"] = segment_kmeans(roi)

    return result
