"""Feature extraction from iridology organ zones."""

import cv2
import numpy as np

from src.iridology.zones import IrisZone, hour_to_degrees


def _normalize_angle(deg: float) -> float:
    return deg % 360.0


def create_zone_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    iris_radius: int,
    zone: IrisZone,
) -> np.ndarray:
    """Build binary mask for an organ sector on the iris."""
    h, w = shape
    cx, cy = center
    mask = np.zeros((h, w), dtype=np.uint8)

    angle_start = _normalize_angle(hour_to_degrees(zone.hour_start))
    angle_end = _normalize_angle(hour_to_degrees(zone.hour_end))
    r_inner = max(1, int(iris_radius * zone.radius_inner))
    r_outer = max(r_inner + 1, int(iris_radius * zone.radius_outer))

    y_idx, x_idx = np.ogrid[:h, :w]
    dx = x_idx - cx
    dy = cy - y_idx
    pixel_r = np.hypot(dx, dy)
    pixel_deg = np.degrees(np.arctan2(dy, dx)) % 360.0

    ring = (pixel_r >= r_inner) & (pixel_r <= r_outer)
    if angle_start <= angle_end:
        sector = (pixel_deg >= angle_start) & (pixel_deg <= angle_end)
    else:
        sector = (pixel_deg >= angle_start) | (pixel_deg <= angle_end)

    mask[ring & sector] = 255
    return mask


def extract_zone_features(gray: np.ndarray, mask: np.ndarray, intensity_threshold: int = 128) -> dict:
    """
    Extract Punia-style features: white/black ratios, SNR, coefficient of variation.
    """
    pixels = gray[mask > 0]
    if pixels.size < 10:
        return {
            "pixel_count": int(pixels.size),
            "white_ratio": 0.0,
            "black_ratio": 0.0,
            "mean_intensity": 0.0,
            "std_intensity": 0.0,
            "coefficient_variation": 0.0,
            "snr": 0.0,
            "valid": False,
        }

    white = np.sum(pixels > intensity_threshold)
    black = pixels.size - white
    total = pixels.size
    mean = float(np.mean(pixels))
    std = float(np.std(pixels))
    cv = float(std / mean) if mean > 1e-6 else 0.0
    snr = float(mean / std) if std > 1e-6 else 0.0

    return {
        "pixel_count": int(total),
        "white_ratio": float(white / total),
        "black_ratio": float(black / total),
        "mean_intensity": mean,
        "std_intensity": std,
        "coefficient_variation": cv,
        "snr": snr,
        "valid": True,
    }


def draw_zone_overlay(
    gray: np.ndarray,
    center: tuple[int, int],
    iris_radius: int,
    zone: IrisZone,
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Draw organ zone sector on iris for visualization."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask = create_zone_mask(gray.shape, center, iris_radius, zone)
    overlay = bgr.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(bgr, 0.65, overlay, 0.35, 0)
