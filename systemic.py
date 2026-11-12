"""Systemic disease screening via iridology (Punia et al. approach)."""

import json
from pathlib import Path

import cv2
import numpy as np

from config import MODELS_DIR
from src.iridology.features import create_zone_mask, draw_zone_overlay, extract_zone_features
from src.iridology.zones import SYSTEMIC_ZONES, IrisZone
from src.segmentation import detect_iris_hough

THRESHOLDS_PATH = MODELS_DIR / "systemic_thresholds.json"

# Defaults from Punia et al. white-ratio thresholds (scaled to 0–1 feature space)
DEFAULT_THRESHOLDS = {
    "heart_disease": {"white_ratio": 0.42, "cv_max": 0.55, "snr_min": 2.0},
    "diabetes": {"white_ratio": 0.48, "cv_max": 0.58, "snr_min": 1.8},
    "cholesterol": {"white_ratio": 0.52, "cv_max": 0.60, "snr_min": 1.6},
    "lung_disease": {"white_ratio": 0.46, "cv_max": 0.57, "snr_min": 1.7},
}


def load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        return json.loads(THRESHOLDS_PATH.read_text())
    return DEFAULT_THRESHOLDS.copy()


def _risk_score(features: dict, thresholds: dict) -> tuple[str, float, list[str]]:
    """
    Classify zone as Normal or At Risk using Punia rule:
    higher white ratio in organ zone suggests abnormality.
    """
    if not features.get("valid"):
        return "Inconclusive", 0.0, ["Insufficient pixels in organ zone"]

    white = features["white_ratio"]
    cv = features["coefficient_variation"]
    snr = features["snr"]

    t_white = thresholds["white_ratio"]
    t_cv = thresholds.get("cv_max", 0.6)
    t_snr = thresholds.get("snr_min", 1.5)

    flags = []
    score = 0.0

    if white > t_white:
        flags.append(f"Elevated zone density ({white:.0%} vs threshold {t_white:.0%})")
        score += 0.45
    if cv > t_cv:
        flags.append(f"High texture variation (CV={cv:.2f})")
        score += 0.30
    if snr < t_snr:
        flags.append(f"Low signal quality (SNR={snr:.2f})")
        score += 0.25

    score = min(1.0, score)
    if score >= 0.45:
        return "At Risk", score, flags
    if score >= 0.25:
        return "Borderline", score, flags
    return "Normal", score, flags


def analyze_zone(gray: np.ndarray, center: tuple, radius: int, zone: IrisZone) -> dict:
    thresholds = load_thresholds().get(zone.condition, DEFAULT_THRESHOLDS[zone.condition])
    mask = create_zone_mask(gray.shape, center, radius, zone)
    features = extract_zone_features(gray, mask)
    status, risk, flags = _risk_score(features, thresholds)
    overlay = draw_zone_overlay(gray, center, radius, zone)

    return {
        "zone": zone.name,
        "label": zone.label,
        "condition": zone.condition,
        "status": status,
        "risk_score": risk,
        "flags": flags,
        "features": features,
        "overlay": overlay,
        "mask": mask,
    }


def analyze_systemic_conditions(gray_image: np.ndarray) -> dict:
    """
    Run iridology analysis for heart, diabetes, cholesterol, and lung conditions.
    Requires a detected iris circle (left iris preferred per Punia et al.).
    """
    detected, circle = detect_iris_hough(gray_image)
    results = {
        "iris_detected": circle is not None,
        "detected_image": detected,
        "circle": circle,
        "conditions": [],
        "zone_results": {},
    }

    if circle is None:
        results["error"] = "Iris not detected — systemic iridology analysis needs a clear iris image."
        return results

    cx, cy, r = circle
    center = (cx, cy)

    for key, zone in SYSTEMIC_ZONES.items():
        zone_result = analyze_zone(gray_image, center, r, zone)
        results["zone_results"][key] = zone_result
        results["conditions"].append(
            {
                "condition": zone.condition,
                "label": zone.label,
                "status": zone_result["status"],
                "risk_score": zone_result["risk_score"],
                "flags": zone_result["flags"],
            }
        )

    return results


def analyze_systemic_from_bgr(bgr_image: np.ndarray) -> dict:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    return analyze_systemic_conditions(gray)
