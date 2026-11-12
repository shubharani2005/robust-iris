"""Inference pipeline for iris disease detection."""

import json

import cv2
import numpy as np
from tensorflow import keras

from config import (
    DEFAULT_MODEL_PROFILE,
    DEFAULT_MODEL_TYPE,
    MODEL_PROFILES,
    class_names_path,
    model_meta_path,
    weights_path,
)
from src.models.build_models import get_model
from src.iridology.systemic import analyze_systemic_conditions
from src.preprocessing import normalize_for_model, preprocess_pipeline
from src.segmentation import segmentation_pipeline
from src.validation import is_external_eye, is_fundus


def _is_eye_photo(image_bgr: np.ndarray, profile: str) -> bool:
    """Reject non-eye uploads before running disease classification.

    external_eye: needs a large, confident Haar eye detection AND a
    dark-iris-inside-brighter-ring circle (Hough). Either alone gives false
    positives on faces, clothing and backgrounds.
    fundus: needs a strongly red-dominant circular retinal field.
    """
    if image_bgr is None or image_bgr.size == 0:
        return False
    if profile == "external_eye":
        ok, _ = is_external_eye(image_bgr)
    else:
        ok, _ = is_fundus(image_bgr)
    return ok


def load_class_names(profile: str = DEFAULT_MODEL_PROFILE) -> list[str]:
    path = class_names_path(profile)
    if path.exists():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return MODEL_PROFILES[profile]["default_classes"].copy()


def _get_model_type(profile: str) -> str:
    path = model_meta_path(profile)
    if path.exists():
        meta = json.loads(path.read_text())
        return meta.get("model_type", DEFAULT_MODEL_TYPE)
    return DEFAULT_MODEL_TYPE


def load_trained_model(profile: str = DEFAULT_MODEL_PROFILE) -> keras.Model | None:
    """
    Rebuild architecture and load weights for the given model profile
    ("external_eye" or "fundus" — see config.MODEL_PROFILES).
    Avoids corrupted .keras files (common when project is in OneDrive).
    """
    path = weights_path(profile)
    if not path.exists():
        return None

    class_names = load_class_names(profile)
    model = get_model(_get_model_type(profile), num_classes=len(class_names))
    model.load_weights(str(path))
    return model


def predict_disease(
    image_bgr: np.ndarray,
    model: keras.Model | None = None,
    profile: str = DEFAULT_MODEL_PROFILE,
) -> dict:
    """Run full analysis: preprocessing, segmentation, and classification."""
    model = model or load_trained_model(profile)
    class_names = load_class_names(profile)

    if not _is_eye_photo(image_bgr, profile):
        if profile == "fundus":
            message = "Invalid image: no retinal/fundus image was detected. Upload a clear fundus-camera photo and try again."
        else:
            message = "Invalid image: no eye was detected. Upload a clear external-eye photo and try again."
        return {"valid_image": False, "validation_error": message}

    preprocessed = preprocess_pipeline(image_bgr)
    seg = segmentation_pipeline(preprocessed["enhanced"])

    prediction = {
        "valid_image": True,
        "preprocessed": preprocessed,
        "segmentation": seg,
        "systemic": None,
        "class_name": None,
        "confidence": None,
        "all_probs": None,
        "model_loaded": model is not None,
    }

    # Iridology/systemic screening is based on external iris texture — it's
    # meaningless on a fundus photo (no iris is visible in one), so only run
    # it for the external-eye pathway.
    if profile == "external_eye":
        # Reuse the same enhanced image segmentation used above — running
        # Hough detection separately on the raw upload made systemic analysis
        # fail even when the iris was clearly detected in the preview above.
        prediction["systemic"] = analyze_systemic_conditions(preprocessed["enhanced"])

    if model is None:
        return prediction

    # Classify on full image — must match training (model was NOT trained on iris ROI crops).
    # Segmentation output is for visualization only.
    tensor = normalize_for_model(image_bgr)

    batch = np.expand_dims(tensor, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))

    prediction["valid_image"] = True
    prediction["class_name"] = class_names[idx] if idx < len(class_names) else f"class_{idx}"
    prediction["confidence"] = float(probs[idx])
    prediction["low_confidence"] = float(probs[idx]) < 0.55
    prediction["all_probs"] = {
        (class_names[i] if i < len(class_names) else f"class_{i}"): float(probs[i])
        for i in range(len(probs))
    }
    return prediction


def format_display_name(class_name: str) -> str:
    mapping = {
        "normal": "Normal",
        "cataract": "Cataract",
        "glaucoma": "Glaucoma",
        "diabetic_retinopathy": "Diabetic Retinopathy",
        "diabetic": "Diabetic Retinopathy",
    }
    return mapping.get(class_name.lower(), class_name.replace("_", " ").title())