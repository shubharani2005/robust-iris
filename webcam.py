"""Webcam capture helpers: eye detection, crop, and quality checks."""

import cv2
import numpy as np

_EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

MIN_BLUR_VARIANCE = 80.0
MIN_BRIGHTNESS = 45.0
MAX_BRIGHTNESS = 210.0
MIN_SIDE_PX = 80


def apply_digital_zoom(
    bgr: np.ndarray,
    zoom: float,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> np.ndarray:
    """
    Center-crop and resize to simulate camera zoom.
    zoom=1.0 no change, zoom=2.0 uses central 50% of the frame, etc.
    """
    if bgr is None or bgr.size == 0:
        return bgr
    if zoom <= 1.0:
        return bgr.copy()

    h, w = bgr.shape[:2]
    crop_w = max(1, int(w / zoom))
    crop_h = max(1, int(h / zoom))

    cx = int(center_x * w)
    cy = int(center_y * h)
    x1 = max(0, min(cx - crop_w // 2, w - crop_w))
    y1 = max(0, min(cy - crop_h // 2, h - crop_h))
    x2, y2 = x1 + crop_w, y1 + crop_h

    cropped = bgr[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)


def draw_crop_overlay(
    bgr: np.ndarray,
    zoom: float,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> np.ndarray:
    """Show the crop rectangle that will be magnified."""
    out = bgr.copy()
    if zoom <= 1.0:
        return out

    h, w = out.shape[:2]
    crop_w = max(1, int(w / zoom))
    crop_h = max(1, int(h / zoom))
    cx = int(center_x * w)
    cy = int(center_y * h)
    x1 = max(0, min(cx - crop_w // 2, w - crop_w))
    y1 = max(0, min(cy - crop_h // 2, h - crop_h))
    cv2.rectangle(out, (x1, y1), (x1 + crop_w, y1 + crop_h), (0, 200, 255), 2)
    cv2.putText(
        out, f"Zoom {zoom:.1f}x area", (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
    )
    return out


def draw_zoom_guides(bgr: np.ndarray, zoom: float) -> np.ndarray:
    """Draw a target circle on the live frame so the user knows where to place the eye."""
    out = draw_crop_overlay(bgr, zoom, 0.5, 0.5)
    h, w = out.shape[:2]
    radius = int(min(h, w) * 0.22 / max(zoom, 1.0))
    cv2.circle(out, (w // 2, h // 2), radius, (0, 255, 0), 2)
    cv2.putText(
        out, "Center your eye here", (max(10, w // 2 - 90), max(30, h // 2 - radius - 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
    )
    return out


def assess_image_quality(bgr: np.ndarray) -> dict:
    """Check blur, brightness, and minimum size for reliable prediction."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    h, w = gray.shape[:2]

    messages: list[str] = []
    ok = True

    if blur < MIN_BLUR_VARIANCE:
        ok = False
        messages.append("Image is blurry — hold the camera steady and tap capture again.")
    if brightness < MIN_BRIGHTNESS:
        ok = False
        messages.append("Too dark — face a light source or increase room lighting.")
    if brightness > MAX_BRIGHTNESS:
        ok = False
        messages.append("Too bright — reduce glare or move away from direct light.")
    if min(h, w) < MIN_SIDE_PX:
        ok = False
        messages.append("Eye region too small — move closer to the camera.")

    return {
        "ok": ok,
        "blur": blur,
        "brightness": brightness,
        "width": w,
        "height": h,
        "messages": messages,
    }


def extract_eye_region(
    bgr: np.ndarray,
    padding: float = 2.0,
    *,
    scale_factor: float = 1.08,
    min_neighbors: int = 6,
    min_size: tuple[int, int] = (40, 40),
    enhance: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Detect the largest eye in a webcam frame and crop a square ROI.
    Falls back to the full frame if no eye is found.
    """
    meta: dict = {
        "eye_detected": False,
        "bbox": None,
        "used_full_frame": True,
        "message": "",
    }

    if bgr is None or bgr.size == 0:
        meta["message"] = "Empty frame."
        return bgr, meta

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if enhance:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    eyes = _EYE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    if len(eyes) == 0:
        meta["message"] = "No eye detected — using full frame. Center your left eye in view."
        return bgr, meta

    x, y, w, h = max(eyes, key=lambda e: e[2] * e[3])
    meta["eye_detected"] = True
    meta["bbox"] = (int(x), int(y), int(w), int(h))
    meta["used_full_frame"] = False

    cx, cy = x + w // 2, y + h // 2
    half = int(max(w, h) * padding)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(bgr.shape[1], cx + half)
    y2 = min(bgr.shape[0], cy + half)
    crop = bgr[y1:y2, x1:x2].copy()
    meta["message"] = "Eye region detected and cropped for analysis."
    return crop, meta


def draw_eye_bbox(bgr: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    """Draw detected eye rectangle on the webcam frame for user feedback."""
    out = bgr.copy()
    if bbox is None:
        return out
    x, y, w, h = bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        out, "Eye detected", (x, max(y - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
    )
    return out


def prepare_webcam_frame(bgr: np.ndarray, zoom: float = 1.0) -> dict:
    """Full webcam pipeline: detect eye, crop, assess quality."""
    annotated, eye_meta = extract_eye_region(bgr)
    quality = assess_image_quality(annotated)
    preview = draw_eye_bbox(bgr, eye_meta.get("bbox"))
    return {
        "analysis_image": annotated,
        "preview": preview,
        "eye_meta": eye_meta,
        "quality": quality,
        "zoom": zoom,
    }


def prepare_webcam_from_capture(
    raw_bgr: np.ndarray,
    zoom: float,
    pan_x: float,
    pan_y: float,
) -> dict:
    """Apply user zoom/pan, then run eye detection and quality checks."""
    zoomed = apply_digital_zoom(raw_bgr, zoom, pan_x, pan_y)
    result = prepare_webcam_frame(zoomed, zoom=zoom)
    result["raw_preview"] = draw_crop_overlay(raw_bgr, zoom, pan_x, pan_y)
    result["zoomed_preview"] = zoomed
    result["pan"] = (pan_x, pan_y)
    return result
