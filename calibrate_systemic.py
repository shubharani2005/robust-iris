"""Calibrate systemic iridology thresholds from normal iris images."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import DATA_DIR, MODELS_DIR  # noqa: E402
from src.iridology.features import create_zone_mask, extract_zone_features  # noqa: E402
from src.iridology.zones import SYSTEMIC_ZONES  # noqa: E402
from src.preprocessing import preprocess_pipeline  # noqa: E402
from src.segmentation import detect_iris_hough  # noqa: E402

THRESHOLDS_PATH = MODELS_DIR / "systemic_thresholds.json"


def collect_zone_stats(data_path: Path, class_folder: str, max_images: int = 80):
    folder = data_path / class_folder
    if not folder.exists():
        return {}

    stats = {k: [] for k in SYSTEMIC_ZONES}
    count = 0
    for img_path in folder.iterdir():
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        gray = preprocess_pipeline(bgr)["enhanced"]
        _, circle = detect_iris_hough(gray)
        if circle is None:
            continue
        cx, cy, r = circle
        for key, zone in SYSTEMIC_ZONES.items():
            mask = create_zone_mask(gray.shape, (cx, cy), r, zone)
            feat = extract_zone_features(gray, mask)
            if feat["valid"]:
                stats[key].append(feat)
        count += 1
        if count >= max_images:
            break
    return stats


def main():
    data_path = DATA_DIR / "eye_disease_repo" / "dataset"
    if not data_path.exists():
        data_path = DATA_DIR / "eye_diseases"

    print(f"Calibrating from normal images in {data_path}...")
    normal_stats = collect_zone_stats(data_path, "normal")

    thresholds = {}
    for key, zone in SYSTEMIC_ZONES.items():
        feats = normal_stats.get(key, [])
        if not feats:
            print(f"  {key}: no data, using defaults")
            continue
        white_ratios = [f["white_ratio"] for f in feats]
        cvs = [f["coefficient_variation"] for f in feats]
        snrs = [f["snr"] for f in feats]
        thresholds[zone.condition] = {
            "white_ratio": float(min(0.92, np.percentile(white_ratios, 75) + 0.05)),
            "cv_max": float(min(0.85, np.percentile(cvs, 75) + 0.08)),
            "snr_min": float(max(1.0, np.percentile(snrs, 25) - 0.3)),
        }
        print(
            f"  {zone.label}: white_ratio={thresholds[zone.condition]['white_ratio']:.3f}, "
            f"cv_max={thresholds[zone.condition]['cv_max']:.3f}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2))
    print(f"Saved to {THRESHOLDS_PATH}")


if __name__ == "__main__":
    main()
