"""Debug predictions per class — full image vs iris ROI."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import load_class_names, load_trained_model
from src.preprocessing import normalize_for_model
from src.segmentation import segmentation_pipeline
from src.preprocessing import preprocess_pipeline

DATA = ROOT / "data" / "cataract_external"


def predict_tensor(model, tensor):
    probs = model.predict(np.expand_dims(tensor, 0), verbose=0)[0]
    names = load_class_names()
    idx = int(np.argmax(probs))
    return names[idx], probs


def main():
    model = load_trained_model()
    names = load_class_names()
    print("Classes:", names)

    for cls in names:
        folder = DATA / cls
        imgs = list(folder.glob("*.jpg"))[:3]
        print(f"\n=== {cls} ===")
        for img_path in imgs:
            bgr = cv2.imread(str(img_path))
            pre = preprocess_pipeline(bgr)
            seg = segmentation_pipeline(pre["enhanced"])

            full_t = normalize_for_model(bgr)
            pred_full, probs_full = predict_tensor(model, full_t)

            if seg["iris_roi"] is not None:
                roi = cv2.cvtColor(seg["iris_roi"], cv2.COLOR_GRAY2BGR)
                roi_t = normalize_for_model(roi)
                pred_roi, probs_roi = predict_tensor(model, roi_t)
                roi_shape = seg["iris_roi"].shape
            else:
                pred_roi, probs_roi = "N/A", None
                roi_shape = None

            print(f"  {img_path.name}:")
            print(f"    full image -> {pred_full} {probs_full.round(3)}")
            print(f"    iris ROI {roi_shape} -> {pred_roi} {probs_roi.round(3) if probs_roi is not None else ''}")


if __name__ == "__main__":
    main()
