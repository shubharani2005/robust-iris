"""Project configuration for iris disease detection."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_MODEL_TYPE = "mobilenet"

# Two separate model pathways, because they need genuinely different input
# photos and detect genuinely different conditions — a single external-eye
# photo has no diabetic-retinopathy or chronic-glaucoma signal in it at all
# (those are retina / optic-nerve findings), so folding them into one model
# just reproduces confident-but-meaningless predictions on whichever photo
# type it wasn't trained on.
MODEL_PROFILES = {
    "external_eye": {
        "label": "External Eye Photo",
        "description": (
            "A regular close-up photo of the eye (phone or webcam). Detects cataract "
            "only — diabetic retinopathy and chronic glaucoma aren't visible from "
            "outside the eye, so they aren't offered in this mode."
        ),
        "dir": MODELS_DIR / "cataract_external",
        "default_classes": ["cataract", "normal"],
        "default_data_dir": DATA_DIR / "cataract_external",
    },
    "fundus": {
        "label": "Fundus / Retinal Photo",
        "description": (
            "A retinal photo taken with a fundus camera (specialized clinical "
            "equipment — not a phone or webcam). Detects cataract, diabetic "
            "retinopathy, glaucoma, and normal."
        ),
        "dir": MODELS_DIR / "fundus_4class",
        "default_classes": ["cataract", "diabetic_retinopathy", "glaucoma", "normal"],
        "default_data_dir": DATA_DIR / "eye_diseases",
    },
}
DEFAULT_MODEL_PROFILE = "external_eye"


def weights_path(profile: str) -> Path:
    return MODEL_PROFILES[profile]["dir"] / "best_weights.weights.h5"


def model_meta_path(profile: str) -> Path:
    return MODEL_PROFILES[profile]["dir"] / "model_meta.json"


def class_names_path(profile: str) -> Path:
    return MODEL_PROFILES[profile]["dir"] / "class_names.txt"


def training_metrics_path(profile: str) -> Path:
    return MODEL_PROFILES[profile]["dir"] / "training_metrics.json"


# Image settings (aligned with MobileNetV2 + CNN paper)
IMG_SIZE = (224, 224)
HOUGH_MIN_DIST = 50
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 30
KMEANS_CLUSTERS = 3

# Training defaults
BATCH_SIZE = 32
EPOCHS = 20
FINE_TUNE_EPOCHS = 10
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
