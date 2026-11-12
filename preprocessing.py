"""Image preprocessing for iris disease detection."""

import cv2
import numpy as np


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to BGR numpy array."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Upload a valid JPG or PNG file.")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (k, k), 0)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_for_model(image: np.ndarray) -> np.ndarray:
    """Prepare RGB image tensor in [0, 1] for Keras models."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = resize_image(rgb, size=(224, 224))
    return rgb.astype(np.float32) / 255.0


def preprocess_pipeline(image: np.ndarray) -> dict[str, np.ndarray]:
    """
    Full preprocessing chain from the project proposal:
    resize, grayscale, denoise, contrast enhancement.
    """
    resized = resize_image(image, (512, 512))
    gray = to_grayscale(resized)
    blurred = apply_gaussian_blur(gray)
    enhanced = enhance_contrast(blurred)
    return {
        "original": resized,
        "grayscale": gray,
        "blurred": blurred,
        "enhanced": enhanced,
    }
