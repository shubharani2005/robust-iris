"""Input validation: reject non-eye and non-fundus uploads before classification.

The classifier only knows its training classes (softmax always picks one), so
its confidence cannot tell us whether the photo is an eye at all. These
heuristics gate the input instead. Thresholds are constants near the top of each
function so they can be tuned against real data.
"""
import cv2
import numpy as np

_EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def _detect_eyes(gray):
    eyes = _EYE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
    return [tuple(map(int, e)) for e in eyes]


def _has_iris_structure(gray):
    """Dark pupil/iris disc surrounded by a brighter ring (sclera/skin)."""
    h, w = gray.shape
    scale = 256.0 / max(h, w)
    small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 5)
    sh, sw = small.shape
    circles = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1.2, minDist=sh // 4,
                               param1=100, param2=30,
                               minRadius=max(8, min(sh, sw) // 12), maxRadius=min(sh, sw) // 2)
    if circles is None:
        return False
    yy, xx = np.ogrid[:sh, :sw]
    for x, y, r in circles[0][:5]:
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        inner = small[d2 <= (0.5 * r) ** 2]
        ring = small[(d2 >= (1.15 * r) ** 2) & (d2 <= (1.6 * r) ** 2)]
        if inner.size < 20 or ring.size < 20:
            continue
        if float(inner.mean()) < 0.75 * float(ring.mean()) and float(inner.mean()) < 110:
            return True
    return False


def is_external_eye(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    eyes = _detect_eyes(gray)
    info = {"n_eyes": len(eyes)}
    if not eyes:
        return False, info
    frac = max(w * h for _, _, w, h in eyes) / gray.size
    info["frac"] = frac
    if frac < 0.02:
        return False, info
    info["iris"] = _has_iris_structure(gray)
    return info["iris"], info


def is_fundus(bgr):
    h, w = bgr.shape[:2]
    scale = 256.0 / max(h, w)
    small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    b, g, r = [c.astype(np.float32) for c in cv2.split(small)]
    yy, xx = np.ogrid[:sh, :sw]
    field = (xx - sw / 2) ** 2 + (yy - sh / 2) ** 2 <= (0.4 * min(sh, sw)) ** 2
    rm, gm, bm = r[field].mean(), g[field].mean(), b[field].mean()
    info = {"r_over_g": rm / (gm + 1), "r_over_b": rm / (bm + 1)}
    # retinal field: strongly red-dominant (skin is only ~1.2-1.4x)
    red_dominant = rm / (gm + 1) >= 1.6 and rm / (bm + 1) >= 2.2
    # circular field: corners are dark/black or clearly different from the centre
    corners = np.concatenate([small[:sh//8, :sw//8].ravel(), small[:sh//8, -sw//8:].ravel(),
                              small[-sh//8:, :sw//8].ravel(), small[-sh//8:, -sw//8:].ravel()])
    info["corner_mean"] = float(corners.mean())
    dark_border = corners.mean() < 45
    # vessel/texture: retinas have dark vessel structure inside the field
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    info["texture"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return bool(red_dominant and (dark_border or info["texture"] > 5)), info