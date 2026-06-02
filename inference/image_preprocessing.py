from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def read_image_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image with Unicode-safe path handling on Windows."""
    path = str(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def save_image_unicode(path: str | Path, image: np.ndarray) -> bool:
    """Write an image with Unicode-safe path handling on Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True


def decode_image_bytes(image_bytes: bytes, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, flags)


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        return np.clip(rgb * alpha + white * (1.0 - alpha), 0, 255).astype(np.uint8)
    return img[:, :, :3]


def _prepare_gray(img_or_path: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(img_or_path, (str, Path)):
        image = read_image_unicode(img_or_path, cv2.IMREAD_UNCHANGED)
    else:
        image = img_or_path

    if image is None:
        raise FileNotFoundError(f"无法读取图片：{img_or_path}")

    image = _ensure_bgr(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    max_side = max(h, w)
    if max_side > 1800:
        scale = 1800.0 / max_side
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return gray


def _estimate_dark_background(gray: np.ndarray) -> bool:
    h, w = gray.shape[:2]
    border = max(8, min(h, w) // 20)
    strips = [
        gray[:border, :],
        gray[-border:, :],
        gray[:, :border],
        gray[:, -border:],
    ]
    border_pixels = np.concatenate([s.reshape(-1) for s in strips])
    return float(border_pixels.mean()) < 127.0


def _normalize_lighting(gray: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    k = max(31, (min(gray.shape[:2]) // 12) | 1)
    background = cv2.GaussianBlur(gray, (k, k), 0)
    normalized = cv2.divide(gray, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(normalized)


def _remove_small_components(binary: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    min_area = max(16, int(binary.shape[0] * binary.shape[1] * 0.00004))
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def _crop_with_padding(binary: np.ndarray) -> np.ndarray:
    coords = cv2.findNonZero(binary)
    if coords is None:
        return binary

    x, y, w, h = cv2.boundingRect(coords)
    pad_x = max(8, int(w * 0.08))
    pad_y = max(8, int(h * 0.12))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(binary.shape[1], x + w + pad_x)
    y1 = min(binary.shape[0], y + h + pad_y)
    return binary[y0:y1, x0:x1]


def _thicken_if_needed(binary: np.ndarray) -> np.ndarray:
    foreground_ratio = float(np.count_nonzero(binary)) / float(binary.size)
    if foreground_ratio < 0.015:
        iterations = 2
    elif foreground_ratio < 0.035:
        iterations = 1
    else:
        iterations = 0

    if iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=iterations)
    return binary


def preprocess_formula_image(img_or_path: str | Path | np.ndarray) -> np.ndarray:
    """Convert real-world handwritten formula images to model-friendly white-on-black binary."""
    gray = _prepare_gray(img_or_path)
    dark_background = _estimate_dark_background(gray)

    if dark_background:
        smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.medianBlur(binary, 3)
    else:
        normalized = _normalize_lighting(gray)
        binary = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            35,
            15,
        )
        binary = cv2.medianBlur(binary, 3)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

    binary = _remove_small_components(binary)
    binary = _crop_with_padding(binary)
    binary = _thicken_if_needed(binary)

    if np.count_nonzero(binary) == 0:
        return np.zeros((64, 64), dtype=np.uint8)

    out_h = max(binary.shape[0] + 16, 64)
    out_w = max(binary.shape[1] + 16, 64)
    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    y = (out_h - binary.shape[0]) // 2
    x = (out_w - binary.shape[1]) // 2
    canvas[y:y + binary.shape[0], x:x + binary.shape[1]] = binary
    return canvas


def preprocess_and_save(
    img_or_path: str | Path | np.ndarray,
    save_path: str | Path,
) -> Tuple[np.ndarray, str]:
    processed = preprocess_formula_image(img_or_path)
    if not save_image_unicode(save_path, processed):
        raise OSError(f"无法保存图片：{save_path}")
    return processed, str(save_path)
