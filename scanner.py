import cv2
import numpy as np


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """
    Flatten paper lighting (shadows/gradients) via background estimation.
    """
    h, w = gray.shape[:2]
    kernel = _odd(max(31, min(h, w) // 6))
    background = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    normalized = cv2.divide(gray, background, scale=255)
    return normalized


def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def enhance_color_scan(image_bgr: np.ndarray) -> np.ndarray:
    """Color mode with cleaner paper and stronger legibility."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_chan = normalize_illumination(l_chan)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    l_chan = cv2.normalize(l_chan, None, 0, 255, cv2.NORM_MINMAX)
    enhanced_lab = cv2.merge([l_chan, a_chan, b_chan])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=30, sigmaSpace=30)
    return unsharp_mask(denoised, sigma=0.8, strength=0.6)


def enhance_clean_gray(image_bgr: np.ndarray) -> np.ndarray:
    """Natural grayscale scan with suppressed shadows."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = normalize_illumination(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


def enhance_bw(image_bgr: np.ndarray) -> np.ndarray:
    """High-contrast black/white mode for printing or OCR."""
    clean_gray = enhance_clean_gray(image_bgr)
    bw = cv2.adaptiveThreshold(
        clean_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        15,
    )
    kernel = np.ones((2, 2), dtype=np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    return bw


def process_document(image_bytes: bytes, scan_mode: str = "color") -> tuple[np.ndarray, dict]:
    """Enhance an image only, without contour detection/perspective correction."""
    file_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image file")

    original_h, original_w = image.shape[:2]

    if scan_mode == "bw":
        scan = enhance_bw(image)
    elif scan_mode == "clean_gray":
        scan = enhance_clean_gray(image)
    else:
        scan = enhance_color_scan(image)

    meta = {
        "pipeline": "enhance_only",
        "scan_mode": scan_mode,
        "original_size": [int(original_w), int(original_h)],
        "result_size": [int(scan.shape[1]), int(scan.shape[0])],
    }
    return scan, meta
