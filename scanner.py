import cv2
import numpy as np

def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def enhance_color_scan(image_bgr: np.ndarray) -> np.ndarray:
    """Color scan style: CLAHE on L channel + light sharpening."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    enhanced_lab = cv2.merge([l_chan, a_chan, b_chan])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return unsharp_mask(enhanced, sigma=0.8, strength=0.8)


def enhance_clean_gray(image_bgr: np.ndarray) -> np.ndarray:
    """Natural grayscale scan."""
    color = enhance_color_scan(image_bgr)
    return cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)


def enhance_bw(image_bgr: np.ndarray) -> np.ndarray:
    """High-contrast black/white mode for printing or OCR."""
    clean_gray = enhance_clean_gray(image_bgr)
    _, bw = cv2.threshold(clean_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
