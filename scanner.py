import cv2
import numpy as np

MAX_SIDE = 2000


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order corners as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def resize_for_detection(image: np.ndarray, max_side: int = MAX_SIDE) -> tuple[np.ndarray, float]:
    """
    Resize image only for contour detection speed.
    Returns resized image and scale_to_original factor.
    """
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, 1.0 / scale
    return image.copy(), 1.0


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def is_reasonable_warp(image: np.ndarray, warped: np.ndarray) -> bool:
    """Reject obviously bad perspective transforms."""
    original_area = image.shape[0] * image.shape[1]
    warped_area = warped.shape[0] * warped.shape[1]
    if warped.shape[0] < 300 or warped.shape[1] < 300:
        return False
    if warped_area < original_area * 0.08:
        return False
    return True


def detect_document_contour(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    Find the largest 4-point contour likely to be a document.
    Returns 4x2 array or None.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    min_area = image_area * 0.2

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area >= min_area:
                return approx.reshape(4, 2).astype("float32")

    return None


def detect_document_by_brightness(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    Detect a document as the largest bright quadrilateral area.
    This is a strong fallback for white paper on darker desks.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Separate bright paper from darker background.
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    min_area = image_area * 0.15

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        if cv2.contourArea(box.astype(np.float32)) >= min_area:
            return box.astype("float32")

    return None


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
    """Run full pipeline and return processed scan image + debug metadata."""
    file_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image file")

    original_h, original_w = image.shape[:2]
    resized, scale_to_original = resize_for_detection(image)
    contour = detect_document_contour(resized)
    detector_used = "canny"

    if contour is None:
        contour = detect_document_by_brightness(resized)
        detector_used = "brightness" if contour is not None else "none"

    if contour is not None:
        contour = contour * scale_to_original
        candidate = four_point_transform(image, contour)
        if is_reasonable_warp(image, candidate):
            warped = candidate
            used_perspective = True
        else:
            warped = image
            used_perspective = False
            detector_used = "rejected_bad_warp"
    else:
        # Fallback: no contour found, continue with original image.
        warped = image
        used_perspective = False

    if scan_mode == "bw":
        scan = enhance_bw(warped)
    elif scan_mode == "clean_gray":
        scan = enhance_clean_gray(warped)
    else:
        scan = enhance_color_scan(warped)

    meta = {
        "used_perspective_correction": used_perspective,
        "detector_used": detector_used,
        "scan_mode": scan_mode,
        "original_size": [int(original_w), int(original_h)],
        "result_size": [int(scan.shape[1]), int(scan.shape[0])],
    }
    return scan, meta
