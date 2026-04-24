import cv2
import numpy as np

MAX_SIDE = 2000


def resize_for_detection(image: np.ndarray, max_side: int = MAX_SIDE) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, 1.0 / scale
    return image.copy(), 1.0


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def quad_metrics(pts: np.ndarray) -> tuple[float, float, float]:
    """Return area_ratio, min_side, max_side for 4-point polygon."""
    rect = order_points(pts.astype(np.float32))
    area = abs(cv2.contourArea(rect))
    side_lengths = np.array(
        [
            np.linalg.norm(rect[0] - rect[1]),
            np.linalg.norm(rect[1] - rect[2]),
            np.linalg.norm(rect[2] - rect[3]),
            np.linalg.norm(rect[3] - rect[0]),
        ],
        dtype=np.float32,
    )
    return area, float(np.min(side_lengths)), float(np.max(side_lengths))


def find_document_contour(image_bgr: np.ndarray) -> np.ndarray | None:
    """Find best 4-point page contour using edges + white-paper fallback."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    h, w = image_bgr.shape[:2]
    image_area = float(h * w)
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    best: np.ndarray | None = None
    best_score = -1e9

    def try_candidate(quad: np.ndarray) -> None:
        nonlocal best, best_score
        area, min_side, max_side = quad_metrics(quad)
        area_ratio = area / image_area
        if area_ratio < 0.18 or area_ratio > 0.97:
            return
        if min_side < min(h, w) * 0.18:
            return
        if max_side / max(min_side, 1.0) > 4.5:
            return

        quad_center = np.mean(quad, axis=0)
        center_penalty = (abs(float(quad_center[0] - center[0])) / w) + (
            abs(float(quad_center[1] - center[1])) / h
        )
        score = area_ratio - 0.35 * center_penalty
        if score > best_score:
            best_score = score
            best = quad.astype(np.float32)

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            try_candidate(approx.reshape(4, 2).astype(np.float32))

    # Fallback 1: rotated rectangle from edge contours.
    for contour in contours[:8]:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        try_candidate(box)

    # Fallback 2: brightest paper-like connected component.
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
        iterations=2,
    )
    b_contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    b_contours = sorted(b_contours, key=cv2.contourArea, reverse=True)[:8]
    for contour in b_contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            try_candidate(approx.reshape(4, 2).astype(np.float32))
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        try_candidate(box)

    return best


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        return image

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def crop_document_if_found(image_bgr: np.ndarray) -> tuple[np.ndarray, bool, str]:
    resized, scale_to_original = resize_for_detection(image_bgr)
    points = find_document_contour(resized)
    if points is None:
        return image_bgr, False, "none"

    points = points * scale_to_original
    warped = four_point_transform(image_bgr, points)
    if warped is image_bgr:
        return image_bgr, False, "invalid_warp"
    src_area = image_bgr.shape[0] * image_bgr.shape[1]
    dst_area = warped.shape[0] * warped.shape[1]
    if dst_area < src_area * 0.18:
        return image_bgr, False, "too_small_warp"
    return warped, True, "canny_contour"


def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 0.8) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Suppress page shadows/gradients before enhancement."""
    h, w = gray.shape[:2]
    kernel = max(31, (min(h, w) // 6) | 1)
    background = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    return cv2.divide(gray, background, scale=255)


def enhance_color_scan(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    norm = normalize_illumination(gray)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    l_chan = clahe.apply(norm)
    l_chan = cv2.normalize(l_chan, None, 0, 255, cv2.NORM_MINMAX)
    merged = cv2.merge([l_chan, a_chan, b_chan])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    enhanced = cv2.bilateralFilter(enhanced, d=7, sigmaColor=35, sigmaSpace=35)
    enhanced = unsharp_mask(enhanced, sigma=0.8, strength=0.95)

    # Light paper whitening without destroying text.
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    paper_mask = cv2.inRange(hsv, (0, 0, 120), (180, 80, 255)).astype(np.float32) / 255.0
    paper_mask = cv2.GaussianBlur(paper_mask, (9, 9), 0)[..., None]
    white_bg = np.full_like(enhanced, 247, dtype=np.float32)
    mixed = enhanced.astype(np.float32) * (1.0 - 0.22 * paper_mask) + white_bg * (0.22 * paper_mask)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def enhance_clean_gray(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(enhance_color_scan(image_bgr), cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=6, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return unsharp_mask(gray, sigma=0.7, strength=0.55)


def enhance_bw(image_bgr: np.ndarray) -> np.ndarray:
    clean_gray = enhance_clean_gray(image_bgr)
    return cv2.adaptiveThreshold(
        clean_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )


def process_document(
    image_bytes: bytes,
    scan_mode: str = "color",
    auto_crop: bool = False,
) -> tuple[np.ndarray, dict]:
    file_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file")

    original_h, original_w = image.shape[:2]
    working = image
    crop_applied = False
    crop_detector = "disabled"

    if auto_crop:
        working, crop_applied, crop_detector = crop_document_if_found(image)

    if scan_mode == "bw":
        scan = enhance_bw(working)
    elif scan_mode == "clean_gray":
        scan = enhance_clean_gray(working)
    else:
        scan = enhance_color_scan(working)

    meta = {
        "pipeline": "auto_crop_enhance" if auto_crop else "enhance_only",
        "scan_mode": scan_mode,
        "auto_crop_requested": auto_crop,
        "auto_crop_applied": crop_applied,
        "crop_detector": crop_detector,
        "original_size": [int(original_w), int(original_h)],
        "result_size": [int(scan.shape[1]), int(scan.shape[0])],
    }
    return scan, meta
